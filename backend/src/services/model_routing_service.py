"""Evidence-aware forecast method selection.

The router decides which existing forecasting method should be attempted. It
does not generate forecasts and it deliberately keeps the legacy demand-pattern
policy as the fallback whenever evidence is weak, stale, incompatible, or too
small to matter.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from repositories.forecast_evaluation_repository import MethodPerformance

logger = logging.getLogger(__name__)


RUNTIME_METHODS = {"ml_lightgbm", "croston", "conservative", "simple_average"}
OFFLINE_METHOD_TO_RUNTIME = {
    "lightgbm": "ml_lightgbm",
    "croston_sba": "croston",
}


@dataclass(frozen=True)
class RoutingDecision:
    selected_method: str
    default_method: str
    selection_source: str
    evidence_level: str
    reason: str
    metric_name: str | None = None
    selected_metric_value: float | None = None
    baseline_metric_value: float | None = None
    evaluation_sample_size: int = 0
    evaluation_count: int = 0
    evidence_age_days: int | None = None
    fallback_used: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_method": self.selected_method,
            "default_method": self.default_method,
            "selection_source": self.selection_source,
            "evidence_level": self.evidence_level,
            "reason": self.reason,
            "metric_name": self.metric_name,
            "selected_metric_value": self.selected_metric_value,
            "baseline_metric_value": self.baseline_metric_value,
            "evaluation_sample_size": self.evaluation_sample_size,
            "evaluation_count": self.evaluation_count,
            "evidence_age_days": self.evidence_age_days,
            "fallback_used": self.fallback_used,
        }


class ModelRoutingService:
    """Select a forecast method from existing, demand-pattern-valid methods."""

    def __init__(
        self,
        *,
        settings: Any,
        forecast_evaluation_repository: Any | None = None,
        offline_evaluation_path: str | Path | None = None,
    ):
        self.settings = settings
        self.repository = forecast_evaluation_repository
        self.offline_evaluation_path = Path(offline_evaluation_path) if offline_evaluation_path else None

    def select_method(
        self,
        *,
        sku_code: str,
        demand_pattern: str,
        forecast_horizon: int,
        as_of_date: date | None = None,
    ) -> RoutingDecision:
        as_of_date = as_of_date or datetime.now(timezone.utc).date()
        default_method = self.default_method_for_pattern(demand_pattern)
        eligible_methods = self.eligible_methods_for_pattern(demand_pattern)

        if not self.settings.evidence_routing_enabled:
            return self._default_decision(
                default_method,
                "Evidence routing disabled; using the legacy demand-pattern policy.",
            )

        if default_method not in eligible_methods:
            return self._default_decision(default_method, "No valid default method for this demand pattern.")

        generated_after = datetime.combine(
            as_of_date - timedelta(days=self.settings.routing_evidence_lookback_days),
            time.min,
            tzinfo=timezone.utc,
        )
        sources = (
            ("logged", "sku", self._logged_sku_evidence(sku_code, forecast_horizon, generated_after)),
            ("logged", "pattern", self._logged_pattern_evidence(demand_pattern, forecast_horizon, generated_after)),
            ("offline", "sku", self._offline_sku_evidence(sku_code, demand_pattern, forecast_horizon, as_of_date)),
            ("offline", "pattern", self._offline_pattern_evidence(demand_pattern, forecast_horizon, as_of_date)),
        )

        for source, level, evidence in sources:
            decision = self._decision_from_evidence(
                evidence=evidence,
                eligible_methods=eligible_methods,
                default_method=default_method,
                source=source,
                level=level,
                as_of_date=as_of_date,
            )
            if decision is not None:
                return decision

        return self._default_decision(
            default_method,
            "No trustworthy matching evaluation evidence was available; using the legacy demand-pattern policy.",
        )

    @staticmethod
    def default_method_for_pattern(demand_pattern: str) -> str:
        if demand_pattern == "regular":
            return "ml_lightgbm"
        if demand_pattern == "intermittent":
            return "croston"
        return "conservative"

    @staticmethod
    def eligible_methods_for_pattern(demand_pattern: str) -> set[str]:
        if demand_pattern == "regular":
            return {"ml_lightgbm", "croston", "simple_average"}
        if demand_pattern == "intermittent":
            return {"croston", "simple_average"}
        return {"conservative"}

    def _decision_from_evidence(
        self,
        *,
        evidence: list[MethodPerformance],
        eligible_methods: set[str],
        default_method: str,
        source: str,
        level: str,
        as_of_date: date,
    ) -> RoutingDecision | None:
        usable = {
            row.method: row
            for row in evidence
            if row.method in eligible_methods
            and row.metric_value is not None
            and row.metric_value >= 0
            and row.sample_size >= self.settings.routing_min_evaluation_points
            and not self._is_stale(row.latest_generated_at, as_of_date)
        }
        default_perf = usable.get(default_method)
        if default_perf is None:
            return None

        best = min(usable.values(), key=lambda row: (row.metric_value, row.method != default_method, row.method))
        age_days = self._age_days(best.latest_generated_at, as_of_date)

        if best.method == default_method:
            return RoutingDecision(
                selected_method=default_method,
                default_method=default_method,
                selection_source=source,
                evidence_level=level,
                reason=(
                    f"Evaluation evidence for {level} {source} routing retained the default "
                    f"{default_method} method on {self.settings.routing_primary_metric}."
                ),
                metric_name=self.settings.routing_primary_metric,
                selected_metric_value=best.metric_value,
                baseline_metric_value=default_perf.metric_value,
                evaluation_sample_size=best.sample_size,
                evaluation_count=best.evaluation_count,
                evidence_age_days=age_days,
                fallback_used=False,
            )

        if default_perf.metric_value <= 0:
            return self._default_decision(
                default_method,
                "Default method evidence had a zero metric value, so relative improvement could not be trusted.",
            )

        relative_improvement = (default_perf.metric_value - best.metric_value) / default_perf.metric_value
        if relative_improvement < self.settings.routing_min_relative_improvement:
            return RoutingDecision(
                selected_method=default_method,
                default_method=default_method,
                selection_source=source,
                evidence_level=level,
                reason=(
                    f"{best.method} improved {self.settings.routing_primary_metric} by "
                    f"{relative_improvement:.1%}, below the configured "
                    f"{self.settings.routing_min_relative_improvement:.1%} threshold; "
                    "retaining the default method for stability."
                ),
                metric_name=self.settings.routing_primary_metric,
                selected_metric_value=default_perf.metric_value,
                baseline_metric_value=default_perf.metric_value,
                evaluation_sample_size=best.sample_size,
                evaluation_count=best.evaluation_count,
                evidence_age_days=age_days,
                fallback_used=False,
            )

        return RoutingDecision(
            selected_method=best.method,
            default_method=default_method,
            selection_source=source,
            evidence_level=level,
            reason=(
                f"{best.method} selected from {level} {source} evidence because it improved "
                f"{self.settings.routing_primary_metric} by {relative_improvement:.1%} "
                f"over default {default_method} with {best.sample_size} evaluated points."
            ),
            metric_name=self.settings.routing_primary_metric,
            selected_metric_value=best.metric_value,
            baseline_metric_value=default_perf.metric_value,
            evaluation_sample_size=best.sample_size,
            evaluation_count=best.evaluation_count,
            evidence_age_days=age_days,
            fallback_used=False,
        )

    def _logged_sku_evidence(
        self,
        sku_code: str,
        forecast_horizon: int,
        generated_after: datetime,
    ) -> list[MethodPerformance]:
        if self.repository is None:
            return []
        try:
            return self.repository.logged_method_performance_for_sku(
                sku_code=sku_code,
                horizon_days=forecast_horizon,
                metric_name=self.settings.routing_primary_metric,
                generated_after=generated_after,
            )
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning("Logged SKU routing evidence unavailable for %s: %s", sku_code, exc)
            return []

    def _logged_pattern_evidence(
        self,
        demand_pattern: str,
        forecast_horizon: int,
        generated_after: datetime,
    ) -> list[MethodPerformance]:
        if self.repository is None:
            return []
        try:
            return self.repository.logged_method_performance_for_pattern(
                demand_class=demand_pattern,
                horizon_days=forecast_horizon,
                metric_name=self.settings.routing_primary_metric,
                generated_after=generated_after,
            )
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning("Logged pattern routing evidence unavailable for %s: %s", demand_pattern, exc)
            return []

    def _offline_sku_evidence(
        self,
        sku_code: str,
        demand_pattern: str,
        forecast_horizon: int,
        as_of_date: date,
    ) -> list[MethodPerformance]:
        payload = self._offline_payload(forecast_horizon)
        if not payload or payload.get("horizon_days") != forecast_horizon:
            return []
        generated_at = self._parse_datetime(payload.get("generated_at"))
        if self._is_stale(generated_at, as_of_date):
            return []
        for row in payload.get("per_sku") or []:
            if row.get("sku") == sku_code and row.get("demand_class") == demand_pattern:
                return self._offline_metrics_to_performance(
                    metrics=row.get("metrics") or {},
                    horizon_days=forecast_horizon,
                    generated_at=generated_at,
                    evidence_level="sku",
                )
        return []

    def _offline_pattern_evidence(
        self,
        demand_pattern: str,
        forecast_horizon: int,
        as_of_date: date,
    ) -> list[MethodPerformance]:
        payload = self._offline_payload(forecast_horizon)
        if not payload or payload.get("horizon_days") != forecast_horizon:
            return []
        generated_at = self._parse_datetime(payload.get("generated_at"))
        if self._is_stale(generated_at, as_of_date):
            return []
        aggregate = (payload.get("aggregates") or {}).get(demand_pattern)
        if not aggregate:
            return []
        return self._offline_metrics_to_performance(
            metrics=aggregate,
            horizon_days=forecast_horizon,
            generated_at=generated_at,
            evidence_level="pattern",
        )

    def _offline_payload(self, forecast_horizon: int | None = None) -> dict[str, Any] | None:
        if self.offline_evaluation_path is None or not self.offline_evaluation_path.exists():
            return None
        try:
            with self.offline_evaluation_path.open() as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - evidence is optional
            logger.warning("Offline routing evidence unreadable: %s", exc)
            return None
        if forecast_horizon is not None:
            horizons = payload.get("horizons")
            if isinstance(horizons, dict):
                matched = horizons.get(str(forecast_horizon))
                if isinstance(matched, dict):
                    return matched

            sibling = self.offline_evaluation_path.with_name("forecast_evaluation_horizons.json")
            if sibling.exists():
                try:
                    with sibling.open() as fh:
                        multi_payload = json.load(fh)
                    matched = (multi_payload.get("horizons") or {}).get(str(forecast_horizon))
                    if isinstance(matched, dict):
                        return matched
                except Exception as exc:  # noqa: BLE001 - optional bootstrap evidence
                    logger.warning("Multi-horizon offline evidence unreadable: %s", exc)
        return payload

    def _offline_metrics_to_performance(
        self,
        *,
        metrics: dict[str, Any],
        horizon_days: int,
        generated_at: datetime | None,
        evidence_level: str,
    ) -> list[MethodPerformance]:
        rows: list[MethodPerformance] = []
        for offline_method, values in metrics.items():
            method = OFFLINE_METHOD_TO_RUNTIME.get(offline_method)
            if method is None or method not in RUNTIME_METHODS:
                continue
            metric_value = values.get(self.settings.routing_primary_metric)
            if metric_value is None:
                continue
            rows.append(
                MethodPerformance(
                    method=method,
                    metric_value=float(metric_value),
                    sample_size=int(values.get("n_test_points") or values.get("n") or 0),
                    evaluation_count=1,
                    mean_bias=float(values["bias"]) if values.get("bias") is not None else None,
                    latest_generated_at=generated_at,
                    horizon_days=horizon_days,
                    evidence_source="offline",
                    evidence_level=evidence_level,
                )
            )
        return rows

    def _is_stale(self, generated_at: datetime | None, as_of_date: date) -> bool:
        age = self._age_days(generated_at, as_of_date)
        return age is None or age > self.settings.routing_evidence_lookback_days

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    @staticmethod
    def _age_days(generated_at: datetime | None, as_of_date: date) -> int | None:
        if generated_at is None:
            return None
        return max(0, (as_of_date - generated_at.date()).days)

    @staticmethod
    def _default_decision(default_method: str, reason: str) -> RoutingDecision:
        return RoutingDecision(
            selected_method=default_method,
            default_method=default_method,
            selection_source="default",
            evidence_level="default",
            reason=reason,
            fallback_used=True,
        )
