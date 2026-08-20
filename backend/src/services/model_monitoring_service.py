"""Phase A model monitoring metrics from completed forecast evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from db.models import ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot
from repositories.model_monitoring_repository import ModelMonitoringRepository


@dataclass(frozen=True)
class Baseline:
    wape: float | None
    provenance: str


@dataclass(frozen=True)
class MonitoringMetrics:
    wape: float | None
    mae: float | None
    rmse: float | None
    bias: float | None
    mase: float | None
    residual_mean: float | None
    residual_std: float | None
    mean_actual_demand: float | None


@dataclass(frozen=True)
class PerformanceState:
    status: str
    reason: str
    message: str
    wape_relative_change: float | None
    bias_ratio: float | None
    consecutive_degradation_count: int


@dataclass(frozen=True)
class MonitoringSnapshotResult:
    snapshot: ModelMonitoringSnapshot
    created: bool


class ModelMonitoringService:
    """Compute and persist rolling monitoring snapshots.

    Phase A intentionally performs no drift detection, retraining decision, or
    lifecycle mutation. It summarizes only completed ``forecast_evaluations``.
    """

    def __init__(
        self,
        *,
        repository: ModelMonitoringRepository,
        settings: Any,
        data_service: Any | None = None,
        offline_evaluation_path: str | Path | None = None,
    ):
        self.repository = repository
        self.settings = settings
        self.data_service = data_service
        self.offline_evaluation_path = Path(offline_evaluation_path) if offline_evaluation_path else None

    def create_snapshot(self, *, model_name: str = "lightgbm_demand_forecast") -> MonitoringSnapshotResult:
        forecasting = self.settings.forecasting
        model_artifact = (
            self.repository.active_model_artifact(model_name)
            or self.repository.latest_evaluated_model_artifact(model_name)
        )
        model_artifact_id = model_artifact.id if model_artifact is not None else None
        model_version = model_artifact.version if model_artifact is not None else None

        generated_after = datetime.now(timezone.utc) - timedelta(
            days=int(forecasting.model_monitoring_lookback_days)
        )
        evaluations = self.repository.get_recent_completed_evaluations(
            model_artifact_id=model_artifact_id,
            model_name=model_name,
            model_version=model_version,
            generated_after=generated_after,
            limit=int(forecasting.model_monitoring_window_evaluations),
        )
        metrics = self._metrics(evaluations)
        baseline = self._baseline(
            model_artifact=model_artifact,
            model_name=model_name,
            model_version=model_version,
            evaluations=evaluations,
        )
        evidence_key = self._evidence_key(
            model_artifact_id=model_artifact_id,
            model_name=model_name,
            model_version=model_version,
            evaluations=evaluations,
            window_size=int(forecasting.model_monitoring_window_evaluations),
            lookback_days=int(forecasting.model_monitoring_lookback_days),
        )
        existing = self.repository.snapshot_by_evidence_key(evidence_key)
        if existing is not None:
            return MonitoringSnapshotResult(snapshot=existing, created=False)

        previous_snapshots = self.repository.recent_snapshots_for_scope(
            model_artifact_id=model_artifact_id,
            model_name=model_name,
            model_version=model_version,
            limit=int(forecasting.model_monitoring_degradation_consecutive_runs),
        )
        state = self._performance_state(
            metrics=metrics,
            baseline=baseline,
            evaluation_count=len(evaluations),
            previous_snapshots=previous_snapshots,
        )

        snapshot = self.repository.create_snapshot(
            {
                "generated_at": datetime.now(timezone.utc),
                "model_artifact_id": model_artifact_id,
                "model_name": model_name,
                "model_version": model_version,
                "window_type": "latest_evaluations",
                "window_size": int(forecasting.model_monitoring_window_evaluations),
                "evaluation_count": len(evaluations),
                "metric_wape": self._decimal(metrics.wape),
                "metric_mae": self._decimal(metrics.mae),
                "metric_rmse": self._decimal(metrics.rmse),
                "metric_bias": self._decimal(metrics.bias),
                "metric_mase": self._decimal(metrics.mase),
                "residual_mean": self._decimal(metrics.residual_mean),
                "residual_std": self._decimal(metrics.residual_std),
                "baseline_wape": self._decimal(baseline.wape),
                "baseline_provenance": baseline.provenance,
                "wape_relative_change": self._decimal(state.wape_relative_change),
                "bias_ratio": self._decimal(state.bias_ratio),
                "degradation_reason": state.reason,
                "degradation_message": state.message,
                "consecutive_degradation_count": state.consecutive_degradation_count,
                "status": state.status,
                "evidence_key": evidence_key,
            }
        )
        return MonitoringSnapshotResult(snapshot=snapshot, created=True)

    def current_snapshot(self, *, model_name: str = "lightgbm_demand_forecast") -> ModelMonitoringSnapshot | None:
        """Return the latest snapshot for the active model scope, if available."""
        active_artifact = self.repository.active_model_artifact(model_name)
        if active_artifact is not None:
            return self.repository.get_latest_snapshot_for_scope(
                model_artifact_id=active_artifact.id,
                model_name=model_name,
                model_version=active_artifact.version,
            )

        fallback_artifact = self.repository.latest_evaluated_model_artifact(model_name)
        if fallback_artifact is not None:
            scoped = self.repository.get_latest_snapshot_for_scope(
                model_artifact_id=fallback_artifact.id,
                model_name=model_name,
                model_version=fallback_artifact.version,
            )
            if scoped is not None:
                return scoped

        return self.repository.get_latest_snapshot(model_name=model_name)

    def snapshot_history(
        self,
        *,
        model_name: str = "lightgbm_demand_forecast",
        limit: int = 20,
        model_artifact_id: int | None = None,
        status: str | None = None,
    ) -> list[ModelMonitoringSnapshot]:
        return self.repository.list_recent_snapshots(
            model_name=model_name,
            limit=limit,
            model_artifact_id=model_artifact_id,
            status=status,
        )

    def _metrics(self, evaluations: list[ForecastEvaluation]) -> MonitoringMetrics:
        if not evaluations:
            return MonitoringMetrics(None, None, None, None, None, None, None, None)

        raw_actuals: list[np.ndarray] = []
        raw_predictions: list[np.ndarray] = []
        for evaluation in evaluations:
            actual, predicted = self._actual_and_predicted(evaluation)
            if actual is not None and predicted is not None:
                raw_actuals.append(actual)
                raw_predictions.append(predicted)

        if len(raw_actuals) == len(evaluations):
            actual_all = np.concatenate(raw_actuals)
            predicted_all = np.concatenate(raw_predictions)
            residuals = actual_all - predicted_all
            absolute_error = np.abs(actual_all - predicted_all)
            actual_demand = np.abs(actual_all).sum()
            return MonitoringMetrics(
                wape=float(absolute_error.sum() / actual_demand) if actual_demand > 0 else None,
                mae=float(absolute_error.mean()) if absolute_error.size else None,
                rmse=float(np.sqrt(np.mean((actual_all - predicted_all) ** 2))) if actual_all.size else None,
                bias=float(np.mean(predicted_all - actual_all)) if actual_all.size else None,
                mase=self._weighted_metric(evaluations, "metric_mase"),
                residual_mean=float(np.mean(residuals)) if residuals.size else None,
                residual_std=float(np.std(residuals, ddof=1)) if residuals.size >= 2 else None,
                mean_actual_demand=float(np.mean(np.abs(actual_all))) if actual_all.size else None,
            )

        return MonitoringMetrics(
            wape=self._aggregate_wape(evaluations),
            mae=self._weighted_metric(evaluations, "metric_mae"),
            rmse=self._aggregate_rmse(evaluations),
            bias=self._weighted_metric(evaluations, "metric_bias"),
            mase=self._weighted_metric(evaluations, "metric_mase"),
            residual_mean=self._residual_mean_from_bias(evaluations),
            residual_std=None,
            mean_actual_demand=self._mean_actual_demand(evaluations),
        )

    def _actual_and_predicted(self, evaluation: ForecastEvaluation) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.data_service is None or evaluation.prediction_log is None:
            return None, None
        prediction = evaluation.prediction_log
        if prediction.target_start_date is None or prediction.target_end_date is None:
            return None, None
        forecast = prediction.forecast_daily or []
        if not forecast:
            return None, None

        series = self.data_service.get_demand_history(prediction.sku_code)
        if series.empty or not isinstance(series.index, pd.DatetimeIndex):
            return None, None
        start = pd.Timestamp(prediction.target_start_date)
        end = pd.Timestamp(prediction.target_end_date)
        actual = series.loc[start:end].astype(float).to_numpy()
        expected_days = (prediction.target_end_date - prediction.target_start_date).days + 1
        if actual.size != expected_days or len(forecast) < expected_days:
            return None, None
        predicted = np.asarray(forecast[:expected_days], dtype=float)
        return actual, predicted

    def _performance_state(
        self,
        *,
        metrics: MonitoringMetrics,
        baseline: Baseline,
        evaluation_count: int,
        previous_snapshots: list[ModelMonitoringSnapshot],
    ) -> PerformanceState:
        forecasting = self.settings.forecasting
        if evaluation_count < int(forecasting.model_monitoring_min_evaluations):
            return PerformanceState(
                status="insufficient_evidence",
                reason="insufficient_evidence",
                message=(
                    f"Only {evaluation_count} completed evaluations are available; "
                    f"{forecasting.model_monitoring_min_evaluations} are required."
                ),
                wape_relative_change=None,
                bias_ratio=self._bias_ratio(metrics),
                consecutive_degradation_count=0,
            )

        wape_relative_change = self._wape_relative_change(metrics.wape, baseline.wape)
        bias_ratio = self._bias_ratio(metrics)
        bias_warning = (
            bias_ratio is not None
            and abs(bias_ratio) >= float(forecasting.model_monitoring_bias_warning_ratio)
        )

        if wape_relative_change is None:
            reason = "baseline_unavailable"
            if baseline.wape == 0:
                reason = "baseline_zero"
            elif bias_warning:
                reason = "bias_warning"
            return PerformanceState(
                status="warning" if bias_warning else "stable",
                reason=reason,
                message=self._message(reason, baseline, wape_relative_change, bias_ratio),
                wape_relative_change=wape_relative_change,
                bias_ratio=bias_ratio,
                consecutive_degradation_count=0,
            )

        degradation_threshold = float(forecasting.model_monitoring_wape_degradation_threshold)
        warning_threshold = float(forecasting.model_monitoring_wape_warning_threshold)
        if wape_relative_change >= degradation_threshold:
            consecutive = self._next_consecutive_degradation_count(previous_snapshots)
            required = int(forecasting.model_monitoring_degradation_consecutive_runs)
            if consecutive >= required:
                reason = "persistent_wape_degradation"
                status = "degraded"
            else:
                reason = "wape_degradation_threshold_exceeded"
                status = "warning"
            return PerformanceState(
                status=status,
                reason=reason,
                message=self._message(reason, baseline, wape_relative_change, bias_ratio),
                wape_relative_change=wape_relative_change,
                bias_ratio=bias_ratio,
                consecutive_degradation_count=consecutive,
            )

        if wape_relative_change >= warning_threshold:
            reason = "wape_warning_threshold_exceeded"
            return PerformanceState(
                status="warning",
                reason=reason,
                message=self._message(reason, baseline, wape_relative_change, bias_ratio),
                wape_relative_change=wape_relative_change,
                bias_ratio=bias_ratio,
                consecutive_degradation_count=0,
            )

        if bias_warning:
            reason = "bias_warning"
            return PerformanceState(
                status="warning",
                reason=reason,
                message=self._message(reason, baseline, wape_relative_change, bias_ratio),
                wape_relative_change=wape_relative_change,
                bias_ratio=bias_ratio,
                consecutive_degradation_count=0,
            )

        reason = "wape_within_baseline"
        return PerformanceState(
            status="stable",
            reason=reason,
            message=self._message(reason, baseline, wape_relative_change, bias_ratio),
            wape_relative_change=wape_relative_change,
            bias_ratio=bias_ratio,
            consecutive_degradation_count=0,
        )

    @staticmethod
    def _wape_relative_change(recent_wape: float | None, baseline_wape: float | None) -> float | None:
        if recent_wape is None or baseline_wape is None or baseline_wape <= 0:
            return None
        return (recent_wape - baseline_wape) / baseline_wape

    @staticmethod
    def _bias_ratio(metrics: MonitoringMetrics) -> float | None:
        if metrics.bias is None or metrics.mean_actual_demand is None or metrics.mean_actual_demand <= 0:
            return None
        return metrics.bias / metrics.mean_actual_demand

    @staticmethod
    def _next_consecutive_degradation_count(previous_snapshots: list[ModelMonitoringSnapshot]) -> int:
        latest = previous_snapshots[0] if previous_snapshots else None
        if latest is None:
            return 1
        if latest.degradation_reason not in {
            "wape_degradation_threshold_exceeded",
            "persistent_wape_degradation",
        }:
            return 1
        return int(latest.consecutive_degradation_count or 0) + 1

    @staticmethod
    def _message(
        reason: str,
        baseline: Baseline,
        wape_relative_change: float | None,
        bias_ratio: float | None,
    ) -> str:
        baseline_note = (
            " Baseline provenance is offline_backtest, so this is a bootstrap comparison, not live production truth."
            if baseline.provenance == "offline_backtest"
            else ""
        )
        if reason == "insufficient_evidence":
            return "Not enough completed evaluations to classify forecast performance."
        if reason == "baseline_unavailable":
            return "Baseline WAPE is unavailable, so WAPE degradation cannot be classified."
        if reason == "baseline_zero":
            return "Baseline WAPE is zero, so relative WAPE degradation cannot be classified safely."
        if reason == "wape_within_baseline":
            return f"Recent WAPE is within configured baseline tolerance.{baseline_note}"
        if reason == "wape_warning_threshold_exceeded":
            return f"Recent WAPE is {wape_relative_change:.1%} worse than baseline.{baseline_note}"
        if reason == "wape_degradation_threshold_exceeded":
            return (
                f"Recent WAPE is {wape_relative_change:.1%} worse than baseline, "
                f"but persistence is not yet established.{baseline_note}"
            )
        if reason == "persistent_wape_degradation":
            return f"Recent WAPE degradation persisted across consecutive monitoring runs.{baseline_note}"
        if reason == "bias_warning":
            return f"Forecast bias ratio is {bias_ratio:.1%}, above the configured warning threshold."
        return reason

    @staticmethod
    def _aggregate_wape(evaluations: list[ForecastEvaluation]) -> float | None:
        numerator = 0.0
        denominator = 0.0
        exact_count = 0
        weighted_values: list[tuple[float, int]] = []
        for evaluation in evaluations:
            if evaluation.metric_wape is None:
                continue
            wape = float(evaluation.metric_wape)
            weighted_values.append((wape, int(evaluation.n_test_points or 0)))
            actual_total = (
                float(evaluation.prediction_log.actual_observed_demand)
                if evaluation.prediction_log is not None
                and evaluation.prediction_log.actual_observed_demand is not None
                else None
            )
            if actual_total is not None and actual_total > 0:
                numerator += wape * actual_total
                denominator += actual_total
                exact_count += 1
        if denominator > 0 and exact_count == len(weighted_values):
            return numerator / denominator
        return ModelMonitoringService._weighted_pairs(weighted_values)

    @staticmethod
    def _aggregate_rmse(evaluations: list[ForecastEvaluation]) -> float | None:
        total = 0.0
        count = 0
        for evaluation in evaluations:
            if evaluation.metric_rmse is None:
                continue
            n = int(evaluation.n_test_points or 0)
            if n <= 0:
                continue
            rmse = float(evaluation.metric_rmse)
            total += (rmse**2) * n
            count += n
        return float(np.sqrt(total / count)) if count else None

    @staticmethod
    def _mean_actual_demand(evaluations: list[ForecastEvaluation]) -> float | None:
        actual_total = 0.0
        point_count = 0
        for evaluation in evaluations:
            prediction = evaluation.prediction_log
            n = int(evaluation.n_test_points or 0)
            if prediction is None or prediction.actual_observed_demand is None or n <= 0:
                continue
            actual_total += abs(float(prediction.actual_observed_demand))
            point_count += n
        if point_count <= 0:
            return None
        return actual_total / point_count

    @staticmethod
    def _weighted_metric(evaluations: list[ForecastEvaluation], field_name: str) -> float | None:
        return ModelMonitoringService._weighted_pairs(
            [
                (float(value), int(evaluation.n_test_points or 0))
                for evaluation in evaluations
                if (value := getattr(evaluation, field_name)) is not None
            ]
        )

    @staticmethod
    def _weighted_pairs(values: list[tuple[float, int]]) -> float | None:
        total_weight = sum(weight for _, weight in values if weight > 0)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in values if weight > 0) / total_weight

    @staticmethod
    def _residual_mean_from_bias(evaluations: list[ForecastEvaluation]) -> float | None:
        bias = ModelMonitoringService._weighted_metric(evaluations, "metric_bias")
        return -bias if bias is not None else None

    def _baseline(
        self,
        *,
        model_artifact: ModelArtifact | None,
        model_name: str,
        model_version: str | None,
        evaluations: list[ForecastEvaluation],
    ) -> Baseline:
        artifact_id = model_artifact.id if model_artifact is not None else None
        promotion = self.repository.latest_baseline_evaluation(
            model_artifact_id=artifact_id,
            model_name=model_name,
            model_version=model_version,
        )
        if promotion is not None and promotion.metric_wape is not None:
            return Baseline(float(promotion.metric_wape), "promotion_evidence")

        if model_artifact is not None and isinstance(model_artifact.training_metrics, dict):
            wape = model_artifact.training_metrics.get("wape")
            if wape is not None:
                return Baseline(float(wape), "artifact_metadata")

        offline = self._offline_baseline(evaluations)
        if offline is not None:
            return Baseline(offline, "offline_backtest")

        return Baseline(None, "unavailable")

    def _offline_baseline(self, evaluations: list[ForecastEvaluation]) -> float | None:
        if self.offline_evaluation_path is None or not self.offline_evaluation_path.exists():
            return None
        horizon = self._first_horizon(evaluations)
        try:
            payload = json.loads(self.offline_evaluation_path.read_text())
        except Exception:  # noqa: BLE001 - baseline is optional
            return None
        if horizon is not None:
            horizons = payload.get("horizons")
            if isinstance(horizons, dict):
                matched = horizons.get(str(horizon))
                value = self._offline_lightgbm_wape(matched)
                if value is not None:
                    return value
            sibling = self.offline_evaluation_path.with_name("forecast_evaluation_horizons.json")
            if sibling.exists():
                try:
                    multi = json.loads(sibling.read_text())
                    value = self._offline_lightgbm_wape((multi.get("horizons") or {}).get(str(horizon)))
                    if value is not None:
                        return value
                except Exception:  # noqa: BLE001 - baseline is optional
                    pass
        return self._offline_lightgbm_wape(payload)

    @staticmethod
    def _offline_lightgbm_wape(payload: Any) -> float | None:
        if not isinstance(payload, dict):
            return None
        value = (((payload.get("aggregates") or {}).get("all") or {}).get("lightgbm") or {}).get("wape")
        return float(value) if value is not None else None

    @staticmethod
    def _first_horizon(evaluations: list[ForecastEvaluation]) -> int | None:
        for evaluation in evaluations:
            if evaluation.horizon_days is not None:
                return int(evaluation.horizon_days)
        return None

    @staticmethod
    def _evidence_key(
        *,
        model_artifact_id: int | None,
        model_name: str,
        model_version: str | None,
        evaluations: list[ForecastEvaluation],
        window_size: int,
        lookback_days: int,
    ) -> str:
        latest = evaluations[0] if evaluations else None
        raw = "|".join(
            [
                str(model_artifact_id or ""),
                model_name,
                model_version or "",
                str(window_size),
                str(lookback_days),
                str(len(evaluations)),
                str(latest.id if latest is not None else ""),
                latest.generated_at.isoformat() if latest is not None and latest.generated_at else "",
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _decimal(value: float | None) -> Decimal | None:
        if value is None or not np.isfinite(value):
            return None
        return Decimal(str(round(float(value), 6)))
