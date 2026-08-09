"""Application service for SKU analysis orchestration.

This module owns the business workflow behind ``POST /api/analyze`` while
keeping FastAPI route handlers focused on HTTP concerns. It deliberately reuses
the existing forecasting and inventory services; it does not reimplement model
or reorder-point algorithms.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from repositories.analysis_repository import AnalysisRepository, serialize_analysis_runs
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository
from services.model_routing_service import ModelRoutingService
from services.model_service import ModelService

logger = logging.getLogger(__name__)


class AnalysisServiceError(Exception):
    """Base class for analysis-service failures."""


class AnalysisServiceUnavailableError(AnalysisServiceError):
    """Raised when the analysis engine is not initialized."""


class AnalysisExecutionError(AnalysisServiceError):
    """Raised when analysis computation fails."""


@dataclass(frozen=True)
class ForecastBlockData:
    p50: float
    p90: float
    daily: list[float]
    full_horizon_daily: list[float]
    horizon_days: int


@dataclass(frozen=True)
class ModelInfoData:
    model_name: str
    model_type: str
    artifact_available: bool
    model_version: str | None = None
    feature_schema_version: str | None = None
    artifact_checksum: str | None = None
    artifact_valid: bool | None = None
    trained_at: str | None = None
    feature_count: int | None = None
    dataset: str | None = None
    evaluation_available: bool = False
    evaluation_generated_at: str | None = None


@dataclass(frozen=True)
class ExplanationBlockData:
    classification_reason: str
    method_reason: str
    risk_reason: str
    confidence_note: str


@dataclass(frozen=True)
class DecisionBlockData:
    lead_time_days: int
    lead_time_demand: float
    safety_stock: float
    safety_stock_method: str
    reorder_point: float
    service_level: float
    inventory_gap: float
    why: str
    constraints: dict[str, Any]


@dataclass(frozen=True)
class AnalyzeResult:
    sku: str
    risk: str
    risk_color: str
    forecast: ForecastBlockData
    current_stock: float
    recommended_order: int
    action: str
    demand_pattern: str
    forecast_method: str
    demand_source: str
    forecast_source: str
    decision: DecisionBlockData
    model_info: ModelInfoData
    explanation: ExplanationBlockData
    routing: dict[str, Any] | None = None

    def to_response_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("routing", None)
        return payload


def stable_hash_int(value: str) -> int:
    """Process-stable hash; Python's built-in hash is salted per process."""
    return int.from_bytes(hashlib.md5(value.encode("utf-8")).digest()[:4], "big")


FORECAST_SOURCE_BY_METHOD = {
    "ml_lightgbm": "model_forecast",
    "croston": "statistical_method",
    "conservative": "statistical_method",
    "simple_average": "rule_based_estimate",
}


def classify_forecast_source(method: str) -> str:
    return FORECAST_SOURCE_BY_METHOD.get(method, "unavailable")


STATISTICAL_METHOD_NAMES = {
    "croston": "Croston (SBA-corrected)",
    "conservative": "Conservative buffer (recent mean x 1.5)",
}


class AnalysisService:
    """Coordinates demand resolution, forecasting, inventory math, and audit writes."""

    def __init__(
        self,
        *,
        inventory_service: Any,
        settings: Any,
        data_service: Any | None = None,
        analysis_repository: AnalysisRepository | None = None,
        model_loaded: bool = False,
        model_dir: str | Path | None = None,
    ):
        self.inventory_service = inventory_service
        self.settings = settings
        self.data_service = data_service
        self.analysis_repository = analysis_repository
        self.model_loaded = model_loaded
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "saved_models"
        self.backend_dir = self.model_dir.parent

    def analyze(self, request: Any) -> dict[str, Any]:
        if self.inventory_service is None:
            raise AnalysisServiceUnavailableError("Inventory service unavailable")

        demand_series, demand_source = self._resolve_demand(request)
        lead_time_days = (
            int(request.lead_time_days)
            if getattr(request, "lead_time_days", None)
            else self.settings.inventory.default_lead_time_days
        )
        service_level = (
            float(request.service_level)
            if getattr(request, "service_level", None)
            else self.settings.inventory.default_service_level
        )

        try:
            decision = self.inventory_service.get_intelligent_reorder_decision(
                sku=request.sku,
                current_stock=request.current_stock,
                demand_history=demand_series,
                lead_time_days=lead_time_days,
                service_level=service_level,
                routing_service=self._routing_service(),
            )
        except Exception as exc:  # noqa: BLE001 - preserve API error behavior at boundary
            raise AnalysisExecutionError(f"Analysis failed for SKU {request.sku}: {exc}") from exc

        result = self._compose_result(
            request=request,
            demand_series=demand_series,
            demand_source=demand_source,
            decision=decision,
        )
        self._persist_analysis(result, demand_series)
        return result.to_response_dict()

    def recent_analyses(self, limit: int = 20) -> dict[str, Any]:
        if self.analysis_repository is not None:
            try:
                rows = self.analysis_repository.recent(limit=limit)
                return {
                    "available": True,
                    "items": serialize_analysis_runs(rows),
                    "total": self.analysis_repository.count(),
                    "source": "sqlalchemy",
                }
            except SQLAlchemyError as exc:
                self.analysis_repository.session.rollback()
                logger.exception(
                    "SQLAlchemy analysis history unavailable",
                    extra={"operation": "recent_analyses", "repository": "AnalysisRepository"},
                )
                raise AnalysisExecutionError("Analysis history persistence is unavailable.") from exc

        return {"available": False, "items": [], "total": 0, "source": "sqlalchemy"}

    def _resolve_demand(self, request: Any) -> tuple[pd.Series, str]:
        if getattr(request, "demand_history", None):
            return pd.Series(request.demand_history, dtype=float), "request"

        if self.data_service is not None:
            history = self.data_service.get_demand_history(request.sku)
            if len(history) > 0:
                return history.tail(60).astype(float), "historical"

        rng = np.random.default_rng(stable_hash_int(request.sku))
        return pd.Series(rng.poisson(20, 30).astype(float)), "synthetic"

    def _compose_result(
        self,
        *,
        request: Any,
        demand_series: pd.Series,
        demand_source: str,
        decision: dict[str, Any],
    ) -> AnalyzeResult:
        demand_values = demand_series.to_numpy(dtype=float)
        p50 = float(np.mean(demand_values))
        p90 = float(np.percentile(demand_values, 90))

        if request.current_stock < p50:
            risk, risk_color = "HIGH", "#ef4444"
        elif request.current_stock < p90:
            risk, risk_color = "MEDIUM", "#eab308"
        else:
            risk, risk_color = "LOW", "#22c55e"

        order_qty = int(decision.get("order_quantity", 0))
        forecast_method = decision.get("intelligence", {}).get("forecast_method", "unknown")
        routing = decision.get("intelligence", {}).get("routing")

        lead_time_demand = float(decision.get("lead_time_demand", 0.0))
        safety_stock = float(decision.get("safety_stock", 0.0))
        safety_stock_method = str(decision.get("safety_stock_method", "traditional"))
        reorder_point = float(decision.get("reorder_point", 0.0))
        service_level = float(decision.get("service_level", 0.95))
        lead_time_days = int(decision.get("lead_time_days", 7))
        daily_forecast = self._daily_forecast(decision)
        full_horizon_forecast = self._full_horizon_forecast(decision, lead_time_days)
        inventory_gap = max(0.0, reorder_point - float(request.current_stock))
        constraints = self._constraint_metadata(decision, inventory_gap)

        decision_block = DecisionBlockData(
            lead_time_days=lead_time_days,
            lead_time_demand=round(lead_time_demand, 2),
            safety_stock=round(safety_stock, 2),
            safety_stock_method=safety_stock_method,
            reorder_point=round(reorder_point, 2),
            service_level=service_level,
            inventory_gap=round(inventory_gap, 2),
            why=self._compose_decision_why(
                order_qty=order_qty,
                current_stock=request.current_stock,
                reorder_point=reorder_point,
                lead_time_demand=lead_time_demand,
                lead_time_days=lead_time_days,
                safety_stock=safety_stock,
                service_level=service_level,
                constraints=constraints,
            ),
            constraints=constraints,
        )

        demand_pattern = decision.get("intelligence", {}).get("demand_pattern", "unknown")
        forecast_source = classify_forecast_source(forecast_method)
        model_info = self._model_info_for_method(forecast_method)
        explanation = self._compose_explanation(
            demand_series=demand_series,
            demand_pattern=demand_pattern,
            forecast_method=forecast_method,
            forecast_source=forecast_source,
            demand_source=demand_source,
            risk=risk,
            p50=p50,
            p90=p90,
            current_stock=float(request.current_stock),
            routing=routing,
        )

        return AnalyzeResult(
            sku=request.sku,
            risk=risk,
            risk_color=risk_color,
            forecast=ForecastBlockData(
                p50=round(p50, 1),
                p90=round(p90, 1),
                daily=daily_forecast,
                full_horizon_daily=full_horizon_forecast,
                horizon_days=lead_time_days,
            ),
            current_stock=request.current_stock,
            recommended_order=order_qty,
            action="PURCHASE" if order_qty > 0 else "NO_ACTION",
            demand_pattern=demand_pattern,
            forecast_method=forecast_method,
            demand_source=demand_source,
            forecast_source=forecast_source,
            decision=decision_block,
            model_info=model_info,
            explanation=explanation,
            routing=routing if isinstance(routing, dict) else None,
        )

    @staticmethod
    def _daily_forecast(decision: dict[str, Any]) -> list[float]:
        forecast_daily_raw = decision.get("forecast_daily") or []
        if forecast_daily_raw:
            daily_forecast = [round(float(v), 2) for v in forecast_daily_raw[:7]]
            if len(daily_forecast) < 7:
                daily_forecast += [daily_forecast[-1]] * (7 - len(daily_forecast))
            return daily_forecast

        lead_time_demand = float(decision.get("lead_time_demand", 0))
        return [round(lead_time_demand / 7, 2)] * 7

    @staticmethod
    def _full_horizon_forecast(decision: dict[str, Any], horizon_days: int) -> list[float]:
        forecast_daily_raw = decision.get("forecast_daily") or []
        if forecast_daily_raw:
            return [round(float(v), 2) for v in forecast_daily_raw[:horizon_days]]

        lead_time_demand = float(decision.get("lead_time_demand", 0))
        per_day = round(lead_time_demand / max(horizon_days, 1), 2)
        return [per_day] * horizon_days

    @staticmethod
    def _constraint_metadata(decision: dict[str, Any], inventory_gap: float) -> dict[str, Any]:
        constraints = decision.get("business_constraints") or {}
        raw_qty = int(constraints.get("original_quantity", decision.get("order_quantity", 0)))
        final_qty = int(constraints.get("final_quantity", decision.get("order_quantity", 0)))
        max_order_quantity = constraints.get("max_order_quantity")
        constraints_applied = list(constraints.get("constraints_applied") or [])

        return {
            "raw_order_quantity": raw_qty,
            "final_order_quantity": final_qty,
            "moq": constraints.get("moq"),
            "order_multiple": constraints.get("order_multiple"),
            "max_order_quantity": max_order_quantity,
            "constraints_applied": constraints_applied,
            "moq_applied": any(item.startswith("MOQ") for item in constraints_applied),
            "order_multiple_applied": any("multiple" in item.lower() for item in constraints_applied),
            "max_order_cap_applied": any("capped at maximum" in item.lower() for item in constraints_applied),
            "constrained": raw_qty != final_qty,
            "remaining_gap_after_order": round(max(0.0, float(inventory_gap) - final_qty), 2),
        }

    def _model_info_for_method(self, method: str) -> ModelInfoData:
        eval_path = self.backend_dir / "data" / "forecast_evaluation.json"
        eval_available = eval_path.exists()
        eval_generated_at: str | None = None
        if eval_available:
            try:
                with eval_path.open() as fh:
                    eval_generated_at = json.load(fh).get("generated_at")
            except Exception:  # noqa: BLE001 - evaluation metadata is non-critical
                eval_generated_at = None

        if method == "ml_lightgbm":
            model_service = ModelService(model_dir=str(self.model_dir))
            meta = model_service.get_model_metadata("lightgbm_demand_forecast") or {}
            artifact_status = model_service.artifact_status("lightgbm_demand_forecast")
            features = meta.get("features") or []
            return ModelInfoData(
                model_name="lightgbm_demand_forecast",
                model_type="ml",
                artifact_available=self.model_loaded,
                model_version=meta.get("version") or self._checksum_version(),
                feature_schema_version=meta.get("feature_schema_version"),
                artifact_checksum=meta.get("artifact_checksum"),
                artifact_valid=bool(artifact_status.get("valid")),
                trained_at=meta.get("saved_at"),
                feature_count=len(features) if features else None,
                dataset=meta.get("dataset"),
                evaluation_available=eval_available,
                evaluation_generated_at=eval_generated_at,
            )

        if method in STATISTICAL_METHOD_NAMES:
            return ModelInfoData(
                model_name=STATISTICAL_METHOD_NAMES[method],
                model_type="statistical_method",
                artifact_available=False,
                evaluation_available=eval_available,
                evaluation_generated_at=eval_generated_at,
            )

        if method == "simple_average":
            return ModelInfoData(
                model_name="Simple 7-day moving average",
                model_type="rule_based_fallback",
                artifact_available=False,
                evaluation_available=eval_available,
                evaluation_generated_at=eval_generated_at,
            )

        return ModelInfoData(
            model_name="unknown",
            model_type="none",
            artifact_available=False,
            evaluation_available=eval_available,
            evaluation_generated_at=eval_generated_at,
        )

    def _compose_explanation(
        self,
        *,
        demand_series: pd.Series,
        demand_pattern: str,
        forecast_method: str,
        forecast_source: str,
        demand_source: str,
        risk: str,
        p50: float,
        p90: float,
        current_stock: float,
        routing: dict[str, Any] | None = None,
    ) -> ExplanationBlockData:
        arr = demand_series.to_numpy(dtype=float)
        zero_share = float((arr == 0).mean()) if arr.size else 0.0
        n_obs = int(arr.size)

        if demand_pattern == "highly_intermittent":
            classification_reason = (
                f"{zero_share * 100:.0f}% of the {n_obs} observed days have zero demand, "
                "which crosses the 80% threshold for the highly-intermittent class."
            )
        elif demand_pattern == "intermittent":
            classification_reason = (
                f"{zero_share * 100:.0f}% of the {n_obs} observed days have zero demand, "
                "which falls between the 50% and 80% thresholds used for the intermittent class."
            )
        else:
            classification_reason = (
                f"Only {zero_share * 100:.0f}% of the {n_obs} observed days have zero demand "
                "(below the 50% threshold), so this SKU is routed as regular demand."
            )

        if forecast_method == "ml_lightgbm":
            method_reason = (
                "Regular-demand SKUs are forecast with the trained LightGBM model using "
                "lag and calendar features - that's the path chosen here."
            )
        elif forecast_method == "croston":
            method_reason = (
                "Intermittent SKUs are forecast with Croston's method (SBA-corrected), "
                "which separately estimates demand size and inter-arrival interval."
            )
        elif forecast_method == "conservative":
            method_reason = (
                "Highly-intermittent SKUs use a conservative buffer (recent mean x 1.5) "
                "because few non-zero days make any model's point forecast unreliable."
            )
        elif forecast_method == "simple_average":
            if not self.model_loaded:
                method_reason = (
                    "The regular-demand SKU would normally be forecast by LightGBM, but "
                    "the trained artifact isn't loaded on this deployment - falling back "
                    "to a 7-day moving average. Treat the recommendation as approximate."
                )
            else:
                method_reason = (
                    "The regular-demand SKU fell back to a 7-day moving average "
                    "(likely due to insufficient history or a feature-build failure); "
                    "see server logs for the exact reason."
                )
        else:
            method_reason = f"Forecast method {forecast_method!r} was used."

        if routing and routing.get("reason"):
            method_reason = f"{method_reason} Routing note: {routing['reason']}"

        if risk == "HIGH":
            risk_reason = (
                f"Current stock ({current_stock:g}) is below the P50 demand estimate "
                f"({p50:g}) - any higher-than-median day risks a stockout."
            )
        elif risk == "MEDIUM":
            risk_reason = (
                f"Current stock ({current_stock:g}) covers median demand ({p50:g}) but not "
                f"the P90 scenario ({p90:g}) - a higher-demand day could cause a stockout."
            )
        else:
            risk_reason = (
                f"Current stock ({current_stock:g}) covers the P90 demand scenario "
                f"({p90:g}), so even a higher-demand day should be fulfillable."
            )

        if demand_source == "synthetic":
            confidence_note = (
                "This SKU isn't in the processed dataset - the demand series is "
                "synthetic demo data, so the recommendation is illustrative only."
            )
        elif forecast_source == "rule_based_estimate":
            confidence_note = (
                "Forecast came from a rule-based fallback, not the trained model - "
                "the recommendation is coarser than the regular-path ML output."
            )
        elif forecast_source == "statistical_method":
            confidence_note = (
                "Forecast came from a statistical method tuned to sparse demand; "
                "point accuracy is inherently limited when most days have zero demand."
            )
        else:
            confidence_note = (
                "Forecast came from the trained LightGBM model; the recommendation "
                "reflects the model's regular-demand path."
            )

        return ExplanationBlockData(
            classification_reason=classification_reason,
            method_reason=method_reason,
            risk_reason=risk_reason,
            confidence_note=confidence_note,
        )

    @staticmethod
    def _compose_decision_why(
        *,
        order_qty: int,
        current_stock: float,
        reorder_point: float,
        lead_time_demand: float,
        lead_time_days: int,
        safety_stock: float,
        service_level: float,
        constraints: dict[str, Any],
    ) -> str:
        service_pct = int(round(service_level * 100))
        lead_time_demand_v = round(float(lead_time_demand), 1)
        safety_stock_v = round(float(safety_stock), 1)
        reorder_point_v = round(float(reorder_point), 1)
        stock_v = round(float(current_stock), 1)
        raw_qty = int(constraints.get("raw_order_quantity", order_qty))
        max_order = constraints.get("max_order_quantity")

        if order_qty > 0:
            if constraints.get("max_order_cap_applied") and max_order:
                return (
                    f"Projected demand over the {lead_time_days}-day lead time is "
                    f"{lead_time_demand_v} units. With safety stock of {safety_stock_v} "
                    f"units (targeting {service_pct}% service level), the reorder point "
                    f"is {reorder_point_v} units. Current stock is {stock_v} units, "
                    f"giving an uncapped requirement of {raw_qty} units. The supplier "
                    f"maximum order quantity is {max_order} units, so the final "
                    f"recommendation is capped at {order_qty} units."
                )
            return (
                f"Projected demand over the {lead_time_days}-day lead time is "
                f"{lead_time_demand_v} units. With a {safety_stock_v}-unit safety "
                f"buffer (targeting {service_pct}% service level) the reorder point "
                f"is {reorder_point_v}. Current stock {stock_v} is below that, so "
                f"ordering {order_qty} units brings the position back above the reorder point."
            )
        return (
            f"Projected demand over the {lead_time_days}-day lead time is "
            f"{lead_time_demand_v} units. Reorder point is {reorder_point_v} "
            f"(incl. {safety_stock_v}-unit safety stock at {service_pct}% service "
            f"level). Current stock {stock_v} already covers the reorder point - no action needed."
        )

    def _persist_analysis(self, result: AnalyzeResult, demand_series: pd.Series) -> None:
        if self.analysis_repository is not None:
            try:
                sku_id = self.analysis_repository.get_sku_id(result.sku)
                model_artifact_id = self._model_artifact_id(result)
                self.analysis_repository.create_analysis_with_prediction(
                    self._analysis_values(result, sku_id=sku_id),
                    self._prediction_values(
                        result,
                        demand_series,
                        sku_id=sku_id,
                        model_artifact_id=model_artifact_id,
                    ),
                )
                return
            except SQLAlchemyError as exc:
                self.analysis_repository.session.rollback()
                logger.exception(
                    "SQLAlchemy analysis persistence failed",
                    extra={
                        "operation": "persist_analysis",
                        "repository": "AnalysisRepository",
                        "sku": result.sku,
                        "exception_type": type(exc).__name__,
                    },
                )
                raise AnalysisExecutionError("Analysis persistence is unavailable.") from exc

    def _analysis_values(self, result: AnalyzeResult, *, sku_id: int | None = None) -> dict[str, Any]:
            return {
            "sku_id": sku_id,
            "sku_code": result.sku,
            "current_stock": Decimal(str(result.current_stock)),
            "recommended_order_quantity": result.recommended_order,
            "action": result.action,
            "risk": result.risk,
            "risk_color": result.risk_color,
            "demand_pattern": result.demand_pattern,
            "demand_source": result.demand_source,
            "forecast_source": result.forecast_source,
            "forecast_method": result.forecast_method,
            "routing_reason": self._routing_reason(result.routing),
            "lead_time_days": result.decision.lead_time_days,
            "service_level": Decimal(str(result.decision.service_level)),
            "lead_time_demand": Decimal(str(result.decision.lead_time_demand)),
            "safety_stock": Decimal(str(result.decision.safety_stock)),
            "safety_stock_method": result.decision.safety_stock_method,
            "reorder_point": Decimal(str(result.decision.reorder_point)),
            "inventory_gap": Decimal(str(result.decision.inventory_gap)),
            "p50": Decimal(str(result.forecast.p50)),
            "p90": Decimal(str(result.forecast.p90)),
            "forecast_daily": result.forecast.full_horizon_daily,
            "explanation": asdict(result.explanation),
        }

    def _prediction_values(
        self,
        result: AnalyzeResult,
        demand_series: pd.Series,
        *,
        sku_id: int | None = None,
        model_artifact_id: int | None = None,
    ) -> dict[str, Any]:
        target_start, target_end = self._target_window(demand_series, result.decision.lead_time_days)
        return {
            "sku_id": sku_id,
            "sku_code": result.sku,
            "target_start_date": target_start,
            "target_end_date": target_end,
            "demand_source": result.demand_source,
            "forecast_method": result.forecast_method,
            "forecast_source": result.forecast_source,
            "routing_reason": self._routing_reason(result.routing),
            "model_name": result.model_info.model_name,
            "model_version": self._model_version(result),
            "feature_schema_version": result.model_info.feature_schema_version,
            "model_artifact_id": model_artifact_id,
            "input_history_length": int(len(demand_series)),
            "forecast_horizon_days": result.decision.lead_time_days,
            "p50": Decimal(str(result.forecast.p50)),
            "p90": Decimal(str(result.forecast.p90)),
            "forecast_daily": result.forecast.full_horizon_daily,
            "recommended_order_quantity": result.recommended_order,
        }

    def _model_artifact_id(self, result: AnalyzeResult) -> int | None:
        if (
            self.analysis_repository is None
            or result.model_info.model_type != "ml"
            or result.forecast_method != "ml_lightgbm"
        ):
            return None

        version = self._model_version(result)
        if not version:
            return None

        metadata_path = self.model_dir / "lightgbm_demand_forecast_metadata.json"
        artifact_path = self.model_dir / "lightgbm_demand_forecast.pkl"
        model_service = ModelService(model_dir=str(self.model_dir))
        meta = model_service.get_model_metadata("lightgbm_demand_forecast") or {}
        artifact = self.analysis_repository.get_or_create_model_artifact(
            {
                "model_name": "lightgbm_demand_forecast",
                "model_family": meta.get("model_family", "lightgbm"),
                "model_type": "ml",
                "version": version,
                "artifact_checksum": meta.get("artifact_checksum") or self._artifact_checksum(),
                "checksum_algorithm": meta.get("checksum_algorithm", "sha256"),
                "artifact_uri": str(artifact_path) if artifact_path.exists() else None,
                "metadata_uri": str(metadata_path) if metadata_path.exists() else None,
                "feature_schema": meta.get("features"),
                "feature_schema_version": meta.get("feature_schema_version"),
                "feature_schema_checksum": meta.get("feature_schema_checksum"),
                "training_dataset": meta.get("dataset"),
                "training_finished_at": self._parse_datetime(result.model_info.trained_at),
                "training_metrics": {
                    "mae": meta.get("mae"),
                    "rmse": meta.get("rmse"),
                },
                "training_metadata": {
                    "train_skus": meta.get("train_skus"),
                    "n_train_rows": meta.get("n_train_rows"),
                    "n_test_rows": meta.get("n_test_rows"),
                    "training_data": meta.get("training_data"),
                    "training_config": meta.get("training_config"),
                },
                "lifecycle_status": meta.get("lifecycle_status", "candidate"),
                "is_active": meta.get("lifecycle_status") == "active",
            }
        )
        return artifact.id

    def _routing_service(self) -> ModelRoutingService | None:
        forecasting_settings = getattr(self.settings, "forecasting", None)
        if forecasting_settings is None:
            return None
        repository = None
        if self.analysis_repository is not None:
            repository = ForecastEvaluationRepository(self.analysis_repository.session)
        return ModelRoutingService(
            settings=forecasting_settings,
            forecast_evaluation_repository=repository,
            offline_evaluation_path=self.backend_dir / "data" / "forecast_evaluation.json",
        )

    @staticmethod
    def _routing_reason(routing: dict[str, Any] | None) -> str | None:
        if not routing:
            return None
        return str(routing.get("reason") or "") or None

    @staticmethod
    def _target_window(demand_series: pd.Series, horizon_days: int) -> tuple[Any | None, Any | None]:
        if demand_series.empty or not isinstance(demand_series.index, pd.DatetimeIndex):
            return None, None
        last_observed = demand_series.index.max()
        if pd.isna(last_observed):
            return None, None
        start = (last_observed + pd.Timedelta(days=1)).date()
        end = (last_observed + pd.Timedelta(days=horizon_days)).date()
        return start, end

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _model_version(self, result: AnalyzeResult) -> str | None:
        if result.model_info.model_type == "ml":
            if result.model_info.model_version:
                return result.model_info.model_version
            checksum_version = self._checksum_version()
            if checksum_version:
                return checksum_version
            return result.model_info.trained_at
        versions = {
            "croston": "croston_sba_alpha_0.1",
            "conservative": "conservative_recent_mean_x1.5",
            "simple_average": "simple_average_7_day_v1",
        }
        return versions.get(result.forecast_method)

    def _artifact_checksum(self) -> str | None:
        artifact_path = self.model_dir / "lightgbm_demand_forecast.pkl"
        if not artifact_path.exists():
            return None
        digest = hashlib.sha256()
        with artifact_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _checksum_version(self) -> str | None:
        checksum = self._artifact_checksum()
        if not checksum:
            return None
        return f"sha256:{checksum[:16]}"
