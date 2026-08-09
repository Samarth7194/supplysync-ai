"""Evaluate logged forecasts after actual demand is available."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import ForecastMetrics, compute_all
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository


@dataclass(frozen=True)
class EvaluationRunSummary:
    predictions_scanned: int
    eligible: int
    evaluated: int
    already_evaluated: int
    actual_demand_unavailable: int
    invalid_prediction: int


class ForecastEvaluationService:
    """Coordinates logged-prediction evaluation against recorded actual demand.

    Convention: if a prediction was made from historical demand ending on day D
    with horizon H, the target window is D+1 through D+H inclusive.
    """

    def __init__(self, *, repository: ForecastEvaluationRepository, data_service: Any):
        self.repository = repository
        self.data_service = data_service

    def evaluate_due_predictions(
        self,
        *,
        as_of: date | None = None,
        limit: int = 500,
    ) -> EvaluationRunSummary:
        as_of = as_of or datetime.now(timezone.utc).date()
        predictions = self.repository.eligible_prediction_logs(as_of=as_of, limit=limit)
        summary = {
            "predictions_scanned": len(predictions),
            "eligible": len(predictions),
            "evaluated": 0,
            "already_evaluated": 0,
            "actual_demand_unavailable": 0,
            "invalid_prediction": 0,
        }

        for prediction in predictions:
            if self.repository.prediction_has_evaluation(prediction.id):
                summary["already_evaluated"] += 1
                continue

            actual = self._actual_window(prediction.sku_code, prediction.target_start_date, prediction.target_end_date)
            if actual is None:
                summary["actual_demand_unavailable"] += 1
                continue

            predicted = self._predicted_values(prediction.forecast_daily, len(actual))
            if predicted is None:
                summary["invalid_prediction"] += 1
                continue

            in_sample = self._in_sample_history(prediction.sku_code, prediction.target_start_date)
            metrics = compute_all(actual, predicted, in_sample=in_sample)
            self._persist_evaluation(prediction, actual, metrics)
            summary["evaluated"] += 1

        return EvaluationRunSummary(**summary)

    def _actual_window(
        self,
        sku_code: str,
        target_start: date | None,
        target_end: date | None,
    ) -> np.ndarray | None:
        if target_start is None or target_end is None or target_end < target_start:
            return None
        series = self.data_service.get_demand_history(sku_code)
        if series.empty or not isinstance(series.index, pd.DatetimeIndex):
            return None

        start_ts = pd.Timestamp(target_start)
        end_ts = pd.Timestamp(target_end)
        if start_ts < series.index.min() or end_ts > series.index.max():
            return None

        actual = series.loc[start_ts:end_ts].astype(float).to_numpy()
        expected_days = (target_end - target_start).days + 1
        if actual.size != expected_days:
            return None
        return actual

    def _in_sample_history(self, sku_code: str, target_start: date | None) -> np.ndarray | None:
        if target_start is None:
            return None
        series = self.data_service.get_demand_history(sku_code)
        if series.empty or not isinstance(series.index, pd.DatetimeIndex):
            return None
        history = series.loc[: pd.Timestamp(target_start) - pd.Timedelta(days=1)]
        return history.astype(float).to_numpy() if len(history) else None

    @staticmethod
    def _predicted_values(values: list[float] | None, expected_len: int) -> np.ndarray | None:
        if not values:
            return None
        predicted = np.asarray(values, dtype=float)
        if predicted.size < expected_len:
            return None
        return predicted[:expected_len]

    def _persist_evaluation(self, prediction: Any, actual: np.ndarray, metrics: ForecastMetrics) -> None:
        metric_values = metrics.as_dict()
        analysis = prediction.analysis_run
        self.repository.create_evaluation(
            {
                "prediction_log_id": prediction.id,
                "model_artifact_id": prediction.model_artifact_id,
                "sku_id": prediction.sku_id,
                "sku_code": prediction.sku_code,
                "demand_class": analysis.demand_pattern if analysis is not None else None,
                "model_name": prediction.model_name or prediction.forecast_method,
                "evaluation_scope": "logged_prediction",
                "metric_mae": self._decimal(metric_values["mae"]),
                "metric_rmse": self._decimal(metric_values["rmse"]),
                "metric_bias": self._decimal(metric_values["bias"]),
                "metric_wape": self._decimal(metric_values["wape"]),
                "metric_mase": self._decimal(metric_values["mase"]),
                "n_skus": 1,
                "n_test_points": metric_values["n"],
                "horizon_days": prediction.forecast_horizon_days,
                "source_path": "logged_prediction",
                "generated_at": datetime.now(timezone.utc),
            }
        )
        prediction.actual_observed_demand = Decimal(str(float(np.sum(actual))))
        prediction.actual_observed_at = datetime.now(timezone.utc)

    @staticmethod
    def _decimal(value: float | None) -> Decimal | None:
        return Decimal(str(value)) if value is not None else None

