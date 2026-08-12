"""Residual-based uncertainty selection for inventory safety stock."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from repositories.forecast_evaluation_repository import ForecastEvaluationRepository


@dataclass(frozen=True)
class UncertaintyEstimate:
    source: str
    sigma: float
    sample_count: int
    lookback_days: int
    fallback_used: bool
    method: str | None = None
    horizon_days: int | None = None
    demand_pattern: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sigma"] = round(float(self.sigma), 6)
        return payload


class ForecastUncertaintyService:
    """Choose the best available sigma without leaking future demand.

    Residual evidence is sourced only through completed ``forecast_evaluations``.
    The residual definition is daily ``actual - prediction`` over the logged
    prediction's target window.
    """

    def __init__(
        self,
        *,
        repository: ForecastEvaluationRepository,
        data_service: Any,
        min_residual_observations: int = 30,
        lookback_days: int = 365,
    ):
        self.repository = repository
        self.data_service = data_service
        self.min_residual_observations = max(2, int(min_residual_observations))
        self.lookback_days = max(1, int(lookback_days))

    def select_sigma(
        self,
        *,
        sku_code: str,
        forecast_method: str,
        demand_pattern: str,
        horizon_days: int,
        historical_sigma: float,
    ) -> UncertaintyEstimate:
        generated_after = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        candidates = (
            (
                "sku_method_residuals",
                {"sku_code": sku_code, "forecast_method": forecast_method},
            ),
            (
                "sku_residuals",
                {"sku_code": sku_code},
            ),
            (
                "pattern_residuals",
                {"demand_class": demand_pattern},
            ),
        )

        for source, filters in candidates:
            residuals = self._residuals(
                horizon_days=horizon_days,
                generated_after=generated_after,
                **filters,
            )
            sigma = self._sigma(residuals)
            if sigma is not None and residuals.size >= self.min_residual_observations:
                return UncertaintyEstimate(
                    source=source,
                    sigma=sigma,
                    sample_count=int(residuals.size),
                    lookback_days=self.lookback_days,
                    fallback_used=False,
                    method=forecast_method,
                    horizon_days=horizon_days,
                    demand_pattern=demand_pattern,
                )

        safe_sigma = float(historical_sigma) if np.isfinite(historical_sigma) and historical_sigma > 0 else 0.0
        return UncertaintyEstimate(
            source="historical_demand_std",
            sigma=safe_sigma,
            sample_count=0,
            lookback_days=self.lookback_days,
            fallback_used=True,
            method=forecast_method,
            horizon_days=horizon_days,
            demand_pattern=demand_pattern,
        )

    def _residuals(
        self,
        *,
        horizon_days: int,
        generated_after: datetime,
        sku_code: str | None = None,
        forecast_method: str | None = None,
        demand_class: str | None = None,
    ) -> np.ndarray:
        predictions = self.repository.evaluated_prediction_logs(
            horizon_days=horizon_days,
            generated_after=generated_after,
            sku_code=sku_code,
            forecast_method=forecast_method,
            demand_class=demand_class,
        )
        values: list[float] = []
        for prediction in predictions:
            values.extend(self._daily_residuals(prediction))
        residuals = np.asarray(values, dtype=float)
        return residuals[np.isfinite(residuals)]

    def _daily_residuals(self, prediction: Any) -> Iterable[float]:
        target_start = prediction.target_start_date
        target_end = prediction.target_end_date
        forecast = prediction.forecast_daily or []
        if target_start is None or target_end is None or target_end < target_start or not forecast:
            return []

        series = self.data_service.get_demand_history(prediction.sku_code)
        if series.empty or not isinstance(series.index, pd.DatetimeIndex):
            return []

        start_ts = pd.Timestamp(target_start)
        end_ts = pd.Timestamp(target_end)
        if start_ts < series.index.min() or end_ts > series.index.max():
            return []

        actual = series.loc[start_ts:end_ts].astype(float).to_numpy()
        expected_days = (target_end - target_start).days + 1
        if actual.size != expected_days or len(forecast) < expected_days:
            return []

        predicted = np.asarray(forecast[:expected_days], dtype=float)
        return list(actual - predicted)

    @staticmethod
    def _sigma(residuals: np.ndarray) -> float | None:
        if residuals.size < 2:
            return None
        sigma = float(np.std(residuals, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            return None
        return sigma
