"""Forecasting and model-artifact settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.common import BACKEND_DIR, env_bool, env_float, env_int, env_path


@dataclass(frozen=True)
class ForecastingSettings:
    model_path: str
    evidence_routing_enabled: bool
    routing_primary_metric: str
    routing_min_evaluation_points: int
    routing_min_relative_improvement: float
    routing_evidence_lookback_days: int
    uncertainty_min_residual_observations: int
    uncertainty_residual_lookback_days: int


def load_forecasting_settings() -> ForecastingSettings:
    primary_metric = os.getenv("ROUTING_PRIMARY_METRIC", "wape").strip().lower()
    min_points = env_int("ROUTING_MIN_EVALUATION_POINTS", 30)
    min_improvement = env_float("ROUTING_MIN_RELATIVE_IMPROVEMENT", 0.05)
    lookback_days = env_int("ROUTING_EVIDENCE_LOOKBACK_DAYS", 365)
    uncertainty_min_residuals = env_int("UNCERTAINTY_MIN_RESIDUAL_OBSERVATIONS", 30)
    uncertainty_lookback_days = env_int("UNCERTAINTY_RESIDUAL_LOOKBACK_DAYS", 365)

    if primary_metric not in {"wape", "mase", "mae", "rmse"}:
        raise ValueError("ROUTING_PRIMARY_METRIC must be one of: wape, mase, mae, rmse.")
    if min_points < 1:
        raise ValueError("ROUTING_MIN_EVALUATION_POINTS must be greater than or equal to 1.")
    if not 0 <= min_improvement <= 1:
        raise ValueError("ROUTING_MIN_RELATIVE_IMPROVEMENT must satisfy 0 <= value <= 1.")
    if lookback_days < 1:
        raise ValueError("ROUTING_EVIDENCE_LOOKBACK_DAYS must be greater than or equal to 1.")
    if uncertainty_min_residuals < 2:
        raise ValueError("UNCERTAINTY_MIN_RESIDUAL_OBSERVATIONS must be greater than or equal to 2.")
    if uncertainty_lookback_days < 1:
        raise ValueError("UNCERTAINTY_RESIDUAL_LOOKBACK_DAYS must be greater than or equal to 1.")

    return ForecastingSettings(
        model_path=env_path("MODEL_PATH", BACKEND_DIR / "saved_models"),
        evidence_routing_enabled=env_bool("EVIDENCE_ROUTING_ENABLED", False),
        routing_primary_metric=primary_metric,
        routing_min_evaluation_points=min_points,
        routing_min_relative_improvement=min_improvement,
        routing_evidence_lookback_days=lookback_days,
        uncertainty_min_residual_observations=uncertainty_min_residuals,
        uncertainty_residual_lookback_days=uncertainty_lookback_days,
    )
