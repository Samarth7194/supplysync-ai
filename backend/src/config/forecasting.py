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
    model_monitoring_enabled: bool
    model_monitoring_window_evaluations: int
    model_monitoring_lookback_days: int
    model_monitoring_min_evaluations: int
    model_monitoring_wape_warning_threshold: float
    model_monitoring_wape_degradation_threshold: float
    model_monitoring_bias_warning_ratio: float
    model_monitoring_degradation_consecutive_runs: int
    auto_retrain_enabled: bool
    model_retrain_min_evaluated_forecast_days: int
    model_retrain_cooldown_days: int
    model_retrain_require_degraded_status: bool


def load_forecasting_settings() -> ForecastingSettings:
    primary_metric = os.getenv("ROUTING_PRIMARY_METRIC", "wape").strip().lower()
    min_points = env_int("ROUTING_MIN_EVALUATION_POINTS", 30)
    min_improvement = env_float("ROUTING_MIN_RELATIVE_IMPROVEMENT", 0.05)
    lookback_days = env_int("ROUTING_EVIDENCE_LOOKBACK_DAYS", 365)
    uncertainty_min_residuals = env_int("UNCERTAINTY_MIN_RESIDUAL_OBSERVATIONS", 30)
    uncertainty_lookback_days = env_int("UNCERTAINTY_RESIDUAL_LOOKBACK_DAYS", 365)
    monitoring_window = env_int("MODEL_MONITORING_WINDOW_EVALUATIONS", 30)
    monitoring_lookback_days = env_int("MODEL_MONITORING_LOOKBACK_DAYS", 90)
    monitoring_min_evaluations = env_int("MODEL_MONITORING_MIN_EVALUATIONS", 30)
    monitoring_wape_warning = env_float("MODEL_MONITORING_WAPE_WARNING_THRESHOLD", 0.15)
    monitoring_wape_degradation = env_float("MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD", 0.25)
    monitoring_bias_warning = env_float("MODEL_MONITORING_BIAS_WARNING_RATIO", 0.20)
    monitoring_consecutive_runs = env_int("MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS", 2)
    retrain_min_forecast_days = env_int("MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS", 100)
    retrain_cooldown_days = env_int("MODEL_RETRAIN_COOLDOWN_DAYS", 14)

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
    if monitoring_window < 1:
        raise ValueError("MODEL_MONITORING_WINDOW_EVALUATIONS must be greater than or equal to 1.")
    if monitoring_lookback_days < 1:
        raise ValueError("MODEL_MONITORING_LOOKBACK_DAYS must be greater than or equal to 1.")
    if monitoring_min_evaluations < 1:
        raise ValueError("MODEL_MONITORING_MIN_EVALUATIONS must be greater than or equal to 1.")
    if monitoring_wape_warning < 0:
        raise ValueError("MODEL_MONITORING_WAPE_WARNING_THRESHOLD must be greater than or equal to 0.")
    if monitoring_wape_degradation < 0:
        raise ValueError("MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD must be greater than or equal to 0.")
    if monitoring_wape_warning >= monitoring_wape_degradation:
        raise ValueError(
            "MODEL_MONITORING_WAPE_WARNING_THRESHOLD must be less than "
            "MODEL_MONITORING_WAPE_DEGRADATION_THRESHOLD."
        )
    if monitoring_bias_warning < 0:
        raise ValueError("MODEL_MONITORING_BIAS_WARNING_RATIO must be greater than or equal to 0.")
    if monitoring_consecutive_runs < 1:
        raise ValueError("MODEL_MONITORING_DEGRADATION_CONSECUTIVE_RUNS must be greater than or equal to 1.")
    if retrain_min_forecast_days < 1:
        raise ValueError("MODEL_RETRAIN_MIN_EVALUATED_FORECAST_DAYS must be greater than or equal to 1.")
    if retrain_cooldown_days < 0:
        raise ValueError("MODEL_RETRAIN_COOLDOWN_DAYS must be greater than or equal to 0.")

    return ForecastingSettings(
        model_path=env_path("MODEL_PATH", BACKEND_DIR / "saved_models"),
        evidence_routing_enabled=env_bool("EVIDENCE_ROUTING_ENABLED", False),
        routing_primary_metric=primary_metric,
        routing_min_evaluation_points=min_points,
        routing_min_relative_improvement=min_improvement,
        routing_evidence_lookback_days=lookback_days,
        uncertainty_min_residual_observations=uncertainty_min_residuals,
        uncertainty_residual_lookback_days=uncertainty_lookback_days,
        model_monitoring_enabled=env_bool("MODEL_MONITORING_ENABLED", True),
        model_monitoring_window_evaluations=monitoring_window,
        model_monitoring_lookback_days=monitoring_lookback_days,
        model_monitoring_min_evaluations=monitoring_min_evaluations,
        model_monitoring_wape_warning_threshold=monitoring_wape_warning,
        model_monitoring_wape_degradation_threshold=monitoring_wape_degradation,
        model_monitoring_bias_warning_ratio=monitoring_bias_warning,
        model_monitoring_degradation_consecutive_runs=monitoring_consecutive_runs,
        auto_retrain_enabled=env_bool("AUTO_RETRAIN_ENABLED", False),
        model_retrain_min_evaluated_forecast_days=retrain_min_forecast_days,
        model_retrain_cooldown_days=retrain_cooldown_days,
        model_retrain_require_degraded_status=env_bool("MODEL_RETRAIN_REQUIRE_DEGRADED_STATUS", True),
    )
