# src/services/adaptive_forecasting_service.py

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from forecasting.forecast_service import forecast_next_days
from features.inference_features import build_inference_features
from services.model_routing_service import RoutingDecision, ModelRoutingService

logger = logging.getLogger(__name__)

def classify_sku_demand_pattern(demand_series: pd.Series) -> str:
    """
    Classify SKU demand pattern based on intermittency.
    
    Parameters:
    - demand_series: Historical demand values for a SKU
    
    Returns:
    - Demand pattern: 'regular', 'intermittent', or 'highly_intermittent'
    """
    
    zero_demand_pct = (demand_series == 0).mean()
    
    if zero_demand_pct > 0.8:
        return "highly_intermittent"
    elif zero_demand_pct > 0.5:
        return "intermittent"
    else:
        return "regular"

def croston_forecast(
    demand_series: pd.Series,
    horizon: int,
    alpha: float = 0.1,
    bias_correct: bool = True,
) -> List[float]:
    """Croston's method for intermittent demand.

    With ``bias_correct=True`` applies the Syntetos-Boylan Approximation
    (SBA) to remove Croston's positive bias: forecast *= (1 - alpha/2).
    """
    demand = demand_series.values
    n = len(demand)

    positive_demands = demand[demand > 0]
    if len(positive_demands) == 0:
        return [0.0] * horizon

    z = float(positive_demands[0])  # demand size
    p = 1.0                         # inter-arrival interval
    q = 1

    forecasts = []
    for i in range(n):
        if demand[i] > 0:
            z = z + alpha * (demand[i] - z)
            p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
        forecasts.append(z / p if p > 0 else 0.0)

    last = forecasts[-1] if forecasts else 0.0
    if bias_correct:
        last *= (1.0 - alpha / 2.0)
    return [last] * horizon

def conservative_forecast(
    demand_series: pd.Series,
    horizon: int,
    buffer: float = 1.5,
) -> List[float]:
    """Conservative forecast for highly intermittent SKUs.

    Uses the full 30-day mean (including zeros) so the per-day rate is
    honest, then applies a buffer. Previously this dropped zeros *and*
    multiplied by 1.5 — effectively double-buffering for sparse SKUs.
    """
    recent = demand_series.tail(30)
    avg = float(recent.mean()) if len(recent) > 0 else 0.0
    return [avg * buffer] * horizon


def simple_average_forecast(demand_series: pd.Series, horizon: int) -> List[float]:
    return [float(demand_series.tail(7).mean())] * horizon


def _result(
    forecast: List[float],
    method: str,
    routing_decision: RoutingDecision,
    include_routing: bool,
) -> Any:
    if include_routing:
        return forecast, method, routing_decision.as_dict()
    return forecast, method

def adaptive_forecast(
    sku: str,
    demand_series: pd.Series,
    horizon: int = 7,
    model=None,
    last_features: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    routing_service: ModelRoutingService | None = None,
    routing_as_of_date=None,
    include_routing: bool = False,
) -> Tuple[List[float], str] | Tuple[List[float], str, dict[str, Any]]:
    """
    Adaptive forecasting based on SKU demand pattern.

    Parameters:
    - sku: SKU identifier
    - demand_series: Historical demand values
    - horizon: Forecast horizon
    - model: ML model (for regular demand)
    - last_features: Pre-built feature row (optional). When absent but
      ``model`` and ``feature_columns`` are provided, the row is built from
      ``demand_series`` using the same pipeline as training.
    - feature_columns: Ordered feature-name schema the model was trained on.

    Returns:
    - Tuple of (forecast_values, method_used)
    """

    demand_pattern = classify_sku_demand_pattern(demand_series)
    if routing_service is not None:
        routing_decision = routing_service.select_method(
            sku_code=sku,
            demand_pattern=demand_pattern,
            forecast_horizon=horizon,
            as_of_date=routing_as_of_date,
        )
    else:
        default_method = ModelRoutingService.default_method_for_pattern(demand_pattern)
        routing_decision = RoutingDecision(
            selected_method=default_method,
            default_method=default_method,
            selection_source="default",
            evidence_level="default",
            reason="Legacy demand-pattern routing used; no routing service was provided.",
            fallback_used=True,
        )

    selected_method = routing_decision.selected_method

    if selected_method == "croston":
        logger.info("SKU %s: %s demand -> croston", sku, demand_pattern)
        return _result(croston_forecast(demand_series, horizon), "croston", routing_decision, include_routing)

    if selected_method == "conservative":
        logger.info("SKU %s: %s demand -> conservative", sku, demand_pattern)
        return _result(conservative_forecast(demand_series, horizon), "conservative", routing_decision, include_routing)

    if selected_method == "simple_average":
        logger.info("SKU %s: %s demand -> simple_average", sku, demand_pattern)
        return _result(simple_average_forecast(demand_series, horizon), "simple_average", routing_decision, include_routing)

    if selected_method == "ml_lightgbm":
        features_df = last_features
        build_failure_reason = None

        if features_df is None and model is not None and feature_columns:
            try:
                features_df = build_inference_features(demand_series, feature_columns)
                if features_df is None:
                    build_failure_reason = "insufficient_history_or_nans"
            except Exception as exc:  # defensive: never let feature-build crash analysis
                logger.warning("Feature build failed for SKU %s: %s", sku, exc)
                build_failure_reason = "feature_build_exception"
                features_df = None

        if model is not None and features_df is not None:
            try:
                forecast = forecast_next_days(model, features_df, horizon)
                logger.info(
                    "SKU %s: %s demand -> trained LightGBM model (ml_lightgbm)",
                    sku,
                    demand_pattern,
                )
                return _result(forecast, "ml_lightgbm", routing_decision, include_routing)
            except Exception as exc:
                logger.warning(
                    "Model inference failed for SKU %s: %s; falling back to simple_average",
                    sku, exc,
                )

        if model is None:
            fallback_reason = "model_unavailable"
        elif feature_columns is None:
            fallback_reason = "no_feature_schema"
        elif features_df is None:
            fallback_reason = build_failure_reason or "feature_unavailable"
        else:
            fallback_reason = "model_inference_error"

        logger.info(
            "SKU %s: %s demand -> simple_average fallback (reason=%s)",
            sku,
            demand_pattern,
            fallback_reason,
        )
        return _result(simple_average_forecast(demand_series, horizon), "simple_average", routing_decision, include_routing)

    logger.info("SKU %s: invalid routing method %s -> legacy conservative fallback", sku, selected_method)
    fallback = conservative_forecast(demand_series, horizon)
    return _result(fallback, "conservative", routing_decision, include_routing)

def get_sku_classification_metadata(demand_series: pd.Series) -> Dict:
    """
    Get detailed classification metadata for a SKU.
    
    Parameters:
    - demand_series: Historical demand values
    
    Returns:
    - Dictionary with classification details
    """
    
    zero_demand_pct = (demand_series == 0).mean()
    demand_pattern = classify_sku_demand_pattern(demand_series)
    
    # Additional statistics
    positive_demands = demand_series[demand_series > 0]
    avg_positive_demand = positive_demands.mean() if len(positive_demands) > 0 else 0
    max_demand = demand_series.max()
    demand_std = demand_series.std()
    
    return {
        "demand_pattern": demand_pattern,
        "zero_demand_percentage": round(zero_demand_pct, 3),
        "average_positive_demand": round(avg_positive_demand, 2),
        "max_demand": int(max_demand),
        "demand_std": round(demand_std, 2),
        "total_days": len(demand_series),
        "positive_demand_days": len(positive_demands),
        "classification_confidence": "high" if zero_demand_pct < 0.3 or zero_demand_pct > 0.7 else "medium"
    }
