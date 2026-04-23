# src/uncertainty/prediction_intervals.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

def compute_residual_statistics(residuals: pd.Series) -> Dict:
    """
    Compute statistics of forecast residuals for uncertainty quantification.
    
    Parameters:
    - residuals: Forecast errors (actual - predicted)
    
    Returns:
    - Dictionary with residual statistics
    """
    
    residuals_clean = residuals.dropna()
    
    if len(residuals_clean) == 0:
        return {"mean": 0, "std": 1, "skewness": 0, "kurtosis": 0}
    
    stats_dict = {
        "mean": residuals_clean.mean(),
        "std": residuals_clean.std(),
        "skewness": residuals_clean.skew(),
        "kurtosis": residuals_clean.kurtosis(),
        "q25": residuals_clean.quantile(0.25),
        "q75": residuals_clean.quantile(0.75),
        "min": residuals_clean.min(),
        "max": residuals_clean.max()
    }
    
    return stats_dict

def compute_prediction_intervals(
    point_forecast: float,
    residual_stats: Dict,
    confidence_levels: List[float] = [0.80, 0.90, 0.95],
    method: str = "normal"
) -> Dict[str, float]:
    """
    Compute prediction intervals for a point forecast.
    
    Parameters:
    - point_forecast: Point forecast value
    - residual_stats: Statistics of forecast residuals
    - confidence_levels: List of confidence levels for intervals
    - method: Method for interval calculation ('normal', 'empirical', 'bootstrap')
    
    Returns:
    - Dictionary with prediction intervals
    """
    
    intervals = {}
    
    if method == "normal":
        # Normal distribution assumption
        std_error = residual_stats["std"]
        
        for confidence in confidence_levels:
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin = z_score * std_error
            
            intervals[f"lower_{int(confidence*100)}"] = max(0, point_forecast - margin)
            intervals[f"upper_{int(confidence*100)}"] = point_forecast + margin
            intervals[f"mean_{int(confidence*100)}"] = point_forecast
    
    elif method == "empirical":
        # Empirical quantiles from residuals
        for confidence in confidence_levels:
            alpha = 1 - confidence
            lower_quantile = residual_stats.get(f"q{int((alpha/2)*100)}", 
                                              residuals_clean.quantile(alpha/2) if 'residuals_clean' in locals() else 0)
            upper_quantile = residual_stats.get(f"q{int((1-alpha/2)*100)}", 
                                              residuals_clean.quantile(1-alpha/2) if 'residuals_clean' in locals() else std_error)
            
            intervals[f"lower_{int(confidence*100)}"] = max(0, point_forecast + lower_quantile)
            intervals[f"upper_{int(confidence*100)}"] = point_forecast + upper_quantile
            intervals[f"mean_{int(confidence*100)}"] = point_forecast
    
    return intervals

def uncertainty_aware_reorder_decision(
    base_decision: Dict,
    prediction_intervals: Dict,
    risk_appetite: str = "moderate"
) -> Dict:
    """
    Adjust reorder decision based on prediction intervals and risk appetite.
    
    Parameters:
    - base_decision: Base reorder decision from point forecast
    - prediction_intervals: Prediction intervals for demand
    - risk_appetite: 'conservative', 'moderate', 'aggressive'
    
    Returns:
    - Uncertainty-adjusted reorder decision
    """
    
    # Select demand forecast based on risk appetite
    if risk_appetite == "conservative":
        # Use upper bound of 95% interval
        demand_forecast = prediction_intervals.get("upper_95", base_decision["lead_time_demand"])
        adjustment_factor = 1.2  # 20% additional buffer
    elif risk_appetite == "aggressive":
        # Use lower bound of 80% interval
        demand_forecast = prediction_intervals.get("lower_80", base_decision["lead_time_demand"])
        adjustment_factor = 0.9  # 10% reduction
    else:  # moderate
        # Use point forecast
        demand_forecast = base_decision["lead_time_demand"]
        adjustment_factor = 1.0
    
    # Recalculate safety stock with uncertainty buffer
    base_safety_stock = base_decision["safety_stock"]
    uncertainty_buffer = 1.1 if risk_appetite == "conservative" else 1.0
    adjusted_safety_stock = base_safety_stock * uncertainty_buffer * adjustment_factor
    
    # Recalculate reorder point and order quantity
    adjusted_reorder_point = demand_forecast + adjusted_safety_stock
    adjusted_order_qty = max(0, adjusted_reorder_point - base_decision["current_stock"])
    
    # Create uncertainty-aware decision
    uncertain_decision = base_decision.copy()
    uncertain_decision.update({
        "lead_time_demand": round(demand_forecast, 2),
        "safety_stock": round(adjusted_safety_stock, 2),
        "reorder_point": round(adjusted_reorder_point, 2),
        "order_quantity": int(round(adjusted_order_qty)),
        "action": "REORDER" if adjusted_order_qty > 0 else "NO_ACTION",
        "uncertainty_adjustment": {
            "risk_appetite": risk_appetite,
            "demand_forecast_used": "upper_95" if risk_appetite == "conservative" else 
                                   "lower_80" if risk_appetite == "aggressive" else "point_estimate",
            "adjustment_factor": adjustment_factor,
            "uncertainty_buffer": uncertainty_buffer,
            "prediction_intervals": prediction_intervals
        }
    })
    
    return uncertain_decision

def compute_forecast_uncertainty_metrics(
    actuals: pd.Series,
    forecasts: pd.Series,
    prediction_intervals: Dict = None
) -> Dict:
    """
    Compute metrics to quantify forecast uncertainty quality.
    
    Parameters:
    - actuals: Actual values
    - forecasts: Point forecasts
    - prediction_intervals: Prediction intervals (optional)
    
    Returns:
    - Dictionary of uncertainty metrics
    """
    
    # Basic accuracy metrics
    mae = np.mean(np.abs(actuals - forecasts))
    rmse = np.sqrt(np.mean((actuals - forecasts) ** 2))
    mape = np.mean(np.abs((actuals - forecasts) / actuals.replace(0, np.nan))) * 100
    
    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "forecast_bias": round(np.mean(actuals - forecasts), 3)
    }
    
    # Interval coverage if available
    if prediction_intervals:
        for confidence in [80, 90, 95]:
            lower_key = f"lower_{confidence}"
            upper_key = f"upper_{confidence}"
            
            if lower_key in prediction_intervals and upper_key in prediction_intervals:
                coverage = np.mean((actuals >= prediction_intervals[lower_key]) & 
                                 (actuals <= prediction_intervals[upper_key]))
                metrics[f"coverage_{confidence}"] = round(coverage, 3)
                metrics[f"target_coverage_{confidence}"] = confidence / 100
    
    return metrics
