# src/services/adaptive_forecasting_service.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from forecasting.forecast_service import forecast_next_days

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

def croston_forecast(demand_series: pd.Series, horizon: int, alpha: float = 0.1) -> List[float]:
    """
    Croston's method for intermittent demand forecasting.
    
    Parameters:
    - demand_series: Historical demand values
    - horizon: Number of days to forecast
    - alpha: Smoothing parameter
    
    Returns:
    - List of forecasted values
    """
    
    demand = demand_series.values
    n = len(demand)
    
    # Initialize
    positive_demands = demand[demand > 0]
    if len(positive_demands) == 0:
        return [0.0] * horizon
    
    z = positive_demands[0]  # demand size
    p = 1                    # interval
    q = 1
    
    forecasts = []
    
    for i in range(n):
        if demand[i] > 0:
            z = z + alpha * (demand[i] - z)
            p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
        
        forecasts.append(z / p if p > 0 else 0)
    
    # Return last forecast repeated for horizon
    last_forecast = forecasts[-1] if forecasts else 0
    return [last_forecast] * horizon

def conservative_forecast(demand_series: pd.Series, horizon: int) -> List[float]:
    """
    Conservative forecasting for highly intermittent SKUs.
    Uses recent average with high safety buffer.
    
    Parameters:
    - demand_series: Historical demand values
    - horizon: Number of days to forecast
    
    Returns:
    - List of conservative forecasted values
    """
    
    # Use recent 30-day average
    recent_demand = demand_series.tail(30)
    avg_demand = recent_demand[recent_demand > 0].mean() if len(recent_demand[recent_demand > 0]) > 0 else 0
    
    # Conservative estimate with 50% buffer
    conservative_forecast = avg_demand * 1.5
    
    return [conservative_forecast] * horizon

def adaptive_forecast(
    sku: str,
    demand_series: pd.Series,
    horizon: int = 7,
    model=None,
    last_features=None
) -> Tuple[List[float], str]:
    """
    Adaptive forecasting based on SKU demand pattern.
    
    Parameters:
    - sku: SKU identifier
    - demand_series: Historical demand values
    - horizon: Forecast horizon
    - model: ML model (for regular demand)
    - last_features: Recent features for ML model
    
    Returns:
    - Tuple of (forecast_values, method_used)
    """
    
    # Classify demand pattern
    demand_pattern = classify_sku_demand_pattern(demand_series)
    
    if demand_pattern == "regular":
        if model is not None and last_features is not None:
            forecast = forecast_next_days(model, last_features, horizon)
            method = "ml_lightgbm"
        else:
            # Fallback to simple average
            forecast = [demand_series.tail(7).mean()] * horizon
            method = "simple_average"
    
    elif demand_pattern == "intermittent":
        forecast = croston_forecast(demand_series, horizon)
        method = "croston"
    
    else:  # highly_intermittent
        forecast = conservative_forecast(demand_series, horizon)
        method = "conservative"
    
    return forecast, method

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
