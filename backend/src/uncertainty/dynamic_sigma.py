# src/uncertainty/dynamic_sigma.py

import numpy as np
import pandas as pd
from typing import Tuple

def compute_rolling_forecast_error(
    actuals: pd.Series, 
    predictions: pd.Series, 
    window_days: int = 30
) -> Tuple[float, float]:
    """
    Compute rolling forecast error statistics.
    
    Parameters:
    - actuals: Actual demand values
    - predictions: Predicted demand values  
    - window_days: Rolling window for error calculation
    
    Returns:
    - Tuple of (rolling_sigma, rolling_mean_error)
    """
    
    # Calculate absolute errors
    errors = actuals - predictions
    abs_errors = np.abs(errors)
    
    # Rolling statistics
    rolling_sigma = abs_errors.rolling(window=window_days, min_periods=1).std().iloc[-1]
    rolling_mean_error = abs_errors.rolling(window=window_days, min_periods=1).mean().iloc[-1]
    
    # Handle NaN for short series
    if pd.isna(rolling_sigma):
        rolling_sigma = abs_errors.std()
    if pd.isna(rolling_mean_error):
        rolling_mean_error = abs_errors.mean()
    
    return rolling_sigma, rolling_mean_error

def compute_dynamic_safety_stock(
    rolling_sigma: float,
    lead_time_days: int,
    service_level: float = 0.95,
    error_buffer: float = 1.2  # 20% buffer for uncertainty
) -> float:
    """
    Compute safety stock using dynamic forecast error.
    
    Parameters:
    - rolling_sigma: Rolling forecast error standard deviation
    - lead_time_days: Supplier lead time
    - service_level: Target service level (0.95 = 95%)
    - error_buffer: Additional buffer for uncertainty
    
    Returns:
    - Dynamic safety stock quantity
    """
    
    # Z-score for service level
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    Z = z_scores.get(service_level, 1.65)
    
    # Dynamic safety stock with uncertainty buffer
    safety_stock = Z * rolling_sigma * np.sqrt(lead_time_days) * error_buffer
    
    return safety_stock

def estimate_demand_variance(
    demand_series: pd.Series,
    window_days: int = 30
) -> float:
    """
    Estimate demand variance from historical data.
    
    Parameters:
    - demand_series: Historical demand values
    - window_days: Rolling window for variance calculation
    
    Returns:
    - Estimated demand variance
    """
    
    # Rolling variance
    rolling_var = demand_series.rolling(window=window_days, min_periods=1).var().iloc[-1]
    
    # Handle NaN for short series
    if pd.isna(rolling_var):
        rolling_var = demand_series.var()
    
    return rolling_var
