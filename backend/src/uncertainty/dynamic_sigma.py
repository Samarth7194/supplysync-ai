# src/uncertainty/dynamic_sigma.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def compute_rolling_forecast_error(
    actuals: pd.Series,
    predictions: pd.Series,
    window_days: int = 30,
) -> Tuple[float, float]:
    """Rolling forecast error stats.

    Returns (rolling_sigma, rolling_mean_abs_error).
    sigma uses signed residuals (actual - predicted) so it reflects true
    error variance — using std of |errors| underestimates it and biases
    safety stock downward.
    """
    residuals = actuals - predictions
    abs_errors = residuals.abs()

    rolling_sigma = residuals.rolling(window=window_days, min_periods=2).std().iloc[-1]
    rolling_mae = abs_errors.rolling(window=window_days, min_periods=1).mean().iloc[-1]

    if pd.isna(rolling_sigma):
        rolling_sigma = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
    if pd.isna(rolling_mae):
        rolling_mae = abs_errors.mean() if len(abs_errors) > 0 else 0.0

    return float(rolling_sigma or 0.0), float(rolling_mae or 0.0)


def compute_dynamic_safety_stock(
    rolling_sigma: float,
    lead_time_days: int,
    service_level: float = 0.95,
    error_buffer: float = 1.2,
) -> float:
    """Safety stock from rolling forecast error, with uncertainty buffer."""
    z = float(stats.norm.ppf(service_level))
    return z * rolling_sigma * np.sqrt(lead_time_days) * error_buffer
