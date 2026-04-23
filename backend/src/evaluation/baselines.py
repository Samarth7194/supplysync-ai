"""Simple forecast baselines for comparison against the trained model.

Each baseline exposes the same interface: given a demand history up to (but
not including) the test day, return a single one-step-ahead prediction.
This mirrors how the live analyze endpoint would reason day by day.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


BaselinePredictor = Callable[[pd.Series], float]


def naive_last_value(history: pd.Series) -> float:
    """Predict tomorrow = last observed day."""
    if len(history) == 0:
        return 0.0
    return float(history.iloc[-1])


def seasonal_naive_7(history: pd.Series) -> float:
    """Predict tomorrow = same-day-of-week a week ago (lag-7 naive)."""
    if len(history) < 7:
        return naive_last_value(history)
    return float(history.iloc[-7])


def moving_average_7(history: pd.Series) -> float:
    """Predict tomorrow = mean of the last 7 observed days."""
    if len(history) == 0:
        return 0.0
    window = history.tail(7)
    return float(window.mean())


def croston_single_step(history: pd.Series, alpha: float = 0.1) -> float:
    """One-step forecast from Croston's method with SBA bias correction.

    Reuses the same estimator as the live intermittent-demand path. The
    live service returns a horizon of equal values; here we return the
    single-day rate.
    """
    values = history.to_numpy(dtype=float)
    n = values.size
    if n == 0:
        return 0.0

    positive = values[values > 0]
    if positive.size == 0:
        return 0.0

    z = float(positive[0])  # demand size
    p = 1.0                 # inter-arrival
    q = 1

    for v in values:
        if v > 0:
            z = z + alpha * (v - z)
            p = p + alpha * (q - p)
            q = 1
        else:
            q += 1

    if p <= 0:
        return 0.0
    rate = z / p
    rate *= 1.0 - alpha / 2.0  # Syntetos-Boylan bias correction
    return float(max(0.0, rate))


BASELINES: dict[str, BaselinePredictor] = {
    "naive_last": naive_last_value,
    "seasonal_naive_7": seasonal_naive_7,
    "moving_avg_7": moving_average_7,
    "croston_sba": croston_single_step,
}


def run_baseline_over_window(
    history: pd.Series,
    test_window: pd.Series,
    baseline: BaselinePredictor,
) -> np.ndarray:
    """Walk day-by-day over ``test_window``, feeding actuals back into history.

    Returns a 1-D numpy array of predictions aligned with ``test_window``.
    """
    running = history.copy()
    out = np.empty(len(test_window), dtype=float)
    for i, (_, actual) in enumerate(test_window.items()):
        out[i] = max(0.0, baseline(running))
        running = pd.concat([running, pd.Series([actual], index=[test_window.index[i]])])
    return out
