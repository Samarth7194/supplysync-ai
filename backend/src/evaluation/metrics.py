"""Forecast-error metrics suited to retail / intermittent demand.

We avoid MAPE as the headline metric because it explodes when actuals are
near zero (a single-unit actual against a two-unit forecast is a 100% error).
The metric set below is standard for demand forecasting:

  * ``mae``    — mean absolute error, units of demand
  * ``rmse``   — root mean squared error, penalizes large misses
  * ``bias``   — mean error (predicted - actual), negative = under-forecast
  * ``wape``   — weighted absolute percentage error: ``sum|err| / sum|actual|``.
                 Well-defined as long as actuals don't sum to zero, and much
                 less noisy than row-wise MAPE on skewed data.
  * ``mase``   — mean absolute scaled error: MAE / in-sample naive MAE.
                 MASE < 1 means the forecast beats a naive repeat-last-value
                 baseline on the training series; MASE >= 1 means it doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    bias: float
    wape: Optional[float]
    mase: Optional[float]
    n: int

    def as_dict(self) -> dict:
        return {
            "mae": _round(self.mae),
            "rmse": _round(self.rmse),
            "bias": _round(self.bias),
            "wape": _round(self.wape),
            "mase": _round(self.mase),
            "n": self.n,
        }


def _round(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return float(round(value, 4))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = _align(actual, predicted)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = _align(actual, predicted)
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean error (predicted - actual). Negative = under-forecast on average."""
    actual, predicted = _align(actual, predicted)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(predicted - actual))


def wape(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """Weighted absolute percentage error: sum|err| / sum|actual|.

    Returns ``None`` if the denominator is zero (e.g. a horizon with zero
    total recorded demand), rather than silently emitting ``inf``.
    """
    actual, predicted = _align(actual, predicted)
    if actual.size == 0:
        return None
    denom = float(np.sum(np.abs(actual)))
    if denom == 0.0:
        return None
    return float(np.sum(np.abs(actual - predicted)) / denom)


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    in_sample: np.ndarray,
    seasonality: int = 1,
) -> Optional[float]:
    """Mean absolute scaled error.

    Scale is the in-sample MAE of a seasonal-naive forecast on ``in_sample``.
    With ``seasonality=1`` this reduces to naive-last-value scaling.
    Returns ``None`` when the scale can't be computed (series too short or
    all-identical values).
    """
    actual, predicted = _align(actual, predicted)
    if actual.size == 0:
        return None
    in_sample = np.asarray(in_sample, dtype=float)
    if in_sample.size <= seasonality:
        return None
    naive_errors = np.abs(in_sample[seasonality:] - in_sample[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale == 0.0:
        return None
    return float(mae(actual, predicted) / scale)


def compute_all(
    actual: np.ndarray,
    predicted: np.ndarray,
    in_sample: Optional[np.ndarray] = None,
    seasonality: int = 1,
) -> ForecastMetrics:
    """Compute the full metric set in one call."""
    actual, predicted = _align(actual, predicted)
    n = int(actual.size)
    if n == 0:
        return ForecastMetrics(
            mae=float("nan"), rmse=float("nan"), bias=float("nan"),
            wape=None, mase=None, n=0,
        )
    return ForecastMetrics(
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        bias=bias(actual, predicted),
        wape=wape(actual, predicted),
        mase=mase(actual, predicted, in_sample, seasonality) if in_sample is not None else None,
        n=n,
    )


def _align(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]
