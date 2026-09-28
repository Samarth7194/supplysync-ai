"""Build model feature rows from a demand series, one forecast day at a time.

``feature_vector`` is the single source of truth for what a feature means: it
computes every column for one target date from the values observed *before*
that date. ``build_inference_features`` wraps it for the common case (a single
next-day row from a raw series); ``forecasting.forecast_service.recursive_forecast``
calls it once per step of a multi-day forecast, extending the series with each
prediction so lags, rolling stats, calendar features, and SKU-profile features
all stay correct and advance with the date across the whole horizon.
``tests/test_recursive_forecast.py`` asserts this agrees exactly with the
vectorised training-time pipeline (``features.lag_features`` +
``features.time_features`` + ``features.sku_features``). A schema that doesn't
reference the SKU-profile columns (``demand_lag_calendar_v1``) is unaffected —
only requested columns are ever computed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

MIN_HISTORY_FOR_INFERENCE = 14  # needed for a non-NaN rolling_mean_14
SKU_FEATURE_WINDOW = 28  # matches features.sku_features.SKU_WINDOW

_NAN = float("nan")


def feature_vector(
    values: Sequence[float] | np.ndarray,
    target_date: pd.Timestamp,
    columns: Sequence[str],
) -> list[float] | None:
    """Feature values (in ``columns`` order) for ``target_date``, given prior ``values``.

    ``values[-1]`` is the demand on the day before ``target_date``. Returns
    ``None`` when history is too short for a requested feature, any value is
    not finite, or a column name isn't recognized.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < MIN_HISTORY_FOR_INFERENCE:
        return None

    cache: dict[str, float] = {}

    def compute(name: str) -> float:
        if name in cache:
            return cache[name]
        if name.startswith("lag_"):
            k = int(name.split("_", 1)[1])
            value = float(arr[-k]) if n >= k else _NAN
        elif name == "rolling_mean_7":
            value = float(arr[-7:].mean()) if n >= 7 else _NAN
        elif name == "rolling_std_7":
            value = float(arr[-7:].std(ddof=1)) if n >= 7 else _NAN
        elif name == "rolling_mean_14":
            value = float(arr[-14:].mean()) if n >= 14 else _NAN
        elif name == "day_of_week":
            value = float(target_date.dayofweek)
        elif name == "month":
            value = float(target_date.month)
        elif name == "is_weekend":
            value = float(target_date.dayofweek >= 5)
        elif name == "day_of_month":
            value = float(target_date.day)
        elif name == "week_of_year":
            value = float(target_date.isocalendar().week)
        elif name in ("zero_share_28", "mean_28", "cv_28"):
            w = SKU_FEATURE_WINDOW
            if n < w:
                value = _NAN
            else:
                window = arr[-w:]
                mean = float(window.mean())
                if name == "zero_share_28":
                    value = float((window == 0).mean())
                elif name == "mean_28":
                    value = mean
                else:
                    value = float(window.std(ddof=1) / mean) if mean > 0 else 0.0
        else:
            raise KeyError(f"Unknown feature column: {name!r}")
        cache[name] = value
        return value

    try:
        row = [compute(name) for name in columns]
    except KeyError:
        return None
    if any(not math.isfinite(v) for v in row):
        return None
    return row


def build_inference_features(
    demand_series: pd.Series,
    feature_columns: List[str],
    end_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.DataFrame]:
    """Return a 1-row DataFrame of next-day features, or ``None``.

    Returns ``None`` when history is too short, a column is unrecognized, or
    the resulting row contains non-finite values.
    """
    if demand_series is None or len(demand_series) < MIN_HISTORY_FOR_INFERENCE:
        return None

    values = pd.to_numeric(demand_series, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    if isinstance(demand_series.index, pd.DatetimeIndex) and not demand_series.index.hasnans:
        anchor = pd.DatetimeIndex(demand_series.index)[-1]
    else:
        anchor = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.utcnow().normalize()

    row = feature_vector(values, anchor + pd.Timedelta(days=1), feature_columns)
    if row is None:
        return None
    return pd.DataFrame([row], columns=list(feature_columns))
