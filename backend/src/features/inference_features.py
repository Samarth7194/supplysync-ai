"""Build a single-row feature DataFrame for next-day model inference.

Mirrors the training pipeline in ``scripts/train_model.py`` so the model sees
identical preprocessing at inference time: we append a placeholder row for the
next day, run ``create_lag_features`` + ``create_time_features``, then return
the last row in the exact column order the model was trained on.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from features.lag_features import create_lag_features
from features.time_features import create_time_features


MIN_HISTORY_FOR_INFERENCE = 14  # needed for a non-NaN rolling_mean_14


def build_inference_features(
    demand_series: pd.Series,
    feature_columns: List[str],
    end_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.DataFrame]:
    """Return a 1-row DataFrame matching ``feature_columns``, or ``None``.

    Returns ``None`` when history is too short, required columns cannot be
    produced, or the resulting row still contains NaNs.
    """
    if demand_series is None or len(demand_series) < MIN_HISTORY_FOR_INFERENCE:
        return None

    values = pd.to_numeric(demand_series, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    if isinstance(demand_series.index, pd.DatetimeIndex) and not demand_series.index.hasnans:
        dates = pd.DatetimeIndex(demand_series.index)
        anchor = dates[-1]
    else:
        anchor = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.utcnow().normalize()
        dates = pd.date_range(end=anchor, periods=len(values), freq="D")

    df = pd.DataFrame({"date": dates, "demand": values})

    # Append one "next-day" row so shifted lag/rolling features align with the
    # date we're actually predicting for.
    next_date = anchor + pd.Timedelta(days=1)
    df = pd.concat(
        [df, pd.DataFrame({"date": [next_date], "demand": [np.nan]})],
        ignore_index=True,
    )

    df = create_lag_features(df, target_col="demand")
    df = create_time_features(df, date_col="date")

    if any(col not in df.columns for col in feature_columns):
        return None

    last_row = df[feature_columns].iloc[[-1]].copy()
    if last_row.isna().any(axis=1).iloc[0]:
        return None

    return last_row
