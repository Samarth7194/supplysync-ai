"""SKU-level demand-profile features, computed from a trailing window.

These describe how sparse and how variable a SKU's *recent* demand is —
context the lag/rolling/calendar features alone can't express, and exactly
what separates the regular / intermittent / highly-intermittent regimes the
router already classifies by. They are trailing statistics shifted by one day
(mirroring ``lag_features.create_lag_features``), so the feature for day ``t``
uses only demand up to ``t-1``; computing them over a SKU's whole history
instead would leak future demand into training rows.

``inference_features.feature_vector`` computes the same three numbers one day
at a time for inference; ``tests/test_recursive_forecast.py`` asserts the two
agree exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.schema import SKU_FEATURE_COLUMNS

SKU_WINDOW = 28

__all__ = ["SKU_WINDOW", "SKU_FEATURE_COLUMNS", "create_sku_features"]


def create_sku_features(df: pd.DataFrame, target_col: str = "demand", window: int = SKU_WINDOW) -> pd.DataFrame:
    df = df.copy()
    shifted = df[target_col].shift(1)
    rolling = shifted.rolling(window)

    mean = rolling.mean()
    std = rolling.std()  # ddof=1, matching rolling_std_7 in lag_features.py

    zero_flag = (shifted == 0).astype(float)
    zero_flag[shifted.isna()] = np.nan

    safe_mean = mean.where(mean > 0, other=1.0)  # denominator only used where mean > 0; avoids a div-by-zero warning
    df["zero_share_28"] = zero_flag.rolling(window).mean()
    df["mean_28"] = mean
    df["cv_28"] = np.where(mean.isna(), np.nan, np.where(mean > 0, std / safe_mean, 0.0))
    return df
