"""Lag-based feature engineering for demand forecasting."""

import pandas as pd


def create_lag_features(df: pd.DataFrame, target_col: str = "demand", lags: list = None) -> pd.DataFrame:
    df = df.copy()

    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7]

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling statistics (shifted by 1 to prevent leakage)
    shifted = df[target_col].shift(1)
    df["rolling_mean_7"] = shifted.rolling(7).mean()
    df["rolling_std_7"] = shifted.rolling(7).std()
    df["rolling_mean_14"] = shifted.rolling(14).mean()

    return df
