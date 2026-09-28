"""Tests for the v2 schema's SKU demand-profile features (zero_share_28, mean_28, cv_28)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features.inference_features import feature_vector  # noqa: E402
from features.schema import FEATURE_COLUMNS_V2, SKU_FEATURE_COLUMNS  # noqa: E402
from features.sku_features import SKU_WINDOW, create_sku_features  # noqa: E402


def _series(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.poisson(6, n).astype(float)
    values[rng.random(n) < 0.3] = 0.0
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def _frame(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"date": series.index, "demand": series.to_numpy()})


# ---------------------------------------------------------- create_sku_features (training)


def test_sku_features_are_nan_before_a_full_window_is_available():
    df = create_sku_features(_frame(_series(n=40)), target_col="demand")
    # shift(1) + rolling(28): first valid value needs 28 prior days, so index 28 (0-based) is first non-NaN.
    assert df["mean_28"].iloc[:28].isna().all()
    assert df["mean_28"].iloc[28:].notna().all()


def test_sku_features_use_only_data_strictly_before_the_row_date():
    series = _series(n=60)
    frame = _frame(series)
    df = create_sku_features(frame, target_col="demand")

    row = 40
    expected_window = series.to_numpy()[row - SKU_WINDOW:row]  # shift(1): excludes today
    assert df["mean_28"].iloc[row] == pytest.approx(expected_window.mean())
    assert df["zero_share_28"].iloc[row] == pytest.approx((expected_window == 0).mean())


def test_zero_share_is_bounded_between_zero_and_one():
    df = create_sku_features(_frame(_series(n=80, seed=2)), target_col="demand")
    valid = df["zero_share_28"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_cv_is_zero_when_mean_is_zero_not_nan_or_inf():
    all_zero = pd.Series(0.0, index=pd.date_range("2020-01-01", periods=40, freq="D"))
    df = create_sku_features(_frame(all_zero), target_col="demand")
    valid = df["cv_28"].iloc[28:]
    assert (valid == 0.0).all()
    assert np.isfinite(valid).all()


def test_constant_positive_series_has_zero_coefficient_of_variation():
    constant = pd.Series(5.0, index=pd.date_range("2020-01-01", periods=40, freq="D"))
    df = create_sku_features(_frame(constant), target_col="demand")
    assert df["cv_28"].iloc[28:].to_numpy() == pytest.approx(0.0, abs=1e-9)
    assert df["mean_28"].iloc[28:].to_numpy() == pytest.approx(5.0)
    assert df["zero_share_28"].iloc[28:].to_numpy() == pytest.approx(0.0)


# ---------------------------------------------------------- feature_vector parity (inference)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_feature_vector_sku_columns_match_create_sku_features(seed):
    """create_sku_features row i uses data through day i-1; feature_vector(arr, target_date, ...)
    uses arr through target_date-1. So row i (date = series.index[i]) matches
    feature_vector(series up to but excluding i, target_date=series.index[i])."""
    series = _series(n=70, seed=seed)
    df = create_sku_features(_frame(series), target_col="demand")
    expected_row = df.iloc[-1]  # date = series.index[-1]

    computed = feature_vector(series.to_numpy()[:-1], series.index[-1], SKU_FEATURE_COLUMNS)
    assert computed == pytest.approx(
        [expected_row["zero_share_28"], expected_row["mean_28"], expected_row["cv_28"]]
    )


def test_full_v2_schema_row_matches_training_pipeline():
    from features.lag_features import create_lag_features
    from features.time_features import create_time_features

    series = _series(n=70, seed=5)
    frame = _frame(series)
    anchor = series.index[-1]
    frame = pd.concat(
        [frame, pd.DataFrame({"date": [anchor + pd.Timedelta(days=1)], "demand": [np.nan]})],
        ignore_index=True,
    )
    frame = create_lag_features(frame, target_col="demand")
    frame = create_time_features(frame, date_col="date")
    frame = create_sku_features(frame, target_col="demand")
    expected = frame[FEATURE_COLUMNS_V2].iloc[-1].astype(float).tolist()

    computed = feature_vector(series.to_numpy(), anchor + pd.Timedelta(days=1), FEATURE_COLUMNS_V2)
    assert computed == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_feature_vector_returns_none_when_sku_window_is_short():
    values = np.arange(20, dtype=float)  # >= MIN_HISTORY_FOR_INFERENCE(14) but < SKU_FEATURE_WINDOW(28)
    assert feature_vector(values, pd.Timestamp("2020-02-01"), SKU_FEATURE_COLUMNS) is None


def test_v1_schema_is_unaffected_by_sku_feature_requirements():
    """A model still declaring the v1 schema must not need 28 days just because v2 exists."""
    from features.schema import FEATURE_COLUMNS_V1

    values = np.arange(14, dtype=float)  # exactly MIN_HISTORY_FOR_INFERENCE
    assert feature_vector(values, pd.Timestamp("2020-02-01"), FEATURE_COLUMNS_V1) is not None
