"""Tests for the forecast-evaluation primitives.

Covers the metric math and the baseline predictors so the evaluation report
can be trusted. These primitives are pure and do not depend on the trained
model or the processed dataset.
"""

import math

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    ForecastMetrics,
    bias,
    compute_all,
    mae,
    mase,
    rmse,
    wape,
)
from evaluation.baselines import (
    BASELINES,
    croston_single_step,
    moving_average_7,
    naive_last_value,
    run_baseline_over_window,
    seasonal_naive_7,
)


# ---- metrics ----------------------------------------------------------------


def test_mae_rmse_bias_basic():
    actual = np.array([10, 12, 8, 14])
    predicted = np.array([11, 10, 9, 13])
    # errors: -1, 2, -1, 1  → |e|: 1,2,1,1
    assert mae(actual, predicted) == pytest.approx(1.25)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt((1 + 4 + 1 + 1) / 4))
    # bias = mean(pred - actual) = mean(1, -2, 1, -1) = -0.25
    assert bias(actual, predicted) == pytest.approx(-0.25)


def test_wape_denominator_handling():
    actual = np.array([5.0, 5.0, 0.0, 10.0])
    predicted = np.array([6.0, 4.0, 1.0, 8.0])
    # sum|err| = 1+1+1+2 = 5 ; sum|actual| = 20 ; WAPE = 0.25
    assert wape(actual, predicted) == pytest.approx(0.25)


def test_wape_returns_none_when_actuals_are_zero():
    actual = np.zeros(5)
    predicted = np.array([1.0, 0.0, 2.0, 0.0, 1.0])
    assert wape(actual, predicted) is None


def test_mase_less_than_one_when_better_than_naive():
    # In-sample with a clear drift: naive-last MAE will be sizable.
    in_sample = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    # naive_errors = |3-1, 5-3, 7-5, 9-7, 11-9| = 2 each, so scale = 2.
    actual = np.array([13.0, 15.0])
    predicted = np.array([13.0, 15.0])  # perfect
    assert mase(actual, predicted, in_sample) == pytest.approx(0.0)

    # Naive-last on the same window → MAE = 2 (each step misses by 2)
    predicted_naive = np.array([11.0, 13.0])
    # MAE = 2 ; scale = 2 ; MASE = 1.0
    assert mase(actual, predicted_naive, in_sample) == pytest.approx(1.0)


def test_mase_none_when_in_sample_is_flat():
    in_sample = np.array([5.0, 5.0, 5.0, 5.0])
    assert mase(np.array([1.0]), np.array([2.0]), in_sample) is None


def test_compute_all_returns_dict_with_expected_keys():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 1.9, 3.2])
    m = compute_all(actual, predicted, in_sample=np.array([1.0, 2.0, 3.0, 4.0]))
    assert isinstance(m, ForecastMetrics)
    d = m.as_dict()
    assert set(d.keys()) == {"mae", "rmse", "bias", "wape", "mase", "n"}
    assert d["n"] == 3


def test_align_filters_non_finite_pairs():
    actual = np.array([1.0, 2.0, np.nan, 4.0])
    predicted = np.array([1.0, np.inf, 3.0, 4.0])
    # Only indices 0 and 3 are fully finite; |err| = 0, 0
    assert mae(actual, predicted) == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae(np.array([1, 2, 3]), np.array([1, 2]))


# ---- baselines --------------------------------------------------------------


def _series(values):
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
        dtype=float,
    )


def test_naive_last_returns_last_value():
    assert naive_last_value(_series([1, 2, 3, 4])) == 4.0


def test_naive_last_zero_history_returns_zero():
    assert naive_last_value(pd.Series(dtype=float)) == 0.0


def test_seasonal_naive_7_uses_lag_7():
    s = _series(range(14))          # 0..13
    assert seasonal_naive_7(s) == 7.0


def test_seasonal_naive_7_falls_back_when_short():
    s = _series([3, 4, 5])
    assert seasonal_naive_7(s) == naive_last_value(s)


def test_moving_average_7_uses_last_7():
    s = _series([10, 20, 30, 10, 20, 30, 40])  # mean = 160/7
    assert moving_average_7(s) == pytest.approx(160 / 7)


def test_croston_zero_history_returns_zero():
    assert croston_single_step(_series([0, 0, 0])) == 0.0


def test_croston_positive_rate_for_intermittent_series():
    s = _series([0, 0, 10, 0, 0, 0, 5, 0, 0, 10, 0])
    assert croston_single_step(s) > 0.0


def test_run_baseline_over_window_feeds_actuals_back():
    history = _series([1, 2, 3, 4, 5])
    test_window = _series([6, 7, 8])
    # naive_last: predicts 5, 6, 7 (after feeding the real 6, 7 back in)
    preds = run_baseline_over_window(history, test_window, naive_last_value)
    assert list(preds) == [5.0, 6.0, 7.0]


def test_baselines_registry_matches_exports():
    # Guard against accidentally renaming a baseline without updating the map.
    assert set(BASELINES.keys()) == {
        "naive_last", "seasonal_naive_7", "moving_avg_7", "croston_sba",
    }
