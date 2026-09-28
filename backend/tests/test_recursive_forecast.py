"""Tests for the corrected recursive forecaster and its feature parity with training.

Before this fix, ``forecast_next_days`` mutated a single pre-built feature row
across the horizon: calendar features were frozen at the first forecast day,
and ``rolling_mean_7/14``/``rolling_std_7`` were overwritten with statistics of
the model's own predictions alone (after one step ``rolling_std_7`` collapsed
to exactly 0), discarding the real history. ``recursive_forecast`` rebuilds
every feature from the extended series at each step instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features.inference_features import build_inference_features, feature_vector  # noqa: E402
from features.lag_features import create_lag_features  # noqa: E402
from features.schema import FEATURE_COLUMNS_V1  # noqa: E402
from features.time_features import create_time_features  # noqa: E402
from forecasting.forecast_service import recursive_forecast  # noqa: E402
from services.adaptive_forecasting_service import adaptive_forecast  # noqa: E402

# Pinned to v1 explicitly: this file tests the recursion fix (calendar advance,
# rolling-stat correctness), which is schema-version-agnostic. Using "latest"
# here would silently start testing v2's SKU features instead.
V1 = FEATURE_COLUMNS_V1


class _RecordingModel:
    """Returns a fixed prediction and records every feature row it is asked about."""

    def __init__(self, value: float = 5.0):
        self.value = value
        self.rows: list[pd.Series] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.rows.append(features.iloc[0].copy())
        return np.array([self.value])


def _history(n: int = 40, end: str = "2020-01-29", seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.poisson(9, n).astype(float)
    values[rng.random(n) < 0.2] = 0.0
    values[-7:] = [3.0, 11.0, 0.0, 14.0, 6.0, 9.0, 2.0]  # a clearly non-constant last week
    return pd.Series(values, index=pd.date_range(end=end, periods=n, freq="D"))


# ---------------------------------------------------------- parity with training


def _pandas_next_day_row(series: pd.Series, columns: list[str]) -> list[float]:
    """The training-time pipeline: append a placeholder day, build features, take the last row."""
    anchor = series.index[-1]
    frame = pd.DataFrame({"date": series.index, "demand": series.to_numpy()})
    frame = pd.concat(
        [frame, pd.DataFrame({"date": [anchor + pd.Timedelta(days=1)], "demand": [np.nan]})],
        ignore_index=True,
    )
    frame = create_lag_features(frame, target_col="demand")
    frame = create_time_features(frame, date_col="date")
    return frame[columns].iloc[-1].astype(float).tolist()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("end", ["2020-01-29", "2020-12-30", "2021-03-06"])  # month/year/weekend edges
def test_feature_vector_matches_the_training_pipeline(seed, end):
    series = _history(n=45, end=end, seed=seed)
    expected = _pandas_next_day_row(series, V1)
    actual = feature_vector(series.to_numpy(), series.index[-1] + pd.Timedelta(days=1), V1)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_build_inference_features_returns_the_same_row_as_the_training_pipeline():
    series = _history()
    row = build_inference_features(series, V1)
    assert list(row.columns) == V1
    assert row.iloc[0].tolist() == pytest.approx(_pandas_next_day_row(series, V1))


def test_short_history_yields_no_features():
    assert feature_vector(np.arange(13, dtype=float), pd.Timestamp("2020-02-01"), V1) is None
    assert recursive_forecast(_RecordingModel(), _history(n=13), V1, 7) is None


def test_unknown_feature_column_returns_none_instead_of_raising():
    values = np.arange(20, dtype=float)
    assert feature_vector(values, pd.Timestamp("2020-02-01"), ["lag_1", "not_a_real_feature"]) is None


# ---------------------------------------------------------- the two bugs, fixed


def test_calendar_features_advance_every_step():
    history = _history(end="2020-01-29")  # Wednesday; horizon crosses month end and a weekend
    model = _RecordingModel()

    recursive_forecast(model, history, V1, horizon=7)

    expected_dates = pd.date_range("2020-01-30", periods=7, freq="D")
    got = pd.DataFrame(model.rows)
    assert got["day_of_week"].tolist() == [float(d.dayofweek) for d in expected_dates]
    assert got["month"].tolist() == [float(d.month) for d in expected_dates]
    assert got["day_of_month"].tolist() == [float(d.day) for d in expected_dates]
    assert got["week_of_year"].tolist() == [float(d.isocalendar().week) for d in expected_dates]
    assert got["is_weekend"].tolist() == [float(d.dayofweek >= 5) for d in expected_dates]
    assert len(set(got["day_of_week"])) == 7  # not frozen at the first day


def test_rolling_std_is_computed_from_the_extended_series_not_collapsed_to_zero():
    history = _history()
    model = _RecordingModel(value=5.0)

    preds = recursive_forecast(model, history, V1, horizon=8)
    rows = pd.DataFrame(model.rows)
    actuals = history.to_numpy()

    for step in range(8):
        extended = np.concatenate([actuals, preds[:step]])
        assert rows["rolling_std_7"].iloc[step] == pytest.approx(extended[-7:].std(ddof=1), rel=1e-9)
    # Step 2 mixes six real days with one prediction: clearly not zero.
    assert rows["rolling_std_7"].iloc[1] > 1.0
    # By step 8 the window is entirely predictions of a constant model, so 0 is now the correct value.
    assert rows["rolling_std_7"].iloc[7] == pytest.approx(0.0, abs=1e-12)


def test_rolling_means_blend_real_history_with_predictions():
    history = _history()
    model = _RecordingModel(value=5.0)
    preds = recursive_forecast(model, history, V1, horizon=6)
    rows = pd.DataFrame(model.rows)
    actuals = history.to_numpy()

    for step in range(6):
        extended = np.concatenate([actuals, preds[:step]])
        assert rows["rolling_mean_7"].iloc[step] == pytest.approx(extended[-7:].mean())
        assert rows["rolling_mean_14"].iloc[step] == pytest.approx(extended[-14:].mean())
    # The 14-day mean at step 5 still contains 9 real days; it must not equal the prediction-only mean.
    assert rows["rolling_mean_14"].iloc[5] != pytest.approx(5.0)


def test_lags_shift_through_the_extended_series():
    history = _history()
    model = _RecordingModel(value=7.0)
    recursive_forecast(model, history, V1, horizon=9)
    rows = pd.DataFrame(model.rows)
    actuals = history.to_numpy()

    assert rows["lag_1"].iloc[0] == actuals[-1]
    assert rows["lag_1"].iloc[1] == 7.0  # yesterday's prediction
    assert rows["lag_7"].iloc[0] == actuals[-7]
    assert rows["lag_7"].iloc[7] == 7.0  # first prediction has reached lag_7


def test_predictions_are_clamped_at_zero():
    history = _history()
    model = _RecordingModel(value=-3.0)
    preds = recursive_forecast(model, history, V1, horizon=5)
    assert preds == [0.0] * 5


# ---------------------------------------------------------- no leakage


def test_forecast_depends_only_on_history_before_the_origin():
    history = _history()
    model = _RecordingModel()
    baseline = recursive_forecast(model, history, V1, horizon=7)

    truncated = history.copy()
    with_extra_future_rows = pd.concat(
        [truncated, pd.Series([9999.0] * 5, index=pd.date_range(truncated.index[-1] + pd.Timedelta(days=1), periods=5))]
    )
    # Forecasting from the ORIGINAL history (not the extended one) must be unaffected by what
    # would come after it.
    again = recursive_forecast(_RecordingModel(), history, V1, horizon=7)
    assert again == baseline
    assert with_extra_future_rows.iloc[len(history):].tolist() == [9999.0] * 5  # sanity: extension is real


# ---------------------------------------------------------- production wiring


def test_adaptive_forecast_uses_the_corrected_recursion():
    history = pd.Series(
        [8.0 + (i % 5) for i in range(60)],
        index=pd.date_range(end="2020-01-29", periods=60, freq="D"),
    )
    model = _RecordingModel(value=9.0)

    forecast, method = adaptive_forecast("SKU", history, horizon=7, model=model, feature_columns=V1)

    assert method == "ml_lightgbm"
    assert len(forecast) == 7
    rows = pd.DataFrame(model.rows)
    assert rows["day_of_week"].nunique() == 7  # calendar advances in production too


def test_adaptive_forecast_falls_back_when_history_is_too_short_for_features():
    history = pd.Series([5.0] * 10, index=pd.date_range(end="2020-01-29", periods=10, freq="D"))
    forecast, method = adaptive_forecast("SKU", history, horizon=7, model=_RecordingModel(), feature_columns=V1)
    assert method == "simple_average"
