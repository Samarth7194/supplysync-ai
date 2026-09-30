"""Tests for the shared rolling-origin, multi-step backtest module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import backtest as bt  # noqa: E402
from features.schema import FEATURE_COLUMNS  # noqa: E402


class _LinearModel:
    """Deterministic stand-in for LightGBM: depends on lags, rolling stats and calendar."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        row = features.iloc[0]
        return np.array(
            [0.5 * row["lag_1"] + 0.3 * row["rolling_mean_7"] + 0.2 * row["rolling_mean_14"] + 0.5 * row["day_of_week"]]
        )


def _series(n: int = 150, seed: int = 0, start: str = "2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.poisson(8, n).astype(float)
    values[rng.random(n) < 0.15] = 0.0
    return pd.Series(values, index=pd.date_range(start, periods=n, freq="D"))


def _all_forecasters() -> dict:
    model = _LinearModel()
    return {
        bt.LIGHTGBM: bt.lightgbm_forecaster(model, FEATURE_COLUMNS),
        bt.PRODUCTION_ROUTED: bt.production_forecaster(model, FEATURE_COLUMNS),
    }


# ---------------------------------------------------------------- leakage


def test_forecast_at_origin_is_identical_whether_or_not_future_data_exists():
    series = _series()
    forecasters = {**bt.reference_forecasters(), **_all_forecasters()}
    t = 100
    horizon = 14

    baseline = bt.forecast_at_origin(series, t, horizon, forecasters, sku="A")

    truncated = series.iloc[:t]
    without_future = bt.forecast_at_origin(truncated, t, horizon, forecasters, sku="A")

    poisoned = series.copy()
    poisoned.iloc[t:] = np.random.default_rng(1).uniform(5_000, 9_000, len(poisoned) - t)
    with_poisoned_future = bt.forecast_at_origin(poisoned, t, horizon, forecasters, sku="A")

    for name in forecasters:
        assert baseline[name] is not None, name
        assert baseline[name].values == without_future[name].values, f"{name} changed when future data removed"
        assert baseline[name].values == with_poisoned_future[name].values, f"{name} changed when future data altered"


def test_forecasters_only_ever_see_history_before_the_origin():
    series = _series()
    seen: list[tuple[pd.Timestamp, int]] = []

    def probe(sku: str, history: pd.Series, horizon: int):
        seen.append((history.index[-1], len(history)))
        return bt.Forecast((1.0,) * horizon, "probe")

    config = bt.BacktestConfig(horizon=7, eval_days=30)
    dataset_end = series.index[-1]
    bt.run_backtest({"A": series}, {"probe": probe}, config, dataset_end=dataset_end)

    cutoff = dataset_end - pd.Timedelta(days=config.eval_days)
    assert seen, "probe was never called"
    for last_seen, length in seen:
        assert last_seen >= cutoff  # origin is inside the evaluation period
        assert last_seen <= dataset_end - pd.Timedelta(days=config.horizon)  # targets fit inside the data
        assert length <= config.history_window  # same trailing window production uses


def test_no_actuals_are_fed_back_between_horizon_steps():
    """A multi-step forecast must not change when the in-horizon actuals change."""
    series = _series()
    forecasters = _all_forecasters()
    t, horizon = 90, 7
    original = bt.forecast_at_origin(series, t, horizon, forecasters, sku="A")

    shifted = series.copy()
    shifted.iloc[t:t + horizon] += 1_000.0  # would change a one-step-ahead walk immediately
    changed = bt.forecast_at_origin(shifted, t, horizon, forecasters, sku="A")

    for name in forecasters:
        assert original[name].values == changed[name].values


# ---------------------------------------------------------------- origins


def test_origin_positions_stay_inside_evaluation_period():
    series = _series(n=150)
    config = bt.BacktestConfig(horizon=7, eval_days=30)
    positions = bt.origin_positions(series, config, dataset_end=series.index[-1])

    assert positions == list(range(150 - 30, 150 - 7 + 1))
    for t in positions:
        first_target = series.index[t]
        assert first_target > series.index[-1] - pd.Timedelta(days=30)
        assert t + 7 <= len(series)


def test_longer_horizon_means_fewer_origins_but_still_multi_step():
    series = _series(n=150)
    short = bt.origin_positions(series, bt.BacktestConfig(horizon=7), dataset_end=series.index[-1])
    long = bt.origin_positions(series, bt.BacktestConfig(horizon=14), dataset_end=series.index[-1])
    assert len(long) == len(short) - 7


def test_config_rejects_eval_window_shorter_than_horizon():
    with pytest.raises(ValueError):
        bt.BacktestConfig(horizon=14, eval_days=7)


# ---------------------------------------------------------------- metrics


def test_predict_zero_scores_wape_of_exactly_one():
    result = bt.run_backtest({"A": _series(), "B": _series(seed=3)}, {}, bt.BacktestConfig(horizon=7))
    zero = result.aggregates["all"]["predict_zero"]
    assert zero["wape"] == 1.0
    assert zero["wape_lead_time_sum"] == 1.0
    assert zero["bias_ratio"] == -1.0


def test_reference_rows_are_always_present_for_every_class_scored():
    result = bt.run_backtest({"A": _series()}, {}, bt.BacktestConfig(horizon=7))
    for bucket, methods in result.aggregates.items():
        for name in bt.REFERENCE_METHODS:
            assert name in methods, f"{name} missing from {bucket}"


def test_constant_demand_hand_computed_metrics():
    series = pd.Series(10.0, index=pd.date_range("2020-01-01", periods=120, freq="D"))
    result = bt.run_backtest({"A": series}, {}, bt.BacktestConfig(horizon=7))
    metrics = result.aggregates["regular"]

    assert metrics["moving_avg_7"]["wape"] == 0.0
    assert metrics["seasonal_naive_7"]["wape"] == 0.0
    # Croston-SBA on a constant positive series: 10 * (1 - alpha/2) = 9.5 -> 5% low.
    assert metrics["croston_sba"]["wape"] == pytest.approx(0.05, abs=1e-4)
    assert metrics["croston_sba"]["wape_lead_time_sum"] == pytest.approx(0.05, abs=1e-4)
    assert metrics["croston_sba"]["bias_ratio"] == pytest.approx(-0.05, abs=1e-4)


def test_lead_time_sum_wape_is_computed_on_the_horizon_total():
    """Offsetting daily errors cancel in the lead-time total but not in daily WAPE."""
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    alternating = pd.Series([0.0, 20.0] * 50, index=index)

    def flat_ten(sku, history, horizon):
        return bt.Forecast((10.0,) * horizon, "flat_ten")

    result = bt.run_backtest({"A": alternating}, {"flat_ten": flat_ten}, bt.BacktestConfig(horizon=6, eval_days=30))
    row = result.aggregates["all"]["flat_ten"]
    assert row["wape"] == pytest.approx(1.0, abs=1e-6)  # every day is off by 10 of mean 10
    assert row["wape_lead_time_sum"] == pytest.approx(0.0, abs=1e-6)  # 3 highs + 3 lows sum to 60 = 6*10


def test_none_from_any_forecaster_skips_the_origin_for_all_methods():
    series = _series()
    calls = {"n": 0}

    def flaky(sku, history, horizon):
        calls["n"] += 1
        return None if calls["n"] % 2 == 0 else bt.Forecast((1.0,) * horizon, "flaky")

    result = bt.run_backtest({"A": series}, {"flaky": flaky}, bt.BacktestConfig(horizon=7))
    assert result.n_skipped_origins > 0
    counts = {m["n_origins"] for m in result.aggregates["all"].values()}
    assert counts == {result.n_origins}


def test_production_routed_records_which_method_was_used():
    result = bt.run_backtest(
        {"A": _series()},
        {bt.PRODUCTION_ROUTED: bt.production_forecaster(_LinearModel(), FEATURE_COLUMNS)},
        bt.BacktestConfig(horizon=7),
    )
    assert sum(result.routed_method_counts.values()) == result.n_origins
    assert set(result.routed_method_counts) <= {"ml_lightgbm", "croston", "conservative", "simple_average"}


def test_production_forecaster_uses_the_routing_service_when_given_one():
    """Real bug: evaluate_forecast.py built production_forecaster() without
    ever passing a routing_service, so "production_routed" always used the
    legacy per-pattern default (ml_lightgbm) no matter what the evidence
    said -- ModelRoutingService.select_method() was never even called. A
    routing_service whose evidence favors Croston must actually change what
    production_forecaster returns."""
    from services.model_routing_service import RoutingDecision

    class _StubRouter:
        def select_method(self, **kwargs):
            return RoutingDecision(
                selected_method="croston", default_method="ml_lightgbm",
                selection_source="offline", evidence_level="pattern",
                reason="test evidence favors croston", fallback_used=False,
            )

    forecaster = bt.production_forecaster(_LinearModel(), FEATURE_COLUMNS, routing_service=_StubRouter())
    forecast = forecaster("A", _series(), 7)

    assert forecast.method == "croston"


# ---------------------------------------------------------------- eval windows


def test_validation_dataset_end_is_held_out_days_before_the_real_end():
    end = pd.Timestamp("2011-12-09")
    assert bt.validation_dataset_end(end) == end - pd.Timedelta(days=bt.HELD_OUT_DAYS)


# ---------------------------------------------------------------- selection


def test_strongest_baseline_includes_predict_zero_and_picks_lowest():
    metrics = {
        "predict_zero": {"wape_lead_time_sum": 1.0},
        "moving_avg_7": {"wape_lead_time_sum": 0.6},
        "croston_sba": {"wape_lead_time_sum": 0.4},
        "lightgbm": {"wape_lead_time_sum": 0.1},  # not a baseline
    }
    assert bt.strongest_baseline(metrics) == ("croston_sba", 0.4)
    assert bt.strongest_baseline({"predict_zero": {"wape_lead_time_sum": 1.0}}) == ("predict_zero", 1.0)
    assert bt.strongest_baseline({}) is None


def test_sku_selection_ignores_demand_after_the_cutoff():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    cutoff = dates[69]
    rows = []
    for d in dates:
        rows.append(("STEADY", d, 5.0))
        rows.append(("LATE_SPIKE", d, 1.0 if d <= cutoff else 10_000.0))
    daily = pd.DataFrame(rows, columns=["StockCode", "date", "demand"])

    ranked = bt.select_eval_skus(daily, cutoff=cutoff, min_active_days=30, max_skus=2)
    assert ranked == ["STEADY", "LATE_SPIKE"]  # ranked on pre-cutoff totals only


def test_prepare_sku_series_can_pad_to_common_end_with_zeros():
    daily = pd.DataFrame(
        {
            "StockCode": ["A", "A", "B"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]),
            "demand": [4.0, 6.0, 9.0],
        }
    )
    end = pd.Timestamp("2020-01-08")
    padded = bt.prepare_sku_series(daily, pad_to=end)
    assert padded["A"].index[-1] == end
    assert padded["A"].tolist() == [4.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    unpadded = bt.prepare_sku_series(daily)
    assert unpadded["A"].index[-1] == pd.Timestamp("2020-01-03")


# ---------------------------------------------------------------- uncertainty


def test_bootstrap_paired_difference_of_identical_methods_is_zero():
    result = bt.run_backtest({f"S{i}": _series(seed=i) for i in range(8)}, {}, bt.BacktestConfig(horizon=7))
    ci = bt.bootstrap_wape_ci(
        result.per_sku, demand_class="all", method="moving_avg_7", reference="moving_avg_7", n_boot=50
    )
    assert ci is not None
    assert ci["point"] == pytest.approx(0.0, abs=1e-12)
    assert ci["lo"] == pytest.approx(0.0, abs=1e-12) and ci["hi"] == pytest.approx(0.0, abs=1e-12)


def test_payload_is_schema_v2_and_backward_compatible_with_consumers():
    result = bt.run_backtest({"A": _series()}, {}, bt.BacktestConfig(horizon=7))
    payload = result.to_payload()
    assert payload["schema_version"] == 2
    assert payload["evaluation_mode"] == "multi_step_rolling_origin"
    assert payload["horizon_days"] == 7
    row = payload["aggregates"]["all"]["moving_avg_7"]
    for key in ("wape", "wape_lead_time_sum", "mae", "rmse", "bias", "mase", "n_skus", "n_test_points"):
        assert key in row
