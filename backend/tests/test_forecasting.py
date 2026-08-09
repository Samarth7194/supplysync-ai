"""Tests for adaptive forecasting service: classification, Croston, and fallback methods."""

import pytest
import pandas as pd
import numpy as np
from services.adaptive_forecasting_service import (
    classify_sku_demand_pattern,
    croston_forecast,
    conservative_forecast,
    adaptive_forecast,
)
from services.intelligent_inventory_service import IntelligentInventoryService
from services.model_routing_service import RoutingDecision
from features.inference_features import (
    build_inference_features,
    MIN_HISTORY_FOR_INFERENCE,
)
from features.schema import FEATURE_COLUMNS


TRAIN_FEATURE_COLUMNS = FEATURE_COLUMNS


class _StubModel:
    """Deterministic stand-in for a trained regressor.

    Records the feature frame it was asked to score so tests can assert the
    column schema the service handed in.
    """

    def __init__(self, constant: float = 42.0):
        self.constant = constant
        self.calls = []

    def predict(self, features):
        self.calls.append(features)
        return np.array([self.constant])


class _RoutingService:
    def __init__(self, selected_method: str, default_method: str = "ml_lightgbm"):
        self.selected_method = selected_method
        self.default_method = default_method

    def select_method(self, **kwargs):
        return RoutingDecision(
            selected_method=self.selected_method,
            default_method=self.default_method,
            selection_source="logged",
            evidence_level="pattern",
            reason=f"unit-test selected {self.selected_method}",
            metric_name="wape",
            selected_metric_value=0.7,
            baseline_metric_value=1.0,
            evaluation_sample_size=30,
            evaluation_count=3,
            evidence_age_days=1,
            fallback_used=False,
        )


# --- SKU Classification Tests ---

def test_classify_regular_demand():
    # Less than 50% zeros
    series = pd.Series([10, 0, 15, 20, 0, 12, 18, 25, 0, 14])
    assert classify_sku_demand_pattern(series) == "regular"


def test_classify_intermittent_demand():
    # 50-80% zeros
    series = pd.Series([0, 0, 0, 10, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 3, 0, 0, 0, 0])
    pattern = classify_sku_demand_pattern(series)
    assert pattern == "intermittent"


def test_classify_highly_intermittent_demand():
    # More than 80% zeros
    series = pd.Series([0] * 18 + [5, 3])
    assert classify_sku_demand_pattern(series) == "highly_intermittent"


# --- Croston Forecast Tests ---

def test_croston_produces_positive_values():
    series = pd.Series([0, 0, 10, 0, 0, 0, 5, 0, 0, 8, 0, 0])
    forecasts = croston_forecast(series, horizon=7)
    assert len(forecasts) == 7
    assert all(v >= 0 for v in forecasts)


def test_croston_all_zeros_returns_zeros():
    series = pd.Series([0] * 20)
    forecasts = croston_forecast(series, horizon=7)
    assert forecasts == [0.0] * 7


def test_croston_length_matches_horizon():
    series = pd.Series([0, 5, 0, 0, 10, 0, 3, 0, 0, 0])
    for horizon in [3, 7, 14]:
        forecasts = croston_forecast(series, horizon=horizon)
        assert len(forecasts) == horizon


# --- Conservative Forecast Tests ---

def test_conservative_has_buffer_over_average():
    series = pd.Series([1, 0, 2, 0, 0, 3, 0, 0, 0, 1])
    forecasts = conservative_forecast(series, horizon=7)
    simple_avg = series.tail(30).mean()
    # Conservative applies 1.5x buffer
    assert all(v >= simple_avg for v in forecasts)


# --- Adaptive Forecast Tests ---

def test_adaptive_regular_no_model_uses_fallback():
    series = pd.Series([10, 15, 12, 18, 20, 14, 16, 22, 11, 19])
    forecasts, method = adaptive_forecast("TEST", series, horizon=7, model=None, last_features=None)
    assert method == "simple_average"
    assert len(forecasts) == 7


def test_adaptive_intermittent_uses_croston():
    series = pd.Series([0, 0, 0, 10, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 3, 0, 0, 0, 0])
    forecasts, method = adaptive_forecast("TEST", series, horizon=7, model=None, last_features=None)
    assert method == "croston"
    assert len(forecasts) == 7


def test_adaptive_highly_intermittent_uses_conservative():
    series = pd.Series([0] * 18 + [5, 3])
    forecasts, method = adaptive_forecast("TEST", series, horizon=7, model=None, last_features=None)
    assert method == "conservative"
    assert len(forecasts) == 7


# --- Inference Feature Builder Tests ---

def _regular_series(length=30, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.integers(5, 25, length).astype(float)
    dates = pd.date_range(end=pd.Timestamp("2024-01-31"), periods=length, freq="D")
    return pd.Series(values, index=dates)


def test_build_inference_features_matches_training_schema():
    series = _regular_series(length=30)
    row = build_inference_features(series, TRAIN_FEATURE_COLUMNS)
    assert row is not None
    assert list(row.columns) == TRAIN_FEATURE_COLUMNS
    assert row.shape == (1, len(TRAIN_FEATURE_COLUMNS))
    assert not row.isna().any(axis=1).iloc[0]
    # lag_1 must equal the most recent observed demand
    assert float(row["lag_1"].iloc[0]) == float(series.iloc[-1])
    # rolling_mean_7 should equal the mean of the last 7 observed values
    assert float(row["rolling_mean_7"].iloc[0]) == pytest.approx(float(series.tail(7).mean()))


def test_build_inference_features_returns_none_when_history_too_short():
    short = pd.Series([1.0] * (MIN_HISTORY_FOR_INFERENCE - 1))
    assert build_inference_features(short, TRAIN_FEATURE_COLUMNS) is None


# --- Adaptive Forecast: Trained-Model Path ---

def test_adaptive_regular_builds_features_and_uses_model():
    model = _StubModel(constant=17.5)
    series = _regular_series(length=30)

    forecasts, method = adaptive_forecast(
        "TEST",
        series,
        horizon=7,
        model=model,
        last_features=None,
        feature_columns=TRAIN_FEATURE_COLUMNS,
    )

    assert method == "ml_lightgbm"
    assert len(forecasts) == 7
    assert all(v >= 0 for v in forecasts)
    # Model must have been invoked with the exact training schema
    assert model.calls, "model.predict was never called"
    first_call_cols = list(model.calls[0].columns)
    assert first_call_cols == TRAIN_FEATURE_COLUMNS


def test_adaptive_regular_falls_back_when_no_feature_schema():
    model = _StubModel(constant=17.5)
    series = _regular_series(length=30)

    _, method = adaptive_forecast(
        "TEST", series, horizon=7, model=model, feature_columns=None,
    )
    assert method == "simple_average"
    assert not model.calls


def test_adaptive_regular_falls_back_when_history_too_short():
    model = _StubModel(constant=17.5)
    series = pd.Series([10.0, 12.0, 8.0, 9.0, 11.0])  # way below MIN_HISTORY_FOR_INFERENCE

    _, method = adaptive_forecast(
        "TEST", series, horizon=7, model=model, feature_columns=TRAIN_FEATURE_COLUMNS,
    )
    assert method == "simple_average"


def test_intelligent_service_uses_model_end_to_end():
    """Regular-demand SKU with sufficient history must drive inference
    through the trained model path when the schema is wired up."""
    model = _StubModel(constant=20.0)
    service = IntelligentInventoryService(
        model=model, model_feature_columns=TRAIN_FEATURE_COLUMNS,
    )
    series = _regular_series(length=30, seed=1)

    decision = service.get_intelligent_reorder_decision(
        sku="TEST", current_stock=50, demand_history=series, lead_time_days=7,
    )

    assert decision["intelligence"]["forecast_method"] == "ml_lightgbm"
    assert decision["intelligence"]["demand_pattern"] == "regular"
    assert "forecast_daily" in decision
    assert len(decision["forecast_daily"]) == 7
    assert model.calls  # trained model actually invoked


def test_intelligent_service_falls_back_without_model():
    service = IntelligentInventoryService(model=None, model_feature_columns=None)
    series = _regular_series(length=30, seed=2)

    decision = service.get_intelligent_reorder_decision(
        sku="TEST", current_stock=50, demand_history=series, lead_time_days=7,
    )
    assert decision["intelligence"]["forecast_method"] == "simple_average"


def test_intelligent_service_intermittent_still_uses_croston():
    """Intermittent SKUs must NOT go through the ML path, even when the model
    and schema are available."""
    model = _StubModel(constant=20.0)
    service = IntelligentInventoryService(
        model=model, model_feature_columns=TRAIN_FEATURE_COLUMNS,
    )
    series = pd.Series([0, 0, 0, 10, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 3, 0, 0, 0, 0])

    decision = service.get_intelligent_reorder_decision(
        sku="TEST-INT", current_stock=10, demand_history=series, lead_time_days=7,
    )
    assert decision["intelligence"]["forecast_method"] == "croston"
    assert not model.calls, "model should not be called on intermittent SKUs"


def test_intelligent_service_executes_router_selected_method():
    model = _StubModel(constant=20.0)
    service = IntelligentInventoryService(
        model=model,
        model_feature_columns=TRAIN_FEATURE_COLUMNS,
    )
    series = _regular_series(length=30, seed=3)

    decision = service.get_intelligent_reorder_decision(
        sku="TEST-ROUTE",
        current_stock=50,
        demand_history=series,
        lead_time_days=7,
        routing_service=_RoutingService("croston"),
    )

    assert decision["intelligence"]["forecast_method"] == "croston"
    assert decision["intelligence"]["routing"]["selected_method"] == "croston"
    assert decision["intelligence"]["routing"]["default_method"] == "ml_lightgbm"
    assert not model.calls, "model should not run when routing selected Croston"


def test_intelligent_service_does_not_crash_on_short_history():
    model = _StubModel(constant=20.0)
    service = IntelligentInventoryService(
        model=model, model_feature_columns=TRAIN_FEATURE_COLUMNS,
    )
    # Tiny series, regular pattern: should fall back gracefully
    series = pd.Series([10.0, 12.0, 9.0, 11.0, 13.0])
    decision = service.get_intelligent_reorder_decision(
        sku="TEST-SHORT", current_stock=5, demand_history=series, lead_time_days=7,
    )
    assert decision["intelligence"]["forecast_method"] == "simple_average"
    assert "order_quantity" in decision
