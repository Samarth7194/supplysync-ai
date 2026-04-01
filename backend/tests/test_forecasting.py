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
