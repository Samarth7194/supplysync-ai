"""Provenance tests for the /api/analyze response.

Ensures the API surfaces truthful ``demand_source`` and ``forecast_source``
values so the UI can label real, model, rule-based, and synthetic paths
distinctly.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))


class _StubDataService:
    def __init__(self, known):
        self._known = known

    def get_demand_history(self, sku):
        if sku in self._known:
            return self._known[sku]
        return pd.Series(dtype=float)

    def get_top_skus(self, n=20):
        return list(self._known.keys())


@pytest.fixture()
def regular_series():
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    # Regular daily demand, no zeros → classified "regular"
    values = [10 + (i % 5) for i in range(45)]
    return pd.Series(values, index=dates, dtype=float)


def test_analyze_marks_synthetic_when_sku_unknown(monkeypatch):
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService(known={})
        res = c.post(
            "/api/analyze",
            json={"sku": "NOT-IN-DATASET", "current_stock": 10},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["demand_source"] == "synthetic"
        # With a synthetic Poisson(20) series the demand pattern should be
        # "regular" (no zero-heavy distribution), so method+source reflect the
        # chosen path, not a hidden fake.
        assert body["forecast_source"] in {
            "model_forecast",
            "rule_based_estimate",
            "statistical_method",
        }


def test_analyze_marks_historical_when_data_available(regular_series):
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService(known={"HAS": regular_series})
        res = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10})
        assert res.status_code == 200
        body = res.json()
        assert body["demand_source"] == "historical"


def test_analyze_marks_request_when_demand_supplied():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService(known={})
        res = c.post(
            "/api/analyze",
            json={
                "sku": "ANY",
                "current_stock": 10,
                "demand_history": [5, 7, 8, 6, 9, 4, 10, 12, 8, 7, 6, 9, 11, 5],
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["demand_source"] == "request"


def test_forecast_source_classification_is_consistent():
    """Method → source classification must be deterministic and complete."""
    from services.analysis_service import classify_forecast_source

    assert classify_forecast_source("ml_lightgbm") == "model_forecast"
    assert classify_forecast_source("croston") == "statistical_method"
    assert classify_forecast_source("conservative") == "statistical_method"
    assert classify_forecast_source("simple_average") == "rule_based_estimate"
    # Anything we haven't deliberately classified must degrade to "unavailable",
    # not masquerade as a real forecast.
    assert classify_forecast_source("something_new") == "unavailable"
    assert classify_forecast_source("") == "unavailable"
