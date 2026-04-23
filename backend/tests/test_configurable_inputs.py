"""Tests for the configurable analyze inputs (lead_time_days, service_level).

Locks the contract that the API accepts per-request overrides, validates
their bounds, and actually routes them through to the decision math so the
reorder point changes.
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
    def __init__(self, series_by_sku):
        self._series = series_by_sku

    def get_demand_history(self, sku):
        return self._series.get(sku, pd.Series(dtype=float))

    def get_top_skus(self, n=20):
        return list(self._series.keys())


def _regular_series(length=45, level=20.0):
    dates = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.Series([level + (i % 5) for i in range(length)], index=dates, dtype=float)


def test_analyze_defaults_are_backward_compatible():
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(":memory:")
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        body = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10}).json()
    assert body["decision"]["lead_time_days"] == 7
    assert body["decision"]["service_level"] == pytest.approx(0.95)


def test_lead_time_override_flows_into_decision(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        week = c.post("/api/analyze",
                      json={"sku": "HAS", "current_stock": 10, "lead_time_days": 7}).json()
        month = c.post("/api/analyze",
                       json={"sku": "HAS", "current_stock": 10, "lead_time_days": 21}).json()
    # A longer horizon should sum more forecast days → larger lead-time demand.
    assert month["decision"]["lead_time_days"] == 21
    assert month["decision"]["lead_time_demand"] > week["decision"]["lead_time_demand"]


def test_service_level_override_increases_safety_stock(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        low = c.post("/api/analyze",
                     json={"sku": "HAS", "current_stock": 10, "service_level": 0.80}).json()
        high = c.post("/api/analyze",
                      json={"sku": "HAS", "current_stock": 10, "service_level": 0.99}).json()
    assert high["decision"]["service_level"] == pytest.approx(0.99)
    # Higher service level demands a larger buffer (traditional formula is
    # monotonic in Z-score).
    assert high["decision"]["safety_stock"] >= low["decision"]["safety_stock"]


@pytest.mark.parametrize(
    "payload",
    [
        {"sku": "HAS", "current_stock": 10, "lead_time_days": 0},
        {"sku": "HAS", "current_stock": 10, "lead_time_days": 365},
        {"sku": "HAS", "current_stock": 10, "service_level": 0.2},
        {"sku": "HAS", "current_stock": 10, "service_level": 1.0},
    ],
)
def test_bad_inputs_rejected_with_422(payload):
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        r = c.post("/api/analyze", json=payload)
        assert r.status_code == 422
