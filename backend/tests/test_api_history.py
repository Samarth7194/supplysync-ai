"""Tests for the /api/skus/{sku}/history endpoint and the SKU detail flow.

These verify that historical demand returned to the frontend comes only from
the processed DataService (never synthesized) and that a missing-history SKU
produces an honest empty response instead of fabricated values.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))


@pytest.fixture()
def client(monkeypatch):
    """Build a TestClient with a stub DataService injected post-lifespan."""
    import main as backend_main

    with TestClient(backend_main.app) as c:
        class _StubDataService:
            def __init__(self):
                dates = pd.date_range("2024-01-01", periods=10, freq="D")
                self._series = pd.Series(
                    [5, 0, 7, 3, 0, 12, 4, 6, 0, 9], index=dates, dtype=float,
                )

            def get_demand_history(self, sku):
                if sku == "HAS-HISTORY":
                    return self._series
                return pd.Series(dtype=float)

            def get_top_skus(self, n=20):
                return ["HAS-HISTORY"]

        backend_main._data_service = _StubDataService()
        yield c


def test_history_endpoint_returns_real_recorded_demand(client):
    res = client.get("/api/skus/HAS-HISTORY/history?days=5")
    assert res.status_code == 200
    body = res.json()

    assert body["sku"] == "HAS-HISTORY"
    assert body["available"] is True
    assert len(body["history"]) == 5

    # Must be the tail of the real recorded series (last 5 values), sorted by date.
    expected = [
        {"date": "2024-01-06", "demand": 12.0},
        {"date": "2024-01-07", "demand": 4.0},
        {"date": "2024-01-08", "demand": 6.0},
        {"date": "2024-01-09", "demand": 0.0},
        {"date": "2024-01-10", "demand": 9.0},
    ]
    assert body["history"] == expected


def test_history_endpoint_empty_when_sku_missing(client):
    res = client.get("/api/skus/UNKNOWN/history")
    assert res.status_code == 200
    body = res.json()
    assert body == {"sku": "UNKNOWN", "available": False, "history": []}


def test_history_endpoint_clamps_days(client):
    # days=0 should clamp up to 1; huge values should clamp to 365
    res_small = client.get("/api/skus/HAS-HISTORY/history?days=0")
    assert res_small.status_code == 200
    assert len(res_small.json()["history"]) == 1

    res_big = client.get("/api/skus/HAS-HISTORY/history?days=9999")
    assert res_big.status_code == 200
    # Our stub only has 10 points, so the clamp caps at what's available
    assert len(res_big.json()["history"]) == 10


def test_history_endpoint_503_when_data_service_down(monkeypatch):
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = None
        res = c.get("/api/skus/ANY/history")
        assert res.status_code == 503
