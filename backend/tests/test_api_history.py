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

    # Back-compat: every row still exposes ``date`` + ``demand``.
    # New: every row also exposes the unambiguous ``units_sold`` alias.
    expected_demand = [
        {"date": "2024-01-06", "demand": 12.0},
        {"date": "2024-01-07", "demand": 4.0},
        {"date": "2024-01-08", "demand": 6.0},
        {"date": "2024-01-09", "demand": 0.0},
        {"date": "2024-01-10", "demand": 9.0},
    ]
    for row, expected in zip(body["history"], expected_demand):
        assert row["date"] == expected["date"]
        assert row["demand"] == expected["demand"]
        assert row["units_sold"] == expected["demand"]


def test_history_endpoint_exposes_provenance_metadata(client):
    res = client.get("/api/skus/HAS-HISTORY/history?days=5")
    assert res.status_code == 200
    body = res.json()

    assert body["series_type"] == "recorded_history"
    assert body["value_meaning"] == "actual_units_sold"
    assert body["source"] == "processed_dataset"
    assert "forecast" not in body["description"].split(".")[0].lower()


def test_history_endpoint_summary_stats_match_series(client):
    res = client.get("/api/skus/HAS-HISTORY/history?days=5")
    body = res.json()
    summary = body["summary"]
    # Last 5 values from the stub series: 12, 4, 6, 0, 9
    assert summary["first_date"] == "2024-01-06"
    assert summary["last_date"] == "2024-01-10"
    assert summary["window_days_returned"] == 5
    assert summary["total_units_sold"] == 31.0
    assert summary["mean_units_per_day"] == 6.2
    assert summary["peak_units_in_one_day"] == 12.0
    assert summary["days_with_sales"] == 4
    assert summary["days_with_zero_sales"] == 1


def test_history_endpoint_empty_when_sku_missing(client):
    res = client.get("/api/skus/UNKNOWN/history")
    assert res.status_code == 200
    body = res.json()
    # Back-compat shape preserved: sku, available, history (empty).
    assert body["sku"] == "UNKNOWN"
    assert body["available"] is False
    assert body["history"] == []
    # Provenance metadata is always present, even for the empty case.
    assert body["series_type"] == "recorded_history"
    assert body["summary"] is None


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
