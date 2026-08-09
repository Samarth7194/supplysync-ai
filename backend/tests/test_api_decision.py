"""Tests for the /api/analyze decision block and the /api/kpis interpretation metadata.

These lock the contract that the API explains *why* a recommendation was
made, not just *what* was recommended, and that /api/kpis ships enough
interpretation metadata for a consumer to render the business meaning of
each KPI.
"""

import json
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
    values = [level + (i % 5) for i in range(length)]  # non-zero, mildly varying
    return pd.Series(values, index=dates, dtype=float)


def test_analyze_returns_decision_block_with_all_fields():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        res = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10})
        assert res.status_code == 200
        body = res.json()

        assert "decision" in body, "decision block must be present"
        d = body["decision"]
        expected_keys = {
            "lead_time_days", "lead_time_demand",
            "safety_stock", "safety_stock_method",
            "reorder_point", "service_level",
            "inventory_gap", "why", "constraints",
        }
        assert set(d.keys()) == expected_keys

        # Consistency checks
        assert d["lead_time_days"] == 7
        assert d["service_level"] == pytest.approx(0.95)
        assert d["inventory_gap"] >= 0
        assert isinstance(d["why"], str) and len(d["why"]) > 40


def test_decision_why_explains_order_when_stock_low():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"HAS": _regular_series(level=30.0)})
        # Very low stock forces the order-path explanation
        res = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 1}).json()
        d = res["decision"]
        assert res["recommended_order"] > 0
        why = d["why"].lower()
        assert "ordering" in why or "order" in why
        # The sentence must namedrop the lead-time horizon and service level
        assert "7-day" in d["why"]
        assert "95%" in d["why"]


def test_decision_why_explains_no_action_when_stock_high():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"HAS": _regular_series(level=5.0)})
        # Enormous stock → no reorder needed
        res = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 9999}).json()
        assert res["recommended_order"] == 0
        assert res["decision"]["inventory_gap"] == 0
        assert "no action needed" in res["decision"]["why"].lower()


def test_inventory_gap_matches_reorder_point_minus_stock():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"HAS": _regular_series(level=40.0)})
        res = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 20}).json()
        d = res["decision"]
        expected_gap = max(0.0, d["reorder_point"] - 20)
        assert d["inventory_gap"] == pytest.approx(expected_gap, rel=1e-3, abs=0.02)


def test_kpis_endpoint_ships_interpretation_metadata(tmp_path, monkeypatch):
    import main as backend_main

    cache = BACKEND_DIR / "data" / "cached_kpis.json"
    assert cache.exists(), "This test expects the committed cached_kpis.json to be present"

    with TestClient(backend_main.app) as c:
        res = c.get("/api/kpis")
        assert res.status_code == 200
        body = res.json()

        assert "interpretation" in body, "KPI response must carry interpretation metadata"
        interp = body["interpretation"]
        for key in ("baseline", "baseline_description", "intelligent_description",
                    "assumptions", "metric_meanings"):
            assert key in interp, f"missing interpretation key: {key}"

        # Metric meanings must cover at least the headline KPIs
        for key in ("cost_savings_pct", "fill_rate", "skus_analyzed"):
            assert key in interp["metric_meanings"]
            assert isinstance(interp["metric_meanings"][key], str)

        # Assumptions must expose the parameters a reviewer would ask about
        assumptions = interp["assumptions"]
        assert assumptions["lead_time_days"] == 7
        assert assumptions["service_level"] == 0.95
