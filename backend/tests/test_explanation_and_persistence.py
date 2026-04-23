"""Tests for the explanation block on /api/analyze, the AnalysisStore
persistence layer, and the /api/analyses/recent endpoint.
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


def _highly_intermittent_series():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    values = [0.0] * 40
    for i in (5, 20, 33):
        values[i] = 8.0
    return pd.Series(values, index=dates, dtype=float)


# ------ ExplanationBlock -----------------------------------------------------


def test_analyze_response_carries_explanation_block(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        body = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10}).json()

    assert "explanation" in body
    expl = body["explanation"]
    for key in ("classification_reason", "method_reason", "risk_reason", "confidence_note"):
        assert key in expl and isinstance(expl[key], str) and len(expl[key]) > 20


def test_explanation_reflects_classification_thresholds(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        # Regular-demand SKU (no zeros)
        backend_main._data_service = _StubDataService({"REG": _regular_series()})
        reg = c.post("/api/analyze", json={"sku": "REG", "current_stock": 10}).json()
        # Highly-intermittent SKU
        backend_main._data_service = _StubDataService({"SPARSE": _highly_intermittent_series()})
        sparse = c.post("/api/analyze", json={"sku": "SPARSE", "current_stock": 10}).json()

    reg_text = reg["explanation"]["classification_reason"].lower()
    sparse_text = sparse["explanation"]["classification_reason"].lower()
    assert "regular demand" in reg_text
    assert "highly-intermittent" in sparse_text or "80%" in sparse_text


def test_explanation_confidence_note_flags_synthetic_demo(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        backend_main._data_service = _StubDataService(series_by_sku={})  # no known SKUs
        body = c.post("/api/analyze", json={"sku": "UNKNOWN", "current_stock": 10}).json()

    assert body["demand_source"] == "synthetic"
    assert "synthetic" in body["explanation"]["confidence_note"].lower()
    assert "illustrative" in body["explanation"]["confidence_note"].lower()


def test_explanation_risk_reason_matches_bucket(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        backend_main._data_service = _StubDataService({"HAS": _regular_series(level=30.0)})
        high = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 1}).json()
        low = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 9999}).json()

    assert high["risk"] == "HIGH"
    assert "p50" in high["explanation"]["risk_reason"].lower()
    assert low["risk"] == "LOW"
    assert "p90" in low["explanation"]["risk_reason"].lower()


# ------ AnalysisStore --------------------------------------------------------


def test_analysis_store_records_and_recalls(tmp_path):
    from storage.analysis_store import AnalysisStore

    store = AnalysisStore(tmp_path / "db.sqlite")
    assert store.count() == 0

    rid = store.record({
        "sku": "ABC", "risk": "HIGH", "action": "PURCHASE",
        "current_stock": 10.0, "recommended_order": 42,
        "demand_pattern": "regular", "forecast_method": "ml_lightgbm",
        "demand_source": "historical", "forecast_source": "model_forecast",
        "model_type": "ml", "model_name": "lightgbm_demand_forecast",
        "artifact_available": True,
        "lead_time_demand": 120.5, "safety_stock": 30.0, "reorder_point": 150.5,
        "inventory_gap": 140.5, "p50": 18.4, "p90": 42.0,
    })
    assert rid > 0
    assert store.count() == 1

    rows = store.recent(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["sku"] == "ABC"
    assert row["risk"] == "HIGH"
    assert row["artifact_available"] == 1  # SQLite bool → int
    assert row["recommended_order"] == 42
    assert row["created_at"] is not None


def test_analysis_store_recent_is_most_recent_first(tmp_path):
    from storage.analysis_store import AnalysisStore

    store = AnalysisStore(tmp_path / "db.sqlite")
    for i, sku in enumerate(["A", "B", "C", "D"]):
        store.record({"sku": sku, "recommended_order": i})

    rows = store.recent(limit=3)
    assert [r["sku"] for r in rows] == ["D", "C", "B"]


def test_analysis_store_clamps_limit(tmp_path):
    from storage.analysis_store import AnalysisStore

    store = AnalysisStore(tmp_path / "db.sqlite")
    for i in range(5):
        store.record({"sku": f"S{i}"})
    # Zero or negative → at least 1 row; huge values capped at 200.
    assert len(store.recent(limit=0)) == 1
    assert len(store.recent(limit=-5)) == 1
    assert len(store.recent(limit=99999)) == 5  # capped at 200 by the clamp, but we only have 5


# ------ /api/analyses/recent endpoint ----------------------------------------


def test_recent_analyses_endpoint_returns_what_was_written(tmp_path):
    import main as backend_main
    from storage.analysis_store import AnalysisStore

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = AnalysisStore(tmp_path / "db.sqlite")
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10})
        c.post("/api/analyze", json={"sku": "HAS", "current_stock": 20})

        body = c.get("/api/analyses/recent?limit=5").json()

    assert body["available"] is True
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["current_stock"] == 20.0  # most-recent first
    assert body["items"][1]["current_stock"] == 10.0
    assert body["items"][0]["artifact_available"] in (True, False, None)


def test_recent_analyses_endpoint_gracefully_handles_no_store():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._analysis_store = None
        body = c.get("/api/analyses/recent").json()

    assert body == {"available": False, "items": [], "total": 0}
