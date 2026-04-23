"""Tests for /api/model-info and the model_info sub-block on /api/analyze.

Locks the contract that:
  - the metadata surface is always truthful (never claims an artifact is
    loaded when it isn't, never fakes a model name on statistical paths),
  - statistical/rule-based paths are labeled correctly,
  - the endpoint shape is stable for frontend consumption.
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


def _intermittent_series():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    values = [0] * 40
    for i in (3, 10, 17, 24, 31, 38):
        values[i] = 8.0
    return pd.Series(values, index=dates, dtype=float)


# ---- /api/model-info --------------------------------------------------------


def test_model_info_endpoint_returns_expected_shape():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        res = c.get("/api/model-info")
        assert res.status_code == 200
        body = res.json()
        for key in (
            "model_name", "model_type", "artifact_available",
            "trained_at", "dataset", "feature_count", "features",
            "train_skus", "training_metrics", "evaluation", "hint",
        ):
            assert key in body, f"/api/model-info missing key {key!r}"
        assert body["model_type"] == "ml"
        assert isinstance(body["artifact_available"], bool)
        # evaluation sub-block must always be present and well-shaped
        assert set(body["evaluation"].keys()) == {"available", "generated_at", "summary"}
        assert isinstance(body["evaluation"]["available"], bool)


def test_model_info_hint_is_actionable_when_artifact_missing():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        # Force the endpoint into the "artifact not loaded" branch AFTER the
        # lifespan has run (otherwise the on-disk pkl is loaded over our stub).
        backend_main._loaded_model = None
        body = c.get("/api/model-info").json()
        assert body["artifact_available"] is False
        assert isinstance(body["hint"], str)
        assert "bootstrap.py" in body["hint"]


# ---- model_info sub-block on /api/analyze ----------------------------------


def test_analyze_model_info_ml_branch_for_regular_sku_with_artifact():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        backend_main._loaded_model = object()  # pretend the artifact is loaded
        body = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10}).json()
        mi = body["model_info"]
        # Either the trained model ran, or we fell back deterministically.
        # Either way the block must truthfully describe what happened.
        if body["forecast_method"] == "ml_lightgbm":
            assert mi["model_type"] == "ml"
            assert mi["model_name"] == "lightgbm_demand_forecast"
            assert mi["artifact_available"] is True
        else:
            # Fallback path — must never pretend to be the ML model.
            assert mi["model_type"] in {"statistical_method", "rule_based_fallback"}
            assert mi["artifact_available"] is False


def test_analyze_model_info_statistical_for_intermittent_sku():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        backend_main._data_service = _StubDataService({"SPARSE": _intermittent_series()})
        body = c.post("/api/analyze", json={"sku": "SPARSE", "current_stock": 10}).json()
        assert body["forecast_method"] in {"croston", "conservative"}
        mi = body["model_info"]
        assert mi["model_type"] == "statistical_method"
        assert "Croston" in mi["model_name"] or "Conservative" in mi["model_name"]
        assert mi["artifact_available"] is False


def test_analyze_model_info_never_claims_artifact_when_model_unloaded():
    """If the ML model isn't loaded, the response must not say it is —
    regardless of which forecast_method was ultimately selected."""
    import main as backend_main

    with TestClient(backend_main.app) as c:
        # Override AFTER the lifespan so we're sure the on-disk pkl doesn't
        # sneak back in.
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        backend_main._loaded_model = None
        body = c.post("/api/analyze", json={"sku": "HAS", "current_stock": 10}).json()
        mi = body["model_info"]
        assert mi["artifact_available"] is False
