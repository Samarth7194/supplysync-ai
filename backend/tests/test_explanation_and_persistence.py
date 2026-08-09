"""Tests for explanation blocks and SQLAlchemy-backed analysis persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


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


def _session():
    from db.models import Base

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return Session()


def _client_with_analysis_service(session, data_service):
    import main as backend_main
    from repositories.analysis_repository import AnalysisRepository
    from services.analysis_service import AnalysisService
    from services.intelligent_inventory_service import IntelligentInventoryService

    backend_main._data_service = data_service
    backend_main._inventory_service = IntelligentInventoryService(model=None, model_feature_columns=None)

    def override_analysis_service():
        return AnalysisService(
            inventory_service=backend_main._inventory_service,
            settings=backend_main.SETTINGS,
            data_service=backend_main._data_service,
            analysis_repository=AnalysisRepository(session),
            model_loaded=False,
            model_dir=backend_main.SETTINGS.forecasting.model_path,
        )

    backend_main.app.dependency_overrides[backend_main.get_analysis_service] = override_analysis_service
    client = TestClient(backend_main.app)
    return backend_main, client


def _cleanup_overrides(backend_main):
    backend_main.app.dependency_overrides.clear()


def test_analyze_response_carries_explanation_block():
    session = _session()
    backend_main, client = _client_with_analysis_service(session, _StubDataService({"HAS": _regular_series()}))
    try:
        body = client.post("/api/analyze", json={"sku": "HAS", "current_stock": 10}).json()
    finally:
        client.close()
        _cleanup_overrides(backend_main)

    assert "explanation" in body
    expl = body["explanation"]
    for key in ("classification_reason", "method_reason", "risk_reason", "confidence_note"):
        assert key in expl and isinstance(expl[key], str) and len(expl[key]) > 20


def test_explanation_reflects_classification_thresholds():
    session = _session()
    data_service = _StubDataService({
        "REG": _regular_series(),
        "SPARSE": _highly_intermittent_series(),
    })
    backend_main, client = _client_with_analysis_service(session, data_service)
    try:
        reg = client.post("/api/analyze", json={"sku": "REG", "current_stock": 10}).json()
        sparse = client.post("/api/analyze", json={"sku": "SPARSE", "current_stock": 10}).json()
    finally:
        client.close()
        _cleanup_overrides(backend_main)

    reg_text = reg["explanation"]["classification_reason"].lower()
    sparse_text = sparse["explanation"]["classification_reason"].lower()
    assert "regular demand" in reg_text
    assert "highly-intermittent" in sparse_text or "80%" in sparse_text


def test_explanation_confidence_note_flags_synthetic_demo():
    session = _session()
    backend_main, client = _client_with_analysis_service(session, _StubDataService(series_by_sku={}))
    try:
        body = client.post("/api/analyze", json={"sku": "UNKNOWN", "current_stock": 10}).json()
    finally:
        client.close()
        _cleanup_overrides(backend_main)

    assert body["demand_source"] == "synthetic"
    assert "synthetic" in body["explanation"]["confidence_note"].lower()
    assert "illustrative" in body["explanation"]["confidence_note"].lower()


def test_explanation_risk_reason_matches_bucket():
    session = _session()
    backend_main, client = _client_with_analysis_service(session, _StubDataService({"HAS": _regular_series(level=30.0)}))
    try:
        high = client.post("/api/analyze", json={"sku": "HAS", "current_stock": 1}).json()
        low = client.post("/api/analyze", json={"sku": "HAS", "current_stock": 9999}).json()
    finally:
        client.close()
        _cleanup_overrides(backend_main)

    assert high["risk"] == "HIGH"
    assert "p50" in high["explanation"]["risk_reason"].lower()
    assert low["risk"] == "LOW"
    assert "p90" in low["explanation"]["risk_reason"].lower()


def test_recent_analyses_endpoint_returns_what_was_written():
    session = _session()
    backend_main, client = _client_with_analysis_service(session, _StubDataService({"HAS": _regular_series()}))
    try:
        first = client.post("/api/analyze", json={"sku": "HAS", "current_stock": 10})
        second = client.post("/api/analyze", json={"sku": "HAS", "current_stock": 20})
        assert first.status_code == 200
        assert second.status_code == 200
        session.commit()

        body = client.get("/api/analyses/recent?limit=5").json()
    finally:
        client.close()
        _cleanup_overrides(backend_main)

    assert body["available"] is True
    assert body["source"] == "sqlalchemy"
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["current_stock"] == 20.0
    assert body["items"][1]["current_stock"] == 10.0


def test_recent_analyses_endpoint_reports_missing_repository():
    import main as backend_main
    from services.analysis_service import AnalysisService
    from services.intelligent_inventory_service import IntelligentInventoryService

    def override_analysis_service():
        return AnalysisService(
            inventory_service=IntelligentInventoryService(model=None, model_feature_columns=None),
            settings=backend_main.SETTINGS,
            data_service=_StubDataService({}),
            analysis_repository=None,
            model_loaded=False,
        )

    backend_main.app.dependency_overrides[backend_main.get_analysis_service] = override_analysis_service
    with TestClient(backend_main.app) as client:
        body = client.get("/api/analyses/recent").json()
    _cleanup_overrides(backend_main)

    assert body == {"available": False, "items": [], "total": 0, "source": "sqlalchemy"}
