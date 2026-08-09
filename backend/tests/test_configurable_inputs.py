"""Tests for configurable analyze inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
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


def _client(data_service):
    import main as backend_main
    from repositories.analysis_repository import AnalysisRepository
    from services.analysis_service import AnalysisService
    from services.intelligent_inventory_service import IntelligentInventoryService

    session = _session()
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
    return backend_main, TestClient(backend_main.app)


def _cleanup(backend_main):
    backend_main.app.dependency_overrides.clear()


def test_analyze_defaults_are_backward_compatible():
    backend_main, client = _client(_StubDataService({"HAS": _regular_series()}))
    try:
        body = client.post("/api/analyze", json={"sku": "HAS", "current_stock": 10}).json()
    finally:
        client.close()
        _cleanup(backend_main)

    assert body["decision"]["lead_time_days"] == 7
    assert body["decision"]["service_level"] == pytest.approx(0.95)


def test_lead_time_override_flows_into_decision():
    backend_main, client = _client(_StubDataService({"HAS": _regular_series()}))
    try:
        week = client.post(
            "/api/analyze",
            json={"sku": "HAS", "current_stock": 10, "lead_time_days": 7},
        ).json()
        month = client.post(
            "/api/analyze",
            json={"sku": "HAS", "current_stock": 10, "lead_time_days": 21},
        ).json()
    finally:
        client.close()
        _cleanup(backend_main)

    assert month["decision"]["lead_time_days"] == 21
    assert month["decision"]["lead_time_demand"] > week["decision"]["lead_time_demand"]
    assert len(week["forecast"]["daily"]) == 7
    assert len(week["forecast"]["full_horizon_daily"]) == 7
    assert week["forecast"]["horizon_days"] == 7
    assert len(month["forecast"]["daily"]) == 7
    assert len(month["forecast"]["full_horizon_daily"]) == 21
    assert month["forecast"]["horizon_days"] == 21
    assert "constraints" in month["decision"]


def test_service_level_override_increases_safety_stock():
    backend_main, client = _client(_StubDataService({"HAS": _regular_series()}))
    try:
        low = client.post(
            "/api/analyze",
            json={"sku": "HAS", "current_stock": 10, "service_level": 0.80},
        ).json()
        high = client.post(
            "/api/analyze",
            json={"sku": "HAS", "current_stock": 10, "service_level": 0.99},
        ).json()
    finally:
        client.close()
        _cleanup(backend_main)

    assert high["decision"]["service_level"] == pytest.approx(0.99)
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

    with TestClient(backend_main.app) as client:
        backend_main._data_service = _StubDataService({"HAS": _regular_series()})
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 422
