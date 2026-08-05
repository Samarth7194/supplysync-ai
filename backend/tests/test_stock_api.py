from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

from db.models import Base  # noqa: E402
from dependencies.stock import get_stock_service  # noqa: E402
from main import app  # noqa: E402
from repositories.stock_repository import StockRepository  # noqa: E402
from services.stock_service import StockService  # noqa: E402


@pytest.fixture()
def stock_service():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    with Session() as session:
        yield StockService(StockRepository(session))


@pytest.fixture()
def client(stock_service):
    app.dependency_overrides[get_stock_service] = lambda: stock_service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_stock_service, None)


def test_put_stock_records_server_side_snapshot(client):
    res = client.put(
        "/api/stock/SKU-1",
        json={"quantity_on_hand": 12, "quantity_reserved": 2},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["sku"] == "SKU-1"
    assert body["quantity_on_hand"] == 12.0
    assert body["quantity_reserved"] == 2.0
    assert body["quantity_available"] == 10.0
    assert body["source"] == "user"


def test_get_stock_returns_latest_snapshot(client):
    client.put("/api/stock/SKU-1", json={"quantity_on_hand": 12})
    client.put("/api/stock/SKU-1", json={"quantity_on_hand": 20})

    res = client.get("/api/stock/SKU-1")

    assert res.status_code == 200
    assert res.json()["quantity_on_hand"] == 20.0


def test_list_stock_returns_latest_per_sku(client):
    client.put("/api/stock/SKU-1", json={"quantity_on_hand": 12})
    client.put("/api/stock/SKU-2", json={"quantity_on_hand": 5})

    res = client.get("/api/stock")

    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "database"
    assert {item["sku"] for item in body["items"]} == {"SKU-1", "SKU-2"}


def test_get_stock_returns_404_when_not_recorded(client):
    res = client.get("/api/stock/UNKNOWN")

    assert res.status_code == 404
    assert "No server-side stock recorded" in res.json()["detail"]


def test_put_stock_rejects_reserved_above_on_hand(client):
    res = client.put(
        "/api/stock/SKU-1",
        json={"quantity_on_hand": 1, "quantity_reserved": 2},
    )

    assert res.status_code == 422
    assert "cannot exceed" in res.json()["detail"]
