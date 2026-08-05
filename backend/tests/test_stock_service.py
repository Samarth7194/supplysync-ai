from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
from repositories.stock_repository import StockRepository
from services.stock_service import InvalidStockLevelError, StockService


@pytest.fixture()
def service():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    with Session() as session:
        yield StockService(StockRepository(session))


def test_record_stock_returns_serializable_snapshot(service):
    snapshot = service.record_stock("SKU-1", 10, quantity_reserved=2)

    assert snapshot.sku == "SKU-1"
    assert snapshot.quantity_on_hand == 10.0
    assert snapshot.quantity_reserved == 2.0
    assert snapshot.quantity_available == 8.0
    assert snapshot.source == "user"


def test_record_stock_validates_reserved_stock(service):
    with pytest.raises(InvalidStockLevelError, match="cannot exceed"):
        service.record_stock("SKU-1", 1, quantity_reserved=2)


def test_get_latest_stock_returns_none_for_unknown_sku(service):
    assert service.get_latest_stock("UNKNOWN") is None

