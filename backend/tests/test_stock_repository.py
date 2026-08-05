from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
from repositories.stock_repository import StockRepository


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    with Session() as session:
        yield session


def test_record_stock_creates_sku_and_snapshot(session):
    repo = StockRepository(session)

    stock = repo.record_stock("SKU-1", Decimal("12"), sku_name="Demo SKU")
    session.commit()

    assert stock.sku.sku_code == "SKU-1"
    assert stock.sku.name == "Demo SKU"
    assert stock.quantity_on_hand == Decimal("12.000")
    assert stock.quantity_available == Decimal("12.000")


def test_latest_for_sku_returns_most_recent_snapshot(session):
    repo = StockRepository(session)

    first = repo.record_stock("SKU-1", Decimal("12"))
    second = repo.record_stock("SKU-1", Decimal("18"))
    session.commit()

    latest = repo.latest_for_sku("SKU-1")

    assert latest is not None
    assert latest.id == second.id
    assert latest.id != first.id
    assert latest.quantity_on_hand == Decimal("18.000")


def test_record_stock_rejects_reserved_above_on_hand(session):
    repo = StockRepository(session)

    with pytest.raises(ValueError, match="quantity_reserved"):
        repo.record_stock(
            "SKU-1",
            Decimal("2"),
            quantity_reserved=Decimal("3"),
        )

