"""Repository for server-side SKU stock levels."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Sku, StockLevel


class StockRepository:
    """Read and append stock snapshots."""

    def __init__(self, session: Session):
        self.session = session

    def get_or_create_sku(self, sku_code: str, name: str | None = None) -> Sku:
        sku = self.session.scalar(select(Sku).where(Sku.sku_code == sku_code))
        if sku is not None:
            if name and not sku.name:
                sku.name = name
                self.session.flush()
            return sku

        sku = Sku(sku_code=sku_code, name=name)
        self.session.add(sku)
        self.session.flush()
        return sku

    def record_stock(
        self,
        sku_code: str,
        quantity_on_hand: Decimal,
        *,
        quantity_reserved: Decimal = Decimal("0"),
        source: str = "user",
        note: str | None = None,
        sku_name: str | None = None,
    ) -> StockLevel:
        sku = self.get_or_create_sku(sku_code, sku_name)
        quantity_available = quantity_on_hand - quantity_reserved
        if quantity_available < 0:
            raise ValueError("quantity_reserved cannot exceed quantity_on_hand")

        stock = StockLevel(
            sku_id=sku.id,
            quantity_on_hand=quantity_on_hand,
            quantity_reserved=quantity_reserved,
            quantity_available=quantity_available,
            source=source,
            note=note,
        )
        self.session.add(stock)
        self.session.flush()
        return stock

    def latest_for_sku(self, sku_code: str) -> StockLevel | None:
        stmt = (
            select(StockLevel)
            .join(Sku)
            .where(Sku.sku_code == sku_code)
            .order_by(StockLevel.recorded_at.desc(), StockLevel.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def latest_for_all(self) -> list[StockLevel]:
        latest_ids = (
            select(
                StockLevel.sku_id,
                func.max(StockLevel.id).label("latest_id"),
            )
            .group_by(StockLevel.sku_id)
            .subquery()
        )
        stmt = (
            select(StockLevel)
            .join(
                latest_ids,
                (StockLevel.sku_id == latest_ids.c.sku_id)
                & (StockLevel.id == latest_ids.c.latest_id),
            )
            .join(Sku)
            .where(Sku.is_active.is_(True))
            .order_by(Sku.sku_code.asc())
        )
        return list(self.session.scalars(stmt))

    @staticmethod
    def serialize(stock: StockLevel) -> dict[str, Any]:
        return {
            "sku": stock.sku.sku_code if stock.sku else None,
            "quantity_on_hand": float(stock.quantity_on_hand),
            "quantity_reserved": float(stock.quantity_reserved),
            "quantity_available": float(stock.quantity_available),
            "source": stock.source,
            "recorded_at": stock.recorded_at.isoformat() if stock.recorded_at else None,
        }
