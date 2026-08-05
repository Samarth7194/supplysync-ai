"""Business service for server-side stock persistence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from repositories.stock_repository import StockRepository


class StockServiceError(Exception):
    """Base class for stock service failures."""


class InvalidStockLevelError(StockServiceError):
    """Raised when stock quantities violate business rules."""


class StockPersistenceUnavailableError(StockServiceError):
    """Raised when the stock database is unavailable or not migrated."""


@dataclass(frozen=True)
class StockSnapshot:
    sku: str
    quantity_on_hand: float
    quantity_reserved: float
    quantity_available: float
    source: str
    recorded_at: str | None


class StockService:
    """Business layer for stock reads and writes.

    Routes should depend on this service rather than calling repositories
    directly. Future validation, permissions, audit logging, or stock-event
    behavior belongs here.
    """

    def __init__(self, repository: StockRepository):
        self.repository = repository

    def get_latest_stock(self, sku_code: str) -> StockSnapshot | None:
        try:
            stock = self.repository.latest_for_sku(sku_code)
        except SQLAlchemyError as exc:
            raise StockPersistenceUnavailableError(
                "Stock persistence is unavailable. Run database migrations before using stock endpoints."
            ) from exc
        return self._to_snapshot(stock) if stock is not None else None

    def list_latest_stock(self) -> list[StockSnapshot]:
        try:
            return [self._to_snapshot(stock) for stock in self.repository.latest_for_all()]
        except SQLAlchemyError as exc:
            raise StockPersistenceUnavailableError(
                "Stock persistence is unavailable. Run database migrations before using stock endpoints."
            ) from exc

    def record_stock(
        self,
        sku_code: str,
        quantity_on_hand: float,
        *,
        quantity_reserved: float = 0.0,
        note: str | None = None,
        sku_name: str | None = None,
    ) -> StockSnapshot:
        on_hand = self._decimal(quantity_on_hand, "quantity_on_hand")
        reserved = self._decimal(quantity_reserved, "quantity_reserved")
        if on_hand < 0:
            raise InvalidStockLevelError("quantity_on_hand must be greater than or equal to 0.")
        if reserved < 0:
            raise InvalidStockLevelError("quantity_reserved must be greater than or equal to 0.")
        if reserved > on_hand:
            raise InvalidStockLevelError("quantity_reserved cannot exceed quantity_on_hand.")

        try:
            stock = self.repository.record_stock(
                sku_code=sku_code,
                quantity_on_hand=on_hand,
                quantity_reserved=reserved,
                source="user",
                note=note,
                sku_name=sku_name,
            )
        except SQLAlchemyError as exc:
            raise StockPersistenceUnavailableError(
                "Stock persistence is unavailable. Run database migrations before using stock endpoints."
            ) from exc
        return self._to_snapshot(stock)

    @staticmethod
    def _decimal(value: float, field_name: str) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise InvalidStockLevelError(f"{field_name} must be a valid number.") from exc

    @staticmethod
    def _to_snapshot(stock: Any) -> StockSnapshot:
        sku = stock.sku.sku_code if stock.sku else ""
        return StockSnapshot(
            sku=sku,
            quantity_on_hand=float(stock.quantity_on_hand),
            quantity_reserved=float(stock.quantity_reserved),
            quantity_available=float(stock.quantity_available),
            source=stock.source,
            recorded_at=stock.recorded_at.isoformat() if stock.recorded_at else None,
        )
