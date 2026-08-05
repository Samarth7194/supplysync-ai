"""Dependency factory for StockService."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from db.session import get_session
from repositories.stock_repository import StockRepository
from services.stock_service import StockService


def get_stock_service(session: Session = Depends(get_session)) -> StockService:
    return StockService(StockRepository(session))

