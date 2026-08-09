"""FastAPI dependencies for analysis persistence."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from db.session import get_session
from repositories.analysis_repository import AnalysisRepository


def get_analysis_repository(session: Session = Depends(get_session)) -> AnalysisRepository:
    return AnalysisRepository(session)

