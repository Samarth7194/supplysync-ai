"""Repository for analysis runs and prediction logs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import AnalysisRun, PredictionLog


class AnalysisRepository:
    """Persist and read inventory analysis decisions."""

    def __init__(self, session: Session):
        self.session = session

    def create_analysis_run(self, values: Mapping[str, Any]) -> AnalysisRun:
        analysis = AnalysisRun(**dict(values))
        self.session.add(analysis)
        self.session.flush()
        return analysis

    def create_prediction_log(self, values: Mapping[str, Any]) -> PredictionLog:
        prediction = PredictionLog(**dict(values))
        self.session.add(prediction)
        self.session.flush()
        return prediction

    def create_analysis_with_prediction(
        self,
        analysis_values: Mapping[str, Any],
        prediction_values: Mapping[str, Any],
    ) -> tuple[AnalysisRun, PredictionLog]:
        analysis = self.create_analysis_run(analysis_values)
        payload = dict(prediction_values)
        payload.setdefault("analysis_run_id", analysis.id)
        prediction = self.create_prediction_log(payload)
        return analysis, prediction

    def recent(self, limit: int = 20) -> list[AnalysisRun]:
        limit = max(1, min(int(limit), 200))
        stmt = select(AnalysisRun).order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def recent_for_sku(self, sku_code: str, limit: int = 20) -> list[AnalysisRun]:
        limit = max(1, min(int(limit), 200))
        stmt = (
            select(AnalysisRun)
            .where(AnalysisRun.sku_code == sku_code)
            .order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

