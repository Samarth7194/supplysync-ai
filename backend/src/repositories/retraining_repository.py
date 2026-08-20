"""Repository for retraining recommendation decisions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from db.models import ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog, RetrainingRun


class RetrainingRepository:
    """Read monitoring/evaluation evidence and persist retraining decisions.

    The repository deliberately contains no recommendation policy. It only
    exposes the rows the service needs and flushes newly-created decisions into
    the caller-owned transaction.
    """

    def __init__(self, session: Session):
        self.session = session

    def active_model_artifact(self, model_name: str) -> ModelArtifact | None:
        stmt = (
            select(ModelArtifact)
            .where(ModelArtifact.model_name == model_name)
            .where(ModelArtifact.is_active.is_(True))
            .where(ModelArtifact.lifecycle_status == "active")
            .order_by(ModelArtifact.activated_at.desc().nullslast(), ModelArtifact.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def latest_monitoring_snapshot(
        self,
        *,
        model_artifact_id: int,
        model_name: str,
        model_version: str | None,
    ) -> ModelMonitoringSnapshot | None:
        stmt = (
            select(ModelMonitoringSnapshot)
            .options(selectinload(ModelMonitoringSnapshot.model_artifact))
            .where(ModelMonitoringSnapshot.model_artifact_id == model_artifact_id)
            .where(ModelMonitoringSnapshot.model_name == model_name)
            .order_by(ModelMonitoringSnapshot.generated_at.desc(), ModelMonitoringSnapshot.id.desc())
            .limit(1)
        )
        if model_version:
            stmt = stmt.where(ModelMonitoringSnapshot.model_version == model_version)
        return self.session.scalar(stmt)

    def latest_retraining_run(self, *, baseline_model_artifact_id: int) -> RetrainingRun | None:
        stmt = (
            select(RetrainingRun)
            .options(
                selectinload(RetrainingRun.baseline_model_artifact),
                selectinload(RetrainingRun.source_monitoring_snapshot),
            )
            .where(RetrainingRun.baseline_model_artifact_id == baseline_model_artifact_id)
            .order_by(RetrainingRun.triggered_at.desc(), RetrainingRun.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def retraining_run_by_evidence_key(self, evidence_key: str) -> RetrainingRun | None:
        stmt = (
            select(RetrainingRun)
            .options(
                selectinload(RetrainingRun.baseline_model_artifact),
                selectinload(RetrainingRun.source_monitoring_snapshot),
            )
            .where(RetrainingRun.evidence_key == evidence_key)
        )
        return self.session.scalar(stmt)

    def completed_evaluated_forecast_days(
        self,
        *,
        model_artifact_id: int,
        generated_after: datetime | None,
        generated_at_or_before: datetime | None,
    ) -> int:
        stmt = (
            select(ForecastEvaluation)
            .join(PredictionLog, ForecastEvaluation.prediction_log_id == PredictionLog.id)
            .options(selectinload(ForecastEvaluation.prediction_log))
            .where(ForecastEvaluation.model_artifact_id == model_artifact_id)
            .where(ForecastEvaluation.evaluation_scope == "logged_prediction")
            .where(ForecastEvaluation.generated_at.is_not(None))
            .order_by(ForecastEvaluation.generated_at.asc(), ForecastEvaluation.id.asc())
        )
        if generated_after is not None:
            stmt = stmt.where(ForecastEvaluation.generated_at > generated_after)
        if generated_at_or_before is not None:
            stmt = stmt.where(ForecastEvaluation.generated_at <= generated_at_or_before)

        total = 0
        for evaluation in self.session.scalars(stmt):
            total += self._forecast_days(evaluation)
        return total

    def create_retraining_run(self, values: Mapping[str, Any]) -> RetrainingRun:
        row = RetrainingRun(**dict(values))
        self.session.add(row)
        self.session.flush()
        return row

    @staticmethod
    def _forecast_days(evaluation: ForecastEvaluation) -> int:
        if evaluation.n_test_points is not None and int(evaluation.n_test_points) > 0:
            return int(evaluation.n_test_points)
        if evaluation.horizon_days is not None and int(evaluation.horizon_days) > 0:
            return int(evaluation.horizon_days)
        prediction = evaluation.prediction_log
        if prediction is not None and int(prediction.forecast_horizon_days or 0) > 0:
            return int(prediction.forecast_horizon_days)
        return 0
