"""Repository for model monitoring snapshots and completed evaluation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from db.models import ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog


class ModelMonitoringRepository:
    """Read completed forecast evaluations and persist monitoring snapshots."""

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

    def get_model_artifact(self, artifact_id: int) -> ModelArtifact | None:
        return self.session.get(ModelArtifact, artifact_id)

    def latest_evaluated_model_artifact(self, model_name: str) -> ModelArtifact | None:
        stmt = (
            select(ModelArtifact)
            .join(PredictionLog, PredictionLog.model_artifact_id == ModelArtifact.id)
            .join(ForecastEvaluation, ForecastEvaluation.prediction_log_id == PredictionLog.id)
            .where(ModelArtifact.model_name == model_name)
            .where(ForecastEvaluation.evaluation_scope == "logged_prediction")
            .order_by(ForecastEvaluation.generated_at.desc().nullslast(), ForecastEvaluation.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def get_recent_completed_evaluations(
        self,
        *,
        model_artifact_id: int | None,
        model_name: str,
        model_version: str | None,
        generated_after: datetime,
        limit: int,
    ) -> list[ForecastEvaluation]:
        """Return recent completed logged-prediction evaluations for one model scope."""
        limit = max(1, min(int(limit), 5000))
        stmt = (
            select(ForecastEvaluation)
            .join(PredictionLog, ForecastEvaluation.prediction_log_id == PredictionLog.id)
            .options(
                selectinload(ForecastEvaluation.prediction_log),
                selectinload(ForecastEvaluation.model_artifact),
            )
            .where(ForecastEvaluation.evaluation_scope == "logged_prediction")
            .where(ForecastEvaluation.generated_at.is_not(None))
            .where(ForecastEvaluation.generated_at >= generated_after)
            .order_by(ForecastEvaluation.generated_at.desc(), ForecastEvaluation.id.desc())
            .limit(limit)
        )
        if model_artifact_id is not None:
            stmt = stmt.where(ForecastEvaluation.model_artifact_id == model_artifact_id)
        elif model_version:
            stmt = stmt.where(PredictionLog.model_name == model_name)
            stmt = stmt.where(PredictionLog.model_version == model_version)
        else:
            stmt = stmt.where(PredictionLog.model_name == model_name)

        return list(self.session.scalars(stmt))

    def latest_baseline_evaluation(
        self,
        *,
        model_artifact_id: int | None,
        model_name: str,
        model_version: str | None,
    ) -> ForecastEvaluation | None:
        """Return the latest non-logged evaluation evidence for a model, if any."""
        stmt = (
            select(ForecastEvaluation)
            .outerjoin(PredictionLog, ForecastEvaluation.prediction_log_id == PredictionLog.id)
            .where(ForecastEvaluation.metric_wape.is_not(None))
            .where(ForecastEvaluation.evaluation_scope != "logged_prediction")
            .order_by(ForecastEvaluation.generated_at.desc().nullslast(), ForecastEvaluation.id.desc())
            .limit(1)
        )
        if model_artifact_id is not None:
            stmt = stmt.where(ForecastEvaluation.model_artifact_id == model_artifact_id)
        elif model_version:
            stmt = stmt.where(ForecastEvaluation.model_name == model_name)
        else:
            stmt = stmt.where(ForecastEvaluation.model_name == model_name)
        return self.session.scalar(stmt)

    def snapshot_by_evidence_key(self, evidence_key: str) -> ModelMonitoringSnapshot | None:
        stmt = select(ModelMonitoringSnapshot).where(ModelMonitoringSnapshot.evidence_key == evidence_key)
        return self.session.scalar(stmt)

    def recent_snapshots_for_scope(
        self,
        *,
        model_artifact_id: int | None,
        model_name: str,
        model_version: str | None,
        limit: int = 5,
    ) -> list[ModelMonitoringSnapshot]:
        limit = max(1, min(int(limit), 100))
        stmt = (
            select(ModelMonitoringSnapshot)
            .where(ModelMonitoringSnapshot.model_name == model_name)
            .order_by(ModelMonitoringSnapshot.generated_at.desc(), ModelMonitoringSnapshot.id.desc())
            .limit(limit)
        )
        if model_artifact_id is not None:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id == model_artifact_id)
        elif model_version:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id.is_(None))
            stmt = stmt.where(ModelMonitoringSnapshot.model_version == model_version)
        else:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id.is_(None))
            stmt = stmt.where(ModelMonitoringSnapshot.model_version.is_(None))
        return list(self.session.scalars(stmt))

    def create_snapshot(self, values: Mapping[str, Any]) -> ModelMonitoringSnapshot:
        snapshot = ModelMonitoringSnapshot(**dict(values))
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def get_latest_snapshot(self, *, model_name: str) -> ModelMonitoringSnapshot | None:
        stmt = (
            select(ModelMonitoringSnapshot)
            .options(selectinload(ModelMonitoringSnapshot.model_artifact))
            .where(ModelMonitoringSnapshot.model_name == model_name)
            .order_by(ModelMonitoringSnapshot.generated_at.desc(), ModelMonitoringSnapshot.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def get_latest_snapshot_for_scope(
        self,
        *,
        model_artifact_id: int | None,
        model_name: str,
        model_version: str | None,
    ) -> ModelMonitoringSnapshot | None:
        stmt = (
            select(ModelMonitoringSnapshot)
            .options(selectinload(ModelMonitoringSnapshot.model_artifact))
            .where(ModelMonitoringSnapshot.model_name == model_name)
            .order_by(ModelMonitoringSnapshot.generated_at.desc(), ModelMonitoringSnapshot.id.desc())
            .limit(1)
        )
        if model_artifact_id is not None:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id == model_artifact_id)
        elif model_version:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id.is_(None))
            stmt = stmt.where(ModelMonitoringSnapshot.model_version == model_version)
        else:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id.is_(None))
            stmt = stmt.where(ModelMonitoringSnapshot.model_version.is_(None))
        return self.session.scalar(stmt)

    def list_recent_snapshots(
        self,
        *,
        model_name: str,
        limit: int = 20,
        model_artifact_id: int | None = None,
        status: str | None = None,
    ) -> list[ModelMonitoringSnapshot]:
        limit = max(1, min(int(limit), 200))
        stmt = (
            select(ModelMonitoringSnapshot)
            .options(selectinload(ModelMonitoringSnapshot.model_artifact))
            .where(ModelMonitoringSnapshot.model_name == model_name)
            .order_by(ModelMonitoringSnapshot.generated_at.desc(), ModelMonitoringSnapshot.id.desc())
            .limit(limit)
        )
        if model_artifact_id is not None:
            stmt = stmt.where(ModelMonitoringSnapshot.model_artifact_id == model_artifact_id)
        if status is not None:
            stmt = stmt.where(ModelMonitoringSnapshot.status == status)
        return list(self.session.scalars(stmt))
