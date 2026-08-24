"""Retraining recommendation logic from monitoring evidence.

Phase E stops at recommendation tracking. It never calls the training script,
registers a candidate artifact, promotes a model, or touches the inference
path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from db.models import ModelArtifact, ModelMonitoringSnapshot, RetrainingRun
from repositories.retraining_repository import RetrainingRepository


@dataclass(frozen=True)
class RetrainingDecision:
    recommended: bool
    reason: str
    message: str
    latest_monitoring_status: str | None
    new_evaluated_forecast_days: int
    minimum_required: int
    cooldown_days: int
    cooldown_remaining_days: int
    baseline_model_artifact_id: int | None
    baseline_model_name: str | None
    baseline_model_version: str | None
    source_monitoring_snapshot_id: int | None
    last_retraining_attempt_at: datetime | None
    automatic_execution_enabled: bool
    retraining_run: RetrainingRun | None = None
    created: bool = False


class RetrainingDecisionService:
    """Decide whether a model retraining run should be recommended."""

    def __init__(self, *, repository: RetrainingRepository, settings: Any):
        self.repository = repository
        self.settings = settings

    def evaluate(
        self,
        *,
        model_name: str = "lightgbm_demand_forecast",
        persist_recommendation: bool = False,
        now: datetime | None = None,
    ) -> RetrainingDecision:
        now = self._as_aware(now or datetime.now(timezone.utc))
        forecasting = self.settings.forecasting
        minimum_days = int(forecasting.model_retrain_min_evaluated_forecast_days)
        cooldown_days = int(forecasting.model_retrain_cooldown_days)
        automatic_execution_enabled = bool(forecasting.auto_retrain_enabled)

        artifact = self.repository.active_model_artifact(model_name)
        if artifact is None:
            return self._decision(
                recommended=False,
                reason="model_unavailable",
                message="No active model artifact is available for retraining evaluation.",
                minimum_required=minimum_days,
                cooldown_days=cooldown_days,
                automatic_execution_enabled=automatic_execution_enabled,
            )

        snapshot = self.repository.latest_monitoring_snapshot(
            model_artifact_id=artifact.id,
            model_name=artifact.model_name,
            model_version=artifact.version,
        )
        if snapshot is None:
            return self._decision(
                recommended=False,
                reason="monitoring_unavailable",
                message="No monitoring snapshot exists for the active model.",
                minimum_required=minimum_days,
                cooldown_days=cooldown_days,
                automatic_execution_enabled=automatic_execution_enabled,
                artifact=artifact,
            )

        evidence_key = self._evidence_key(artifact, snapshot)
        existing = self.repository.retraining_run_by_evidence_key(evidence_key)
        if existing is not None:
            return self._decision(
                recommended=True,
                reason="retraining_recommended",
                message="Retraining has already been recommended for this monitoring evidence.",
                minimum_required=minimum_days,
                cooldown_days=cooldown_days,
                automatic_execution_enabled=automatic_execution_enabled,
                artifact=artifact,
                snapshot=snapshot,
                retraining_run=existing,
                created=False,
                new_evaluated_forecast_days=int(existing.new_evaluated_forecast_days or 0),
                last_retraining_attempt_at=existing.triggered_at,
            )

        if bool(forecasting.model_retrain_require_degraded_status) and snapshot.status != "degraded":
            return self._decision(
                recommended=False,
                reason="monitoring_not_degraded",
                message=f"Monitoring status is {snapshot.status}; retraining is recommended only after degradation.",
                minimum_required=minimum_days,
                cooldown_days=cooldown_days,
                automatic_execution_enabled=automatic_execution_enabled,
                artifact=artifact,
                snapshot=snapshot,
            )

        last_run = self.repository.latest_retraining_run(baseline_model_artifact_id=artifact.id)
        cooldown_remaining = self._cooldown_remaining_days(last_run, now, cooldown_days)
        if cooldown_remaining > 0:
            return self._decision(
                recommended=False,
                reason="cooldown_active",
                message=f"Retraining was already recommended or attempted recently; cooldown has {cooldown_remaining} day(s) remaining.",
                minimum_required=minimum_days,
                cooldown_days=cooldown_days,
                cooldown_remaining_days=cooldown_remaining,
                automatic_execution_enabled=automatic_execution_enabled,
                artifact=artifact,
                snapshot=snapshot,
                last_retraining_attempt_at=last_run.triggered_at if last_run is not None else None,
            )

        evidence_start = self._evidence_start(artifact, last_run)
        forecast_days = self.repository.completed_evaluated_forecast_days(
            model_artifact_id=artifact.id,
            generated_after=evidence_start,
            generated_at_or_before=snapshot.generated_at,
        )
        if forecast_days < minimum_days:
            return self._decision(
                recommended=False,
                reason="insufficient_new_evidence",
                message=f"Only {forecast_days} new evaluated forecast-day(s) are available; {minimum_days} are required.",
                minimum_required=minimum_days,
                cooldown_days=cooldown_days,
                automatic_execution_enabled=automatic_execution_enabled,
                artifact=artifact,
                snapshot=snapshot,
                new_evaluated_forecast_days=forecast_days,
                last_retraining_attempt_at=last_run.triggered_at if last_run is not None else None,
            )

        run = None
        created = False
        if persist_recommendation:
            run = self.repository.create_retraining_run(
                {
                    "triggered_at": now,
                    "trigger_reason": "retraining_recommended",
                    "status": "recommended",
                    "baseline_model_artifact_id": artifact.id,
                    "source_monitoring_snapshot_id": snapshot.id,
                    "new_evaluated_forecast_days": forecast_days,
                    "promotion_recommended": False,
                    "evidence_key": evidence_key,
                }
            )
            created = True

        return self._decision(
            recommended=True,
            reason="retraining_recommended",
            message="Persistent forecast degradation has enough new completed evaluation evidence for retraining review.",
            minimum_required=minimum_days,
            cooldown_days=cooldown_days,
            automatic_execution_enabled=automatic_execution_enabled,
            artifact=artifact,
            snapshot=snapshot,
            retraining_run=run,
            created=created,
            new_evaluated_forecast_days=forecast_days,
            last_retraining_attempt_at=last_run.triggered_at if last_run is not None else None,
        )

    @staticmethod
    def _evidence_start(artifact: ModelArtifact, last_run: RetrainingRun | None) -> datetime | None:
        if last_run is not None:
            return RetrainingDecisionService._as_aware(last_run.triggered_at)
        for value in (artifact.activated_at, artifact.training_finished_at, artifact.created_at):
            if value is not None:
                return RetrainingDecisionService._as_aware(value)
        return None

    @staticmethod
    def _cooldown_remaining_days(last_run: RetrainingRun | None, now: datetime, cooldown_days: int) -> int:
        if last_run is None or cooldown_days <= 0:
            return 0
        triggered_at = RetrainingDecisionService._as_aware(last_run.triggered_at)
        cooldown_until = triggered_at + timedelta(days=cooldown_days)
        if cooldown_until <= now:
            return 0
        remaining = cooldown_until - now
        return max(1, int((remaining.total_seconds() + 86399) // 86400))

    @staticmethod
    def _evidence_key(artifact: ModelArtifact, snapshot: ModelMonitoringSnapshot) -> str:
        raw = "|".join(
            [
                str(artifact.id),
                artifact.model_name,
                artifact.version,
                str(snapshot.id),
                snapshot.evidence_key,
                str(snapshot.evaluation_count),
                snapshot.generated_at.isoformat() if snapshot.generated_at else "",
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _as_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _decision(
        *,
        recommended: bool,
        reason: str,
        message: str,
        minimum_required: int,
        cooldown_days: int,
        automatic_execution_enabled: bool,
        artifact: ModelArtifact | None = None,
        snapshot: ModelMonitoringSnapshot | None = None,
        retraining_run: RetrainingRun | None = None,
        created: bool = False,
        new_evaluated_forecast_days: int = 0,
        cooldown_remaining_days: int = 0,
        last_retraining_attempt_at: datetime | None = None,
    ) -> RetrainingDecision:
        return RetrainingDecision(
            recommended=recommended,
            reason=reason,
            message=message,
            latest_monitoring_status=snapshot.status if snapshot is not None else None,
            new_evaluated_forecast_days=new_evaluated_forecast_days,
            minimum_required=minimum_required,
            cooldown_days=cooldown_days,
            cooldown_remaining_days=cooldown_remaining_days,
            baseline_model_artifact_id=artifact.id if artifact is not None else None,
            baseline_model_name=artifact.model_name if artifact is not None else None,
            baseline_model_version=artifact.version if artifact is not None else None,
            source_monitoring_snapshot_id=snapshot.id if snapshot is not None else None,
            last_retraining_attempt_at=last_retraining_attempt_at,
            automatic_execution_enabled=automatic_execution_enabled,
            retraining_run=retraining_run,
            created=created,
        )
