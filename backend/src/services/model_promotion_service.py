"""Human-controlled model promotion and rollback service.

No scheduler, monitoring job, or public endpoint calls this service. It is used
by explicit operator flows such as scripts/promote_model.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import ModelArtifact, ModelPromotionEvent, RetrainingRun
from repositories.model_artifact_repository import ModelArtifactRepository
from services.runtime_model_service import LoadedRuntimeModel, RuntimeModelLoadError, load_model_artifact

logger = logging.getLogger("supplysync.mlops.promotion")


@dataclass(frozen=True)
class PromotionResult:
    artifact: ModelArtifact
    previous_artifact: ModelArtifact | None
    event: ModelPromotionEvent | None
    loaded_model: LoadedRuntimeModel
    changed: bool


class ModelPromotionServiceError(Exception):
    """Raised when promotion or rollback cannot be completed safely."""


RuntimeHandoff = Callable[[LoadedRuntimeModel], None]


class ModelPromotionService:
    def __init__(self, *, session: Session, settings: Any, runtime_handoff: RuntimeHandoff | None = None):
        self.session = session
        self.settings = settings
        self.runtime_handoff = runtime_handoff
        self.repository = ModelArtifactRepository(session)

    def promote_candidate(
        self,
        artifact_id: int,
        *,
        initiated_by: str = "manual_cli",
        reason: str | None = None,
    ) -> PromotionResult:
        candidate = self.repository.get(artifact_id)
        if candidate is None:
            raise ModelPromotionServiceError(f"Model artifact {artifact_id} does not exist.")
        if candidate.lifecycle_status == "active" and candidate.is_active:
            loaded = self._load(candidate)
            return PromotionResult(candidate, candidate, None, loaded, changed=False)
        if candidate.lifecycle_status != "candidate" or candidate.is_active:
            raise ModelPromotionServiceError("Only inactive candidate artifacts can be promoted.")

        run = self._eligible_retraining_run(candidate)
        if run is None:
            raise ModelPromotionServiceError("Candidate promotion requires completed eligible candidate-evaluation evidence.")

        loaded = self._load(candidate)
        previous = self.repository.active_for_name(candidate.model_name)
        original_state = self._artifact_state(candidate)
        event = self._activate_artifact(
            target=candidate,
            previous=previous,
            event_type="promotion",
            retraining_run_id=run.id,
            initiated_by=initiated_by,
            reason=reason or "candidate_meets_promotion_evidence_gate",
        )
        try:
            self._handoff(loaded)
        except Exception as exc:  # noqa: BLE001 - restore DB state before surfacing failure
            self._restore_previous(candidate, previous, original_state)
            event.outcome = "handoff_failed_restored"
            self.session.flush()
            logger.exception(
                "Model promotion handoff failed and DB lifecycle was restored: candidate_id=%s previous_id=%s event_id=%s",
                candidate.id,
                previous.id if previous else None,
                event.id,
            )
            raise ModelPromotionServiceError(f"Runtime handoff failed; database lifecycle was restored: {exc}") from exc
        event.outcome = "succeeded"
        self.session.flush()
        logger.info(
            "Model promotion succeeded: candidate_id=%s previous_id=%s event_id=%s",
            candidate.id,
            previous.id if previous else None,
            event.id,
        )
        return PromotionResult(candidate, previous, event, loaded, changed=True)

    def rollback_to_artifact(
        self,
        artifact_id: int,
        *,
        initiated_by: str = "manual_cli",
        reason: str | None = None,
    ) -> PromotionResult:
        target = self.repository.get(artifact_id)
        if target is None:
            raise ModelPromotionServiceError(f"Model artifact {artifact_id} does not exist.")
        if target.lifecycle_status == "failed":
            raise ModelPromotionServiceError("Failed artifacts cannot be restored.")

        loaded = self._load(target)
        previous = self.repository.active_for_name(target.model_name)
        if previous is not None and previous.id == target.id:
            return PromotionResult(target, previous, None, loaded, changed=False)

        original_state = self._artifact_state(target)
        event = self._activate_artifact(
            target=target,
            previous=previous,
            event_type="rollback",
            retraining_run_id=None,
            initiated_by=initiated_by,
            reason=reason or "manual_rollback",
        )
        try:
            self._handoff(loaded)
        except Exception as exc:  # noqa: BLE001
            self._restore_previous(target, previous, original_state)
            event.outcome = "handoff_failed_restored"
            self.session.flush()
            logger.exception(
                "Model rollback handoff failed and DB lifecycle was restored: target_id=%s previous_id=%s event_id=%s",
                target.id,
                previous.id if previous else None,
                event.id,
            )
            raise ModelPromotionServiceError(f"Runtime handoff failed; database lifecycle was restored: {exc}") from exc
        event.outcome = "succeeded"
        self.session.flush()
        logger.info(
            "Model rollback succeeded: target_id=%s previous_id=%s event_id=%s",
            target.id,
            previous.id if previous else None,
            event.id,
        )
        return PromotionResult(target, previous, event, loaded, changed=True)

    def _load(self, artifact: ModelArtifact) -> LoadedRuntimeModel:
        try:
            return load_model_artifact(artifact, model_dir=self.settings.forecasting.model_path)
        except RuntimeModelLoadError as exc:
            raise ModelPromotionServiceError(str(exc)) from exc

    def _eligible_retraining_run(self, candidate: ModelArtifact) -> RetrainingRun | None:
        stmt = (
            select(RetrainingRun)
            .where(RetrainingRun.candidate_model_artifact_id == candidate.id)
            .where(RetrainingRun.status == "completed")
            .where(RetrainingRun.promotion_recommended.is_(True))
            .order_by(RetrainingRun.finished_at.desc().nullslast(), RetrainingRun.id.desc())
            .limit(1)
        )
        run = self.session.scalar(stmt)
        if run is None:
            return None

        evidence = ((candidate.training_metadata or {}).get("candidate_evaluation") or {})
        if evidence.get("promotion_eligible") is not True:
            return None
        if not evidence.get("candidate_metrics") or not evidence.get("active_metrics"):
            return None
        min_points = int(getattr(self.settings.forecasting, "routing_min_evaluation_points", 1))
        if int(evidence.get("test_points") or 0) < min_points:
            return None
        if not self._horizon_compatible(evidence.get("horizon_days")):
            return None
        return run

    def _horizon_compatible(self, horizon_days: Any) -> bool:
        if horizon_days is None:
            return False
        expected = getattr(getattr(self.settings, "inventory", None), "default_lead_time_days", None)
        if expected is None:
            return int(horizon_days) > 0
        return int(horizon_days) == int(expected)

    def _activate_artifact(
        self,
        *,
        target: ModelArtifact,
        previous: ModelArtifact | None,
        event_type: str,
        retraining_run_id: int | None,
        initiated_by: str,
        reason: str | None,
    ) -> ModelPromotionEvent:
        now = datetime.now(timezone.utc)

        active_stmt = select(ModelArtifact).where(
            ModelArtifact.model_name == target.model_name,
            ModelArtifact.is_active.is_(True),
        )

        # Deactivate currently active artifacts first so the one-active-model
        # unique constraint never observes two active artifacts at once.
        for current in self.session.scalars(active_stmt):
            if current.id == target.id:
                continue

            current.is_active = False
            current.lifecycle_status = "retired"
            current.retired_at = now

        self.session.flush()

        target.is_active = True
        target.lifecycle_status = "active"
        target.activated_at = now
        target.retired_at = None

        event = ModelPromotionEvent(
            event_type=event_type,
            model_name=target.model_name,
            promoted_model_artifact_id=target.id,
            previous_model_artifact_id=previous.id if previous and previous.id != target.id else None,
            retraining_run_id=retraining_run_id,
            outcome="pending",
            initiated_by=initiated_by,
            reason=reason,
        )

        self.session.add(event)
        self.session.flush()

        return event

    def _artifact_state(self, artifact: ModelArtifact) -> dict[str, Any]:
        return {
            "is_active": artifact.is_active,
            "lifecycle_status": artifact.lifecycle_status,
            "activated_at": artifact.activated_at,
            "retired_at": artifact.retired_at,
        }

    def _restore_previous(
        self,
        target: ModelArtifact,
        previous: ModelArtifact | None,
        target_original_state: dict[str, Any],
    ) -> None:
        # Deactivate target in its own flush first. SQLAlchemy does not order
        # same-table UPDATEs within a flush, so setting target inactive and
        # previous active in one flush can emit previous's UPDATE before
        # target's and momentarily violate the "one active artifact per
        # model_name" partial unique index.
        target.is_active = False
        self.session.flush()

        target.lifecycle_status = target_original_state["lifecycle_status"]
        target.activated_at = target_original_state["activated_at"]
        target.retired_at = target_original_state["retired_at"]
        if previous is not None:
            previous.is_active = True
            previous.lifecycle_status = "active"
            previous.retired_at = None
        target.is_active = target_original_state["is_active"]
        self.session.flush()

    def _handoff(self, loaded: LoadedRuntimeModel) -> None:
        if self.runtime_handoff is not None:
            self.runtime_handoff(loaded)


