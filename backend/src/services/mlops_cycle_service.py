"""Production-safe MLOps operational cycle orchestration.

The cycle intentionally stops at retraining recommendation. It never trains a
candidate, promotes a model, rolls back a model, or changes inference runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from services.forecast_evaluation_service import EvaluationRunSummary, ForecastEvaluationService
from services.model_monitoring_service import ModelMonitoringService
from services.retraining_decision_service import RetrainingDecision, RetrainingDecisionService


@dataclass(frozen=True)
class EvaluationStageReport:
    due_count: int
    evaluated_count: int
    skipped_count: int
    error_count: int
    already_evaluated: int
    actual_demand_unavailable: int
    invalid_prediction: int


@dataclass(frozen=True)
class MonitoringStageReport:
    model_artifact_id: int | None
    model_name: str
    model_version: str | None
    status: str
    created: bool
    evaluation_count: int
    recent_wape: float | None
    baseline_wape: float | None
    degradation_reason: str | None


@dataclass(frozen=True)
class RetrainingStageReport:
    recommended: bool
    reason: str
    message: str
    new_evaluated_forecast_days: int
    cooldown_remaining_days: int
    retraining_run_id: int | None
    created: bool
    automatic_execution_enabled: bool


@dataclass(frozen=True)
class MLOpsCycleReport:
    cycle_started_at: datetime
    cycle_finished_at: datetime
    dry_run: bool
    evaluations: EvaluationStageReport
    monitoring: MonitoringStageReport
    retraining: RetrainingStageReport


class MLOpsCycleService:
    """Run evaluation, monitoring, and retraining recommendation in order."""

    def __init__(
        self,
        *,
        session: Session,
        evaluation_service: ForecastEvaluationService,
        monitoring_service: ModelMonitoringService,
        retraining_service: RetrainingDecisionService,
    ):
        self.session = session
        self.evaluation_service = evaluation_service
        self.monitoring_service = monitoring_service
        self.retraining_service = retraining_service

    def run(self, *, as_of: date | None = None, dry_run: bool = False) -> MLOpsCycleReport:
        started = datetime.now(timezone.utc)
        if dry_run:
            try:
                return self._run_stages(started=started, as_of=as_of, commit_stages=False)
            finally:
                self.session.rollback()
        return self._run_stages(started=started, as_of=as_of, commit_stages=True)

    def _run_stages(self, *, started: datetime, as_of: date | None, commit_stages: bool) -> MLOpsCycleReport:
        evaluation_summary = self.evaluation_service.evaluate_due_predictions(as_of=as_of)
        if commit_stages:
            self.session.commit()

        monitoring_result = self.monitoring_service.create_snapshot()
        if commit_stages:
            self.session.commit()

        retraining_decision = self.retraining_service.evaluate(persist_recommendation=True)
        if commit_stages:
            self.session.commit()

        return MLOpsCycleReport(
            cycle_started_at=started,
            cycle_finished_at=datetime.now(timezone.utc),
            dry_run=not commit_stages,
            evaluations=self._evaluation_report(evaluation_summary),
            monitoring=self._monitoring_report(monitoring_result),
            retraining=self._retraining_report(retraining_decision),
        )

    @staticmethod
    def _evaluation_report(summary: EvaluationRunSummary) -> EvaluationStageReport:
        skipped = summary.already_evaluated + summary.actual_demand_unavailable + summary.invalid_prediction
        return EvaluationStageReport(
            due_count=summary.eligible,
            evaluated_count=summary.evaluated,
            skipped_count=skipped,
            error_count=0,
            already_evaluated=summary.already_evaluated,
            actual_demand_unavailable=summary.actual_demand_unavailable,
            invalid_prediction=summary.invalid_prediction,
        )

    @staticmethod
    def _monitoring_report(result: Any) -> MonitoringStageReport:
        snapshot = result.snapshot
        return MonitoringStageReport(
            model_artifact_id=snapshot.model_artifact_id,
            model_name=snapshot.model_name,
            model_version=snapshot.model_version,
            status=snapshot.status,
            created=bool(result.created),
            evaluation_count=int(snapshot.evaluation_count or 0),
            recent_wape=_float(snapshot.metric_wape),
            baseline_wape=_float(snapshot.baseline_wape),
            degradation_reason=snapshot.degradation_reason,
        )

    @staticmethod
    def _retraining_report(decision: RetrainingDecision) -> RetrainingStageReport:
        return RetrainingStageReport(
            recommended=decision.recommended,
            reason=decision.reason,
            message=decision.message,
            new_evaluated_forecast_days=decision.new_evaluated_forecast_days,
            cooldown_remaining_days=decision.cooldown_remaining_days,
            retraining_run_id=decision.retraining_run.id if decision.retraining_run is not None else None,
            created=decision.created,
            automatic_execution_enabled=decision.automatic_execution_enabled,
        )


def _float(value: Any) -> float | None:
    return float(value) if value is not None else None
