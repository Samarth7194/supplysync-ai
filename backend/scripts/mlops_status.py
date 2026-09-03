"""Read-only MLOps status summary for operators and demos."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from config.settings import load_settings  # noqa: E402
from db.models import ModelArtifact, ModelPromotionEvent  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from repositories.model_monitoring_repository import ModelMonitoringRepository  # noqa: E402
from repositories.retraining_repository import RetrainingRepository  # noqa: E402
from services.model_monitoring_service import ModelMonitoringService  # noqa: E402
from services.retraining_decision_service import RetrainingDecisionService  # noqa: E402
from services.runtime_model_service import MODEL_NAME, load_runtime_model  # noqa: E402


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _latest_candidate(session):
    stmt = (
        select(ModelArtifact)
        .where(ModelArtifact.model_name == MODEL_NAME)
        .where(ModelArtifact.lifecycle_status == "candidate")
        .where(ModelArtifact.is_active.is_(False))
        .order_by(ModelArtifact.created_at.desc(), ModelArtifact.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def _latest_event(session):
    stmt = (
        select(ModelPromotionEvent)
        .where(ModelPromotionEvent.model_name == MODEL_NAME)
        .order_by(ModelPromotionEvent.created_at.desc(), ModelPromotionEvent.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def main() -> int:
    settings = load_settings()
    runtime = load_runtime_model(settings=settings, session_factory=SessionLocal)
    with SessionLocal() as session:
        try:
            monitoring_service = ModelMonitoringService(
                repository=ModelMonitoringRepository(session),
                settings=settings,
                offline_evaluation_path=BACKEND_DIR / "data" / "forecast_evaluation.json",
            )
            retraining_service = RetrainingDecisionService(
                repository=RetrainingRepository(session),
                settings=settings,
            )
            active = RetrainingRepository(session).active_model_artifact(MODEL_NAME)
            snapshot = monitoring_service.current_snapshot(model_name=MODEL_NAME)
            decision = retraining_service.evaluate(model_name=MODEL_NAME, persist_recommendation=False)
            candidate = _latest_candidate(session)
            event = _latest_event(session)
        except SQLAlchemyError as exc:
            print("MLOps status unavailable: database operation failed.", file=sys.stderr)
            print(f"Database error: {type(exc).__name__}", file=sys.stderr)
            return 1

    print("SupplySync MLOps Status")
    print("")
    print("Runtime model")
    print(f"Source: {_fmt(runtime.status.get('source'))}")
    print(f"Artifact: {_fmt(runtime.status.get('artifact_id'))}")
    print(f"Model: {_fmt(runtime.model_name)}")
    print(f"Version: {_fmt(runtime.model_version)}")
    print(f"Loadable: {_fmt(runtime.status.get('loadable'))}")
    print("")
    print("DB active model")
    print(f"Artifact: {_fmt(active.id if active else None)}")
    print(f"Version: {_fmt(active.version if active else None)}")
    print(f"Lifecycle: {_fmt(active.lifecycle_status if active else None)}")
    print("")
    print("Monitoring")
    print(f"State: {_fmt(snapshot.status if snapshot else None)}")
    print(f"Latest WAPE: {_pct(snapshot.metric_wape if snapshot else None)}")
    print(f"Baseline WAPE: {_pct(snapshot.baseline_wape if snapshot else None)}")
    print(f"Evaluations: {_fmt(snapshot.evaluation_count if snapshot else None)}")
    print("")
    print("Retraining recommendation")
    print(f"Recommended: {'YES' if decision.recommended else 'NO'}")
    print(f"Reason: {decision.reason}")
    print(f"Run: {_fmt(decision.retraining_run.id if decision.retraining_run else None)}")
    print("")
    print("Latest candidate")
    print(f"Artifact: {_fmt(candidate.id if candidate else None)}")
    print(f"Version: {_fmt(candidate.version if candidate else None)}")
    evidence = ((candidate.training_metadata or {}).get("candidate_evaluation") or {}) if candidate else {}
    print(f"Promotion eligible: {_fmt(evidence.get('promotion_eligible'))}")
    print("")
    print("Last promotion event")
    print(f"Event: {_fmt(event.id if event else None)}")
    print(f"Type: {_fmt(event.event_type if event else None)}")
    print(f"Outcome: {_fmt(event.outcome if event else None)}")
    print(f"Target artifact: {_fmt(event.promoted_model_artifact_id if event else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


