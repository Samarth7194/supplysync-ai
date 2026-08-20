"""Check whether retraining should be recommended.

This Phase E script persists or reuses a recommendation record when the current
monitoring evidence qualifies. It never runs training, registers a candidate,
promotes a model, or changes production inference.
"""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from config.settings import load_settings  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from repositories.retraining_repository import RetrainingRepository  # noqa: E402
from services.retraining_decision_service import RetrainingDecisionService  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402


def _fmt(value) -> str:
    return "n/a" if value is None else str(value)


def main() -> int:
    settings = load_settings()
    with SessionLocal() as session:
        service = RetrainingDecisionService(
            repository=RetrainingRepository(session),
            settings=settings,
        )
        try:
            decision = service.evaluate(persist_recommendation=True)
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            print("Retraining recommendation failed: database operation failed.")
            print(f"Database error: {type(exc).__name__}")
            return 1

    print("Retraining decision")
    print("")
    print(f"Model: {_fmt(decision.baseline_model_name)}")
    print(f"Version: {_fmt(decision.baseline_model_version)}")
    print(f"Monitoring status: {_fmt(decision.latest_monitoring_status)}")
    print(f"New evaluated forecast-days: {decision.new_evaluated_forecast_days}")
    print(f"Required forecast-days: {decision.minimum_required}")
    print(
        "Cooldown: "
        + ("satisfied" if decision.cooldown_remaining_days == 0 else f"{decision.cooldown_remaining_days} day(s) remaining")
    )
    print(f"Automatic execution enabled: {decision.automatic_execution_enabled}")
    print("")
    print(f"Decision: {'RETRAINING RECOMMENDED' if decision.recommended else 'NOT RECOMMENDED'}")
    print(f"Reason: {decision.reason}")
    print(f"Message: {decision.message}")
    if decision.retraining_run is not None:
        print(f"Retraining run id: {decision.retraining_run.id}")
        print(f"Created: {decision.created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
