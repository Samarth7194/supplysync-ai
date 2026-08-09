"""Evaluate persisted prediction logs against recorded actual demand."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from db.session import SessionLocal  # noqa: E402
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository  # noqa: E402
from services.data_service import DataService  # noqa: E402
from services.forecast_evaluation_service import ForecastEvaluationService  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402


def main() -> int:
    try:
        data_service = DataService.get_instance()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot evaluate logged predictions: data service unavailable ({exc})")
        return 1

    with SessionLocal() as session:
        service = ForecastEvaluationService(
            repository=ForecastEvaluationRepository(session),
            data_service=data_service,
        )
        try:
            summary = service.evaluate_due_predictions()
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            print(
                "Cannot evaluate logged predictions: SQLAlchemy database is not ready. "
                "Run `cd backend && python -m alembic upgrade head` first."
            )
            print(f"Database error: {exc}")
            return 1

    print(f"Predictions scanned: {summary.predictions_scanned}")
    print(f"Eligible: {summary.eligible}")
    print(f"Evaluated: {summary.evaluated}")
    print(f"Already evaluated: {summary.already_evaluated}")
    print(f"Actual demand unavailable: {summary.actual_demand_unavailable}")
    print(f"Invalid prediction: {summary.invalid_prediction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
