"""Run the safe SupplySync MLOps operational cycle.

This script evaluates due predictions, creates/reuses one monitoring snapshot,
and persists a retraining recommendation when evidence qualifies. It never
trains a candidate, promotes a model, rolls back a model, or deploys artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from config.settings import load_settings  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository  # noqa: E402
from repositories.model_monitoring_repository import ModelMonitoringRepository  # noqa: E402
from repositories.retraining_repository import RetrainingRepository  # noqa: E402
from services.data_service import DataService  # noqa: E402
from services.forecast_evaluation_service import ForecastEvaluationService  # noqa: E402
from services.mlops_cycle_service import MLOpsCycleService  # noqa: E402
from services.model_monitoring_service import ModelMonitoringService  # noqa: E402
from services.retraining_decision_service import RetrainingDecisionService  # noqa: E402


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _build_cycle_service(session, settings) -> MLOpsCycleService:
    data_service = DataService.get_instance()
    return MLOpsCycleService(
        session=session,
        evaluation_service=ForecastEvaluationService(
            repository=ForecastEvaluationRepository(session),
            data_service=data_service,
        ),
        monitoring_service=ModelMonitoringService(
            repository=ModelMonitoringRepository(session),
            settings=settings,
            data_service=data_service,
            offline_evaluation_path=BACKEND_DIR / "data" / "forecast_evaluation.json",
        ),
        retraining_service=RetrainingDecisionService(
            repository=RetrainingRepository(session),
            settings=settings,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the safe SupplySync MLOps cycle.")
    parser.add_argument("--as-of", type=str, default=None, help="Evaluate predictions due on or before YYYY-MM-DD.")
    parser.add_argument("--dry-run", action="store_true", help="Run inside a rollback-only transaction.")
    parser.add_argument("--json", action="store_true", help="Print the cycle report as JSON.")
    args = parser.parse_args()

    settings = load_settings()
    with SessionLocal() as session:
        try:
            service = _build_cycle_service(session, settings)
            report = service.run(as_of=_parse_date(args.as_of), dry_run=args.dry_run)
        except SQLAlchemyError as exc:
            session.rollback()
            print("SupplySync MLOps cycle failed: database operation failed.", file=sys.stderr)
            print(f"Database error: {type(exc).__name__}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - CLI should fail clearly for operators
            session.rollback()
            print(f"SupplySync MLOps cycle failed: {exc}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(asdict(report), default=_json_default, indent=2, sort_keys=True))
        return 0

    print("SupplySync MLOps Cycle")
    print(f"Mode: {'DRY RUN' if report.dry_run else 'LIVE'}")
    print("")
    print("Evaluation")
    print(f"Due predictions: {report.evaluations.due_count}")
    print(f"Evaluated: {report.evaluations.evaluated_count}")
    print(f"Skipped: {report.evaluations.skipped_count}")
    print(f"Already evaluated: {report.evaluations.already_evaluated}")
    print(f"Actual demand unavailable: {report.evaluations.actual_demand_unavailable}")
    print(f"Invalid predictions: {report.evaluations.invalid_prediction}")
    print("")
    print("Monitoring")
    print(f"Model: {report.monitoring.model_name}")
    print(f"Artifact: {_fmt(report.monitoring.model_artifact_id)}")
    print(f"Version: {_fmt(report.monitoring.model_version)}")
    print(f"State: {report.monitoring.status}")
    print(f"Recent WAPE: {_pct(report.monitoring.recent_wape)}")
    print(f"Baseline WAPE: {_pct(report.monitoring.baseline_wape)}")
    print(f"Evaluations: {report.monitoring.evaluation_count}")
    print(f"Created snapshot: {report.monitoring.created}")
    print("")
    print("Retraining")
    print(f"Recommended: {'YES' if report.retraining.recommended else 'NO'}")
    print(f"Reason: {report.retraining.reason}")
    print(f"New evaluated forecast-days: {report.retraining.new_evaluated_forecast_days}")
    print(f"Cooldown remaining: {report.retraining.cooldown_remaining_days}")
    print(f"Retraining run: {_fmt(report.retraining.retraining_run_id)}")
    print(f"Automatic execution enabled: {report.retraining.automatic_execution_enabled}")
    print("")
    print("IMPORTANT:")
    print("No model was trained.")
    print("No model was promoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
