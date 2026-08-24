"""Create a Phase A model monitoring snapshot from completed evaluations."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SRC_DIR))

from config.settings import load_settings  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from repositories.model_monitoring_repository import ModelMonitoringRepository  # noqa: E402
from services.data_service import DataService  # noqa: E402
from services.model_monitoring_service import ModelMonitoringService  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402


def _fmt(value) -> str:
    return "n/a" if value is None else str(value)


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.1f}%"


def main() -> int:
    settings = load_settings()
    if not settings.forecasting.model_monitoring_enabled:
        print("Model monitoring is disabled by MODEL_MONITORING_ENABLED=false")
        return 0

    try:
        data_service = DataService.get_instance()
    except Exception:  # noqa: BLE001 - residual std can be unavailable without failing monitoring
        data_service = None

    with SessionLocal() as session:
        service = ModelMonitoringService(
            repository=ModelMonitoringRepository(session),
            settings=settings,
            data_service=data_service,
            offline_evaluation_path=BACKEND_DIR / "data" / "forecast_evaluation.json",
        )
        try:
            result = service.create_snapshot()
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            print("Model monitoring failed: database operation failed.")
            print(f"Database error: {type(exc).__name__}")
            return 1

    snapshot = result.snapshot
    print("Model monitoring snapshot")
    print("")
    print(f"Model: {snapshot.model_name}")
    print(f"Version: {_fmt(snapshot.model_version)}")
    print(f"Evaluations: {snapshot.evaluation_count}")
    print("")
    print(f"Recent WAPE: {_fmt(snapshot.metric_wape)}")
    print(f"Baseline WAPE: {_fmt(snapshot.baseline_wape)}")
    print(f"Relative change: {_pct(snapshot.wape_relative_change)}")
    print("")
    print(f"MAE: {_fmt(snapshot.metric_mae)}")
    print(f"RMSE: {_fmt(snapshot.metric_rmse)}")
    print(f"Bias: {_fmt(snapshot.metric_bias)}")
    print(f"MASE: {_fmt(snapshot.metric_mase)}")
    print(f"Residual mean: {_fmt(snapshot.residual_mean)}")
    print(f"Residual std: {_fmt(snapshot.residual_std)}")
    print(f"Bias ratio: {_pct(snapshot.bias_ratio)}")
    print("")
    print(f"State: {snapshot.status}")
    print(f"Reason: {snapshot.degradation_reason}")
    print(f"Message: {snapshot.degradation_message}")
    print(f"Consecutive degradation count: {snapshot.consecutive_degradation_count}")
    print(f"Baseline provenance: {snapshot.baseline_provenance}")
    print(f"Created: {result.created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
