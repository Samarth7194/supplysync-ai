from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base, ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog
from repositories.model_monitoring_repository import ModelMonitoringRepository
from services.model_monitoring_service import ModelMonitoringService


@dataclass
class _ForecastingSettings:
    model_monitoring_enabled: bool = True
    model_monitoring_window_evaluations: int = 30
    model_monitoring_lookback_days: int = 90
    model_monitoring_min_evaluations: int = 30
    model_monitoring_wape_warning_threshold: float = 0.15
    model_monitoring_wape_degradation_threshold: float = 0.25
    model_monitoring_bias_warning_ratio: float = 0.20
    model_monitoring_degradation_consecutive_runs: int = 2


@dataclass
class _Settings:
    forecasting: _ForecastingSettings


class _DataService:
    def __init__(self, series_by_sku):
        self.series_by_sku = series_by_sku

    def get_demand_history(self, sku):
        return self.series_by_sku.get(sku, pd.Series(dtype=float))


def _session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return Session()


def _artifact(session, *, version="v1", active=True, training_metrics=None):
    row = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version=version,
        lifecycle_status="active" if active else "candidate",
        is_active=active,
        training_metrics=training_metrics or {},
    )
    session.add(row)
    session.flush()
    return row


def _logged_eval(
    session,
    *,
    artifact,
    sku="SKU-1",
    generated_at,
    actual_total=30,
    forecast=None,
    actual=None,
    horizon=3,
    mae=None,
    rmse=None,
    bias=None,
    wape=None,
    mase=Decimal("0.5"),
):
    if forecast is None:
        forecast = [8.0, 9.0, 10.0]
    if actual is None:
        actual = [10.0, 10.0, 10.0]
    mae = Decimal(str(mae if mae is not None else sum(abs(a - p) for a, p in zip(actual, forecast)) / horizon))
    rmse = Decimal(str(rmse if rmse is not None else (sum((a - p) ** 2 for a, p in zip(actual, forecast)) / horizon) ** 0.5))
    bias = Decimal(str(bias if bias is not None else sum(p - a for a, p in zip(actual, forecast)) / horizon))
    wape = Decimal(str(wape if wape is not None else sum(abs(a - p) for a, p in zip(actual, forecast)) / actual_total))
    prediction = PredictionLog(
        sku_code=sku,
        target_start_date=date(2024, 1, 1),
        target_end_date=date(2024, 1, horizon),
        demand_source="historical",
        forecast_method="ml_lightgbm",
        forecast_source="model_forecast",
        model_name="lightgbm_demand_forecast",
        model_version=artifact.version,
        model_artifact_id=artifact.id,
        input_history_length=30,
        forecast_horizon_days=horizon,
        forecast_daily=forecast,
        recommended_order_quantity=10,
        actual_observed_demand=Decimal(str(actual_total)),
        actual_observed_at=generated_at,
    )
    session.add(prediction)
    session.flush()
    evaluation = ForecastEvaluation(
        prediction_log_id=prediction.id,
        model_artifact_id=artifact.id,
        sku_code=sku,
        demand_class="regular",
        model_name="lightgbm_demand_forecast",
        evaluation_scope="logged_prediction",
        metric_mae=mae,
        metric_rmse=rmse,
        metric_bias=bias,
        metric_wape=wape,
        metric_mase=mase,
        n_skus=1,
        n_test_points=horizon,
        horizon_days=horizon,
        generated_at=generated_at,
    )
    session.add(evaluation)
    session.flush()
    return evaluation


def _service(session, *, data_service=None, settings=None, offline_path=None):
    return ModelMonitoringService(
        repository=ModelMonitoringRepository(session),
        settings=settings or _Settings(_ForecastingSettings()),
        data_service=data_service,
        offline_evaluation_path=offline_path,
    )


def test_metrics_are_calculated_from_completed_evaluations_with_daily_residuals():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _logged_eval(session, artifact=artifact, generated_at=now, forecast=[8, 9, 10])
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=1), forecast=[10, 10, 10])
    data = _DataService({"SKU-1": pd.Series([10.0, 10.0, 10.0], index=pd.date_range("2024-01-01", periods=3))})

    result = _service(session, data_service=data).create_snapshot()

    assert result.snapshot.evaluation_count == 2
    assert result.snapshot.metric_wape == pytest.approx(Decimal("0.05"), abs=Decimal("0.0001"))
    assert result.snapshot.metric_mae == pytest.approx(Decimal("0.5"), abs=Decimal("0.0001"))
    assert result.snapshot.metric_rmse == pytest.approx(Decimal("0.913"), abs=Decimal("0.001"))
    assert result.snapshot.metric_bias == pytest.approx(Decimal("-0.5"), abs=Decimal("0.0001"))
    assert result.snapshot.residual_mean == pytest.approx(Decimal("0.5"), abs=Decimal("0.0001"))
    assert result.snapshot.residual_std is not None


def test_unevaluated_predictions_are_excluded():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _logged_eval(session, artifact=artifact, generated_at=now)
    session.add(
        PredictionLog(
            sku_code="SKU-1",
            target_start_date=date(2024, 2, 1),
            target_end_date=date(2024, 2, 3),
            demand_source="historical",
            forecast_method="ml_lightgbm",
            forecast_source="model_forecast",
            model_name="lightgbm_demand_forecast",
            model_version=artifact.version,
            model_artifact_id=artifact.id,
            input_history_length=30,
            forecast_horizon_days=3,
            forecast_daily=[100.0, 100.0, 100.0],
            recommended_order_quantity=10,
        )
    )
    session.flush()

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.evaluation_count == 1
    assert snapshot.metric_mae == Decimal("1.000000")


def test_monitoring_scopes_to_exact_active_model_artifact():
    session = _session()
    active = _artifact(session, version="v1", active=True)
    old = _artifact(session, version="v0", active=False)
    now = datetime.now(timezone.utc)
    _logged_eval(session, artifact=active, generated_at=now, mae=1, rmse=1, bias=0, wape=Decimal("0.1"))
    _logged_eval(session, artifact=old, generated_at=now, mae=99, rmse=99, bias=99, wape=Decimal("9.9"))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.model_artifact_id == active.id
    assert snapshot.evaluation_count == 1
    assert snapshot.metric_mae == Decimal("1.000000")


def test_latest_30_evaluations_are_respected():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    for idx in range(35):
        _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=idx), mae=idx, rmse=idx, bias=0, wape=Decimal("0.1"))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.evaluation_count == 30
    assert snapshot.metric_mae == Decimal("14.500000")


def test_90_day_lookback_is_respected():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _logged_eval(session, artifact=artifact, generated_at=now, mae=1, rmse=1, bias=0, wape=Decimal("0.1"))
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(days=120), mae=99, rmse=99, bias=0, wape=Decimal("0.1"))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.evaluation_count == 1
    assert snapshot.metric_mae == Decimal("1.000000")


def test_insufficient_evidence_status():
    session = _session()
    artifact = _artifact(session)
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.status == "insufficient_evidence"


def test_healthy_data_status_when_minimum_evidence_is_met():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    for idx in range(2):
        _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=idx))
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=2))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "stable"


def test_wape_aggregation_uses_total_actual_demand_not_blind_average():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _logged_eval(session, artifact=artifact, generated_at=now, actual_total=100, wape=Decimal("0.1"))
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=1), actual_total=10, wape=Decimal("1.0"))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.metric_wape == pytest.approx(Decimal("0.181818"), abs=Decimal("0.000001"))


def test_rmse_bias_mase_and_missing_mase_are_handled():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _logged_eval(session, artifact=artifact, generated_at=now, rmse=3, bias=-2, mase=Decimal("1.0"))
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=1), rmse=4, bias=2, mase=None)

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.metric_rmse == Decimal("3.535534")
    assert snapshot.metric_bias == Decimal("0.000000")
    assert snapshot.metric_mase == Decimal("1.000000")
    assert snapshot.residual_mean == Decimal("0.000000")


def test_missing_baseline_is_handled_safely():
    session = _session()
    artifact = _artifact(session)
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.baseline_wape is None
    assert snapshot.baseline_provenance == "unavailable"


def test_offline_baseline_provenance_is_labeled(tmp_path):
    session = _session()
    artifact = _artifact(session)
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), horizon=3)
    path = tmp_path / "forecast_evaluation.json"
    path.write_text(
        """
        {
          "horizons": {
            "3": {
              "aggregates": {
                "all": {
                  "lightgbm": {"wape": 0.42}
                }
              }
            }
          }
        }
        """
    )

    snapshot = _service(session, offline_path=path).create_snapshot().snapshot

    assert snapshot.baseline_wape == Decimal("0.420000")
    assert snapshot.baseline_provenance == "offline_backtest"


def test_stable_when_wape_below_warning_threshold():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), wape=Decimal("1.10"))
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1, model_monitoring_window_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "stable"
    assert snapshot.degradation_reason == "wape_within_baseline"
    assert snapshot.wape_relative_change == Decimal("0.100000")


def test_warning_when_wape_exceeds_warning_threshold():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), wape=Decimal("1.16"))
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1, model_monitoring_window_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "warning"
    assert snapshot.degradation_reason == "wape_warning_threshold_exceeded"


def test_first_degradation_threshold_breach_is_warning_not_degraded():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), wape=Decimal("1.30"))
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "warning"
    assert snapshot.degradation_reason == "wape_degradation_threshold_exceeded"
    assert snapshot.consecutive_degradation_count == 1


def test_second_consecutive_breach_with_newer_evidence_becomes_degraded():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    now = datetime.now(timezone.utc)
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))
    service = _service(session, settings=settings)
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=1), wape=Decimal("1.30"))
    first = service.create_snapshot().snapshot
    _logged_eval(session, artifact=artifact, generated_at=now, wape=Decimal("1.30"))

    second = service.create_snapshot().snapshot

    assert first.status == "warning"
    assert second.status == "degraded"
    assert second.degradation_reason == "persistent_wape_degradation"
    assert second.consecutive_degradation_count == 2


def test_repeated_run_over_identical_evidence_does_not_increase_consecutive_count():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), wape=Decimal("1.30"))
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))
    service = _service(session, settings=settings)

    first = service.create_snapshot()
    second = service.create_snapshot()

    assert first.created is True
    assert second.created is False
    assert second.snapshot.consecutive_degradation_count == 1
    assert second.snapshot.status == "warning"


def test_recovery_resets_consecutive_degradation_sequence():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    now = datetime.now(timezone.utc)
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1, model_monitoring_window_evaluations=1))
    service = _service(session, settings=settings)
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=2), wape=Decimal("1.30"))
    service.create_snapshot()
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=1), wape=Decimal("1.0"))
    recovery = service.create_snapshot().snapshot
    _logged_eval(session, artifact=artifact, generated_at=now, wape=Decimal("1.30"))

    next_breach = service.create_snapshot().snapshot

    assert recovery.status == "stable"
    assert next_breach.status == "warning"
    assert next_breach.consecutive_degradation_count == 1


def test_insufficient_evidence_overrides_threshold_calculation():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), wape=Decimal("9.0"))

    snapshot = _service(session).create_snapshot().snapshot

    assert snapshot.status == "insufficient_evidence"
    assert snapshot.degradation_reason == "insufficient_evidence"


def test_zero_baseline_is_handled_safely():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 0.0})
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc), wape=Decimal("1.0"))
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "stable"
    assert snapshot.degradation_reason == "baseline_zero"
    assert snapshot.wape_relative_change is None


def test_previous_model_snapshots_do_not_influence_new_model():
    session = _session()
    first = _artifact(session, version="v1", active=True, training_metrics={"wape": 1.0})
    now = datetime.now(timezone.utc)
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))
    service = _service(session, settings=settings)
    _logged_eval(session, artifact=first, generated_at=now - timedelta(minutes=2), wape=Decimal("1.30"))
    service.create_snapshot()
    first.is_active = False
    first.lifecycle_status = "retired"
    second = _artifact(session, version="v2", active=True, training_metrics={"wape": 1.0})
    _logged_eval(session, artifact=second, generated_at=now, wape=Decimal("1.30"))

    snapshot = service.create_snapshot().snapshot

    assert snapshot.model_artifact_id == second.id
    assert snapshot.status == "warning"
    assert snapshot.consecutive_degradation_count == 1


def test_bias_warning_below_threshold_remains_stable():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(
        session,
        artifact=artifact,
        generated_at=datetime.now(timezone.utc),
        actual_total=300,
        bias=Decimal("10"),
        wape=Decimal("1.0"),
    )
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "stable"
    assert snapshot.bias_ratio == Decimal("0.100000")


def test_bias_warning_above_threshold_does_not_mark_degraded():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(
        session,
        artifact=artifact,
        generated_at=datetime.now(timezone.utc),
        actual_total=300,
        bias=Decimal("30"),
        wape=Decimal("1.0"),
    )
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "warning"
    assert snapshot.degradation_reason == "bias_warning"
    assert snapshot.bias_ratio == Decimal("0.300000")


def test_zero_actual_demand_bias_normalization_is_safe():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    _logged_eval(
        session,
        artifact=artifact,
        generated_at=datetime.now(timezone.utc),
        actual_total=0,
        bias=Decimal("30"),
        wape=Decimal("1.0"),
    )
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))

    snapshot = _service(session, settings=settings).create_snapshot().snapshot

    assert snapshot.status == "stable"
    assert snapshot.bias_ratio is None


def test_warning_and_degraded_reasons_are_deterministic():
    session = _session()
    artifact = _artifact(session, training_metrics={"wape": 1.0})
    now = datetime.now(timezone.utc)
    settings = _Settings(_ForecastingSettings(model_monitoring_min_evaluations=1))
    service = _service(session, settings=settings)
    _logged_eval(session, artifact=artifact, generated_at=now - timedelta(minutes=1), wape=Decimal("1.30"))
    first = service.create_snapshot().snapshot
    _logged_eval(session, artifact=artifact, generated_at=now, wape=Decimal("1.30"))
    second = service.create_snapshot().snapshot

    assert first.degradation_reason == "wape_degradation_threshold_exceeded"
    assert second.degradation_reason == "persistent_wape_degradation"


def test_duplicate_monitoring_run_reuses_snapshot():
    session = _session()
    artifact = _artifact(session)
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc))
    service = _service(session)

    first = service.create_snapshot()
    second = service.create_snapshot()

    assert first.created is True
    assert second.created is False
    assert first.snapshot.id == second.snapshot.id
    assert len(session.scalars(select(ModelMonitoringSnapshot)).all()) == 1


def test_repository_flushes_without_committing_transaction():
    session = _session()
    artifact = _artifact(session)
    _logged_eval(session, artifact=artifact, generated_at=datetime.now(timezone.utc))

    snapshot = _service(session).create_snapshot().snapshot
    assert snapshot.id is not None
    session.rollback()

    assert session.scalar(select(ModelMonitoringSnapshot)) is None
