from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base, ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog, RetrainingRun
from repositories.retraining_repository import RetrainingRepository
from services.retraining_decision_service import RetrainingDecisionService


@dataclass
class _ForecastingSettings:
    auto_retrain_enabled: bool = False
    model_retrain_min_evaluated_forecast_days: int = 100
    model_retrain_cooldown_days: int = 14
    model_retrain_require_degraded_status: bool = True


@dataclass
class _Settings:
    forecasting: _ForecastingSettings


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


def _settings(**kwargs):
    return _Settings(_ForecastingSettings(**kwargs))


def _service(session, *, settings=None):
    return RetrainingDecisionService(
        repository=RetrainingRepository(session),
        settings=settings or _settings(),
    )


def _artifact(session, *, version="v1", active=True, activated_at=None):
    now = activated_at or datetime.now(timezone.utc) - timedelta(days=30)
    artifact = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version=version,
        lifecycle_status="active" if active else "candidate",
        is_active=active,
        activated_at=now if active else None,
        training_finished_at=now - timedelta(hours=1),
        training_metrics={"wape": 1.0},
    )
    session.add(artifact)
    session.flush()
    return artifact


def _snapshot(session, *, artifact, status="degraded", generated_at=None, evidence_key="snapshot"):
    row = ModelMonitoringSnapshot(
        generated_at=generated_at or datetime.now(timezone.utc),
        model_artifact_id=artifact.id,
        model_name=artifact.model_name,
        model_version=artifact.version,
        window_type="latest_evaluations",
        window_size=30,
        evaluation_count=30,
        metric_wape=Decimal("1.3"),
        metric_mae=Decimal("2.0"),
        metric_rmse=Decimal("3.0"),
        metric_bias=Decimal("0.1"),
        metric_mase=Decimal("0.8"),
        residual_mean=Decimal("-0.1"),
        residual_std=Decimal("1.5"),
        baseline_wape=Decimal("1.0"),
        baseline_provenance="artifact_metadata",
        wape_relative_change=Decimal("0.3"),
        bias_ratio=Decimal("0.01"),
        degradation_reason="persistent_wape_degradation" if status == "degraded" else "wape_within_baseline",
        degradation_message="test message",
        consecutive_degradation_count=2 if status == "degraded" else 0,
        status=status,
        evidence_key=evidence_key,
    )
    session.add(row)
    session.flush()
    return row


def _evaluation(session, *, artifact, generated_at, forecast_days=7, n_test_points=None, horizon_days=None):
    prediction = PredictionLog(
        sku_code="SKU-RT",
        target_start_date=date(2026, 1, 1),
        target_end_date=date(2026, 1, forecast_days),
        demand_source="historical",
        forecast_method="ml_lightgbm",
        forecast_source="model_forecast",
        model_name=artifact.model_name,
        model_version=artifact.version,
        model_artifact_id=artifact.id,
        input_history_length=30,
        forecast_horizon_days=forecast_days,
        forecast_daily=[10.0] * forecast_days,
        recommended_order_quantity=10,
        actual_observed_demand=Decimal(str(10 * forecast_days)),
        actual_observed_at=generated_at,
    )
    session.add(prediction)
    session.flush()
    row = ForecastEvaluation(
        prediction_log_id=prediction.id,
        model_artifact_id=artifact.id,
        sku_code="SKU-RT",
        demand_class="regular",
        model_name=artifact.model_name,
        evaluation_scope="logged_prediction",
        metric_mae=Decimal("1.0"),
        metric_rmse=Decimal("1.0"),
        metric_bias=Decimal("0.0"),
        metric_wape=Decimal("1.3"),
        metric_mase=Decimal("1.0"),
        n_skus=1,
        n_test_points=n_test_points if n_test_points is not None else forecast_days,
        horizon_days=horizon_days if horizon_days is not None else forecast_days,
        generated_at=generated_at,
    )
    session.add(row)
    session.flush()
    return row


def _run(session, *, artifact, snapshot, triggered_at, evidence_key="existing-run"):
    row = RetrainingRun(
        triggered_at=triggered_at,
        trigger_reason="retraining_recommended",
        status="recommended",
        baseline_model_artifact_id=artifact.id,
        source_monitoring_snapshot_id=snapshot.id,
        new_evaluated_forecast_days=100,
        promotion_recommended=False,
        evidence_key=evidence_key,
    )
    session.add(row)
    session.flush()
    return row


def test_degraded_sufficient_evidence_and_cooldown_satisfied_recommends():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=artifact, generated_at=now)
    for idx in range(15):
        _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=idx), forecast_days=7)

    decision = _service(session).evaluate(now=now)

    assert decision.recommended is True
    assert decision.reason == "retraining_recommended"
    assert decision.new_evaluated_forecast_days == 105


@pytest.mark.parametrize("status", ["stable", "warning", "insufficient_evidence"])
def test_non_degraded_monitoring_statuses_do_not_recommend(status):
    session = _session()
    artifact = _artifact(session)
    _snapshot(session, artifact=artifact, status=status)

    decision = _service(session).evaluate()

    assert decision.recommended is False
    assert decision.reason == "monitoring_not_degraded"


def test_unavailable_monitoring_does_not_recommend():
    session = _session()
    _artifact(session)

    decision = _service(session).evaluate()

    assert decision.recommended is False
    assert decision.reason == "monitoring_unavailable"


def test_degraded_with_insufficient_new_forecast_days_does_not_recommend():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=artifact, generated_at=now)
    _evaluation(session, artifact=artifact, generated_at=now, forecast_days=7)

    decision = _service(session).evaluate(now=now)

    assert decision.recommended is False
    assert decision.reason == "insufficient_new_evidence"
    assert decision.new_evaluated_forecast_days == 7


def test_cooldown_active_does_not_recommend():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    snapshot = _snapshot(session, artifact=artifact, generated_at=now, evidence_key="new")
    _run(session, artifact=artifact, snapshot=snapshot, triggered_at=now - timedelta(days=3), evidence_key="old")

    decision = _service(session).evaluate(now=now)

    assert decision.recommended is False
    assert decision.reason == "cooldown_active"
    assert decision.cooldown_remaining_days > 0


def test_recommendation_becomes_allowed_after_cooldown():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    snapshot = _snapshot(session, artifact=artifact, generated_at=now, evidence_key="new")
    _run(session, artifact=artifact, snapshot=snapshot, triggered_at=now - timedelta(days=20), evidence_key="old")
    for idx in range(15):
        _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=idx), forecast_days=7)

    decision = _service(session).evaluate(now=now)

    assert decision.recommended is True
    assert decision.cooldown_remaining_days == 0


def test_repeated_same_evidence_reuses_recommendation_record():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=artifact, generated_at=now)
    for idx in range(15):
        _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=idx), forecast_days=7)
    service = _service(session)

    first = service.evaluate(persist_recommendation=True, now=now)
    second = service.evaluate(persist_recommendation=True, now=now + timedelta(minutes=1))

    assert first.created is True
    assert second.created is False
    assert first.retraining_run.id == second.retraining_run.id
    assert len(session.scalars(select(RetrainingRun)).all()) == 1


def test_previous_model_evidence_does_not_count_for_new_model():
    session = _session()
    old = _artifact(session, version="v1", active=False)
    new = _artifact(session, version="v2", active=True)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=new, generated_at=now)
    for idx in range(15):
        _evaluation(session, artifact=old, generated_at=now - timedelta(minutes=idx), forecast_days=7)

    decision = _service(session).evaluate(now=now)

    assert decision.recommended is False
    assert decision.reason == "insufficient_new_evidence"
    assert decision.new_evaluated_forecast_days == 0


def test_forecast_day_counting_prefers_test_points_then_horizon_then_prediction():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=artifact, generated_at=now)
    _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=2), forecast_days=5, n_test_points=11, horizon_days=5)
    _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=1), forecast_days=5, n_test_points=0, horizon_days=13)
    _evaluation(session, artifact=artifact, generated_at=now, forecast_days=17, n_test_points=0, horizon_days=0)

    decision = _service(
        session,
        settings=_settings(model_retrain_min_evaluated_forecast_days=41),
    ).evaluate(now=now)

    assert decision.new_evaluated_forecast_days == 41
    assert decision.recommended is True


def test_missing_model_artifact_is_handled_safely():
    session = _session()

    decision = _service(session).evaluate()

    assert decision.recommended is False
    assert decision.reason == "model_unavailable"


def test_auto_retrain_disabled_does_not_suppress_recommendation_or_execute_training():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=artifact, generated_at=now)
    for idx in range(15):
        _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=idx), forecast_days=7)

    decision = _service(session, settings=_settings(auto_retrain_enabled=False)).evaluate(
        persist_recommendation=True,
        now=now,
    )

    assert decision.recommended is True
    assert decision.automatic_execution_enabled is False
    assert session.scalar(select(ModelArtifact).where(ModelArtifact.lifecycle_status == "candidate")) is None
    assert session.get(ModelArtifact, artifact.id).lifecycle_status == "active"


def test_repository_flushes_without_committing_transaction():
    session = _session()
    artifact = _artifact(session)
    now = datetime.now(timezone.utc)
    _snapshot(session, artifact=artifact, generated_at=now)
    for idx in range(15):
        _evaluation(session, artifact=artifact, generated_at=now - timedelta(minutes=idx), forecast_days=7)

    decision = _service(session).evaluate(persist_recommendation=True, now=now)
    assert decision.retraining_run.id is not None
    session.rollback()

    assert session.scalar(select(RetrainingRun)) is None


def test_retraining_migration_is_chained_after_monitoring_degradation_state():
    migration = _load_retraining_migration()

    assert migration.revision == "2f5c8d9e7a41"
    assert migration.down_revision == "e6b9d2c4f8a1"


def test_retraining_migration_declares_upgrade_and_downgrade():
    migration = _load_retraining_migration()

    assert callable(migration.upgrade)
    assert callable(migration.downgrade)


def _load_retraining_migration():
    path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "2f5c8d9e7a41_add_retraining_runs.py"
    spec = importlib.util.spec_from_file_location("phase_e_retraining_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
