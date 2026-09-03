from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import AnalysisRun, Base, ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, ModelPromotionEvent, PredictionLog, RetrainingRun
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository
from repositories.model_monitoring_repository import ModelMonitoringRepository
from repositories.retraining_repository import RetrainingRepository
from services.forecast_evaluation_service import ForecastEvaluationService
from services.mlops_cycle_service import MLOpsCycleService
from services.model_monitoring_service import ModelMonitoringService
from services.retraining_decision_service import RetrainingDecisionService


class _DataService:
    def __init__(self, series_by_sku):
        self.series_by_sku = series_by_sku

    def get_demand_history(self, sku):
        return self.series_by_sku.get(sku, pd.Series(dtype=float))


@dataclass(frozen=True)
class _ForecastingSettings:
    model_path: str = "unused"
    model_monitoring_window_evaluations: int = 30
    model_monitoring_lookback_days: int = 365
    model_monitoring_min_evaluations: int = 1
    model_monitoring_wape_warning_threshold: float = 0.15
    model_monitoring_wape_degradation_threshold: float = 0.25
    model_monitoring_bias_warning_ratio: float = 0.20
    model_monitoring_degradation_consecutive_runs: int = 1
    auto_retrain_enabled: bool = False
    model_retrain_min_evaluated_forecast_days: int = 1
    model_retrain_cooldown_days: int = 0
    model_retrain_require_degraded_status: bool = True


@dataclass(frozen=True)
class _Settings:
    forecasting: _ForecastingSettings = _ForecastingSettings()


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


def _active_artifact(session, *, baseline_wape: float = 0.2):
    artifact = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version="active-v1",
        artifact_checksum="a" * 64,
        checksum_algorithm="sha256",
        artifact_uri="active.pkl",
        feature_schema=["f1"],
        feature_schema_version="test_schema",
        feature_schema_checksum="schema-checksum",
        training_metrics={"wape": baseline_wape},
        lifecycle_status="active",
        is_active=True,
        activated_at=datetime.now(timezone.utc),
    )
    session.add(artifact)
    session.flush()
    return artifact


def _prediction(session, artifact, *, forecast, sku="SKU-1"):
    analysis = AnalysisRun(
        sku_code=sku,
        current_stock=Decimal("5"),
        recommended_order_quantity=10,
        action="PURCHASE",
        risk="HIGH",
        demand_pattern="regular",
        demand_source="historical",
        forecast_source="model",
        forecast_method="lightgbm",
        lead_time_days=3,
        service_level=Decimal("0.95"),
    )
    session.add(analysis)
    session.flush()
    prediction = PredictionLog(
        analysis_run_id=analysis.id,
        model_artifact_id=artifact.id,
        sku_code=sku,
        target_start_date=date(2024, 1, 3),
        target_end_date=date(2024, 1, 5),
        demand_source="historical",
        forecast_method="lightgbm",
        forecast_source="model",
        model_name=artifact.model_name,
        model_version=artifact.version,
        input_history_length=2,
        forecast_horizon_days=3,
        forecast_daily=forecast,
        recommended_order_quantity=10,
    )
    session.add(prediction)
    session.flush()
    return prediction


def _cycle(session, data_service, settings=None, retraining_service=None):
    settings = settings or _Settings()
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
        ),
        retraining_service=retraining_service
        or RetrainingDecisionService(repository=RetrainingRepository(session), settings=settings),
    )


def test_cycle_evaluates_monitors_and_recommends_without_training_or_promotion():
    session = _session()
    artifact = _active_artifact(session, baseline_wape=0.1)
    _prediction(session, artifact, forecast=[0.0, 0.0, 0.0])
    data = _DataService({"SKU-1": pd.Series([5, 5, 10, 10, 10], index=pd.date_range("2024-01-01", periods=5))})

    report = _cycle(session, data).run(as_of=date(2024, 1, 5))

    assert report.evaluations.evaluated_count == 1
    assert report.monitoring.status == "degraded"
    assert report.retraining.recommended is True
    assert report.retraining.created is True
    assert report.retraining.automatic_execution_enabled is False
    assert session.query(ForecastEvaluation).count() == 1
    assert session.query(ModelMonitoringSnapshot).count() == 1
    assert session.query(RetrainingRun).count() == 1
    assert session.query(ModelPromotionEvent).count() == 0
    assert session.query(ModelArtifact).count() == 1


def test_cycle_is_idempotent_for_same_prediction_and_monitoring_evidence():
    session = _session()
    artifact = _active_artifact(session, baseline_wape=0.1)
    _prediction(session, artifact, forecast=[0.0, 0.0, 0.0])
    data = _DataService({"SKU-1": pd.Series([5, 5, 10, 10, 10], index=pd.date_range("2024-01-01", periods=5))})

    first = _cycle(session, data).run(as_of=date(2024, 1, 5))
    second = _cycle(session, data).run(as_of=date(2024, 1, 5))

    assert first.evaluations.evaluated_count == 1
    assert second.evaluations.already_evaluated == 1
    assert second.monitoring.created is False
    assert second.retraining.created is False
    assert session.query(ForecastEvaluation).count() == 1
    assert session.query(ModelMonitoringSnapshot).count() == 1
    assert session.query(RetrainingRun).count() == 1


def test_cycle_dry_run_rolls_back_all_stage_writes():
    session = _session()
    artifact = _active_artifact(session, baseline_wape=0.1)
    _prediction(session, artifact, forecast=[0.0, 0.0, 0.0])
    data = _DataService({"SKU-1": pd.Series([5, 5, 10, 10, 10], index=pd.date_range("2024-01-01", periods=5))})

    report = _cycle(session, data).run(as_of=date(2024, 1, 5), dry_run=True)

    assert report.dry_run is True
    assert report.evaluations.evaluated_count == 1
    assert session.query(ForecastEvaluation).count() == 0
    assert session.query(ModelMonitoringSnapshot).count() == 0
    assert session.query(RetrainingRun).count() == 0
    assert session.query(ModelPromotionEvent).count() == 0


def test_stable_cycle_does_not_recommend_retraining():
    session = _session()
    artifact = _active_artifact(session, baseline_wape=0.5)
    _prediction(session, artifact, forecast=[9.0, 10.0, 11.0])
    data = _DataService({"SKU-1": pd.Series([5, 5, 10, 10, 10], index=pd.date_range("2024-01-01", periods=5))})

    report = _cycle(session, data).run(as_of=date(2024, 1, 5))

    assert report.monitoring.status == "stable"
    assert report.retraining.recommended is False
    assert report.retraining.reason == "monitoring_not_degraded"
    assert session.query(RetrainingRun).count() == 0


class _FailingRetrainingService:
    def evaluate(self, *, persist_recommendation=False):
        raise RuntimeError("decision store unavailable")


def test_retraining_failure_keeps_evaluation_and_monitoring_committed():
    session = _session()
    artifact = _active_artifact(session, baseline_wape=0.1)
    _prediction(session, artifact, forecast=[0.0, 0.0, 0.0])
    data = _DataService({"SKU-1": pd.Series([5, 5, 10, 10, 10], index=pd.date_range("2024-01-01", periods=5))})

    with pytest.raises(RuntimeError, match="decision store unavailable"):
        _cycle(session, data, retraining_service=_FailingRetrainingService()).run(as_of=date(2024, 1, 5))

    assert session.query(ForecastEvaluation).count() == 1
    assert session.query(ModelMonitoringSnapshot).count() == 1
    assert session.query(RetrainingRun).count() == 0
