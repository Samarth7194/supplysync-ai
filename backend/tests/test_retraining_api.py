from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError

import main as backend_main
from db.models import ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog, RetrainingRun
from db.session import SessionLocal


def _clean():
    with SessionLocal() as session:
        for model in (RetrainingRun, ModelMonitoringSnapshot, ForecastEvaluation, PredictionLog, ModelArtifact):
            session.execute(delete(model))
        session.commit()


@pytest.fixture(autouse=True)
def _isolate_retraining_api_rows():
    _clean()
    yield
    _clean()


def _artifact_and_snapshot(status="degraded"):
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        artifact = ModelArtifact(
            model_name="lightgbm_demand_forecast",
            model_family="lightgbm",
            model_type="ml",
            version="api-v1",
            lifecycle_status="active",
            is_active=True,
            activated_at=now - timedelta(days=30),
            training_metrics={"wape": 1.0},
        )
        session.add(artifact)
        session.flush()
        snapshot = ModelMonitoringSnapshot(
            generated_at=now,
            model_artifact_id=artifact.id,
            model_name=artifact.model_name,
            model_version=artifact.version,
            window_type="latest_evaluations",
            window_size=30,
            evaluation_count=30,
            metric_wape=Decimal("1.3"),
            baseline_wape=Decimal("1.0"),
            baseline_provenance="artifact_metadata",
            wape_relative_change=Decimal("0.3"),
            degradation_reason="persistent_wape_degradation" if status == "degraded" else "wape_within_baseline",
            degradation_message="test message",
            consecutive_degradation_count=2 if status == "degraded" else 0,
            status=status,
            evidence_key=f"api-{status}",
        )
        session.add(snapshot)
        for idx in range(15):
            prediction = PredictionLog(
                sku_code=f"SKU-{idx}",
                target_start_date=date(2026, 1, 1),
                target_end_date=date(2026, 1, 7),
                demand_source="historical",
                forecast_method="ml_lightgbm",
                forecast_source="model_forecast",
                model_name=artifact.model_name,
                model_version=artifact.version,
                model_artifact_id=artifact.id,
                input_history_length=30,
                forecast_horizon_days=7,
                forecast_daily=[10.0] * 7,
                recommended_order_quantity=10,
            )
            session.add(prediction)
            session.flush()
            session.add(
                ForecastEvaluation(
                    prediction_log_id=prediction.id,
                    model_artifact_id=artifact.id,
                    sku_code=f"SKU-{idx}",
                    demand_class="regular",
                    model_name=artifact.model_name,
                    evaluation_scope="logged_prediction",
                    n_test_points=7,
                    horizon_days=7,
                    generated_at=now - timedelta(minutes=idx),
                )
            )
        session.commit()


def test_retraining_status_api_reports_recommendation_without_persisting():
    _artifact_and_snapshot()

    with TestClient(backend_main.app) as client:
        body = client.get("/api/model-retraining/status").json()

    assert body["recommended"] is True
    assert body["reason"] == "retraining_recommended"
    assert body["latest_monitoring_status"] == "degraded"
    assert body["new_evaluated_forecast_days"] == 105
    assert body["minimum_required"] == 100
    assert body["baseline_model"]["model_name"] == "lightgbm_demand_forecast"
    with SessionLocal() as session:
        assert session.query(RetrainingRun).count() == 0


def test_retraining_status_api_reports_not_recommended_for_warning():
    _artifact_and_snapshot(status="warning")

    with TestClient(backend_main.app) as client:
        body = client.get("/api/model-retraining/status").json()

    assert body["recommended"] is False
    assert body["reason"] == "monitoring_not_degraded"


def test_retraining_status_api_does_not_affect_analyze():
    _artifact_and_snapshot()

    with TestClient(backend_main.app) as client:
        assert client.get("/api/model-retraining/status").status_code == 200
        response = client.post(
            "/api/analyze",
            json={
                "sku": "SKU-0",
                "current_stock": 10,
                "demand_history": [10, 11, 9, 12, 10, 8, 11, 10, 9, 12, 10, 11, 9, 10],
            },
        )

    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert "recommended_order" in response.json()


def test_retraining_status_db_error_uses_api_error_convention():
    class _FailingService:
        def evaluate(self, *, persist_recommendation=False):
            raise SQLAlchemyError("boom")

    backend_main.app.dependency_overrides[backend_main.get_retraining_decision_service] = lambda: _FailingService()
    try:
        with TestClient(backend_main.app) as client:
            response = client.get("/api/model-retraining/status")
    finally:
        backend_main.app.dependency_overrides.clear()

    assert response.status_code == 503


def test_retraining_endpoint_appears_in_openapi():
    with TestClient(backend_main.app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/model-retraining/status" in paths
