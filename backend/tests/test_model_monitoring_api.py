from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError

import main as backend_main
from db.models import ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog, RetrainingRun
from db.session import SessionLocal


class _StubDataService:
    def __init__(self, series_by_sku):
        self._series = series_by_sku

    def get_demand_history(self, sku):
        return self._series.get(sku, pd.Series(dtype=float))

    def get_top_skus(self, n=20):
        return list(self._series.keys())


def _clean():
    with SessionLocal() as session:
        for model in (RetrainingRun, ModelMonitoringSnapshot, ForecastEvaluation, PredictionLog, ModelArtifact):
            session.execute(delete(model))
        session.commit()


@pytest.fixture(autouse=True)
def _isolate_monitoring_api_rows():
    _clean()
    yield
    _clean()


def _artifact(*, version="v1", active=True, training_metrics=None):
    with SessionLocal() as session:
        artifact = ModelArtifact(
            model_name="lightgbm_demand_forecast",
            model_family="lightgbm",
            model_type="ml",
            version=version,
            lifecycle_status="active" if active else "candidate",
            is_active=active,
            training_metrics=training_metrics or {"wape": 1.0},
        )
        session.add(artifact)
        session.commit()
        return artifact.id


def _snapshot(
    *,
    artifact_id=None,
    version="v1",
    status="stable",
    generated_at=None,
    evidence_key="snapshot",
    wape=Decimal("1.0"),
    baseline=Decimal("1.0"),
    degradation_reason="wape_within_baseline",
):
    generated_at = generated_at or datetime.now(timezone.utc)
    with SessionLocal() as session:
        row = ModelMonitoringSnapshot(
            generated_at=generated_at,
            model_artifact_id=artifact_id,
            model_name="lightgbm_demand_forecast",
            model_version=version,
            window_type="latest_evaluations",
            window_size=30,
            evaluation_count=30,
            metric_wape=wape,
            metric_mae=Decimal("2.0"),
            metric_rmse=Decimal("3.0"),
            metric_bias=Decimal("0.1"),
            metric_mase=Decimal("0.8"),
            residual_mean=Decimal("-0.1"),
            residual_std=Decimal("1.5"),
            baseline_wape=baseline,
            baseline_provenance="offline_backtest",
            wape_relative_change=Decimal("0.0"),
            bias_ratio=Decimal("0.01"),
            degradation_reason=degradation_reason,
            degradation_message="test monitoring message",
            consecutive_degradation_count=2 if status == "degraded" else 0,
            status=status,
            evidence_key=evidence_key,
        )
        session.add(row)
        session.commit()
        return row.id


def _logged_evaluation(*, artifact_id, version="v1", generated_at=None, wape=Decimal("1.3")):
    generated_at = generated_at or datetime.now(timezone.utc)
    with SessionLocal() as session:
        prediction = PredictionLog(
            model_artifact_id=artifact_id,
            sku_code="SKU-API",
            target_start_date=date(2024, 1, 1),
            target_end_date=date(2024, 1, 3),
            demand_source="historical",
            forecast_method="ml_lightgbm",
            forecast_source="model_forecast",
            model_name="lightgbm_demand_forecast",
            model_version=version,
            input_history_length=30,
            forecast_horizon_days=3,
            forecast_daily=[8.0, 9.0, 10.0],
            recommended_order_quantity=10,
            actual_observed_demand=Decimal("30"),
            actual_observed_at=generated_at,
        )
        session.add(prediction)
        session.flush()
        session.add(
            ForecastEvaluation(
                prediction_log_id=prediction.id,
                model_artifact_id=artifact_id,
                sku_code="SKU-API",
                demand_class="regular",
                model_name="lightgbm_demand_forecast",
                evaluation_scope="logged_prediction",
                metric_mae=Decimal("1.0"),
                metric_rmse=Decimal("1.2"),
                metric_bias=Decimal("0.1"),
                metric_wape=wape,
                metric_mase=Decimal("0.8"),
                n_skus=1,
                n_test_points=3,
                horizon_days=3,
                generated_at=generated_at,
            )
        )
        session.commit()


def test_current_monitoring_returns_latest_snapshot():
    _clean()
    artifact_id = _artifact()
    older = datetime.now(timezone.utc) - timedelta(days=1)
    _snapshot(artifact_id=artifact_id, status="warning", generated_at=older, evidence_key="old")
    _snapshot(artifact_id=artifact_id, status="degraded", evidence_key="new", degradation_reason="persistent_wape_degradation")

    with TestClient(backend_main.app) as client:
        body = client.get("/api/model-monitoring").json()

    assert body["status"] == "degraded"
    assert body["degradation_reason"] == "persistent_wape_degradation"
    assert body["lifecycle_status"] == "active"
    assert body["baseline_provenance"] == "offline_backtest"


def test_current_monitoring_unavailable_when_no_snapshot_exists():
    _clean()

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring")

    assert res.status_code == 200
    assert res.json()["status"] == "unavailable"


def test_current_monitoring_exposes_all_persisted_states():
    for status in ("insufficient_evidence", "stable", "warning", "degraded"):
        _clean()
        artifact_id = _artifact()
        reason = "persistent_wape_degradation" if status == "degraded" else "wape_within_baseline"
        if status == "insufficient_evidence":
            reason = "insufficient_evidence"
        _snapshot(artifact_id=artifact_id, status=status, evidence_key=status, degradation_reason=reason)
        with TestClient(backend_main.app) as client:
            body = client.get("/api/model-monitoring").json()
        assert body["status"] == status
        assert "metric_wape" in body
        assert "wape_relative_change" in body


def test_history_is_newest_first_and_limit_is_applied():
    _clean()
    artifact_id = _artifact()
    now = datetime.now(timezone.utc)
    for idx in range(3):
        _snapshot(
            artifact_id=artifact_id,
            generated_at=now - timedelta(minutes=idx),
            evidence_key=f"history-{idx}",
            status="stable",
        )

    with TestClient(backend_main.app) as client:
        body = client.get("/api/model-monitoring/history?limit=2").json()

    assert body["count"] == 2
    assert body["limit"] == 2
    assert body["items"][0]["generated_at"] > body["items"][1]["generated_at"]


def test_history_limit_validation():
    _clean()

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring/history?limit=101")

    assert res.status_code == 422


def test_history_filters_by_artifact_and_status():
    _clean()
    artifact_a = _artifact(version="v1", active=True)
    artifact_b = _artifact(version="v2", active=False)
    _snapshot(artifact_id=artifact_a, version="v1", status="stable", evidence_key="a")
    _snapshot(artifact_id=artifact_b, version="v2", status="warning", evidence_key="b")

    with TestClient(backend_main.app) as client:
        by_artifact = client.get(f"/api/model-monitoring/history?model_artifact_id={artifact_b}").json()
        by_status = client.get("/api/model-monitoring/history?status=warning").json()

    assert by_artifact["count"] == 1
    assert by_artifact["items"][0]["model_artifact_id"] == artifact_b
    assert by_status["count"] == 1
    assert by_status["items"][0]["status"] == "warning"


def test_post_evaluate_creates_and_reuses_snapshot_without_lifecycle_change():
    _clean()
    artifact_id = _artifact(training_metrics={"wape": 1.0})
    _logged_evaluation(artifact_id=artifact_id)
    original_data_service = backend_main._data_service
    series = pd.Series([10.0, 10.0, 10.0], index=pd.date_range("2024-01-01", periods=3))

    try:
        with TestClient(backend_main.app) as client:
            backend_main._data_service = _StubDataService({"SKU-API": series})
            first = client.post("/api/model-monitoring/evaluate").json()
            second = client.post("/api/model-monitoring/evaluate").json()
    finally:
        backend_main._data_service = original_data_service

    assert first["created"] is True
    assert second["created"] is False
    assert second["status"] == "insufficient_evidence"
    with SessionLocal() as session:
        artifact = session.get(ModelArtifact, artifact_id)
        assert artifact.lifecycle_status == "active"
        assert artifact.is_active is True


def test_post_evaluate_does_not_affect_analyze_inference():
    _clean()
    artifact_id = _artifact(training_metrics={"wape": 1.0})
    _logged_evaluation(artifact_id=artifact_id)
    series = pd.Series([20.0] * 45, index=pd.date_range("2024-01-01", periods=45))

    with TestClient(backend_main.app) as client:
        backend_main._data_service = _StubDataService({"SKU-API": series})
        assert client.post("/api/model-monitoring/evaluate").status_code == 200
        analyze = client.post("/api/analyze", json={"sku": "SKU-API", "current_stock": 10})

    assert analyze.status_code == 200
    assert "recommended_order" in analyze.json()


def test_monitoring_db_error_uses_api_error_convention():
    class _FailingService:
        def current_snapshot(self):
            raise SQLAlchemyError("boom")

    backend_main.app.dependency_overrides[backend_main.get_model_monitoring_service] = lambda: _FailingService()
    try:
        with TestClient(backend_main.app) as client:
            res = client.get("/api/model-monitoring")
    finally:
        backend_main.app.dependency_overrides.clear()

    assert res.status_code == 503


def test_monitoring_endpoints_appear_in_openapi():
    with TestClient(backend_main.app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/model-monitoring" in paths
    assert "/api/model-monitoring/history" in paths
    assert "/api/model-monitoring/evaluate" in paths
