from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

import main as backend_main
from db.models import ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, PredictionLog, RetrainingRun
from db.session import SessionLocal


def _clean():
    with SessionLocal() as session:
        for model in (RetrainingRun, ModelMonitoringSnapshot, ForecastEvaluation, PredictionLog, ModelArtifact):
            session.execute(delete(model))
        session.commit()


@pytest.fixture(autouse=True)
def _isolate_replay_api_rows():
    _clean()
    yield
    _clean()


@pytest.fixture(autouse=True)
def _restore_replay_path():
    original = backend_main._HISTORICAL_REPLAY_PATH
    yield
    backend_main._HISTORICAL_REPLAY_PATH = original


def _write_replay_file(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "historical_monitoring_replay.json"
    path.write_text(json.dumps(payload))
    return path


def _sample_payload(**overrides) -> dict:
    payload = {
        "provenance": "historical_replay",
        "generated_at": "2026-09-03T09:37:33+00:00",
        "model_name": "lightgbm_demand_forecast",
        "model_artifact_id": None,
        "model_version": "lightgbm_demand_forecast-20260523T084002Z-ea7815fa9a33",
        "horizon_days": 7,
        "sku_count": 59,
        "status": "warning",
        "degradation_reason": "wape_warning_threshold_exceeded",
        "degradation_message": "Recent WAPE is 17.4% worse than baseline.",
        "evaluation_count": 39,
        "metric_wape": 1.25,
        "metric_mae": 92.2,
        "metric_rmse": 228.5,
        "metric_bias": 21.1,
        "metric_mase": 1.23,
        "baseline_wape": 1.0655,
        "baseline_provenance": "offline_backtest",
        "historical_period": {"start": "2011-11-19", "end": "2011-12-09"},
        "method_breakdown": {
            "ml_lightgbm": {"sku_count": 47, "evaluation_count": 133, "wape": 1.13},
            "croston": {"sku_count": 11, "evaluation_count": 31, "wape": 0.79},
        },
    }
    payload.update(overrides)
    return payload


def test_replay_endpoint_unavailable_when_no_file_generated(tmp_path):
    backend_main._HISTORICAL_REPLAY_PATH = tmp_path / "does_not_exist.json"

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring/replay")

    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert body["mode"] == "historical_replay"
    assert "run_historical_monitoring_replay.py" in body["message"]


def test_replay_endpoint_serves_generated_replay_read_only(tmp_path):
    backend_main._HISTORICAL_REPLAY_PATH = _write_replay_file(tmp_path, _sample_payload())

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring/replay")

    assert res.status_code == 200
    body = res.json()
    assert body["available"] is True
    assert body["provenance"] == "historical_replay"
    assert body["status"] == "warning"
    assert body["metrics"]["wape"] == pytest.approx(1.25)
    assert body["baseline_wape"] == pytest.approx(1.0655)
    assert body["horizon_days"] == 7
    assert body["forecast_days"] == 7
    assert body["historical_period"]["start"] == "2011-11-19"
    assert body["historical_period"]["end"] == "2011-12-09"
    assert "ml_lightgbm" in body["method_breakdown"]
    assert "live_monitoring" in body
    assert body["live_monitoring"]["available"] is False


def test_replay_endpoint_reports_live_monitoring_availability_alongside_replay(tmp_path):
    backend_main._HISTORICAL_REPLAY_PATH = _write_replay_file(tmp_path, _sample_payload())
    with SessionLocal() as session:
        artifact = ModelArtifact(
            model_name="lightgbm_demand_forecast",
            model_family="lightgbm",
            model_type="ml",
            version="live-v1",
            lifecycle_status="active",
            is_active=True,
        )
        session.add(artifact)
        session.flush()
        snapshot = ModelMonitoringSnapshot(
            generated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            model_artifact_id=artifact.id,
            model_name="lightgbm_demand_forecast",
            model_version="live-v1",
            window_type="latest_evaluations",
            window_size=30,
            evaluation_count=30,
            baseline_provenance="offline_backtest",
            degradation_reason="wape_within_baseline",
            degradation_message="ok",
            status="stable",
            evidence_key="live-evidence",
        )
        session.add(snapshot)
        session.commit()

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring/replay")

    body = res.json()
    # The replay endpoint itself always serves the replay artifact — the
    # frontend is responsible for the live-takes-precedence display rule —
    # but it must truthfully report that live evidence exists and how much.
    assert body["available"] is True
    assert body["live_monitoring"]["available"] is True
    assert body["live_monitoring"]["evaluation_count"] == 30


def test_replay_endpoint_handles_unreadable_file_gracefully(tmp_path):
    path = tmp_path / "historical_monitoring_replay.json"
    path.write_text("{not valid json")
    backend_main._HISTORICAL_REPLAY_PATH = path

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring/replay")

    assert res.status_code == 200
    assert res.json()["available"] is False


def test_replay_endpoint_appears_in_openapi():
    with TestClient(backend_main.app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/model-monitoring/replay" in paths


def test_replay_endpoint_never_triggers_replay_generation(tmp_path, monkeypatch):
    """The endpoint must only read a pre-generated file — it must never
    import or call the replay service itself (which would make a GET request
    expensive and could be abused for repeated recomputation)."""
    backend_main._HISTORICAL_REPLAY_PATH = tmp_path / "does_not_exist.json"

    def _forbidden(*args, **kwargs):
        raise AssertionError("replay generation must not be triggered by the API")

    monkeypatch.setattr(
        "services.historical_monitoring_replay_service.HistoricalMonitoringReplayService.run",
        _forbidden,
        raising=False,
    )

    with TestClient(backend_main.app) as client:
        res = client.get("/api/model-monitoring/replay")

    assert res.status_code == 200
