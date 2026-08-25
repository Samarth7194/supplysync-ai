from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

import pandas as pd
import pytest
from sqlalchemy import select

from db.models import Base, ForecastEvaluation, ModelArtifact, ModelMonitoringSnapshot, RetrainingRun
from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum
from services.candidate_evaluation_service import CandidateEvaluationResult, CandidateEvaluationService
from services.candidate_training_service import CandidateTrainingError, CandidateTrainingService
from services.model_service import ModelService
from tests.test_retraining_decision_service import _session


@dataclass
class _ForecastingSettings:
    model_retrain_min_evaluated_forecast_days: int = 100
    routing_min_evaluation_points: int = 100
    routing_min_relative_improvement: float = 0.05


@dataclass
class _InventorySettings:
    default_lead_time_days: int = 7


@dataclass
class _Settings:
    forecasting: _ForecastingSettings
    inventory: _InventorySettings | None = None


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features):
        return [self.value]


class _TinyRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [10.0] * len(X)


def _settings(*, inventory_days: int | None = None, **kwargs):
    inventory = _InventorySettings(inventory_days) if inventory_days is not None else None
    return _Settings(_ForecastingSettings(**kwargs), inventory=inventory)


def _artifact(session, *, version: str, active: bool, artifact_uri: str | None = None, checksum: str | None = None):
    artifact = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version=version,
        artifact_checksum=checksum,
        checksum_algorithm="sha256" if checksum else None,
        artifact_uri=artifact_uri,
        metadata_uri=None,
        feature_schema=FEATURE_COLUMNS,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_checksum=feature_schema_checksum(FEATURE_COLUMNS),
        training_metrics={"wape": 1.0},
        training_metadata={"test": True},
        lifecycle_status="active" if active else "candidate",
        is_active=active,
        activated_at=datetime.now(timezone.utc) - timedelta(days=30) if active else None,
    )
    session.add(artifact)
    session.flush()
    return artifact


def _snapshot(session, artifact):
    row = ModelMonitoringSnapshot(
        generated_at=datetime.now(timezone.utc),
        model_artifact_id=artifact.id,
        model_name=artifact.model_name,
        model_version=artifact.version,
        window_type="latest_evaluations",
        window_size=30,
        evaluation_count=30,
        baseline_provenance="artifact_metadata",
        degradation_reason="persistent_wape_degradation",
        degradation_message="test degradation",
        consecutive_degradation_count=2,
        status="degraded",
        evidence_key="candidate-training-evidence",
    )
    session.add(row)
    session.flush()
    return row


def _run(session, artifact, snapshot, *, status="recommended", evidence_days=120):
    row = RetrainingRun(
        triggered_at=datetime.now(timezone.utc),
        trigger_reason="retraining_recommended",
        status=status,
        baseline_model_artifact_id=artifact.id,
        source_monitoring_snapshot_id=snapshot.id,
        new_evaluated_forecast_days=evidence_days,
        promotion_recommended=False,
        evidence_key="candidate-training-key",
    )
    session.add(row)
    session.flush()
    return row


def _daily_demand(*, sku_count=4, days=90, value=10.0):
    rows = []
    start = pd.Timestamp("2026-01-01")
    for sku_idx in range(sku_count):
        for day in range(days):
            rows.append({"StockCode": f"SKU-{sku_idx}", "date": start + pd.Timedelta(days=day), "demand": value})
    return pd.DataFrame(rows)


def _fake_trainer(tmp_path: Path, *, candidate_value=10.0):
    def trainer(*, model_dir, artifact_file, metadata_file, **kwargs):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / artifact_file
        with model_path.open("wb") as fh:
            pickle.dump(_ConstantModel(candidate_value), fh)
        checksum = ModelService.checksum_file(model_path)
        metadata = {
            "model_name": "lightgbm_demand_forecast",
            "model_family": "lightgbm",
            "model_type": "ml",
            "version": f"candidate-{checksum[:12]}",
            "artifact_file": artifact_file,
            "metadata_file": metadata_file,
            "artifact_checksum": checksum,
            "checksum_algorithm": "sha256",
            "features": FEATURE_COLUMNS,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_checksum": feature_schema_checksum(FEATURE_COLUMNS),
            "dataset": "unit-test.csv",
            "n_train_rows": 120,
            "n_test_rows": 30,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "training_data": {"date_start": "2026-01-01", "date_end": "2026-03-31", "row_count": 360},
            "training_config": {"objective": "regression", "metric": "mae"},
        }
        metadata_path = model_dir / metadata_file
        metadata_path.write_text("{}")

        class Result:
            pass

        result = Result()
        result.metadata = metadata
        result.model_path = model_path
        result.metadata_path = metadata_path
        return result

    return trainer


class _FakeEvaluator:
    def __init__(self, *, eligible=True):
        self.eligible = eligible
        self.calls = 0

    def evaluate(self, *, candidate_artifact, active_artifact, daily_demand, horizon_days=30, persist=True):
        self.calls += 1
        result = CandidateEvaluationResult(
            horizon_days=horizon_days,
            test_points=120,
            candidate_metrics={"wape": 0.8, "mae": 1.0, "rmse": 1.0, "bias": 0.0, "mase": 0.8, "n_test_points": 120},
            active_metrics={"wape": 1.0, "mae": 2.0, "rmse": 2.0, "bias": 0.1, "mase": 1.0, "n_test_points": 120},
            benchmark_metrics={"croston_sba": {"wape": 0.9}, "moving_avg_7": {"wape": 1.1}, "seasonal_naive_7": {"wape": 1.2}},
            relative_wape_improvement=0.2,
            promotion_eligible=self.eligible,
            eligibility_reason="candidate_meets_promotion_evidence_gate" if self.eligible else "candidate_worse_than_active",
            temporal_split={"method": "per_sku_last_n_days_holdout"},
        )
        candidate_artifact.training_metadata = dict(candidate_artifact.training_metadata or {})
        candidate_artifact.training_metadata["candidate_evaluation"] = result.as_dict()
        candidate_artifact.training_metrics = dict(candidate_artifact.training_metrics or {})
        candidate_artifact.training_metrics["promotion_eligible"] = self.eligible
        return result


def test_valid_retraining_run_trains_registers_and_links_candidate(tmp_path):
    session = _session()
    active = _artifact(session, version="active-v1", active=True)
    snapshot = _snapshot(session, active)
    run = _run(session, active, snapshot)
    evaluator = _FakeEvaluator(eligible=True)

    result = CandidateTrainingService(
        session=session,
        settings=_settings(),
        model_dir=tmp_path,
        trainer=_fake_trainer(tmp_path),
        evaluator=evaluator,
    ).train_candidate(retraining_run_id=run.id, daily_demand=_daily_demand())

    assert result.created is True
    assert result.candidate_artifact.lifecycle_status == "candidate"
    assert result.candidate_artifact.is_active is False
    assert result.retraining_run.status == "completed"
    assert result.retraining_run.candidate_model_artifact_id == result.candidate_artifact.id
    assert result.retraining_run.promotion_recommended is True
    assert session.get(ModelArtifact, active.id).lifecycle_status == "active"
    assert session.get(ModelArtifact, active.id).is_active is True
    assert evaluator.calls == 1


def test_duplicate_execution_reuses_existing_candidate(tmp_path):
    session = _session()
    active = _artifact(session, version="active-v1", active=True)
    snapshot = _snapshot(session, active)
    run = _run(session, active, snapshot)
    service = CandidateTrainingService(
        session=session,
        settings=_settings(),
        model_dir=tmp_path,
        trainer=_fake_trainer(tmp_path),
        evaluator=_FakeEvaluator(),
    )

    first = service.train_candidate(retraining_run_id=run.id, daily_demand=_daily_demand())
    second = service.train_candidate(retraining_run_id=run.id, daily_demand=_daily_demand())

    assert first.candidate_artifact.id == second.candidate_artifact.id
    assert second.created is False
    assert len(session.scalars(select(ModelArtifact).where(ModelArtifact.lifecycle_status == "candidate")).all()) == 1


def test_invalid_and_insufficient_retraining_runs_are_rejected(tmp_path):
    session = _session()
    active = _artifact(session, version="active-v1", active=True)
    snapshot = _snapshot(session, active)
    insufficient = _run(session, active, snapshot, evidence_days=7)

    service = CandidateTrainingService(
        session=session,
        settings=_settings(),
        model_dir=tmp_path,
        trainer=_fake_trainer(tmp_path),
        evaluator=_FakeEvaluator(),
    )

    with pytest.raises(CandidateTrainingError, match="does not exist"):
        service.train_candidate(retraining_run_id=999, daily_demand=_daily_demand())
    with pytest.raises(CandidateTrainingError, match="enough evaluated"):
        service.train_candidate(retraining_run_id=insufficient.id, daily_demand=_daily_demand())


def test_training_failure_marks_run_failed_and_active_unchanged(tmp_path):
    session = _session()
    active = _artifact(session, version="active-v1", active=True)
    snapshot = _snapshot(session, active)
    run = _run(session, active, snapshot)

    def failing_trainer(**kwargs):
        raise RuntimeError("training boom")

    service = CandidateTrainingService(
        session=session,
        settings=_settings(),
        model_dir=tmp_path,
        trainer=failing_trainer,
        evaluator=_FakeEvaluator(),
    )

    with pytest.raises(CandidateTrainingError, match="training boom"):
        service.train_candidate(retraining_run_id=run.id, daily_demand=_daily_demand())

    assert run.status == "failed"
    assert "training boom" in run.failure_reason
    assert active.lifecycle_status == "active"
    assert active.is_active is True


def test_candidate_artifact_checksum_and_feature_schema_validate(tmp_path):
    session = _session()
    active = _artifact(session, version="active-v1", active=True)
    snapshot = _snapshot(session, active)
    run = _run(session, active, snapshot)

    result = CandidateTrainingService(
        session=session,
        settings=_settings(),
        model_dir=tmp_path,
        trainer=_fake_trainer(tmp_path),
        evaluator=_FakeEvaluator(),
    ).train_candidate(retraining_run_id=run.id, daily_demand=_daily_demand())

    artifact = result.candidate_artifact
    assert artifact.artifact_checksum == ModelService.checksum_file(artifact.artifact_uri)
    assert artifact.feature_schema == FEATURE_COLUMNS
    assert artifact.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert artifact.feature_schema_checksum == feature_schema_checksum(FEATURE_COLUMNS)


def test_candidate_evaluation_temporal_split_metrics_benchmarks_and_eligibility():
    session = _session()
    candidate = _artifact(session, version="candidate-v1", active=False)
    active = _artifact(session, version="active-v1", active=True)
    models = {candidate.id: _ConstantModel(10.0), active.id: _ConstantModel(11.0)}

    service = CandidateEvaluationService(
        session=session,
        settings=_settings(),
        model_loader=lambda artifact: models[artifact.id],
    )
    result = service.evaluate(
        candidate_artifact=candidate,
        active_artifact=active,
        daily_demand=_daily_demand(sku_count=4, days=90, value=10.0),
        horizon_days=30,
    )

    assert result.temporal_split["method"] == "per_sku_last_n_days_holdout"
    assert result.test_points == 120
    assert result.candidate_metrics["wape"] == 0.0
    assert result.active_metrics["wape"] == 0.1
    assert result.relative_wape_improvement == 1.0
    assert result.promotion_eligible is True
    assert "croston_sba" in result.benchmark_metrics
    assert "moving_avg_7" in result.benchmark_metrics
    assert "seasonal_naive_7" in result.benchmark_metrics
    assert session.scalar(select(ForecastEvaluation).where(ForecastEvaluation.model_name == "candidate_lightgbm")) is not None
    assert active.lifecycle_status == "active"
    assert active.is_active is True


def test_candidate_evaluation_blocks_incompatible_horizon():
    session = _session()
    service = CandidateEvaluationService(session=session, settings=_settings(inventory_days=7))

    eligible, reason = service._promotion_eligibility(
        {"wape": 0.8, "bias": 0.0, "n_test_points": 120},
        {"wape": 1.0, "bias": 0.0, "n_test_points": 120},
        0.2,
        30,
    )

    assert eligible is False
    assert reason == "incompatible_horizon:30!=7"

def test_candidate_evaluation_blocks_small_improvement_insufficient_points_worse_bias_and_worse_candidate():
    session = _session()
    candidate = _artifact(session, version="candidate-v1", active=False)
    active = _artifact(session, version="active-v1", active=True)
    service = CandidateEvaluationService(session=session, settings=_settings())

    assert service._promotion_eligibility(
        {"wape": 0.96, "bias": 0.0, "n_test_points": 120},
        {"wape": 1.0, "bias": 0.0, "n_test_points": 120},
        0.04,
        30,
    )[0] is False
    assert service._promotion_eligibility(
        {"wape": 0.8, "bias": 0.0, "n_test_points": 30},
        {"wape": 1.0, "bias": 0.0, "n_test_points": 30},
        0.2,
        30,
    )[0] is False
    assert service._promotion_eligibility(
        {"wape": 0.8, "bias": 10.0, "n_test_points": 120},
        {"wape": 1.0, "bias": 1.0, "n_test_points": 120},
        0.2,
        30,
    )[0] is False
    assert service._promotion_eligibility(
        {"wape": 1.2, "bias": 0.0, "n_test_points": 120},
        {"wape": 1.0, "bias": 0.0, "n_test_points": 120},
        -0.2,
        30,
    )[0] is False
    assert candidate.lifecycle_status == "candidate"
    assert active.lifecycle_status == "active"



def test_training_pipeline_does_not_replace_global_model_service_cache(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from services import model_service as model_service_module

    global_service = model_service_module.get_model_service(model_dir=str(tmp_path / "active"))
    sentinel = _ConstantModel(123.0)
    global_service._model_cache["lightgbm_demand_forecast"] = sentinel
    global_service._model_metadata["lightgbm_demand_forecast"] = {"model_name": "lightgbm_demand_forecast"}

    module_name = "supplysync_train_model_test"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    train_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = train_module
    try:
        spec.loader.exec_module(train_module)
    finally:
        sys.modules.pop(module_name, None)

    daily = _daily_demand(sku_count=2, days=90, value=10.0)

    monkeypatch.setattr(train_module, "load_and_clean_retail_data", lambda **kwargs: daily)
    monkeypatch.setattr(train_module, "aggregate_daily_demand", lambda raw_df, output_path=None: daily)
    monkeypatch.setattr(train_module, "get_top_skus", lambda daily_df, min_days=60, top_n=20: ["SKU-0", "SKU-1"])
    monkeypatch.setattr(
        train_module,
        "load_sku_demand",
        lambda sku, parquet_path=None: daily[daily["StockCode"] == sku][["date", "demand"]].copy(),
    )
    monkeypatch.setattr(train_module, "LGBMRegressor", _TinyRegressor)

    result = train_module.train_lightgbm_demand_model(
        model_dir=tmp_path / "candidate",
        artifact_file="candidate.pkl",
        metadata_file="candidate_metadata.json",
        verbose=False,
    )

    assert result.model_path.name == "candidate.pkl"
    assert result.metadata_path.name == "candidate_metadata.json"
    assert model_service_module._model_service is global_service
    assert global_service._model_cache["lightgbm_demand_forecast"] is sentinel


def test_candidate_training_does_not_affect_analyze(tmp_path):
    import main as backend_main
    from fastapi.testclient import TestClient

    session = _session()
    active = _artifact(session, version="active-v1", active=True)
    snapshot = _snapshot(session, active)
    run = _run(session, active, snapshot)
    CandidateTrainingService(
        session=session,
        settings=_settings(),
        model_dir=tmp_path,
        trainer=_fake_trainer(tmp_path),
        evaluator=_FakeEvaluator(),
    ).train_candidate(retraining_run_id=run.id, daily_demand=_daily_demand())

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"sku": "SKU-PHASE-F", "current_stock": 10, "demand_history": [10, 11, 9, 10, 12, 8, 10] * 4},
        )

    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert "recommended_order" in response.json()
