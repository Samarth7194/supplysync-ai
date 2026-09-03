from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle

import pytest
from sqlalchemy import select

from db.models import ModelArtifact, ModelPromotionEvent, RetrainingRun, ModelMonitoringSnapshot
from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum
from services.model_promotion_service import ModelPromotionService, ModelPromotionServiceError
from services.model_service import ModelService
from services.runtime_model_service import load_runtime_model, resolve_artifact_path
from tests.test_retraining_decision_service import _session


@dataclass
class _ForecastingSettings:
    model_path: str


@dataclass
class _Settings:
    forecasting: _ForecastingSettings


class _TinyRuntimeModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features):
        return [self.value] * len(features)


def _settings(model_dir: Path) -> _Settings:
    return _Settings(_ForecastingSettings(str(model_dir)))


def _write_model(path: Path, value: float = 10.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(_TinyRuntimeModel(value), fh)
    return ModelService.checksum_file(path)


def _artifact(session, path: Path, *, version: str, status: str, value: float | None = None, checksum: str | None = None):
    model_value = float(value if value is not None else len(version) + len(path.name))
    actual_checksum = checksum if checksum is not None else _write_model(path, model_value)
    row = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version=version,
        artifact_checksum=actual_checksum,
        checksum_algorithm="sha256",
        artifact_uri=str(path),
        metadata_uri=None,
        feature_schema=FEATURE_COLUMNS,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_checksum=feature_schema_checksum(FEATURE_COLUMNS),
        training_metrics={"candidate_wape": 0.8, "active_wape": 1.0},
        training_metadata={
            "candidate_evaluation": {
                "horizon_days": 7,
                "test_points": 120,
                "promotion_eligible": True,
                "candidate_metrics": {"wape": 0.8, "n_test_points": 120},
                "active_metrics": {"wape": 1.0, "n_test_points": 120},
            }
        },
        lifecycle_status=status,
        is_active=status == "active",
        activated_at=datetime.now(timezone.utc) if status == "active" else None,
    )
    session.add(row)
    session.flush()
    return row


def _snapshot(session, active, *, key_suffix: str | int | None = None):
    row = ModelMonitoringSnapshot(
        generated_at=datetime.now(timezone.utc),
        model_artifact_id=active.id,
        model_name=active.model_name,
        model_version=active.version,
        window_type="latest_evaluations",
        window_size=30,
        evaluation_count=30,
        baseline_provenance="artifact_metadata",
        degradation_reason="persistent_wape_degradation",
        degradation_message="test",
        consecutive_degradation_count=2,
        status="degraded",
        evidence_key=f"phase-g-{active.id}-{key_suffix or active.version}",
    )
    session.add(row)
    session.flush()
    return row


def _eligible_run(session, active, candidate, *, recommended=True, status="completed"):
    snapshot = _snapshot(session, active, key_suffix=candidate.id)
    row = RetrainingRun(
        triggered_at=datetime.now(timezone.utc),
        trigger_reason="retraining_recommended",
        status=status,
        baseline_model_artifact_id=active.id,
        source_monitoring_snapshot_id=snapshot.id,
        new_evaluated_forecast_days=120,
        candidate_model_artifact_id=candidate.id,
        promotion_recommended=recommended,
        evidence_key=f"phase-g-run-{candidate.id}",
    )
    session.add(row)
    session.flush()
    return row


def _active_count(session):
    return len(session.scalars(select(ModelArtifact).where(ModelArtifact.is_active.is_(True))).all())


def test_eligible_candidate_promotion_succeeds_retires_previous_and_records_event(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "active.pkl", version="active-v1", status="active", value=1)
    candidate = _artifact(session, tmp_path / "candidate.pkl", version="candidate-v2", status="candidate", value=2)
    run = _eligible_run(session, active, candidate)
    handed = []

    result = ModelPromotionService(
        session=session,
        settings=_settings(tmp_path),
        runtime_handoff=lambda loaded: handed.append(loaded.model_version),
    ).promote_candidate(candidate.id, initiated_by="test", reason="approved")

    assert result.changed is True
    assert result.loaded_model.model_version == "candidate-v2"
    assert handed == ["candidate-v2"]
    assert candidate.lifecycle_status == "active"
    assert candidate.is_active is True
    assert active.lifecycle_status == "retired"
    assert active.is_active is False
    assert _active_count(session) == 1
    event = session.scalar(select(ModelPromotionEvent))
    assert event.event_type == "promotion"
    assert event.promoted_model_artifact_id == candidate.id
    assert event.previous_model_artifact_id == active.id
    assert event.retraining_run_id == run.id
    assert event.outcome == "succeeded"


def test_ineligible_candidate_cannot_promote(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "active.pkl", version="active-v1", status="active")
    candidate = _artifact(session, tmp_path / "candidate.pkl", version="candidate-v2", status="candidate")
    _eligible_run(session, active, candidate, recommended=False)

    with pytest.raises(ModelPromotionServiceError, match="eligible candidate-evaluation"):
        ModelPromotionService(session=session, settings=_settings(tmp_path)).promote_candidate(candidate.id)
    assert active.is_active is True
    assert candidate.is_active is False


def test_non_candidate_and_candidate_without_evidence_cannot_promote(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "active.pkl", version="active-v1", status="active")
    retired = _artifact(session, tmp_path / "retired.pkl", version="retired-v0", status="retired")
    candidate = _artifact(session, tmp_path / "candidate.pkl", version="candidate-v2", status="candidate")

    service = ModelPromotionService(session=session, settings=_settings(tmp_path))
    with pytest.raises(ModelPromotionServiceError, match="Only inactive candidate"):
        service.promote_candidate(retired.id)
    with pytest.raises(ModelPromotionServiceError, match="eligible candidate-evaluation"):
        service.promote_candidate(candidate.id)
    assert active.is_active is True


def test_invalid_checksum_schema_and_missing_artifact_block_promotion_without_retiring_active(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "active.pkl", version="active-v1", status="active")
    bad_checksum = _artifact(session, tmp_path / "bad-checksum.pkl", version="bad-checksum", status="candidate", checksum="0" * 64)
    bad_schema = _artifact(session, tmp_path / "bad-schema.pkl", version="bad-schema", status="candidate")
    bad_schema.feature_schema_version = "old_schema"
    missing = _artifact(session, tmp_path / "missing.pkl", version="missing", status="candidate")
    Path(missing.artifact_uri).unlink()
    for candidate in (bad_checksum, bad_schema, missing):
        _eligible_run(session, active, candidate)

    service = ModelPromotionService(session=session, settings=_settings(tmp_path))
    for candidate in (bad_checksum, bad_schema, missing):
        with pytest.raises(ModelPromotionServiceError):
            service.promote_candidate(candidate.id)
        assert active.is_active is True
        assert active.lifecycle_status == "active"
        assert candidate.is_active is False


def test_runtime_handoff_failure_restores_database_state(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "active.pkl", version="active-v1", status="active")
    candidate = _artifact(session, tmp_path / "candidate.pkl", version="candidate-v2", status="candidate")
    _eligible_run(session, active, candidate)

    def fail(_loaded):
        raise RuntimeError("boom")

    with pytest.raises(ModelPromotionServiceError, match="Runtime handoff failed"):
        ModelPromotionService(session=session, settings=_settings(tmp_path), runtime_handoff=fail).promote_candidate(candidate.id)

    assert active.is_active is True
    assert active.lifecycle_status == "active"
    assert candidate.is_active is False
    assert candidate.lifecycle_status == "candidate"


def test_duplicate_promotion_is_safe_noop(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "active.pkl", version="active-v1", status="active")

    result = ModelPromotionService(session=session, settings=_settings(tmp_path)).promote_candidate(active.id)

    assert result.changed is False
    assert result.event is None
    assert active.is_active is True
    assert _active_count(session) == 1


def test_rollback_restores_previous_valid_artifact_and_does_not_delete_files(tmp_path):
    session = _session()
    current = _artifact(session, tmp_path / "current.pkl", version="current-v2", status="active", value=2)
    previous = _artifact(session, tmp_path / "previous.pkl", version="previous-v1", status="retired", value=1)
    previous_path = Path(previous.artifact_uri)

    result = ModelPromotionService(session=session, settings=_settings(tmp_path)).rollback_to_artifact(previous.id, reason="bad deploy")

    assert result.changed is True
    assert previous.is_active is True
    assert previous.lifecycle_status == "active"
    assert current.is_active is False
    assert current.lifecycle_status == "retired"
    assert previous_path.exists()
    event = session.scalar(select(ModelPromotionEvent).where(ModelPromotionEvent.event_type == "rollback"))
    assert event.promoted_model_artifact_id == previous.id
    assert event.previous_model_artifact_id == current.id
    assert event.outcome == "succeeded"


def test_rollback_validates_target_before_state_change(tmp_path):
    session = _session()
    current = _artifact(session, tmp_path / "current.pkl", version="current-v2", status="active", value=2)
    previous = _artifact(session, tmp_path / "previous.pkl", version="previous-v1", status="retired", checksum="1" * 64)

    with pytest.raises(ModelPromotionServiceError):
        ModelPromotionService(session=session, settings=_settings(tmp_path)).rollback_to_artifact(previous.id)

    assert current.is_active is True
    assert previous.is_active is False


def test_rollback_handoff_failure_restores_original_target_state(tmp_path):
    session = _session()
    current = _artifact(session, tmp_path / "current.pkl", version="current-v2", status="active", value=2)
    previous = _artifact(session, tmp_path / "previous.pkl", version="previous-v1", status="retired", value=1)

    def fail(_loaded):
        raise RuntimeError("handoff failed")

    with pytest.raises(ModelPromotionServiceError, match="Runtime handoff failed"):
        ModelPromotionService(session=session, settings=_settings(tmp_path), runtime_handoff=fail).rollback_to_artifact(
            previous.id
        )

    assert current.is_active is True
    assert current.lifecycle_status == "active"
    assert previous.is_active is False
    assert previous.lifecycle_status == "retired"
    assert _active_count(session) == 1


def test_runtime_loader_prefers_valid_db_active_artifact(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "db-active.pkl", version="db-active", status="active", value=4)

    loaded = load_runtime_model(settings=_settings(tmp_path), session_factory=lambda: session)

    assert loaded.model_version == "db-active"
    assert loaded.status["source"] == "db_active_artifact"
    assert loaded.status["artifact_id"] == active.id


def test_runtime_loader_falls_back_when_db_active_unavailable(tmp_path):
    session = _session()
    active = _artifact(session, tmp_path / "missing-active.pkl", version="db-active", status="active", value=4)
    Path(active.artifact_uri).unlink()
    service = ModelService(model_dir=str(tmp_path))
    service.save_model(
        _TinyRuntimeModel(8),
        "lightgbm_demand_forecast",
        metadata={
            "features": FEATURE_COLUMNS,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_checksum": feature_schema_checksum(FEATURE_COLUMNS),
            "lifecycle_status": "active",
        },
    )

    loaded = load_runtime_model(settings=_settings(tmp_path), session_factory=lambda: session)

    assert loaded.model is not None
    assert loaded.status["source"] == "configured_runtime_artifact"
    assert loaded.status["artifact_id"] is None
    assert "db_active_error" in loaded.status


def test_artifact_uri_resolution_uses_portable_filename_fallback(tmp_path):
    checksum = _write_model(tmp_path / "candidate.pkl")
    windows_uri = r"C:\Users\dev\project\backend\saved_models\candidate.pkl"

    resolved = resolve_artifact_path(windows_uri, model_dir=tmp_path)

    assert resolved == tmp_path / "candidate.pkl"
    assert ModelService.checksum_file(resolved) == checksum


def test_rollback_activates_initial_baseline_when_no_active_artifact_exists(tmp_path):
    """Covers the zero-active-artifact bootstrap edge case: a single valid
    'candidate' artifact with no active row anywhere for the model. This is
    the exact production scenario where a model was deployed by placing a
    file on disk without ever going through DB registration/promotion."""
    session = _session()
    baseline = _artifact(session, tmp_path / "baseline.pkl", version="baseline-v1", status="candidate", value=1)
    baseline_path = Path(baseline.artifact_uri)

    assert _active_count(session) == 0

    result = ModelPromotionService(
        session=session,
        settings=_settings(tmp_path),
        runtime_handoff=lambda loaded: None,
    ).rollback_to_artifact(baseline.id, initiated_by="test", reason="initial baseline activation")

    assert result.changed is True
    assert baseline.lifecycle_status == "active"
    assert baseline.is_active is True
    assert baseline.activated_at is not None
    assert baseline.retired_at is None
    assert baseline_path.exists()
    assert _active_count(session) == 1

    events = session.scalars(select(ModelPromotionEvent)).all()
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "rollback"
    assert event.promoted_model_artifact_id == baseline.id
    assert event.previous_model_artifact_id is None
    assert event.outcome == "succeeded"

    # Idempotency: re-running against the now-active same artifact must be a
    # safe no-op — no duplicate event, no lifecycle churn.
    second_result = ModelPromotionService(
        session=session,
        settings=_settings(tmp_path),
        runtime_handoff=lambda loaded: None,
    ).rollback_to_artifact(baseline.id, initiated_by="test", reason="re-run")

    assert second_result.changed is False
    assert second_result.event is None
    assert baseline.is_active is True
    assert baseline.lifecycle_status == "active"
    assert _active_count(session) == 1
    assert len(session.scalars(select(ModelPromotionEvent)).all()) == 1


def test_rollback_bootstrap_still_enforces_preflight_when_checksum_invalid(tmp_path):
    """The zero-active-artifact bootstrap path must not skip the normal
    checksum/schema/deserialization preflight, and must not weaken the
    separate Phase-F promotion evidence gate used by promote_candidate."""
    session = _session()
    tampered = _artifact(
        session,
        tmp_path / "tampered.pkl",
        version="tampered-v1",
        status="candidate",
        checksum="0" * 64,
    )

    with pytest.raises(ModelPromotionServiceError):
        ModelPromotionService(session=session, settings=_settings(tmp_path)).rollback_to_artifact(tampered.id)

    assert tampered.is_active is False
    assert tampered.lifecycle_status == "candidate"
    assert _active_count(session) == 0

    # promote_candidate's evidence gate is untouched by the bootstrap path:
    # a bare candidate with no RetrainingRun/candidate_evaluation is still
    # rejected by promotion, exactly as before.
    with pytest.raises(ModelPromotionServiceError, match="eligible candidate-evaluation"):
        ModelPromotionService(session=session, settings=_settings(tmp_path)).promote_candidate(tampered.id)







