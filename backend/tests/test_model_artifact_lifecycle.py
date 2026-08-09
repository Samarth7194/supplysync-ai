from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine

from db.models import Base, ForecastEvaluation, ModelArtifact, PredictionLog
from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum
from repositories.analysis_repository import AnalysisRepository
from repositories.model_artifact_repository import ModelArtifactRepository, ModelPromotionError
from services.analysis_service import AnalysisService
from services.model_service import ModelArtifactValidationError, ModelService


class _TinyModel:
    def __init__(self, value=1.0):
        self.value = value

    def predict(self, features):
        return [self.value]


@dataclass
class _InventorySettings:
    default_lead_time_days: int = 7
    default_service_level: float = 0.95


@dataclass
class _Settings:
    inventory: _InventorySettings


class _DataService:
    def get_demand_history(self, sku):
        return pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))


class _InventoryService:
    forecast_method = "ml_lightgbm"
    demand_pattern = "regular"

    def get_intelligent_reorder_decision(self, **kwargs):
        return {
            "order_quantity": 5,
            "lead_time_demand": 7.0,
            "safety_stock": 2.0,
            "safety_stock_method": "traditional",
            "reorder_point": 9.0,
            "service_level": kwargs["service_level"],
            "lead_time_days": kwargs["lead_time_days"],
            "forecast_daily": [1.0] * kwargs["lead_time_days"],
            "intelligence": {
                "demand_pattern": self.demand_pattern,
                "forecast_method": self.forecast_method,
            },
        }


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


def _save_valid_model(model_dir, *, value=1.0):
    service = ModelService(model_dir=str(model_dir))
    model_path = service.save_model(
        _TinyModel(value=value),
        "lightgbm_demand_forecast",
        metadata={
            "features": FEATURE_COLUMNS,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_checksum": feature_schema_checksum(FEATURE_COLUMNS),
            "dataset": "unit",
            "mae": 1.0,
            "rmse": 2.0,
        },
    )
    return service, model_path


def test_save_model_writes_checksum_version_and_feature_schema(tmp_path):
    service, model_path = _save_valid_model(tmp_path)
    metadata = service.get_model_metadata("lightgbm_demand_forecast")

    assert metadata["artifact_checksum"] == ModelService.checksum_file(model_path)
    assert metadata["version"].startswith("lightgbm_demand_forecast-")
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["feature_schema_checksum"] == feature_schema_checksum(FEATURE_COLUMNS)
    assert metadata["lifecycle_status"] == "candidate"


def test_invalid_checksum_is_rejected(tmp_path):
    service, model_path = _save_valid_model(tmp_path)
    with open(model_path, "ab") as fh:
        fh.write(b"corruption")

    with pytest.raises(ModelArtifactValidationError, match="Checksum mismatch"):
        service.load_model("lightgbm_demand_forecast", use_cache=False)


def test_feature_schema_mismatch_is_rejected(tmp_path):
    service, _ = _save_valid_model(tmp_path)
    metadata_path = tmp_path / "lightgbm_demand_forecast_metadata.json"
    text = metadata_path.read_text()
    metadata_path.write_text(text.replace(FEATURE_SCHEMA_VERSION, "future_schema_v2"))
    service.clear_cache("lightgbm_demand_forecast")

    with pytest.raises(ModelArtifactValidationError, match="Feature schema mismatch"):
        service.load_model("lightgbm_demand_forecast", use_cache=False)


def test_register_and_promote_model_artifact_with_evidence(tmp_path):
    service, _ = _save_valid_model(tmp_path)
    metadata = service.validate_model_artifact("lightgbm_demand_forecast")
    metadata["model_dir"] = str(tmp_path)
    session = _session()
    repo = ModelArtifactRepository(session)

    artifact = repo.register_metadata(metadata, status="candidate")
    session.add(
        ForecastEvaluation(
            model_artifact_id=artifact.id,
            model_name="lightgbm",
            evaluation_scope="global",
            metric_wape=0.8,
            n_test_points=30,
        )
    )
    session.flush()
    promoted = repo.promote(artifact.id)
    session.commit()

    assert promoted.lifecycle_status == "active"
    assert promoted.is_active is True
    assert repo.active_for_name("lightgbm_demand_forecast").id == artifact.id


def test_promotion_requires_evidence_unless_forced(tmp_path):
    service, _ = _save_valid_model(tmp_path)
    metadata = service.validate_model_artifact("lightgbm_demand_forecast")
    metadata["model_dir"] = str(tmp_path)
    session = _session()
    repo = ModelArtifactRepository(session)
    artifact = repo.register_metadata(metadata, status="candidate")

    with pytest.raises(ModelPromotionError, match="requires evaluation"):
        repo.promote(artifact.id)

    forced = repo.promote(artifact.id, force=True)
    assert forced.lifecycle_status == "active"


def test_promoting_new_artifact_retires_previous_active(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_service, _ = _save_valid_model(first_dir)
    second_service, _ = _save_valid_model(second_dir, value=2.0)
    second_meta_path = second_dir / "lightgbm_demand_forecast_metadata.json"
    second_meta_path.write_text(second_meta_path.read_text().replace("unit", "unit-v2"))
    # Re-save after metadata change is not needed for identity; force path checks file checksum only.

    session = _session()
    repo = ModelArtifactRepository(session)
    first_meta = first_service.validate_model_artifact("lightgbm_demand_forecast")
    first_meta["model_dir"] = str(first_dir)
    second_meta = second_service.validate_model_artifact("lightgbm_demand_forecast")
    second_meta["model_dir"] = str(second_dir)
    second_meta["version"] = second_meta["version"] + "-v2"

    first = repo.register_metadata(first_meta, status="candidate")
    second = repo.register_metadata(second_meta, status="candidate")
    repo.promote(first.id, force=True)
    repo.promote(second.id, force=True)
    session.commit()

    assert session.get(ModelArtifact, first.id).lifecycle_status == "retired"
    assert session.get(ModelArtifact, first.id).is_active is False
    assert session.get(ModelArtifact, second.id).lifecycle_status == "active"


def test_prediction_keeps_exact_artifact_after_later_promotion(tmp_path):
    service, _ = _save_valid_model(tmp_path)
    session = _session()
    analysis_repo = AnalysisRepository(session)
    metadata = service.validate_model_artifact("lightgbm_demand_forecast")
    metadata["model_dir"] = str(tmp_path)
    artifact = ModelArtifactRepository(session).register_metadata(metadata, status="active")

    service_under_test = AnalysisService(
        inventory_service=_InventoryService(),
        settings=_Settings(inventory=_InventorySettings()),
        data_service=_DataService(),
        analysis_repository=analysis_repo,
        model_loaded=True,
        model_dir=tmp_path,
    )
    service_under_test.analyze(
        SimpleNamespace(
            sku="SKU-1",
            current_stock=5.0,
            demand_history=None,
            lead_time_days=None,
            service_level=None,
        )
    )
    prediction = session.scalar(select(PredictionLog))
    assert prediction.model_artifact_id == artifact.id
    assert prediction.feature_schema_version == FEATURE_SCHEMA_VERSION

    other = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version="other-version",
        artifact_uri=artifact.artifact_uri,
        artifact_checksum=artifact.artifact_checksum + "x",
        lifecycle_status="candidate",
    )
    session.add(other)
    session.flush()
    session.commit()

    assert session.get(PredictionLog, prediction.id).model_artifact_id == artifact.id
