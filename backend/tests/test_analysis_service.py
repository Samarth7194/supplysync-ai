from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import AnalysisRun, Base, InventoryPolicy, ModelArtifact, PredictionLog, Sku
from repositories.analysis_repository import AnalysisRepository
from services.analysis_service import AnalysisService


@dataclass
class _InventorySettings:
    default_lead_time_days: int = 7
    default_service_level: float = 0.95


@dataclass
class _Settings:
    inventory: _InventorySettings
    forecasting: object = field(
        default_factory=lambda: SimpleNamespace(
            evidence_routing_enabled=False,
            routing_primary_metric="wape",
            routing_min_evaluation_points=30,
            routing_min_relative_improvement=0.05,
            routing_evidence_lookback_days=365,
            uncertainty_min_residual_observations=30,
            uncertainty_residual_lookback_days=365,
        )
    )


class _DataService:
    def __init__(self, series_by_sku):
        self.series_by_sku = series_by_sku

    def get_demand_history(self, sku):
        return self.series_by_sku.get(sku, pd.Series(dtype=float))


class _InventoryService:
    forecast_method = "simple_average"
    demand_pattern = "regular"
    routing = None

    def get_intelligent_reorder_decision(
        self,
        *,
        sku,
        current_stock,
        demand_history,
        lead_time_days,
        service_level,
        routing_service=None,
        supplier_constraints=None,
        policy_source=None,
        uncertainty_service=None,
    ):
        forecast = [10.0] * lead_time_days
        intelligence = {
            "demand_pattern": self.demand_pattern,
            "forecast_method": self.forecast_method,
        }
        if self.routing is not None:
            intelligence["routing"] = self.routing
        return {
            "order_quantity": 42,
            "lead_time_demand": sum(forecast),
            "safety_stock": 12.5,
            "safety_stock_method": "traditional",
            "reorder_point": sum(forecast) + 12.5,
            "service_level": service_level,
            "lead_time_days": lead_time_days,
            "forecast_daily": forecast,
            "intelligence": intelligence,
        }


class _ConstrainedInventoryService(_InventoryService):
    def __init__(self, *, original_quantity: int, final_quantity: int, max_order_quantity: int | None = None):
        self.original_quantity = original_quantity
        self.final_quantity = final_quantity
        self.max_order_quantity = max_order_quantity

    def get_intelligent_reorder_decision(
        self,
        *,
        sku,
        current_stock,
        demand_history,
        lead_time_days,
        service_level,
        routing_service=None,
        supplier_constraints=None,
        policy_source=None,
        uncertainty_service=None,
    ):
        forecast = [10.0] * lead_time_days
        lead_time_demand = sum(forecast)
        reorder_point = float(current_stock + self.original_quantity)
        constraints_applied = []
        if self.final_quantity != self.original_quantity and self.max_order_quantity is not None:
            constraints_applied.append(f"Capped at maximum {self.max_order_quantity}")
        return {
            "order_quantity": self.final_quantity,
            "lead_time_demand": lead_time_demand,
            "safety_stock": round(reorder_point - lead_time_demand, 2),
            "safety_stock_method": "traditional",
            "reorder_point": reorder_point,
            "service_level": service_level,
            "lead_time_days": lead_time_days,
            "forecast_daily": forecast,
            "business_constraints": {
                "moq": 10,
                "order_multiple": 5,
                "max_order_quantity": self.max_order_quantity,
                "constraints_applied": constraints_applied,
                "original_quantity": self.original_quantity,
                "final_quantity": self.final_quantity,
            },
            "intelligence": {
                "demand_pattern": "regular",
                "forecast_method": "simple_average",
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


def _request(**overrides):
    values = {
        "sku": "SKU-1",
        "current_stock": 5.0,
        "demand_history": None,
        "lead_time_days": None,
        "service_level": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _service(session, data_service, *, inventory_service=None, model_loaded=False, model_dir=None):
    return AnalysisService(
        inventory_service=inventory_service or _InventoryService(),
        settings=_Settings(inventory=_InventorySettings()),
        data_service=data_service,
        analysis_repository=AnalysisRepository(session),
        model_loaded=model_loaded,
        model_dir=model_dir,
    )


def test_analysis_service_preserves_response_contract_and_persists_prediction():
    session = _session()
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data_service = _DataService({"SKU-1": pd.Series([8.0, 9.0, 10.0] * 10, index=dates)})
    service = _service(session, data_service)

    response = service.analyze(_request())
    session.commit()

    assert set(response) == {
        "sku",
        "risk",
        "risk_color",
        "forecast",
        "current_stock",
        "recommended_order",
        "action",
        "demand_pattern",
        "forecast_method",
        "demand_source",
        "forecast_source",
        "decision",
        "model_info",
        "explanation",
    }
    assert response["sku"] == "SKU-1"
    assert response["demand_source"] == "historical"
    assert response["forecast_source"] == "rule_based_estimate"
    assert response["forecast"]["daily"] == [10.0] * 7
    assert response["forecast"]["full_horizon_daily"] == [10.0] * 7
    assert response["forecast"]["horizon_days"] == 7
    assert response["decision"]["constraints"]["raw_order_quantity"] == 42
    assert response["decision"]["constraints"]["final_order_quantity"] == 42
    assert response["decision"]["constraints"]["constrained"] is False
    assert response["decision"]["uncertainty"]["source"] == "historical_demand_std"

    analysis = session.scalar(select(AnalysisRun))
    prediction = session.scalar(select(PredictionLog))
    assert analysis is not None
    assert prediction is not None
    assert prediction.analysis_run_id == analysis.id
    assert prediction.sku_code == "SKU-1"
    assert prediction.target_start_date.isoformat() == "2024-01-31"
    assert prediction.target_end_date.isoformat() == "2024-02-06"
    assert prediction.input_history_length == 30
    assert prediction.forecast_horizon_days == 7
    assert prediction.recommended_order_quantity == 42


def test_analysis_service_exposes_uncapped_constraint_metadata():
    session = _session()
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(
        session,
        data_service,
        inventory_service=_ConstrainedInventoryService(original_quantity=420, final_quantity=420),
    )

    response = service.analyze(_request(current_stock=100, lead_time_days=7))

    constraints = response["decision"]["constraints"]
    assert response["recommended_order"] == 420
    assert constraints["raw_order_quantity"] == 420
    assert constraints["final_order_quantity"] == 420
    assert constraints["constraints_applied"] == []
    assert constraints["constrained"] is False
    assert constraints["max_order_cap_applied"] is False
    assert response["forecast"]["horizon_days"] == 7
    assert len(response["forecast"]["daily"]) == 7
    assert len(response["forecast"]["full_horizon_daily"]) == 7


def test_analysis_service_explains_max_order_cap_without_claiming_gap_closed():
    session = _session()
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(
        session,
        data_service,
        inventory_service=_ConstrainedInventoryService(
            original_quantity=5649,
            final_quantity=1000,
            max_order_quantity=1000,
        ),
    )

    response = service.analyze(_request(current_stock=64, lead_time_days=19))

    constraints = response["decision"]["constraints"]
    why = response["decision"]["why"].lower()
    assert response["recommended_order"] == 1000
    assert response["decision"]["lead_time_days"] == 19
    assert constraints["raw_order_quantity"] == 5649
    assert constraints["final_order_quantity"] == 1000
    assert constraints["max_order_quantity"] == 1000
    assert constraints["constraints_applied"] == ["Capped at maximum 1000"]
    assert constraints["max_order_cap_applied"] is True
    assert constraints["remaining_gap_after_order"] == 4649
    assert "uncapped requirement of 5649 units" in why
    assert "capped at 1000 units" in why
    assert "brings the position back above the reorder point" not in why
    assert len(response["forecast"]["daily"]) == 7
    assert len(response["forecast"]["full_horizon_daily"]) == 19
    assert response["forecast"]["horizon_days"] == 19


def test_analysis_service_uses_request_history_before_dataset():
    session = _session()
    data_service = _DataService({"SKU-1": pd.Series([99.0] * 30)})
    service = _service(session, data_service)

    response = service.analyze(_request(demand_history=[1, 2, 3, 4, 5, 6, 7]))

    assert response["demand_source"] == "request"


def test_recent_analyses_reads_sqlalchemy_shape():
    session = _session()
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30)})
    service = _service(session, data_service)
    service.analyze(_request(current_stock=5))
    service.analyze(_request(current_stock=9))
    session.commit()

    body = service.recent_analyses(limit=10)

    assert body["available"] is True
    assert body["source"] == "sqlalchemy"
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["sku"] == "SKU-1"
    assert body["items"][0]["recommended_order"] == 42


def test_prediction_links_existing_sku_without_creating_fake_sku():
    session = _session()
    session.add(Sku(sku_code="SKU-1", name="Known SKU"))
    session.commit()
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(session, data_service)

    service.analyze(_request())
    session.commit()

    analysis = session.scalar(select(AnalysisRun))
    prediction = session.scalar(select(PredictionLog))
    assert analysis.sku_id is not None
    assert prediction.sku_id == analysis.sku_id


def test_prediction_leaves_sku_id_null_when_sku_not_persisted():
    session = _session()
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(session, data_service)

    service.analyze(_request())
    session.commit()

    assert session.scalar(select(Sku)) is None
    assert session.scalar(select(PredictionLog)).sku_id is None


def test_lightgbm_prediction_links_model_artifact(tmp_path):
    session = _session()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "lightgbm_demand_forecast.pkl").write_bytes(b"fake-model-bytes")
    (model_dir / "lightgbm_demand_forecast_metadata.json").write_text(
        '{"saved_at":"2024-01-01T00:00:00+00:00","features":["lag_1"],"dataset":"unit"}'
    )
    inventory = _InventoryService()
    inventory.forecast_method = "ml_lightgbm"
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(
        session,
        data_service,
        inventory_service=inventory,
        model_loaded=True,
        model_dir=model_dir,
    )

    service.analyze(_request())
    session.commit()

    prediction = session.scalar(select(PredictionLog))
    artifact = session.scalar(select(ModelArtifact))
    assert artifact is not None
    assert prediction.model_artifact_id == artifact.id
    assert prediction.model_version.startswith("sha256:")


def test_statistical_prediction_does_not_require_model_artifact():
    session = _session()
    inventory = _InventoryService()
    inventory.forecast_method = "croston"
    inventory.demand_pattern = "intermittent"
    data_service = _DataService({"SKU-1": pd.Series([0, 0, 5.0] * 10, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(session, data_service, inventory_service=inventory)

    service.analyze(_request())
    session.commit()

    prediction = session.scalar(select(PredictionLog))
    assert prediction.model_artifact_id is None
    assert prediction.model_version == "croston_sba_alpha_0.1"


def test_routing_reason_is_persisted_when_available():
    session = _session()
    inventory = _InventoryService()
    inventory.routing = {
        "selected_method": "croston",
        "default_method": "ml_lightgbm",
        "selection_source": "logged",
        "reason": "Croston had sufficient logged evidence.",
    }
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(session, data_service, inventory_service=inventory)

    service.analyze(_request())
    session.commit()

    analysis = session.scalar(select(AnalysisRun))
    prediction = session.scalar(select(PredictionLog))
    assert analysis.routing_reason == "Croston had sufficient logged evidence."
    assert prediction.routing_reason == "Croston had sufficient logged evidence."


def test_persisted_inventory_policy_overrides_pattern_constraints_and_defaults():
    session = _session()
    sku = Sku(sku_code="SKU-1", name="Known SKU")
    session.add(sku)
    session.flush()
    session.add(
        InventoryPolicy(
            sku_id=sku.id,
            lead_time_days=9,
            service_level=0.90,
            moq=25,
            order_multiple=10,
            max_order_quantity=250,
            is_active=True,
        )
    )
    session.flush()
    data_service = _DataService({"SKU-1": pd.Series([8.0] * 30, index=pd.date_range("2024-01-01", periods=30))})
    service = _service(
        session,
        data_service,
        inventory_service=_ConstrainedInventoryService(
            original_quantity=420,
            final_quantity=250,
            max_order_quantity=250,
        ),
    )

    response = service.analyze(_request(current_stock=100))

    assert response["decision"]["lead_time_days"] == 9
    assert response["decision"]["service_level"] == 0.90
    assert response["decision"]["constraints"]["policy_source"] == "sku_policy"
    assert response["decision"]["constraints"]["max_order_quantity"] == 250
