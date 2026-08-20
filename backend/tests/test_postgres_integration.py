"""PostgreSQL integration tests for SQLAlchemy repositories.

These tests require POSTGRES_TEST_DATABASE_URL and intentionally refuse to run
against non-PostgreSQL or non-test databases. CI provides the database service;
local developers can run them with Docker Compose or any isolated Postgres DB.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import delete, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from db.models import (
    AnalysisRun,
    Base,
    ForecastEvaluation,
    InventoryPolicy,
    ModelArtifact,
    ModelMonitoringSnapshot,
    PredictionLog,
    RetrainingRun,
    Sku,
    StockLevel,
)
from db.session import build_engine
from repositories.analysis_repository import AnalysisRepository
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository
from repositories.model_artifact_repository import ModelArtifactRepository
from repositories.stock_repository import StockRepository


def _postgres_url() -> str:
    url = os.getenv("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is not set.")
    parsed = make_url(url)
    if not parsed.drivername.startswith("postgresql"):
        pytest.skip("POSTGRES_TEST_DATABASE_URL must use a PostgreSQL SQLAlchemy driver.")
    if "test" not in (parsed.database or "").lower():
        pytest.skip("Refusing to run PostgreSQL integration tests against a non-test database.")
    return url


@pytest.fixture()
def session():
    engine = build_engine(_postgres_url())
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with Session() as session:
        _clean(session)
        yield session
        session.rollback()
        _clean(session)


def _clean(session) -> None:
    for model in (
        RetrainingRun,
        ModelMonitoringSnapshot,
        ForecastEvaluation,
        PredictionLog,
        AnalysisRun,
        StockLevel,
        InventoryPolicy,
        ModelArtifact,
        Sku,
    ):
        session.execute(delete(model))
    session.commit()


def test_postgres_schema_supports_prediction_evaluation_link(session):
    inspector = inspect(session.bind)

    columns = {column["name"] for column in inspector.get_columns("forecast_evaluations")}
    assert "prediction_log_id" in columns

    foreign_keys = inspector.get_foreign_keys("forecast_evaluations")
    assert any(
        fk["referred_table"] == "prediction_logs"
        and fk["constrained_columns"] == ["prediction_log_id"]
        for fk in foreign_keys
    )

    indexes = inspector.get_indexes("forecast_evaluations")
    assert any(
        index["unique"] and index["column_names"] == ["prediction_log_id"]
        for index in indexes
    )

    session.add_all(
        [
            ForecastEvaluation(
                prediction_log_id=None,
                model_name="offline_a",
                evaluation_scope="offline",
            ),
            ForecastEvaluation(
                prediction_log_id=None,
                model_name="offline_b",
                evaluation_scope="offline",
            ),
        ]
    )
    session.commit()
    assert session.query(ForecastEvaluation).count() == 2


def test_postgres_schema_supports_model_artifact_lifecycle(session):
    inspector = inspect(session.bind)
    model_columns = {column["name"] for column in inspector.get_columns("model_artifacts")}
    prediction_columns = {column["name"] for column in inspector.get_columns("prediction_logs")}
    assert {
        "model_family",
        "artifact_checksum",
        "feature_schema_version",
        "feature_schema_checksum",
        "training_metadata",
        "lifecycle_status",
        "activated_at",
        "retired_at",
    }.issubset(model_columns)
    assert "feature_schema_version" in prediction_columns

    repo = ModelArtifactRepository(session)
    first = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version="pg-v1",
        artifact_uri=__file__,
        artifact_checksum=None,
        checksum_algorithm="sha256",
        lifecycle_status="candidate",
    )
    second = ModelArtifact(
        model_name="lightgbm_demand_forecast",
        model_family="lightgbm",
        model_type="ml",
        version="pg-v2",
        artifact_uri=__file__,
        artifact_checksum=None,
        checksum_algorithm="sha256",
        lifecycle_status="candidate",
    )
    session.add_all([first, second])
    session.flush()

    repo.promote(first.id, force=True)
    repo.promote(second.id, force=True)
    session.commit()

    assert session.get(ModelArtifact, first.id).lifecycle_status == "retired"
    assert session.get(ModelArtifact, second.id).lifecycle_status == "active"
    assert repo.active_for_name("lightgbm_demand_forecast").id == second.id


def test_postgres_schema_supports_model_monitoring_snapshots(session):
    inspector = inspect(session.bind)
    columns = {column["name"] for column in inspector.get_columns("model_monitoring_snapshots")}
    assert {
        "generated_at",
        "model_artifact_id",
        "model_name",
        "model_version",
        "window_type",
        "window_size",
        "evaluation_count",
        "metric_wape",
        "metric_mae",
        "metric_rmse",
        "metric_bias",
        "metric_mase",
        "residual_mean",
        "residual_std",
        "baseline_wape",
        "baseline_provenance",
        "wape_relative_change",
        "bias_ratio",
        "degradation_reason",
        "degradation_message",
        "consecutive_degradation_count",
        "status",
        "evidence_key",
    }.issubset(columns)

    foreign_keys = inspector.get_foreign_keys("model_monitoring_snapshots")
    assert any(
        fk["referred_table"] == "model_artifacts"
        and fk["constrained_columns"] == ["model_artifact_id"]
        for fk in foreign_keys
    )

    indexes = inspector.get_indexes("model_monitoring_snapshots")
    assert any(
        index["column_names"] == ["model_artifact_id", "generated_at"]
        for index in indexes
    )


def test_postgres_schema_supports_retraining_runs(session):
    inspector = inspect(session.bind)
    columns = {column["name"] for column in inspector.get_columns("retraining_runs")}
    assert {
        "triggered_at",
        "trigger_reason",
        "status",
        "baseline_model_artifact_id",
        "source_monitoring_snapshot_id",
        "new_evaluated_forecast_days",
        "started_at",
        "finished_at",
        "candidate_model_artifact_id",
        "promotion_recommended",
        "failure_reason",
        "evidence_key",
        "created_at",
        "updated_at",
    }.issubset(columns)

    foreign_keys = inspector.get_foreign_keys("retraining_runs")
    assert any(
        fk["referred_table"] == "model_artifacts"
        and fk["constrained_columns"] == ["baseline_model_artifact_id"]
        for fk in foreign_keys
    )
    assert any(
        fk["referred_table"] == "model_monitoring_snapshots"
        and fk["constrained_columns"] == ["source_monitoring_snapshot_id"]
        for fk in foreign_keys
    )

    indexes = inspector.get_indexes("retraining_runs")
    assert any(
        index["column_names"] == ["status", "triggered_at"]
        for index in indexes
    )


def test_analysis_repository_persists_relationships_and_recent_ordering(session):
    sku = Sku(sku_code="SKU-PG", name="Postgres SKU")
    session.add(sku)
    session.flush()
    artifact = AnalysisRepository(session).get_or_create_model_artifact(
        {
            "model_name": "lightgbm_demand_forecast",
            "model_type": "ml",
            "version": "sha256:test",
            "artifact_uri": "memory://artifact",
            "is_active": True,
        }
    )
    repo = AnalysisRepository(session)

    first, first_prediction = repo.create_analysis_with_prediction(
        _analysis_values(sku.id, "SKU-PG", current_stock="5", routing_reason="default kept"),
        _prediction_values(
            sku.id,
            "SKU-PG",
            artifact.id,
            forecast_method="ml_lightgbm",
            routing_reason="default kept",
        ),
    )
    second, _ = repo.create_analysis_with_prediction(
        _analysis_values(sku.id, "SKU-PG", current_stock="9", routing_reason="evidence selected"),
        _prediction_values(
            sku.id,
            "SKU-PG",
            artifact.id,
            forecast_method="ml_lightgbm",
            routing_reason="evidence selected",
        ),
    )
    session.commit()

    assert first_prediction.analysis_run_id == first.id
    assert first_prediction.sku_id == sku.id
    assert first_prediction.model_artifact_id == artifact.id
    assert first_prediction.routing_reason == "default kept"

    recent = AnalysisRepository(session).recent(limit=2)
    assert [row.id for row in recent] == [second.id, first.id]
    assert AnalysisRepository(session).count() == 2


def test_stock_repository_remains_valid_on_postgres(session):
    repo = StockRepository(session)

    first = repo.record_stock("SKU-STOCK", Decimal("12"), sku_name="Stocked SKU")
    second = repo.record_stock("SKU-STOCK", Decimal("18"), quantity_reserved=Decimal("3"))
    session.commit()

    latest = repo.latest_for_sku("SKU-STOCK")
    assert latest is not None
    assert latest.id == second.id
    assert latest.quantity_available == Decimal("15.000")
    assert first.sku_id == second.sku_id
    assert [row.sku.sku_code for row in repo.latest_for_all()] == ["SKU-STOCK"]


def test_forecast_evaluation_repository_queries_postgres_evidence(session):
    sku = Sku(sku_code="SKU-EVAL")
    session.add(sku)
    session.flush()
    repo = AnalysisRepository(session)
    prediction_a = repo.create_prediction_log(
        _prediction_values(sku.id, "SKU-EVAL", None, forecast_method="ml_lightgbm")
    )
    prediction_b = repo.create_prediction_log(
        _prediction_values(sku.id, "SKU-EVAL", None, forecast_method="croston")
    )
    prediction_old = repo.create_prediction_log(
        _prediction_values(sku.id, "SKU-EVAL", None, forecast_method="simple_average")
    )
    session.flush()

    evaluation_repo = ForecastEvaluationRepository(session)
    evaluation_repo.create_evaluation(
        _evaluation_values(prediction_a.id, sku.id, "SKU-EVAL", "regular", "ml_lightgbm", "1.000", 7, "2026-08-01")
    )
    evaluation_repo.create_evaluation(
        _evaluation_values(prediction_b.id, sku.id, "SKU-EVAL", "regular", "croston", "0.700", 7, "2026-08-01")
    )
    evaluation_repo.create_evaluation(
        _evaluation_values(prediction_old.id, sku.id, "SKU-EVAL", "regular", "simple_average", "0.400", 14, "2026-08-01")
    )
    session.commit()

    rows = evaluation_repo.logged_method_performance_for_pattern(
        demand_class="regular",
        horizon_days=7,
        metric_name="wape",
        generated_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    by_method = {row.method: row for row in rows}
    assert set(by_method) == {"ml_lightgbm", "croston"}
    assert by_method["croston"].metric_value == pytest.approx(0.7)
    assert by_method["croston"].sample_size == 7

    sku_rows = evaluation_repo.logged_method_performance_for_sku(
        sku_code="SKU-EVAL",
        horizon_days=7,
        metric_name="wape",
    )
    assert {row.method for row in sku_rows} == {"ml_lightgbm", "croston"}


def test_duplicate_logged_evaluation_is_rejected_on_postgres(session):
    prediction = AnalysisRepository(session).create_prediction_log(
        _prediction_values(None, "SKU-DUP", None, forecast_method="simple_average")
    )
    repo = ForecastEvaluationRepository(session)
    repo.create_evaluation(
        _evaluation_values(prediction.id, None, "SKU-DUP", "regular", "simple_average", "1.000", 7, "2026-08-01")
    )
    session.commit()

    with pytest.raises(IntegrityError):
        repo.create_evaluation(
            _evaluation_values(prediction.id, None, "SKU-DUP", "regular", "simple_average", "0.900", 7, "2026-08-02")
        )


def _analysis_values(sku_id: int, sku_code: str, *, current_stock: str, routing_reason: str) -> dict:
    return {
        "sku_id": sku_id,
        "sku_code": sku_code,
        "current_stock": Decimal(current_stock),
        "recommended_order_quantity": 10,
        "action": "PURCHASE",
        "risk": "HIGH",
        "risk_color": "#ef4444",
        "demand_pattern": "regular",
        "demand_source": "historical",
        "forecast_source": "model_forecast",
        "forecast_method": "ml_lightgbm",
        "routing_reason": routing_reason,
        "lead_time_days": 7,
        "service_level": Decimal("0.9500"),
        "lead_time_demand": Decimal("70"),
        "safety_stock": Decimal("10"),
        "safety_stock_method": "traditional",
        "reorder_point": Decimal("80"),
        "inventory_gap": Decimal("75"),
        "p50": Decimal("10"),
        "p90": Decimal("15"),
        "forecast_daily": [10.0] * 7,
        "explanation": {"method_reason": "test"},
    }


def _prediction_values(
    sku_id: int | None,
    sku_code: str,
    model_artifact_id: int | None,
    *,
    forecast_method: str,
    routing_reason: str = "test routing",
) -> dict:
    return {
        "sku_id": sku_id,
        "sku_code": sku_code,
        "target_start_date": datetime(2026, 8, 2, tzinfo=timezone.utc).date(),
        "target_end_date": datetime(2026, 8, 8, tzinfo=timezone.utc).date(),
        "demand_source": "historical",
        "forecast_method": forecast_method,
        "forecast_source": "model_forecast" if forecast_method == "ml_lightgbm" else "statistical_method",
        "routing_reason": routing_reason,
        "model_name": forecast_method,
        "model_version": "test-version",
        "feature_schema_version": "demand_lag_calendar_v1" if forecast_method == "ml_lightgbm" else None,
        "model_artifact_id": model_artifact_id,
        "input_history_length": 60,
        "forecast_horizon_days": 7,
        "p50": Decimal("10"),
        "p90": Decimal("15"),
        "forecast_daily": [10.0] * 7,
        "recommended_order_quantity": 10,
    }


def _evaluation_values(
    prediction_id: int | None,
    sku_id: int | None,
    sku_code: str,
    demand_class: str,
    model_name: str,
    wape: str,
    horizon_days: int,
    generated_at: str,
) -> dict:
    return {
        "prediction_log_id": prediction_id,
        "sku_id": sku_id,
        "sku_code": sku_code,
        "demand_class": demand_class,
        "model_name": model_name,
        "evaluation_scope": "logged_prediction" if prediction_id is not None else "offline",
        "metric_mae": Decimal("1.0"),
        "metric_rmse": Decimal("1.0"),
        "metric_bias": Decimal("0.0"),
        "metric_wape": Decimal(wape),
        "metric_mase": Decimal("1.0"),
        "n_skus": 1,
        "n_test_points": 7,
        "horizon_days": horizon_days,
        "source_path": "test",
        "generated_at": datetime.fromisoformat(f"{generated_at}T00:00:00+00:00"),
    }
