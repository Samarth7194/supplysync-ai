from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import AnalysisRun, Base, ForecastEvaluation, PredictionLog, Sku
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository
from services.forecast_evaluation_service import ForecastEvaluationService


class _DataService:
    def __init__(self, series_by_sku):
        self.series_by_sku = series_by_sku

    def get_demand_history(self, sku):
        return self.series_by_sku.get(sku, pd.Series(dtype=float))


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


def _analysis(session, sku_id=None):
    row = AnalysisRun(
        sku_id=sku_id,
        sku_code="SKU-1",
        current_stock=Decimal("5"),
        recommended_order_quantity=10,
        action="PURCHASE",
        risk="HIGH",
        demand_pattern="regular",
        demand_source="historical",
        forecast_source="rule_based_estimate",
        forecast_method="simple_average",
        lead_time_days=3,
        service_level=Decimal("0.95"),
    )
    session.add(row)
    session.flush()
    return row


def _prediction(session, *, target_end=date(2024, 1, 5), forecast=None):
    analysis = _analysis(session)
    row = PredictionLog(
        analysis_run_id=analysis.id,
        sku_code="SKU-1",
        target_start_date=date(2024, 1, 3),
        target_end_date=target_end,
        demand_source="historical",
        forecast_method="simple_average",
        forecast_source="rule_based_estimate",
        model_name="Simple 7-day moving average",
        model_version="simple_average_7_day_v1",
        input_history_length=2,
        forecast_horizon_days=3,
        forecast_daily=forecast or [2.0, 3.0, 4.0],
        recommended_order_quantity=10,
    )
    session.add(row)
    session.flush()
    return row


def _service(session, series):
    return ForecastEvaluationService(
        repository=ForecastEvaluationRepository(session),
        data_service=_DataService({"SKU-1": series}),
    )


def test_horizon_not_completed_is_not_evaluated():
    session = _session()
    _prediction(session, target_end=date(2024, 1, 10))
    series = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5))
    service = _service(session, series)

    summary = service.evaluate_due_predictions(as_of=date(2024, 1, 5))

    assert summary.predictions_scanned == 0
    assert summary.evaluated == 0


def test_completed_prediction_creates_evaluation_and_updates_actual():
    session = _session()
    prediction = _prediction(session)
    series = pd.Series([9, 8, 2, 5, 4], index=pd.date_range("2024-01-01", periods=5))
    service = _service(session, series)

    summary = service.evaluate_due_predictions(as_of=date(2024, 1, 5))
    session.commit()

    evaluation = session.scalar(select(ForecastEvaluation))
    assert summary.evaluated == 1
    assert evaluation.prediction_log_id == prediction.id
    assert evaluation.metric_mae == pytest.approx(Decimal("0.6667"), abs=Decimal("0.001"))
    assert evaluation.metric_bias == pytest.approx(Decimal("-0.6667"), abs=Decimal("0.001"))
    assert evaluation.metric_wape is not None
    assert prediction.actual_observed_demand == Decimal("11.0")
    assert prediction.actual_observed_at is not None


def test_zero_actual_demand_keeps_wape_null():
    session = _session()
    _prediction(session, forecast=[1.0, 1.0, 1.0])
    series = pd.Series([9, 8, 0, 0, 0], index=pd.date_range("2024-01-01", periods=5))
    service = _service(session, series)

    service.evaluate_due_predictions(as_of=date(2024, 1, 5))
    evaluation = session.scalar(select(ForecastEvaluation))

    assert evaluation.metric_wape is None


def test_missing_actual_demand_is_skipped_safely():
    session = _session()
    _prediction(session)
    series = pd.Series([1, 2], index=pd.date_range("2024-01-01", periods=2))
    service = _service(session, series)

    summary = service.evaluate_due_predictions(as_of=date(2024, 1, 5))

    assert summary.evaluated == 0
    assert summary.actual_demand_unavailable == 1
    assert session.scalar(select(ForecastEvaluation)) is None


def test_already_evaluated_prediction_is_not_duplicated():
    session = _session()
    prediction = _prediction(session)
    series = pd.Series([9, 8, 2, 5, 4], index=pd.date_range("2024-01-01", periods=5))
    service = _service(session, series)

    first = service.evaluate_due_predictions(as_of=date(2024, 1, 5))
    second = service.evaluate_due_predictions(as_of=date(2024, 1, 5))

    assert first.evaluated == 1
    assert second.already_evaluated == 1
    assert len(session.scalars(select(ForecastEvaluation)).all()) == 1


def test_evaluation_preserves_sku_relationship_when_present():
    session = _session()
    sku = Sku(sku_code="SKU-1")
    session.add(sku)
    session.flush()
    analysis = _analysis(session, sku_id=sku.id)
    prediction = PredictionLog(
        analysis_run_id=analysis.id,
        sku_id=sku.id,
        sku_code="SKU-1",
        target_start_date=date(2024, 1, 3),
        target_end_date=date(2024, 1, 5),
        demand_source="historical",
        forecast_method="simple_average",
        forecast_source="rule_based_estimate",
        input_history_length=2,
        forecast_horizon_days=3,
        forecast_daily=[2.0, 3.0, 4.0],
        recommended_order_quantity=10,
    )
    session.add(prediction)
    session.flush()
    series = pd.Series([9, 8, 2, 5, 4], index=pd.date_range("2024-01-01", periods=5))
    service = _service(session, series)

    service.evaluate_due_predictions(as_of=date(2024, 1, 5))

    evaluation = session.scalar(select(ForecastEvaluation))
    assert evaluation.sku_id == sku.id


def test_repository_aggregates_logged_method_performance_by_runtime_method():
    session = _session()
    analysis = _analysis(session)
    for method, wape in (("ml_lightgbm", Decimal("1.000")), ("croston", Decimal("0.750"))):
        prediction = PredictionLog(
            analysis_run_id=analysis.id,
            sku_code="SKU-1",
            target_start_date=date(2024, 1, 3),
            target_end_date=date(2024, 1, 9),
            demand_source="historical",
            forecast_method=method,
            forecast_source="model_forecast" if method == "ml_lightgbm" else "statistical_method",
            input_history_length=30,
            forecast_horizon_days=7,
            forecast_daily=[1.0] * 7,
            recommended_order_quantity=10,
        )
        session.add(prediction)
        session.flush()
        session.add(
            ForecastEvaluation(
                prediction_log_id=prediction.id,
                sku_code="SKU-1",
                demand_class="regular",
                model_name=method,
                evaluation_scope="logged_prediction",
                metric_wape=wape,
                metric_bias=Decimal("0.0"),
                n_test_points=7,
                horizon_days=7,
                generated_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
            )
        )
    session.flush()

    rows = ForecastEvaluationRepository(session).logged_method_performance_for_pattern(
        demand_class="regular",
        horizon_days=7,
        metric_name="wape",
    )

    by_method = {row.method: row for row in rows}
    assert by_method["ml_lightgbm"].metric_value == pytest.approx(1.0)
    assert by_method["croston"].metric_value == pytest.approx(0.75)
    assert by_method["croston"].sample_size == 7
    assert by_method["croston"].evidence_source == "logged"
