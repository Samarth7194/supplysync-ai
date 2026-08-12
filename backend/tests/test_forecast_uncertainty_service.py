from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import AnalysisRun, Base, ForecastEvaluation, PredictionLog
from repositories.forecast_evaluation_repository import ForecastEvaluationRepository
from services.forecast_uncertainty_service import ForecastUncertaintyService


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


def _evaluated_prediction(
    session,
    *,
    sku="SKU-1",
    method="simple_average",
    pattern="regular",
    horizon=3,
    start=date(2024, 1, 3),
    forecast=None,
):
    analysis = AnalysisRun(
        sku_code=sku,
        current_stock=Decimal("5"),
        recommended_order_quantity=10,
        action="PURCHASE",
        risk="HIGH",
        demand_pattern=pattern,
        demand_source="historical",
        forecast_source="rule_based_estimate",
        forecast_method=method,
        lead_time_days=horizon,
        service_level=Decimal("0.95"),
    )
    session.add(analysis)
    session.flush()
    prediction = PredictionLog(
        analysis_run_id=analysis.id,
        sku_code=sku,
        target_start_date=start,
        target_end_date=date(start.year, start.month, start.day + horizon - 1),
        demand_source="historical",
        forecast_method=method,
        forecast_source="rule_based_estimate",
        input_history_length=30,
        forecast_horizon_days=horizon,
        forecast_daily=forecast or [1.0, 1.0, 1.0],
        recommended_order_quantity=10,
    )
    session.add(prediction)
    session.flush()
    session.add(
        ForecastEvaluation(
            prediction_log_id=prediction.id,
            sku_code=sku,
            demand_class=pattern,
            model_name=method,
            evaluation_scope="logged_prediction",
            metric_mae=Decimal("1.0"),
            metric_rmse=Decimal("1.0"),
            metric_bias=Decimal("0.0"),
            metric_wape=Decimal("0.5"),
            n_skus=1,
            n_test_points=horizon,
            horizon_days=horizon,
            source_path="logged_prediction",
            generated_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
    )
    session.flush()


def _service(session, data, *, min_obs=3):
    return ForecastUncertaintyService(
        repository=ForecastEvaluationRepository(session),
        data_service=_DataService(data),
        min_residual_observations=min_obs,
        lookback_days=3650,
    )


def test_residual_sigma_prefers_sku_method_compatible_horizon():
    session = _session()
    series = pd.Series([9, 9, 2, 4, 7], index=pd.date_range("2024-01-01", periods=5))
    _evaluated_prediction(session, forecast=[1.0, 1.0, 1.0])
    service = _service(session, {"SKU-1": series})

    estimate = service.select_sigma(
        sku_code="SKU-1",
        forecast_method="simple_average",
        demand_pattern="regular",
        horizon_days=3,
        historical_sigma=99.0,
    )

    # actual [2, 4, 7] - forecast [1, 1, 1] = [1, 3, 6]
    assert estimate.source == "sku_method_residuals"
    assert estimate.sample_count == 3
    assert estimate.sigma == pytest.approx(pd.Series([1.0, 3.0, 6.0]).std())
    assert estimate.fallback_used is False


def test_insufficient_residuals_falls_back_to_historical_sigma():
    session = _session()
    series = pd.Series([9, 9, 2, 4, 7], index=pd.date_range("2024-01-01", periods=5))
    _evaluated_prediction(session, forecast=[1.0, 1.0, 1.0])
    service = _service(session, {"SKU-1": series}, min_obs=4)

    estimate = service.select_sigma(
        sku_code="SKU-1",
        forecast_method="simple_average",
        demand_pattern="regular",
        horizon_days=3,
        historical_sigma=12.5,
    )

    assert estimate.source == "historical_demand_std"
    assert estimate.sigma == 12.5
    assert estimate.fallback_used is True


def test_wrong_horizon_residuals_are_ignored():
    session = _session()
    series = pd.Series([9, 9, 2, 4, 7], index=pd.date_range("2024-01-01", periods=5))
    _evaluated_prediction(session, horizon=3, forecast=[1.0, 1.0, 1.0])
    service = _service(session, {"SKU-1": series}, min_obs=2)

    estimate = service.select_sigma(
        sku_code="SKU-1",
        forecast_method="simple_average",
        demand_pattern="regular",
        horizon_days=7,
        historical_sigma=8.0,
    )

    assert estimate.source == "historical_demand_std"
    assert estimate.sigma == 8.0
