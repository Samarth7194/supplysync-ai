"""Repository for logged-prediction forecast evaluations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from db.models import ForecastEvaluation, PredictionLog


@dataclass(frozen=True)
class MethodPerformance:
    method: str
    metric_value: float
    sample_size: int
    evaluation_count: int
    mean_bias: float | None
    latest_generated_at: datetime | None
    horizon_days: int
    evidence_source: str
    evidence_level: str


class ForecastEvaluationRepository:
    """Read prediction logs and persist evaluation evidence."""

    def __init__(self, session: Session):
        self.session = session

    def eligible_prediction_logs(self, as_of: date, limit: int = 500) -> list[PredictionLog]:
        limit = max(1, min(int(limit), 5000))
        stmt = (
            select(PredictionLog)
            .options(selectinload(PredictionLog.analysis_run))
            .where(PredictionLog.target_end_date.is_not(None))
            .where(PredictionLog.target_end_date <= as_of)
            .order_by(PredictionLog.predicted_at.asc(), PredictionLog.id.asc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def prediction_has_evaluation(self, prediction_log_id: int) -> bool:
        stmt = select(ForecastEvaluation.id).where(
            ForecastEvaluation.prediction_log_id == prediction_log_id
        )
        return self.session.scalar(stmt) is not None

    def create_evaluation(self, values: Mapping[str, Any]) -> ForecastEvaluation:
        evaluation = ForecastEvaluation(**dict(values))
        self.session.add(evaluation)
        self.session.flush()
        return evaluation

    def evaluated_prediction_logs(
        self,
        *,
        horizon_days: int,
        generated_after: datetime | None = None,
        sku_code: str | None = None,
        forecast_method: str | None = None,
        demand_class: str | None = None,
        limit: int = 500,
    ) -> list[PredictionLog]:
        """Return prediction logs that already have completed evaluations.

        This deliberately reads through ``forecast_evaluations`` rather than
        raw ``prediction_logs`` so uncertainty estimates only use predictions
        whose horizon has completed and whose actuals were resolved by
        ``ForecastEvaluationService``.
        """
        limit = max(1, min(int(limit), 5000))
        stmt = (
            select(PredictionLog)
            .join(ForecastEvaluation, ForecastEvaluation.prediction_log_id == PredictionLog.id)
            .options(selectinload(PredictionLog.analysis_run))
            .where(ForecastEvaluation.evaluation_scope == "logged_prediction")
            .where(ForecastEvaluation.horizon_days == horizon_days)
            .where(PredictionLog.forecast_horizon_days == horizon_days)
            .where(PredictionLog.forecast_daily.is_not(None))
            .order_by(ForecastEvaluation.generated_at.desc().nullslast(), PredictionLog.predicted_at.desc())
            .limit(limit)
        )
        if generated_after is not None:
            stmt = stmt.where(ForecastEvaluation.generated_at.is_not(None))
            stmt = stmt.where(ForecastEvaluation.generated_at >= generated_after)
        if sku_code is not None:
            stmt = stmt.where(PredictionLog.sku_code == sku_code)
        if forecast_method is not None:
            stmt = stmt.where(PredictionLog.forecast_method == forecast_method)
        if demand_class is not None:
            stmt = stmt.where(ForecastEvaluation.demand_class == demand_class)

        return list(self.session.scalars(stmt))

    def logged_method_performance_for_sku(
        self,
        *,
        sku_code: str,
        horizon_days: int,
        metric_name: str,
        generated_after: datetime | None = None,
    ) -> list[MethodPerformance]:
        return self._logged_method_performance(
            sku_code=sku_code,
            demand_class=None,
            horizon_days=horizon_days,
            metric_name=metric_name,
            generated_after=generated_after,
            evidence_level="sku",
        )

    def logged_method_performance_for_pattern(
        self,
        *,
        demand_class: str,
        horizon_days: int,
        metric_name: str,
        generated_after: datetime | None = None,
    ) -> list[MethodPerformance]:
        return self._logged_method_performance(
            sku_code=None,
            demand_class=demand_class,
            horizon_days=horizon_days,
            metric_name=metric_name,
            generated_after=generated_after,
            evidence_level="pattern",
        )

    def _logged_method_performance(
        self,
        *,
        sku_code: str | None,
        demand_class: str | None,
        horizon_days: int,
        metric_name: str,
        generated_after: datetime | None,
        evidence_level: str,
    ) -> list[MethodPerformance]:
        metric_column = self._metric_column(metric_name)
        stmt = (
            select(
                PredictionLog.forecast_method.label("method"),
                func.avg(metric_column).label("metric_value"),
                func.coalesce(func.sum(ForecastEvaluation.n_test_points), 0).label("sample_size"),
                func.count(ForecastEvaluation.id).label("evaluation_count"),
                func.avg(ForecastEvaluation.metric_bias).label("mean_bias"),
                func.max(ForecastEvaluation.generated_at).label("latest_generated_at"),
            )
            .join(PredictionLog, ForecastEvaluation.prediction_log_id == PredictionLog.id)
            .where(ForecastEvaluation.evaluation_scope == "logged_prediction")
            .where(ForecastEvaluation.horizon_days == horizon_days)
            .where(metric_column.is_not(None))
            .group_by(PredictionLog.forecast_method)
        )
        if sku_code is not None:
            stmt = stmt.where(ForecastEvaluation.sku_code == sku_code)
        if demand_class is not None:
            stmt = stmt.where(ForecastEvaluation.demand_class == demand_class)
        if generated_after is not None:
            stmt = stmt.where(ForecastEvaluation.generated_at.is_not(None))
            stmt = stmt.where(ForecastEvaluation.generated_at >= generated_after)

        rows = self.session.execute(stmt).all()
        return [
            MethodPerformance(
                method=str(row.method),
                metric_value=float(row.metric_value),
                sample_size=int(row.sample_size or 0),
                evaluation_count=int(row.evaluation_count or 0),
                mean_bias=float(row.mean_bias) if row.mean_bias is not None else None,
                latest_generated_at=row.latest_generated_at,
                horizon_days=horizon_days,
                evidence_source="logged",
                evidence_level=evidence_level,
            )
            for row in rows
            if row.metric_value is not None
        ]

    @staticmethod
    def _metric_column(metric_name: str):
        columns = {
            "mae": ForecastEvaluation.metric_mae,
            "rmse": ForecastEvaluation.metric_rmse,
            "wape": ForecastEvaluation.metric_wape,
            "mase": ForecastEvaluation.metric_mase,
        }
        try:
            return columns[metric_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported routing metric: {metric_name}") from exc
