"""SQLAlchemy models for the target persistence schema.

These mappings mirror docs/database-design.md. Runtime code reaches them
through repository and service layers, keeping SQLAlchemy out of route handlers.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid


class Base(DeclarativeBase):
    pass


class Sku(Base):
    __tablename__ = "skus"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    stock_levels: Mapped[list["StockLevel"]] = relationship(back_populates="sku", cascade="all, delete-orphan")
    inventory_policies: Mapped[list["InventoryPolicy"]] = relationship(back_populates="sku", cascade="all, delete-orphan")
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship(back_populates="sku")
    prediction_logs: Mapped[list["PredictionLog"]] = relationship(back_populates="sku")
    forecast_evaluations: Mapped[list["ForecastEvaluation"]] = relationship(back_populates="sku")


class StockLevel(Base):
    __tablename__ = "stock_levels"
    __table_args__ = (
        CheckConstraint("quantity_on_hand >= 0", name="ck_stock_levels_on_hand_non_negative"),
        CheckConstraint("quantity_reserved >= 0", name="ck_stock_levels_reserved_non_negative"),
        CheckConstraint("quantity_available >= 0", name="ck_stock_levels_available_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sku_id: Mapped[int] = mapped_column(ForeignKey("skus.id"), nullable=False, index=True)
    quantity_on_hand: Mapped[Decimal] = mapped_column(Numeric(14, 3), nullable=False)
    quantity_reserved: Mapped[Decimal] = mapped_column(Numeric(14, 3), nullable=False, default=0, server_default="0")
    quantity_available: Mapped[Decimal] = mapped_column(Numeric(14, 3), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    sku: Mapped[Sku] = relationship(back_populates="stock_levels")


class InventoryPolicy(Base):
    __tablename__ = "inventory_policies"
    __table_args__ = (
        CheckConstraint("lead_time_days between 1 and 365", name="ck_inventory_policies_lead_time_range"),
        CheckConstraint("service_level > 0 and service_level < 1", name="ck_inventory_policies_service_level_range"),
        CheckConstraint("moq >= 0", name="ck_inventory_policies_moq_non_negative"),
        CheckConstraint("order_multiple >= 1", name="ck_inventory_policies_order_multiple_positive"),
        CheckConstraint("max_order_quantity is null or max_order_quantity > 0", name="ck_inventory_policies_max_order_positive"),
        CheckConstraint("holding_cost_per_unit is null or holding_cost_per_unit >= 0", name="ck_inventory_policies_holding_cost_non_negative"),
        CheckConstraint("stockout_cost_per_unit is null or stockout_cost_per_unit >= 0", name="ck_inventory_policies_stockout_cost_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sku_id: Mapped[int] = mapped_column(ForeignKey("skus.id"), nullable=False, index=True)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False)
    service_level: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    moq: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    order_multiple: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    max_order_quantity: Mapped[int | None] = mapped_column(Integer)
    holding_cost_per_unit: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    stockout_cost_per_unit: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true", index=True)
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    sku: Mapped[Sku] = relationship(back_populates="inventory_policies")


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_analysis_runs_request_id"),
        CheckConstraint("current_stock >= 0", name="ck_analysis_runs_current_stock_non_negative"),
        CheckConstraint("recommended_order_quantity >= 0", name="ck_analysis_runs_recommended_order_non_negative"),
        CheckConstraint("service_level > 0 and service_level < 1", name="ck_analysis_runs_service_level_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, default=uuid.uuid4)
    sku_id: Mapped[int | None] = mapped_column(ForeignKey("skus.id"), index=True)
    sku_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    current_stock: Mapped[Decimal] = mapped_column(Numeric(14, 3), nullable=False)
    recommended_order_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    risk: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    risk_color: Mapped[str | None] = mapped_column(String(16))
    demand_pattern: Mapped[str] = mapped_column(String(32), nullable=False)
    demand_source: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_source: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_method: Mapped[str] = mapped_column(String(64), nullable=False)
    routing_reason: Mapped[str | None] = mapped_column(Text)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False)
    service_level: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    lead_time_demand: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    safety_stock: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    safety_stock_method: Mapped[str | None] = mapped_column(String(32))
    reorder_point: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    inventory_gap: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    p50: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    p90: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    forecast_daily: Mapped[list[float] | None] = mapped_column(JSON)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    sku: Mapped[Sku | None] = relationship(back_populates="analysis_runs")
    prediction_logs: Mapped[list["PredictionLog"]] = relationship(back_populates="analysis_run")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    __table_args__ = (
        CheckConstraint("input_history_length >= 0", name="ck_prediction_logs_history_length_non_negative"),
        CheckConstraint("forecast_horizon_days > 0", name="ck_prediction_logs_horizon_positive"),
        CheckConstraint("recommended_order_quantity >= 0", name="ck_prediction_logs_recommended_order_non_negative"),
        CheckConstraint("actual_observed_demand is null or actual_observed_demand >= 0", name="ck_prediction_logs_actual_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    sku_id: Mapped[int | None] = mapped_column(ForeignKey("skus.id"), index=True)
    sku_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    target_start_date: Mapped[date | None] = mapped_column(Date)
    target_end_date: Mapped[date | None] = mapped_column(Date)
    demand_source: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_method: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    forecast_source: Mapped[str] = mapped_column(String(32), nullable=False)
    routing_reason: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(128))
    feature_schema_version: Mapped[str | None] = mapped_column(String(64))
    model_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), index=True)
    input_history_length: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    p50: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    p90: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    forecast_daily: Mapped[list[float] | None] = mapped_column(JSON)
    recommended_order_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_observed_demand: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    actual_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    analysis_run: Mapped[AnalysisRun | None] = relationship(back_populates="prediction_logs")
    sku: Mapped[Sku | None] = relationship(back_populates="prediction_logs")
    model_artifact: Mapped["ModelArtifact | None"] = relationship(back_populates="prediction_logs")


class ForecastEvaluation(Base):
    __tablename__ = "forecast_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_log_id: Mapped[int | None] = mapped_column(
        ForeignKey("prediction_logs.id"),
        unique=True,
        index=True,
    )
    model_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), index=True)
    sku_id: Mapped[int | None] = mapped_column(ForeignKey("skus.id"), index=True)
    sku_code: Mapped[str | None] = mapped_column(String(64))
    demand_class: Mapped[str | None] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    evaluation_scope: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    metric_mae: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_rmse: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_bias: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_wape: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_mase: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    n_skus: Mapped[int | None] = mapped_column(Integer)
    n_test_points: Mapped[int | None] = mapped_column(Integer)
    horizon_days: Mapped[int | None] = mapped_column(Integer)
    source_path: Mapped[str | None] = mapped_column(Text)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    model_artifact: Mapped["ModelArtifact | None"] = relationship(back_populates="forecast_evaluations")
    sku: Mapped[Sku | None] = relationship(back_populates="forecast_evaluations")
    prediction_log: Mapped[PredictionLog | None] = relationship()


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_artifacts_name_version"),
        UniqueConstraint("artifact_checksum", name="uq_model_artifacts_artifact_checksum"),
        CheckConstraint(
            "lifecycle_status in ('candidate', 'active', 'retired', 'failed')",
            name="ck_model_artifacts_lifecycle_status",
        ),
        Index(
            "uq_model_artifacts_one_active_per_model",
            "model_name",
            unique=True,
            postgresql_where=text("is_active = true"),
            sqlite_where=text("is_active = 1"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_family: Mapped[str | None] = mapped_column(String(64), index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_checksum: Mapped[str | None] = mapped_column(String(64))
    checksum_algorithm: Mapped[str | None] = mapped_column(String(16))
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    metadata_uri: Mapped[str | None] = mapped_column(Text)
    feature_schema: Mapped[list[str] | None] = mapped_column(JSON)
    feature_schema_version: Mapped[str | None] = mapped_column(String(64))
    feature_schema_checksum: Mapped[str | None] = mapped_column(String(64))
    training_dataset: Mapped[str | None] = mapped_column(Text)
    training_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    training_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    lifecycle_status: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate", server_default="candidate", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false", index=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    forecast_evaluations: Mapped[list[ForecastEvaluation]] = relationship(back_populates="model_artifact")
    prediction_logs: Mapped[list[PredictionLog]] = relationship(back_populates="model_artifact")
    monitoring_snapshots: Mapped[list["ModelMonitoringSnapshot"]] = relationship(back_populates="model_artifact")
    retraining_runs_as_baseline: Mapped[list["RetrainingRun"]] = relationship(
        back_populates="baseline_model_artifact",
        foreign_keys="RetrainingRun.baseline_model_artifact_id",
    )
    retraining_runs_as_candidate: Mapped[list["RetrainingRun"]] = relationship(
        back_populates="candidate_model_artifact",
        foreign_keys="RetrainingRun.candidate_model_artifact_id",
    )


class ModelPromotionEvent(Base):
    __tablename__ = "model_promotion_events"
    __table_args__ = (
        CheckConstraint("event_type in ('promotion', 'rollback')", name="ck_model_promotion_events_type"),
        CheckConstraint(
            "outcome in ('pending', 'succeeded', 'handoff_failed_restored')",
            name="ck_model_promotion_events_outcome",
        ),
        Index("ix_model_promotion_events_model_created", "model_name", "created_at"),
        Index("ix_model_promotion_events_promoted", "promoted_model_artifact_id"),
        Index("ix_model_promotion_events_previous", "previous_model_artifact_id"),
        Index("ix_model_promotion_events_retraining", "retraining_run_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    promoted_model_artifact_id: Mapped[int] = mapped_column(ForeignKey("model_artifacts.id"), nullable=False)
    previous_model_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"))
    retraining_run_id: Mapped[int | None] = mapped_column(ForeignKey("retraining_runs.id"))
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="succeeded", server_default="succeeded")
    initiated_by: Mapped[str] = mapped_column(String(128), nullable=False, default="manual_cli", server_default="manual_cli")
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

class ModelMonitoringSnapshot(Base):
    __tablename__ = "model_monitoring_snapshots"
    __table_args__ = (
        CheckConstraint("window_size > 0", name="ck_model_monitoring_window_size_positive"),
        CheckConstraint("evaluation_count >= 0", name="ck_model_monitoring_evaluation_count_non_negative"),
        CheckConstraint(
            "status in ('insufficient_evidence', 'stable', 'warning', 'degraded')",
            name="ck_model_monitoring_status",
        ),
        CheckConstraint("consecutive_degradation_count >= 0", name="ck_model_monitoring_consecutive_non_negative"),
        UniqueConstraint("evidence_key", name="uq_model_monitoring_snapshots_evidence_key"),
        Index("ix_model_monitoring_artifact_generated", "model_artifact_id", "generated_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_version: Mapped[str | None] = mapped_column(String(128))
    window_type: Mapped[str] = mapped_column(String(32), nullable=False)
    window_size: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_wape: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_mae: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_rmse: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_bias: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    metric_mase: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    residual_mean: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    residual_std: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    baseline_wape: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    baseline_provenance: Mapped[str] = mapped_column(String(32), nullable=False, default="unavailable", server_default="unavailable")
    wape_relative_change: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    bias_ratio: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    degradation_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    degradation_message: Mapped[str] = mapped_column(Text, nullable=False)
    consecutive_degradation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    evidence_key: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    model_artifact: Mapped[ModelArtifact | None] = relationship(back_populates="monitoring_snapshots")
    retraining_runs: Mapped[list["RetrainingRun"]] = relationship(back_populates="source_monitoring_snapshot")


class RetrainingRun(Base):
    __tablename__ = "retraining_runs"
    __table_args__ = (
        CheckConstraint(
            "status in ('recommended', 'pending', 'running', 'completed', 'failed', 'rejected')",
            name="ck_retraining_runs_status",
        ),
        CheckConstraint("new_evaluated_forecast_days >= 0", name="ck_retraining_runs_evidence_days_non_negative"),
        UniqueConstraint("evidence_key", name="uq_retraining_runs_evidence_key"),
        Index("ix_retraining_runs_status_triggered", "status", "triggered_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    trigger_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    baseline_model_artifact_id: Mapped[int] = mapped_column(ForeignKey("model_artifacts.id"), nullable=False, index=True)
    source_monitoring_snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("model_monitoring_snapshots.id"),
        nullable=False,
        index=True,
    )
    new_evaluated_forecast_days: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    candidate_model_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), index=True)
    promotion_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    failure_reason: Mapped[str | None] = mapped_column(Text)
    evidence_key: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    baseline_model_artifact: Mapped[ModelArtifact] = relationship(
        back_populates="retraining_runs_as_baseline",
        foreign_keys=[baseline_model_artifact_id],
    )
    candidate_model_artifact: Mapped[ModelArtifact | None] = relationship(
        back_populates="retraining_runs_as_candidate",
        foreign_keys=[candidate_model_artifact_id],
    )
    source_monitoring_snapshot: Mapped[ModelMonitoringSnapshot] = relationship(back_populates="retraining_runs")
