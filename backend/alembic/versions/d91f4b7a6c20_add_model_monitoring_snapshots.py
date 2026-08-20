"""add model monitoring snapshots

Revision ID: d91f4b7a6c20
Revises: c4f2a9e1d8b3
Create Date: 2026-08-20 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d91f4b7a6c20"
down_revision: Union[str, None] = "c4f2a9e1d8b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_monitoring_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_artifact_id", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=True),
        sa.Column("window_type", sa.String(length=32), nullable=False),
        sa.Column("window_size", sa.Integer(), nullable=False),
        sa.Column("evaluation_count", sa.Integer(), nullable=False),
        sa.Column("metric_wape", sa.Numeric(14, 6), nullable=True),
        sa.Column("metric_mae", sa.Numeric(14, 6), nullable=True),
        sa.Column("metric_rmse", sa.Numeric(14, 6), nullable=True),
        sa.Column("metric_bias", sa.Numeric(14, 6), nullable=True),
        sa.Column("metric_mase", sa.Numeric(14, 6), nullable=True),
        sa.Column("residual_mean", sa.Numeric(14, 6), nullable=True),
        sa.Column("residual_std", sa.Numeric(14, 6), nullable=True),
        sa.Column("baseline_wape", sa.Numeric(14, 6), nullable=True),
        sa.Column("baseline_provenance", sa.String(length=32), server_default="unavailable", nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("evidence_key", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("evaluation_count >= 0", name="ck_model_monitoring_evaluation_count_non_negative"),
        sa.CheckConstraint(
            "status in ('healthy_data', 'insufficient_evidence')",
            name="ck_model_monitoring_status",
        ),
        sa.CheckConstraint("window_size > 0", name="ck_model_monitoring_window_size_positive"),
        sa.ForeignKeyConstraint(["model_artifact_id"], ["model_artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evidence_key", name="uq_model_monitoring_snapshots_evidence_key"),
    )
    op.create_index("ix_model_monitoring_artifact_generated", "model_monitoring_snapshots", ["model_artifact_id", "generated_at"], unique=False)
    op.create_index(op.f("ix_model_monitoring_snapshots_generated_at"), "model_monitoring_snapshots", ["generated_at"], unique=False)
    op.create_index(op.f("ix_model_monitoring_snapshots_model_artifact_id"), "model_monitoring_snapshots", ["model_artifact_id"], unique=False)
    op.create_index(op.f("ix_model_monitoring_snapshots_model_name"), "model_monitoring_snapshots", ["model_name"], unique=False)
    op.create_index(op.f("ix_model_monitoring_snapshots_status"), "model_monitoring_snapshots", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_model_monitoring_snapshots_status"), table_name="model_monitoring_snapshots")
    op.drop_index(op.f("ix_model_monitoring_snapshots_model_name"), table_name="model_monitoring_snapshots")
    op.drop_index(op.f("ix_model_monitoring_snapshots_model_artifact_id"), table_name="model_monitoring_snapshots")
    op.drop_index(op.f("ix_model_monitoring_snapshots_generated_at"), table_name="model_monitoring_snapshots")
    op.drop_index("ix_model_monitoring_artifact_generated", table_name="model_monitoring_snapshots")
    op.drop_table("model_monitoring_snapshots")
