"""add retraining runs

Revision ID: 2f5c8d9e7a41
Revises: e6b9d2c4f8a1
Create Date: 2026-08-20 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "2f5c8d9e7a41"
down_revision = "e6b9d2c4f8a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retraining_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("trigger_reason", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("baseline_model_artifact_id", sa.Integer(), nullable=False),
        sa.Column("source_monitoring_snapshot_id", sa.Integer(), nullable=False),
        sa.Column("new_evaluated_forecast_days", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("candidate_model_artifact_id", sa.Integer(), nullable=True),
        sa.Column("promotion_recommended", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("evidence_key", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status in ('recommended', 'pending', 'running', 'completed', 'failed', 'rejected')",
            name="ck_retraining_runs_status",
        ),
        sa.CheckConstraint("new_evaluated_forecast_days >= 0", name="ck_retraining_runs_evidence_days_non_negative"),
        sa.ForeignKeyConstraint(["baseline_model_artifact_id"], ["model_artifacts.id"]),
        sa.ForeignKeyConstraint(["candidate_model_artifact_id"], ["model_artifacts.id"]),
        sa.ForeignKeyConstraint(["source_monitoring_snapshot_id"], ["model_monitoring_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evidence_key", name="uq_retraining_runs_evidence_key"),
    )
    op.create_index("ix_retraining_runs_baseline_model_artifact_id", "retraining_runs", ["baseline_model_artifact_id"])
    op.create_index("ix_retraining_runs_candidate_model_artifact_id", "retraining_runs", ["candidate_model_artifact_id"])
    op.create_index("ix_retraining_runs_source_monitoring_snapshot_id", "retraining_runs", ["source_monitoring_snapshot_id"])
    op.create_index("ix_retraining_runs_status", "retraining_runs", ["status"])
    op.create_index("ix_retraining_runs_status_triggered", "retraining_runs", ["status", "triggered_at"])
    op.create_index("ix_retraining_runs_triggered_at", "retraining_runs", ["triggered_at"])


def downgrade() -> None:
    op.drop_index("ix_retraining_runs_triggered_at", table_name="retraining_runs")
    op.drop_index("ix_retraining_runs_status_triggered", table_name="retraining_runs")
    op.drop_index("ix_retraining_runs_status", table_name="retraining_runs")
    op.drop_index("ix_retraining_runs_source_monitoring_snapshot_id", table_name="retraining_runs")
    op.drop_index("ix_retraining_runs_candidate_model_artifact_id", table_name="retraining_runs")
    op.drop_index("ix_retraining_runs_baseline_model_artifact_id", table_name="retraining_runs")
    op.drop_table("retraining_runs")
