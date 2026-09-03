"""add model promotion events

Revision ID: 9b1a4c6d8e2f
Revises: 2f5c8d9e7a41
Create Date: 2026-08-25 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "9b1a4c6d8e2f"
down_revision = "2f5c8d9e7a41"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_promotion_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("promoted_model_artifact_id", sa.Integer(), sa.ForeignKey("model_artifacts.id"), nullable=False),
        sa.Column("previous_model_artifact_id", sa.Integer(), sa.ForeignKey("model_artifacts.id"), nullable=True),
        sa.Column("retraining_run_id", sa.Integer(), sa.ForeignKey("retraining_runs.id"), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=False, server_default="succeeded"),
        sa.Column("initiated_by", sa.String(length=128), nullable=False, server_default="manual_cli"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("event_type in ('promotion', 'rollback')", name="ck_model_promotion_events_type"),
        sa.CheckConstraint(
            "outcome in ('pending', 'succeeded', 'handoff_failed_restored')",
            name="ck_model_promotion_events_outcome",
        ),
    )
    op.create_index(
        "uq_model_artifacts_one_active_per_model",
        "model_artifacts",
        ["model_name"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
        sqlite_where=sa.text("is_active = 1"),
    )
    op.create_index("ix_model_promotion_events_model_created", "model_promotion_events", ["model_name", "created_at"])
    op.create_index("ix_model_promotion_events_promoted", "model_promotion_events", ["promoted_model_artifact_id"])
    op.create_index("ix_model_promotion_events_previous", "model_promotion_events", ["previous_model_artifact_id"])
    op.create_index("ix_model_promotion_events_retraining", "model_promotion_events", ["retraining_run_id"])


def downgrade() -> None:
    op.drop_index("ix_model_promotion_events_retraining", table_name="model_promotion_events")
    op.drop_index("ix_model_promotion_events_previous", table_name="model_promotion_events")
    op.drop_index("ix_model_promotion_events_promoted", table_name="model_promotion_events")
    op.drop_index("ix_model_promotion_events_model_created", table_name="model_promotion_events")
    op.drop_table("model_promotion_events")
    op.drop_index("uq_model_artifacts_one_active_per_model", table_name="model_artifacts")

