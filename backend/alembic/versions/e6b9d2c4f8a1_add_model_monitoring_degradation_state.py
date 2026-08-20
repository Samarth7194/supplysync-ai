"""add model monitoring degradation state

Revision ID: e6b9d2c4f8a1
Revises: d91f4b7a6c20
Create Date: 2026-08-20 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e6b9d2c4f8a1"
down_revision: Union[str, None] = "d91f4b7a6c20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_monitoring_snapshots") as batch_op:
        batch_op.drop_constraint("ck_model_monitoring_status", type_="check")

    op.execute(
        "update model_monitoring_snapshots "
        "set status = case when status = 'healthy_data' then 'stable' else status end"
    )

    with op.batch_alter_table("model_monitoring_snapshots") as batch_op:
        batch_op.add_column(sa.Column("wape_relative_change", sa.Numeric(14, 6), nullable=True))
        batch_op.add_column(sa.Column("bias_ratio", sa.Numeric(14, 6), nullable=True))
        batch_op.add_column(sa.Column("degradation_reason", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("degradation_message", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("consecutive_degradation_count", sa.Integer(), server_default="0", nullable=False)
        )
        batch_op.create_check_constraint(
            "ck_model_monitoring_status",
            "status in ('insufficient_evidence', 'stable', 'warning', 'degraded')",
        )
        batch_op.create_check_constraint(
            "ck_model_monitoring_consecutive_non_negative",
            "consecutive_degradation_count >= 0",
        )
    op.execute(
        "update model_monitoring_snapshots "
        "set degradation_reason = case "
        "when status = 'insufficient_evidence' then 'insufficient_evidence' "
        "else 'baseline_unavailable' end "
        "where degradation_reason is null"
    )
    op.execute(
        "update model_monitoring_snapshots "
        "set degradation_message = case "
        "when status = 'insufficient_evidence' then 'Not enough completed evaluations to classify performance.' "
        "else 'Phase A snapshot backfilled before degradation classification was available.' end "
        "where degradation_message is null"
    )

    with op.batch_alter_table("model_monitoring_snapshots") as batch_op:
        batch_op.alter_column("degradation_reason", existing_type=sa.String(length=64), nullable=False)
        batch_op.alter_column("degradation_message", existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("model_monitoring_snapshots") as batch_op:
        batch_op.drop_constraint("ck_model_monitoring_consecutive_non_negative", type_="check")
        batch_op.drop_constraint("ck_model_monitoring_status", type_="check")

    op.execute(
        "update model_monitoring_snapshots "
        "set status = case when status in ('stable', 'warning', 'degraded') then 'healthy_data' else status end"
    )

    with op.batch_alter_table("model_monitoring_snapshots") as batch_op:
        batch_op.create_check_constraint(
            "ck_model_monitoring_status",
            "status in ('healthy_data', 'insufficient_evidence')",
        )
        batch_op.drop_column("consecutive_degradation_count")
        batch_op.drop_column("degradation_message")
        batch_op.drop_column("degradation_reason")
        batch_op.drop_column("bias_ratio")
        batch_op.drop_column("wape_relative_change")
