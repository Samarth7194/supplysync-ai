"""link forecast evaluations to prediction logs

Revision ID: 8a7f2d4c9b31
Revises: 50f995297bfe
Create Date: 2026-08-08 01:20:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8a7f2d4c9b31"
down_revision: Union[str, None] = "50f995297bfe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("forecast_evaluations") as batch_op:
        batch_op.add_column(sa.Column("prediction_log_id", sa.Integer(), nullable=True))
        batch_op.create_index(
            batch_op.f("ix_forecast_evaluations_prediction_log_id"),
            ["prediction_log_id"],
            unique=True,
        )
        batch_op.create_foreign_key(
            "fk_forecast_evaluations_prediction_log_id_prediction_logs",
            "prediction_logs",
            ["prediction_log_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("forecast_evaluations") as batch_op:
        batch_op.drop_constraint(
            "fk_forecast_evaluations_prediction_log_id_prediction_logs",
            type_="foreignkey",
        )
        batch_op.drop_index(batch_op.f("ix_forecast_evaluations_prediction_log_id"))
        batch_op.drop_column("prediction_log_id")
