"""rename p50/p90 to historical_mean_60d/historical_p90_60d

p50/p90 were never true forecast percentiles — they are a 60-day historical
mean and a 60-day historical 90th percentile of recorded demand, computed in
analysis_service.py and used only as a display/risk-context statistic. The
"P50"/"median" naming implied a forecast quantile that was never actually
computed, which is misleading next to the real forecast values. Renaming to
what they actually are; no value change, no data loss.

The API keeps p50/p90 as deprecated aliases (same values) for one release —
this migration only renames the underlying storage.

Revision ID: b4c8e1f6a930
Revises: 9b1a4c6d8e2f
Create Date: 2026-09-29 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "b4c8e1f6a930"
down_revision = "9b1a4c6d8e2f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_runs") as batch_op:
        batch_op.alter_column("p50", new_column_name="historical_mean_60d", existing_type=sa.Numeric(14, 3))
        batch_op.alter_column("p90", new_column_name="historical_p90_60d", existing_type=sa.Numeric(14, 3))
    with op.batch_alter_table("prediction_logs") as batch_op:
        batch_op.alter_column("p50", new_column_name="historical_mean_60d", existing_type=sa.Numeric(14, 3))
        batch_op.alter_column("p90", new_column_name="historical_p90_60d", existing_type=sa.Numeric(14, 3))


def downgrade() -> None:
    with op.batch_alter_table("prediction_logs") as batch_op:
        batch_op.alter_column("historical_p90_60d", new_column_name="p90", existing_type=sa.Numeric(14, 3))
        batch_op.alter_column("historical_mean_60d", new_column_name="p50", existing_type=sa.Numeric(14, 3))
    with op.batch_alter_table("analysis_runs") as batch_op:
        batch_op.alter_column("historical_p90_60d", new_column_name="p90", existing_type=sa.Numeric(14, 3))
        batch_op.alter_column("historical_mean_60d", new_column_name="p50", existing_type=sa.Numeric(14, 3))
