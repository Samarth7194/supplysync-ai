"""model artifact lifecycle

Revision ID: c4f2a9e1d8b3
Revises: 8a7f2d4c9b31
Create Date: 2026-08-08 14:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4f2a9e1d8b3"
down_revision: Union[str, None] = "8a7f2d4c9b31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_artifacts") as batch_op:
        batch_op.add_column(sa.Column("model_family", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("artifact_checksum", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("checksum_algorithm", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("feature_schema_version", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("feature_schema_checksum", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("training_metadata", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column("lifecycle_status", sa.String(length=32), server_default="candidate", nullable=False)
        )
        batch_op.add_column(sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_check_constraint(
            "ck_model_artifacts_lifecycle_status",
            "lifecycle_status in ('candidate', 'active', 'retired', 'failed')",
        )
        batch_op.create_unique_constraint(
            "uq_model_artifacts_artifact_checksum",
            ["artifact_checksum"],
        )
        batch_op.create_index(batch_op.f("ix_model_artifacts_lifecycle_status"), ["lifecycle_status"], unique=False)
        batch_op.create_index(batch_op.f("ix_model_artifacts_model_family"), ["model_family"], unique=False)
    with op.batch_alter_table("prediction_logs") as batch_op:
        batch_op.add_column(sa.Column("feature_schema_version", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("prediction_logs") as batch_op:
        batch_op.drop_column("feature_schema_version")
    with op.batch_alter_table("model_artifacts") as batch_op:
        batch_op.drop_index(batch_op.f("ix_model_artifacts_model_family"))
        batch_op.drop_index(batch_op.f("ix_model_artifacts_lifecycle_status"))
        batch_op.drop_constraint("uq_model_artifacts_artifact_checksum", type_="unique")
        batch_op.drop_constraint("ck_model_artifacts_lifecycle_status", type_="check")
        batch_op.drop_column("retired_at")
        batch_op.drop_column("activated_at")
        batch_op.drop_column("lifecycle_status")
        batch_op.drop_column("training_metadata")
        batch_op.drop_column("feature_schema_checksum")
        batch_op.drop_column("feature_schema_version")
        batch_op.drop_column("checksum_algorithm")
        batch_op.drop_column("artifact_checksum")
        batch_op.drop_column("model_family")
