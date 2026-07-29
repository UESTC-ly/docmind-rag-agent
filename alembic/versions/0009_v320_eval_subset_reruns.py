"""separate diagnostic subset reruns from full release evaluations

Revision ID: 0009_v320_eval_subset_reruns
Revises: 0008_v320_eval_source_snapshot
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_v320_eval_subset_reruns"
down_revision: str | None = "0008_v320_eval_source_snapshot"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("eval_runs") as batch:
        batch.add_column(
            sa.Column(
                "evaluation_scope",
                sa.String(length=16),
                nullable=False,
                server_default="full",
            )
        )
        batch.add_column(sa.Column("sample_filter", sa.Text(), nullable=True))
        batch.add_column(sa.Column("source_run_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_eval_runs_source_run_id",
            "eval_runs",
            ["source_run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            "ix_eval_runs_source_run_id",
            ["source_run_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("eval_runs") as batch:
        batch.drop_index("ix_eval_runs_source_run_id")
        batch.drop_constraint(
            "fk_eval_runs_source_run_id",
            type_="foreignkey",
        )
        batch.drop_column("source_run_id")
        batch.drop_column("sample_filter")
        batch.drop_column("evaluation_scope")
