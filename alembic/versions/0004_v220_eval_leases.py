"""Make evaluation execution retry-safe and concurrency-safe.

Revision ID: 0004_v220_eval_leases
Revises: 0003_v220_fts
Create Date: 2026-07-13

The lease fields support atomic task claiming on both PostgreSQL and SQLite.
The unique constraint is a final database guard against duplicate delivery.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0004_v220_eval_leases"
down_revision: str | Sequence[str] | None = "0003_v220_fts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("eval_runs") as batch:
        batch.add_column(sa.Column("task_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("lease_token", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True)
        )

    with op.batch_alter_table("eval_results") as batch:
        batch.create_unique_constraint(
            "uq_eval_results_run_sample",
            ["run_id", "sample_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("eval_results") as batch:
        batch.drop_constraint("uq_eval_results_run_sample", type_="unique")

    with op.batch_alter_table("eval_runs") as batch:
        batch.drop_column("heartbeat_at")
        batch.drop_column("lease_token")
        batch.drop_column("task_id")
