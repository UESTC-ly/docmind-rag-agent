"""Add reproducible RAG pipeline and citation evaluation fields.

Revision ID: 0005_v320_rag_experiments
Revises: 0004_v220_eval_leases
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_v320_rag_experiments"
down_revision: str | Sequence[str] | None = "0004_v220_eval_leases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("eval_runs") as batch:
        batch.add_column(sa.Column("citation_precision", sa.Float(), nullable=True))
        batch.add_column(sa.Column("citation_recall", sa.Float(), nullable=True))
        batch.add_column(
            sa.Column("unsupported_claim_rate", sa.Float(), nullable=True)
        )
        batch.add_column(sa.Column("pipeline_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("pipeline_spec", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("pipeline_fingerprint", sa.Text(), nullable=True)
        )

    with op.batch_alter_table("eval_results") as batch:
        batch.add_column(sa.Column("citation_precision", sa.Float(), nullable=True))
        batch.add_column(sa.Column("citation_recall", sa.Float(), nullable=True))
        batch.add_column(
            sa.Column("unsupported_claim_rate", sa.Float(), nullable=True)
        )
        batch.add_column(sa.Column("citation_report", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("eval_results") as batch:
        batch.drop_column("citation_report")
        batch.drop_column("unsupported_claim_rate")
        batch.drop_column("citation_recall")
        batch.drop_column("citation_precision")

    with op.batch_alter_table("eval_runs") as batch:
        batch.drop_column("pipeline_fingerprint")
        batch.drop_column("pipeline_spec")
        batch.drop_column("pipeline_id")
        batch.drop_column("unsupported_claim_rate")
        batch.drop_column("citation_recall")
        batch.drop_column("citation_precision")
