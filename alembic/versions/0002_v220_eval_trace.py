"""Add evaluation retrieval trace fields.

Revision ID: 0002_v220_search
Revises: 0001_v210_baseline
Create Date: 2026-07-13

Keep transactional schema changes separate from the PostgreSQL concurrent
index revision.  If the index build is interrupted, this revision remains
fully committed and stamped, so a retry never repeats these ``ADD COLUMN``
operations.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_v220_search"
down_revision: str | Sequence[str] | None = "0001_v210_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("eval_results", sa.Column("retrieval_mode", sa.Text(), nullable=True))
    op.add_column("eval_results", sa.Column("reranker_mode", sa.Text(), nullable=True))
    op.add_column("eval_results", sa.Column("retrieval_trace", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("eval_results", "retrieval_trace")
    op.drop_column("eval_results", "reranker_mode")
    op.drop_column("eval_results", "retrieval_mode")
