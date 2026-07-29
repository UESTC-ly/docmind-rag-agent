"""Add page, paragraph, and character source locations to document chunks.

Revision ID: 0010_v320_chunk_source_locations
Revises: 0009_v320_eval_subset_reruns
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_v320_chunk_source_locations"
down_revision: str | Sequence[str] | None = "0009_v320_eval_subset_reruns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Keep legacy chunks valid while new ingestion writes precise locators."""

    with op.batch_alter_table("document_chunks") as batch:
        batch.add_column(sa.Column("page_start", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("page_end", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("paragraph_start", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("paragraph_end", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("char_start", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("char_end", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("locator_version", sa.String(length=32), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("document_chunks") as batch:
        batch.drop_column("locator_version")
        batch.drop_column("char_end")
        batch.drop_column("char_start")
        batch.drop_column("paragraph_end")
        batch.drop_column("paragraph_start")
        batch.drop_column("page_end")
        batch.drop_column("page_start")
