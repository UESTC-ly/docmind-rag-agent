"""Add document provenance and freshness policy metadata.

Revision ID: 0007_v320_document_provenance
Revises: 0006_v320_evalops_contracts
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_v320_document_provenance"
down_revision: str | Sequence[str] | None = "0006_v320_evalops_contracts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("source_uri", sa.Text(), nullable=True))
        batch.add_column(sa.Column("source_version", sa.Text(), nullable=True))
        batch.add_column(sa.Column("authority", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("content_fingerprint", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "source_status",
                sa.String(length=32),
                nullable=False,
                server_default="unknown",
            )
        )
        batch.add_column(
            sa.Column("supersedes_document_id", sa.Integer(), nullable=True)
        )
        batch.create_foreign_key(
            "fk_documents_supersedes_document_id",
            "documents",
            ["supersedes_document_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            "ix_documents_supersedes_document_id",
            ["supersedes_document_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.drop_index("ix_documents_supersedes_document_id")
        batch.drop_constraint(
            "fk_documents_supersedes_document_id",
            type_="foreignkey",
        )
        batch.drop_column("supersedes_document_id")
        batch.drop_column("source_status")
        batch.drop_column("effective_to")
        batch.drop_column("effective_from")
        batch.drop_column("content_fingerprint")
        batch.drop_column("authority")
        batch.drop_column("source_version")
        batch.drop_column("source_uri")
