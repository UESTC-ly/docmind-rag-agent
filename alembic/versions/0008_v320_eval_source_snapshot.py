"""retain raw public evaluation source snapshot fingerprints

Revision ID: 0008_v320_eval_source_snapshot
Revises: 0007_v320_document_provenance
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_v320_eval_source_snapshot"
down_revision: str | None = "0007_v320_document_provenance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "eval_datasets",
        sa.Column(
            "source_snapshot_fingerprint",
            sa.String(length=64),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("eval_datasets", "source_snapshot_fingerprint")
