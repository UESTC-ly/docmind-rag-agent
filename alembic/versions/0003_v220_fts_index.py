"""Build the PostgreSQL full-text index without blocking document writes.

Revision ID: 0003_v220_fts
Revises: 0002_v220_search
Create Date: 2026-07-13

``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction.  This dedicated
revision is deliberately retry-safe around PostgreSQL's partial outcomes:

* a valid index with the expected definition is accepted and the revision is
  stamped (covers a crash after index creation but before the version update);
* an invalid index left by an interrupted build is dropped concurrently and
  rebuilt;
* a valid index with a different definition fails visibly as schema drift.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0003_v220_fts"
down_revision: str | Sequence[str] | None = "0002_v220_search"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_FTS_INDEX = "ix_document_chunks_content_fts"
_EXPECTED_EXPRESSION = "to_tsvector('simple'::regconfig, content)"
_CREATE_INDEX = (
    f"CREATE INDEX CONCURRENTLY {_FTS_INDEX} "
    "ON document_chunks USING gin "
    f"({_EXPECTED_EXPRESSION})"
)
_INDEX_QUERY = sa.text(
    """
    SELECT
        index_state.indisvalid,
        pg_get_indexdef(index_state.indexrelid) AS definition,
        access_method.amname AS access_method,
        table_class.relname AS table_name,
        pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate,
        pg_get_expr(index_state.indexprs, index_state.indrelid) AS expression,
        index_state.indnatts,
        index_state.indnkeyatts,
        index_state.indkey::text AS indkey
    FROM pg_class AS index_class
    JOIN pg_namespace AS namespace ON namespace.oid = index_class.relnamespace
    JOIN pg_index AS index_state ON index_state.indexrelid = index_class.oid
    JOIN pg_class AS table_class ON table_class.oid = index_state.indrelid
    JOIN pg_am AS access_method ON access_method.oid = index_class.relam
    WHERE index_class.relname = :index_name
      AND namespace.nspname = current_schema()
    """
)


def _normalize_expression(expression: str) -> str:
    return " ".join(expression.lower().split())


def _expected_state(state: dict) -> bool:
    expression = _normalize_expression(str(state.get("expression") or ""))
    expected = _normalize_expression(_EXPECTED_EXPRESSION)
    return (
        state.get("access_method") == "gin"
        and state.get("table_name") == "document_chunks"
        and state.get("predicate") is None
        and int(state.get("indnatts") or 0) == 1
        and int(state.get("indnkeyatts") or 0) == 1
        and str(state.get("indkey") or "").strip() == "0"
        and expression in {expected, f"({expected})"}
    )


def _index_state(bind) -> dict | None:
    row = bind.execute(_INDEX_QUERY, {"index_name": _FTS_INDEX}).mappings().one_or_none()
    if row is None:
        return None
    return dict(row)


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    context = op.get_context()
    if context.as_sql:
        with context.autocommit_block():
            op.execute(_CREATE_INDEX)
        return

    state = _index_state(op.get_bind())
    if state is not None and bool(state["indisvalid"]):
        if _expected_state(state):
            return
        raise RuntimeError(
            f"{_FTS_INDEX} already exists with an unexpected valid definition: "
            f"{state['definition']}"
        )

    with context.autocommit_block():
        if state is not None:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_FTS_INDEX}")
        op.execute(_CREATE_INDEX)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_FTS_INDEX}")
