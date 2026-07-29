"""Alembic migration-chain regression tests."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import create_engine, inspect, text


ROOT = Path(__file__).resolve().parents[1]
APPLICATION_TABLES = {
    "users",
    "documents",
    "document_chunks",
    "conversations",
    "messages",
    "eval_datasets",
    "eval_samples",
    "eval_runs",
    "eval_results",
    "eval_corpus_documents",
    "eval_metric_results",
    "eval_regression_gates",
    "eval_regression_results",
}


def _config(url: str, *, output_buffer: StringIO | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output_buffer)
    config.set_main_option("sqlalchemy.url", url)
    return config


def test_revision_chain_has_single_current_head() -> None:
    scripts = ScriptDirectory.from_config(_config("sqlite://"))

    assert scripts.get_heads() == ["0010_v320_chunk_source_locations"]
    assert scripts.get_revision("0010_v320_chunk_source_locations").down_revision == (
        "0009_v320_eval_subset_reruns"
    )
    assert scripts.get_revision("0009_v320_eval_subset_reruns").down_revision == (
        "0008_v320_eval_source_snapshot"
    )
    assert scripts.get_revision("0008_v320_eval_source_snapshot").down_revision == (
        "0007_v320_document_provenance"
    )
    assert scripts.get_revision("0007_v320_document_provenance").down_revision == (
        "0006_v320_evalops_contracts"
    )
    assert scripts.get_revision("0006_v320_evalops_contracts").down_revision == (
        "0005_v320_rag_experiments"
    )
    assert scripts.get_revision("0005_v320_rag_experiments").down_revision == (
        "0004_v220_eval_leases"
    )
    assert scripts.get_revision("0004_v220_eval_leases").down_revision == (
        "0003_v220_fts"
    )
    assert scripts.get_revision("0003_v220_fts").down_revision == "0002_v220_search"
    assert scripts.get_revision("0002_v220_search").down_revision == (
        "0001_v210_baseline"
    )


def test_upgrade_and_downgrade_complete_schema_on_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "migration-test.db"
    url = f"sqlite:///{database}"
    config = _config(url)

    command.upgrade(config, "head")

    engine = create_engine(url)
    inspector = inspect(engine)
    assert APPLICATION_TABLES <= set(inspector.get_table_names())
    eval_result_columns = {
        column["name"] for column in inspector.get_columns("eval_results")
    }
    assert {"retrieval_mode", "reranker_mode", "retrieval_trace"} <= (
        eval_result_columns
    )
    eval_run_columns = {column["name"] for column in inspector.get_columns("eval_runs")}
    assert {"task_id", "lease_token", "heartbeat_at"} <= eval_run_columns
    assert {
        "pipeline_id",
        "pipeline_spec",
        "pipeline_fingerprint",
        "citation_precision",
        "citation_recall",
        "unsupported_claim_rate",
        "map_score",
        "ndcg",
        "experiment_key",
        "baseline_run_id",
        "environment_fingerprint",
        "evaluation_scope",
        "sample_filter",
        "source_run_id",
    } <= eval_run_columns
    assert {
        "citation_precision",
        "citation_recall",
        "unsupported_claim_rate",
        "citation_report",
        "average_precision_at_k",
        "ndcg_at_k",
    } <= eval_result_columns
    assert {
        "source_name",
        "source_version",
        "corpus_fingerprint",
        "source_snapshot_fingerprint",
        "label_source",
        "release_eligible",
    } <= {
        column["name"] for column in inspector.get_columns("eval_datasets")
    }
    assert {
        "external_id",
        "document_qrels",
        "answerable",
        "slice_tags",
        "temporal_labels",
        "conflict_labels",
    } <= {
        column["name"] for column in inspector.get_columns("eval_samples")
    }
    assert {
        "source_uri",
        "source_version",
        "content_fingerprint",
        "effective_from",
        "effective_to",
        "source_status",
        "supersedes_document_id",
    } <= {
        column["name"] for column in inspector.get_columns("documents")
    }
    assert {
        "page_start",
        "page_end",
        "paragraph_start",
        "paragraph_end",
        "char_start",
        "char_end",
        "locator_version",
    } <= {
        column["name"] for column in inspector.get_columns("document_chunks")
    }
    assert "uq_eval_results_run_sample" in {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("eval_results")
    }
    assert "ix_document_chunks_content_fts" not in {
        index["name"] for index in inspector.get_indexes("document_chunks")
    }
    engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(url)
    assert APPLICATION_TABLES.isdisjoint(inspect(engine).get_table_names())
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(url)
    assert APPLICATION_TABLES <= set(inspect(engine).get_table_names())
    engine.dispose()


def test_existing_v210_schema_can_be_adopted_without_recreating_tables(
    tmp_path: Path,
) -> None:
    """Model the documented stamp path for a pre-Alembic v2.1 database."""
    database = tmp_path / "existing-v210.db"
    url = f"sqlite:///{database}"
    config = _config(url)

    # Revision 0001 is the immutable v2.1 create_all schema.  Removing only
    # Alembic's version table reproduces an existing pre-migration database.
    command.upgrade(config, "0001_v210_baseline")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))
    assert "users" in inspect(engine).get_table_names()
    assert "alembic_version" not in inspect(engine).get_table_names()
    engine.dispose()

    command.stamp(config, "0001_v210_baseline")
    command.upgrade(config, "head")

    engine = create_engine(url)
    inspector = inspect(engine)
    assert APPLICATION_TABLES <= set(inspector.get_table_names())
    eval_result_columns = {
        column["name"] for column in inspector.get_columns("eval_results")
    }
    assert {"retrieval_mode", "reranker_mode", "retrieval_trace"} <= (
        eval_result_columns
    )
    assert {"task_id", "lease_token", "heartbeat_at"} <= {
        column["name"] for column in inspector.get_columns("eval_runs")
    }
    assert {"pipeline_id", "pipeline_spec", "pipeline_fingerprint"} <= {
        column["name"] for column in inspector.get_columns("eval_runs")
    }
    assert {
        "map_score",
        "ndcg",
        "experiment_key",
        "baseline_run_id",
        "evaluation_scope",
        "sample_filter",
        "source_run_id",
    } <= {
        column["name"] for column in inspector.get_columns("eval_runs")
    }
    engine.dispose()


def test_postgresql_offline_sql_contains_concurrent_fts_index() -> None:
    output = StringIO()
    command.upgrade(
        _config(
            "postgresql+asyncpg://user:pass@localhost/docmind", output_buffer=output
        ),
        "head",
        sql=True,
    )

    sql = output.getvalue()
    assert "CREATE TABLE document_chunks" in sql
    assert "ALTER TABLE eval_results ADD COLUMN retrieval_mode TEXT" in sql
    assert "ALTER TABLE eval_results ADD COLUMN reranker_mode TEXT" in sql
    assert "ALTER TABLE eval_results ADD COLUMN retrieval_trace TEXT" in sql
    assert "CREATE INDEX CONCURRENTLY ix_document_chunks_content_fts" in sql
    assert "ix_document_chunks_content_fts" in sql
    assert "to_tsvector('simple'::regconfig, content)" in sql
    assert sql.index("ALTER TABLE eval_results ADD COLUMN retrieval_trace TEXT") < (
        sql.index("CREATE INDEX CONCURRENTLY ix_document_chunks_content_fts")
    )


def _load_fts_revision():
    path = ROOT / "alembic" / "versions" / "0003_v220_fts_index.py"
    spec = importlib.util.spec_from_file_location("docmind_fts_revision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeMigrationResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def one_or_none(self):
        return self.row


class _FakeMigrationBind:
    dialect = SimpleNamespace(name="postgresql")

    def __init__(self, row):
        self.row = row

    def execute(self, _statement, _parameters):
        return _FakeMigrationResult(self.row)


class _FakeMigrationContext:
    as_sql = False

    @contextmanager
    def autocommit_block(self):
        yield


class _FakeMigrationOp:
    def __init__(self, row):
        self.bind = _FakeMigrationBind(row)
        self.context = _FakeMigrationContext()
        self.statements: list[str] = []

    def get_bind(self):
        return self.bind

    def get_context(self):
        return self.context

    def execute(self, statement):
        self.statements.append(str(statement))


def test_concurrent_fts_revision_recovers_partial_postgresql_outcomes() -> None:
    revision = _load_fts_revision()

    expected_state = {
        "indisvalid": True,
        "definition": (
            "CREATE INDEX ix_document_chunks_content_fts ON public.document_chunks "
            "USING gin (to_tsvector('simple'::regconfig, content))"
        ),
        "access_method": "gin",
        "table_name": "document_chunks",
        "predicate": None,
        "expression": "to_tsvector('simple'::regconfig, content)",
        "indnatts": 1,
        "indnkeyatts": 1,
        "indkey": "0",
    }

    invalid = _FakeMigrationOp(
        {
            **expected_state,
            "indisvalid": False,
            "definition": "CREATE INDEX interrupted_build",
        }
    )
    revision.op = invalid
    revision.upgrade()
    assert invalid.statements == [
        "DROP INDEX CONCURRENTLY IF EXISTS ix_document_chunks_content_fts",
        revision._CREATE_INDEX,
    ]

    expected = _FakeMigrationOp(expected_state)
    revision.op = expected
    revision.upgrade()
    assert expected.statements == []

    drift = _FakeMigrationOp(
        {
            **expected_state,
            "indisvalid": True,
            "definition": (
                "CREATE INDEX ix_document_chunks_content_fts ON public.document_chunks "
                "USING btree (content)"
            ),
            "access_method": "btree",
            "expression": None,
            "indkey": "2",
        }
    )
    revision.op = drift
    with pytest.raises(RuntimeError, match="unexpected valid definition"):
        revision.upgrade()

    partial = _FakeMigrationOp(
        {
            **expected_state,
            "definition": f"{expected_state['definition']} WHERE (document_id > 0)",
            "predicate": "(document_id > 0)",
        }
    )
    revision.op = partial
    with pytest.raises(RuntimeError, match="unexpected valid definition"):
        revision.upgrade()

    extra_expression = _FakeMigrationOp(
        {
            **expected_state,
            "definition": (
                "CREATE INDEX ix_document_chunks_content_fts ON public.document_chunks "
                "USING gin (to_tsvector('simple'::regconfig, content), "
                "to_tsvector('simple'::regconfig, filename))"
            ),
            "expression": (
                "to_tsvector('simple'::regconfig, content), "
                "to_tsvector('simple'::regconfig, filename)"
            ),
            "indnatts": 2,
            "indnkeyatts": 2,
            "indkey": "0 0",
        }
    )
    revision.op = extra_expression
    with pytest.raises(RuntimeError, match="unexpected valid definition"):
        revision.upgrade()


def test_concurrent_index_isolated_from_transactional_schema_revision() -> None:
    schema_source = (
        ROOT / "alembic" / "versions" / "0002_v220_eval_trace.py"
    ).read_text(encoding="utf-8")
    env_source = (ROOT / "alembic" / "env.py").read_text(encoding="utf-8")

    assert "autocommit_block" not in schema_source
    assert "CREATE INDEX" not in schema_source
    assert '"transaction_per_migration": True' in env_source


def test_application_startup_does_not_create_schema() -> None:
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert "create_all" not in source
    assert "Base.metadata" not in source
