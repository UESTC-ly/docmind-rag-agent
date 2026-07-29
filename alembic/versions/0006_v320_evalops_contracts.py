"""Add versioned public evaluation and regression contracts.

Revision ID: 0006_v320_evalops_contracts
Revises: 0005_v320_rag_experiments
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_v320_evalops_contracts"
down_revision: str | Sequence[str] | None = "0005_v320_rag_experiments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("eval_datasets") as batch:
        batch.add_column(sa.Column("source_name", sa.Text(), nullable=True))
        batch.add_column(sa.Column("source_uri", sa.Text(), nullable=True))
        batch.add_column(sa.Column("source_version", sa.Text(), nullable=True))
        batch.add_column(sa.Column("license_name", sa.Text(), nullable=True))
        batch.add_column(sa.Column("split", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("corpus_fingerprint", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("transform_spec", sa.Text(), nullable=True))
        batch.add_column(sa.Column("language", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("domain", sa.Text(), nullable=True))
        batch.add_column(sa.Column("task_type", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "label_source",
                sa.String(length=32),
                nullable=False,
                server_default="synthetic",
            )
        )
        batch.add_column(
            sa.Column(
                "release_eligible",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch.add_column(sa.Column("metadata_json", sa.Text(), nullable=True))

    with op.batch_alter_table("eval_samples") as batch:
        batch.add_column(sa.Column("external_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("document_qrels", sa.Text(), nullable=True))
        batch.add_column(sa.Column("chunk_qrels", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "answerable",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch.add_column(sa.Column("expected_claims", sa.Text(), nullable=True))
        batch.add_column(sa.Column("expected_citations", sa.Text(), nullable=True))
        batch.add_column(sa.Column("temporal_labels", sa.Text(), nullable=True))
        batch.add_column(sa.Column("conflict_labels", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("fact_inference_labels", sa.Text(), nullable=True)
        )
        batch.add_column(sa.Column("difficulty", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("slice_tags", sa.Text(), nullable=True))
        batch.add_column(sa.Column("metadata_json", sa.Text(), nullable=True))

    with op.batch_alter_table("eval_runs") as batch:
        batch.add_column(sa.Column("map_score", sa.Float(), nullable=True))
        batch.add_column(sa.Column("ndcg", sa.Float(), nullable=True))
        batch.add_column(
            sa.Column("experiment_key", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("run_label", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "comparison_role",
                sa.String(length=32),
                nullable=False,
                server_default="standalone",
            )
        )
        batch.add_column(sa.Column("baseline_run_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_eval_runs_baseline_run_id",
            "eval_runs",
            ["baseline_run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.add_column(sa.Column("code_revision", sa.Text(), nullable=True))
        batch.add_column(sa.Column("embedding_model", sa.Text(), nullable=True))
        batch.add_column(sa.Column("generation_model", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("generation_prompt_version", sa.Text(), nullable=True)
        )
        batch.add_column(sa.Column("judge_model", sa.Text(), nullable=True))
        batch.add_column(sa.Column("judge_rubric_version", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("environment_fingerprint", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("latency_ms", sa.Float(), nullable=True))
        batch.add_column(sa.Column("estimated_cost", sa.Float(), nullable=True))
        batch.create_index(
            "ix_eval_runs_experiment_key",
            ["experiment_key"],
            unique=False,
        )
        batch.create_index(
            "ix_eval_runs_baseline_run_id",
            ["baseline_run_id"],
            unique=False,
        )

    with op.batch_alter_table("eval_results") as batch:
        batch.add_column(
            sa.Column(
                "average_precision_at_k",
                sa.Float(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "ndcg_at_k",
                sa.Float(),
                nullable=False,
                server_default="0",
            )
        )

    op.create_table(
        "eval_corpus_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("source_version", sa.Text(), nullable=True),
        sa.Column("content_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("authority", sa.Text(), nullable=True),
        sa.Column(
            "source_status",
            sa.String(length=32),
            nullable=False,
            server_default="current",
        ),
        sa.Column("supersedes_public_id", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["eval_datasets.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id",
            "public_id",
            name="uq_eval_corpus_documents_dataset_public_id",
        ),
    )
    op.create_index(
        "ix_eval_corpus_documents_dataset_id",
        "eval_corpus_documents",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        "ix_eval_corpus_documents_document_id",
        "eval_corpus_documents",
        ["document_id"],
        unique=False,
    )

    op.create_table(
        "eval_metric_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column(
            "subject_type",
            sa.String(length=16),
            nullable=False,
            server_default="sample",
        ),
        sa.Column("subject_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column(
            "metric_version",
            sa.String(length=64),
            nullable=False,
            server_default="v1",
        ),
        sa.Column(
            "evaluator_kind",
            sa.String(length=32),
            nullable=False,
            server_default="deterministic",
        ),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["eval_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "subject_type",
            "subject_id",
            "metric_name",
            "metric_version",
            name="uq_eval_metric_result_subject_metric",
        ),
    )
    op.create_index(
        "ix_eval_metric_results_run_id",
        "eval_metric_results",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_eval_metric_results_metric_name",
        "eval_metric_results",
        ["metric_name"],
        unique=False,
    )

    op.create_table(
        "eval_regression_gates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("metric_version", sa.String(length=64), nullable=True),
        sa.Column("comparison", sa.String(length=32), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column(
            "severity",
            sa.String(length=16),
            nullable=False,
            server_default="error",
        ),
        sa.Column("slice_filter", sa.Text(), nullable=True),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["eval_datasets.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id",
            "name",
            name="uq_eval_regression_gates_dataset_name",
        ),
    )
    op.create_index(
        "ix_eval_regression_gates_dataset_id",
        "eval_regression_gates",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        "ix_eval_regression_gates_metric_name",
        "eval_regression_gates",
        ["metric_name"],
        unique=False,
    )

    op.create_table(
        "eval_regression_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_run_id", sa.Integer(), nullable=False),
        sa.Column("baseline_run_id", sa.Integer(), nullable=True),
        sa.Column("gate_id", sa.Integer(), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("baseline_score", sa.Float(), nullable=True),
        sa.Column("candidate_score", sa.Float(), nullable=True),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["candidate_run_id"],
            ["eval_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["baseline_run_id"],
            ["eval_runs.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["gate_id"],
            ["eval_regression_gates.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_run_id",
            "gate_id",
            name="uq_eval_regression_results_candidate_gate",
        ),
    )
    op.create_index(
        "ix_eval_regression_results_candidate_run_id",
        "eval_regression_results",
        ["candidate_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_eval_regression_results_gate_id",
        "eval_regression_results",
        ["gate_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_eval_regression_results_gate_id",
        table_name="eval_regression_results",
    )
    op.drop_index(
        "ix_eval_regression_results_candidate_run_id",
        table_name="eval_regression_results",
    )
    op.drop_table("eval_regression_results")

    op.drop_index(
        "ix_eval_regression_gates_metric_name",
        table_name="eval_regression_gates",
    )
    op.drop_index(
        "ix_eval_regression_gates_dataset_id",
        table_name="eval_regression_gates",
    )
    op.drop_table("eval_regression_gates")

    op.drop_index(
        "ix_eval_metric_results_metric_name",
        table_name="eval_metric_results",
    )
    op.drop_index(
        "ix_eval_metric_results_run_id",
        table_name="eval_metric_results",
    )
    op.drop_table("eval_metric_results")

    op.drop_index(
        "ix_eval_corpus_documents_document_id",
        table_name="eval_corpus_documents",
    )
    op.drop_index(
        "ix_eval_corpus_documents_dataset_id",
        table_name="eval_corpus_documents",
    )
    op.drop_table("eval_corpus_documents")

    with op.batch_alter_table("eval_results") as batch:
        batch.drop_column("ndcg_at_k")
        batch.drop_column("average_precision_at_k")

    with op.batch_alter_table("eval_runs") as batch:
        batch.drop_index("ix_eval_runs_baseline_run_id")
        batch.drop_index("ix_eval_runs_experiment_key")
        batch.drop_constraint(
            "fk_eval_runs_baseline_run_id",
            type_="foreignkey",
        )
        batch.drop_column("estimated_cost")
        batch.drop_column("latency_ms")
        batch.drop_column("environment_fingerprint")
        batch.drop_column("judge_rubric_version")
        batch.drop_column("judge_model")
        batch.drop_column("generation_prompt_version")
        batch.drop_column("generation_model")
        batch.drop_column("embedding_model")
        batch.drop_column("code_revision")
        batch.drop_column("baseline_run_id")
        batch.drop_column("comparison_role")
        batch.drop_column("run_label")
        batch.drop_column("experiment_key")
        batch.drop_column("ndcg")
        batch.drop_column("map_score")

    with op.batch_alter_table("eval_samples") as batch:
        batch.drop_column("metadata_json")
        batch.drop_column("slice_tags")
        batch.drop_column("difficulty")
        batch.drop_column("fact_inference_labels")
        batch.drop_column("conflict_labels")
        batch.drop_column("temporal_labels")
        batch.drop_column("expected_citations")
        batch.drop_column("expected_claims")
        batch.drop_column("answerable")
        batch.drop_column("chunk_qrels")
        batch.drop_column("document_qrels")
        batch.drop_column("external_id")

    with op.batch_alter_table("eval_datasets") as batch:
        batch.drop_column("metadata_json")
        batch.drop_column("release_eligible")
        batch.drop_column("label_source")
        batch.drop_column("task_type")
        batch.drop_column("domain")
        batch.drop_column("language")
        batch.drop_column("transform_spec")
        batch.drop_column("corpus_fingerprint")
        batch.drop_column("split")
        batch.drop_column("license_name")
        batch.drop_column("source_version")
        batch.drop_column("source_uri")
        batch.drop_column("source_name")
