"""评估模块数据模型。

表结构：
  EvalDataset  ── 一个评估数据集（绑定某篇文档，含若干样本）
  EvalSample   ── 数据集里的一条样本：问题 + 标准答案 + 相关 chunk 索引
  EvalRun      ── 一次完整的评估运行，记录聚合分数
  EvalResult   ── 某次运行里单条样本的检索/生成明细分数
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalDataset(Base):
    __tablename__ = "eval_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(Text)  # 数据集名称，如 "技术手册v1-eval"
    # Public benchmark provenance. Existing synthetic datasets remain readable,
    # but only release_eligible datasets may be used by release gates.
    source_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    license_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    split: Mapped[str | None] = mapped_column(Text, nullable=True)
    corpus_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    source_snapshot_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    transform_spec: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    domain: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    label_source: Mapped[str] = mapped_column(
        String(32),
        default="synthetic",
    )
    release_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    samples: Mapped[list["EvalSample"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    runs: Mapped[list["EvalRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    corpus_documents: Mapped[list["EvalCorpusDocument"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class EvalCorpusDocument(Base):
    """A versioned public document included in an evaluation corpus snapshot."""

    __tablename__ = "eval_corpus_documents"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "public_id",
            name="uq_eval_corpus_documents_dataset_public_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        index=True,
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    public_id: Mapped[str] = mapped_column(Text)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    authority: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_status: Mapped[str] = mapped_column(
        String(32),
        default="current",
    )
    supersedes_public_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    dataset: Mapped["EvalDataset"] = relationship(
        back_populates="corpus_documents"
    )


class EvalSample(Base):
    """一条评估样本（三元组）。

    relevant_chunk_ids 存 JSON 字符串，如 "[0, 3, 7]"，
    表示该问题的正确答案来自文档的第 0、3、7 块。
    """

    __tablename__ = "eval_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    ground_truth_answer: Mapped[str] = mapped_column(Text)
    relevant_chunk_ids: Mapped[str] = mapped_column(Text)  # JSON: [int, ...]
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_qrels: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_qrels: Mapped[str | None] = mapped_column(Text, nullable=True)
    answerable: Mapped[bool] = mapped_column(Boolean, default=True)
    expected_claims: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_citations: Mapped[str | None] = mapped_column(Text, nullable=True)
    temporal_labels: Mapped[str | None] = mapped_column(Text, nullable=True)
    conflict_labels: Mapped[str | None] = mapped_column(Text, nullable=True)
    fact_inference_labels: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    difficulty: Mapped[str | None] = mapped_column(String(32), nullable=True)
    slice_tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dataset: Mapped["EvalDataset"] = relationship(back_populates="samples")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )


class EvalRun(Base):
    """一次评估运行。聚合分数在全部样本跑完后写入。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.PENDING, index=True
    )
    # 聚合检索指标（完成后填入）
    hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 聚合生成指标
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    unsupported_claim_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    map_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Exact RAG experiment snapshot.  The fingerprint includes component
    # parameters and the embedding/index compatibility manifest.
    pipeline_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline_spec: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    experiment_key: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    run_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    comparison_role: Mapped[str] = mapped_column(
        String(32),
        default="standalone",
    )
    # Full runs are comparable release evidence. Subset runs are diagnostic
    # reruns and must never silently enter pipeline promotion/regression gates.
    evaluation_scope: Mapped[str] = mapped_column(
        String(16),
        default="full",
    )
    sample_filter: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    baseline_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Environment/model metadata needed to reproduce model-judged results.
    code_revision: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation_prompt_version: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    judge_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    judge_rubric_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Background workers claim a run with a random lease token. ``task_id`` is
    # stable across a Celery redelivery, while ``lease_token`` changes on every
    # claim so an old worker cannot commit after a replacement has taken over.
    task_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    dataset: Mapped["EvalDataset"] = relationship(back_populates="runs")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    metric_results: Mapped[list["EvalMetricResult"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class EvalResult(Base):
    """单条样本在某次运行中的明细分数。"""

    __tablename__ = "eval_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "sample_id",
            name="uq_eval_results_run_sample",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    sample_id: Mapped[int] = mapped_column(
        ForeignKey("eval_samples.id", ondelete="CASCADE"), index=True
    )
    # 检索明细（该样本的）
    hit: Mapped[int] = mapped_column(Integer, default=0)  # 0 or 1
    reciprocal_rank: Mapped[float] = mapped_column(Float, default=0.0)
    recall_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    precision_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    average_precision_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    ndcg_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    # 生成明细
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    unsupported_claim_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    citation_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 检索器实际返回的 chunk_index 列表（JSON），便于调试
    retrieved_chunk_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 可复现检索轨迹：策略、实际 reranker，以及不含正文的候选/最终排序 JSON。
    retrieval_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    reranker_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_trace: Mapped[str | None] = mapped_column(Text, nullable=True)
    # LLM 生成的答案（评估时实际跑一遍 RAG）
    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["EvalRun"] = relationship(back_populates="results")
    sample: Mapped["EvalSample"] = relationship(back_populates="results")


class EvalMetricResult(Base):
    """Versioned metric result for either a full run or one sample."""

    __tablename__ = "eval_metric_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "subject_type",
            "subject_id",
            "metric_name",
            "metric_version",
            name="uq_eval_metric_result_subject_metric",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        index=True,
    )
    # subject_id=0 identifies the aggregate run; sample rows use EvalSample.id.
    subject_type: Mapped[str] = mapped_column(String(16), default="sample")
    subject_id: Mapped[int] = mapped_column(Integer, default=0)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_version: Mapped[str] = mapped_column(String(64), default="v1")
    evaluator_kind: Mapped[str] = mapped_column(
        String(32),
        default="deterministic",
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    run: Mapped["EvalRun"] = relationship(back_populates="metric_results")


class EvalRegressionGate(Base):
    """A configurable absolute or baseline-relative release gate."""

    __tablename__ = "eval_regression_gates"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "name",
            name="uq_eval_regression_gates_dataset_name",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(Text)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # absolute_min, absolute_max, max_drop, max_increase
    comparison: Mapped[str] = mapped_column(String(32))
    threshold: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16), default="error")
    slice_filter: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class EvalRegressionResult(Base):
    """One persisted gate verdict for a candidate run."""

    __tablename__ = "eval_regression_results"
    __table_args__ = (
        UniqueConstraint(
            "candidate_run_id",
            "gate_id",
            name="uq_eval_regression_results_candidate_gate",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        index=True,
    )
    baseline_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    gate_id: Mapped[int] = mapped_column(
        ForeignKey("eval_regression_gates.id", ondelete="CASCADE"),
        index=True,
    )
    metric_name: Mapped[str] = mapped_column(String(128))
    baseline_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
