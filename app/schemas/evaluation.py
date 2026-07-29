"""评估模块的 Pydantic 请求/响应 Schema。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.evaluation import RunStatus


# ── 数据集 ──────────────────────────────────────────────────────────────────

class DatasetGenerateRequest(BaseModel):
    """触发 LLM 从文档自动生成评估数据集。"""

    document_id: int
    name: str = Field(min_length=1, max_length=200)
    sample_count: int = Field(default=10, ge=3, le=50)


class SampleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: str  # JSON 字符串，前端自行解析
    external_id: str | None = None
    document_qrels: str | None = None
    chunk_qrels: str | None = None
    answerable: bool = True
    expected_claims: str | None = None
    expected_citations: str | None = None
    temporal_labels: str | None = None
    conflict_labels: str | None = None
    fact_inference_labels: str | None = None
    difficulty: str | None = None
    slice_tags: str | None = None


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    document_id: int
    created_at: datetime
    sample_count: int = 0  # 由 service 层填入
    source_name: str | None = None
    source_uri: str | None = None
    source_version: str | None = None
    license_name: str | None = None
    split: str | None = None
    corpus_fingerprint: str | None = None
    source_snapshot_fingerprint: str | None = None
    language: str | None = None
    domain: str | None = None
    task_type: str | None = None
    label_source: str = "synthetic"
    release_eligible: bool = False


# ── 评估运行 ─────────────────────────────────────────────────────────────────

class RunCreateRequest(BaseModel):
    """触发一次评估运行。"""

    dataset_id: int
    pipeline_id: str = Field(default="configured", min_length=1, max_length=128)
    baseline_run_id: int | None = Field(default=None, ge=1)
    run_label: str | None = Field(default=None, min_length=1, max_length=200)


class ExperimentCreateRequest(BaseModel):
    """Run the same dataset against several immutable pipeline snapshots."""

    dataset_id: int
    pipeline_ids: list[str] = Field(min_length=2, max_length=8)
    baseline_pipeline_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    name: str | None = Field(default=None, min_length=1, max_length=200)


class BadcaseRerunRequest(BaseModel):
    """Run only selected results for diagnosis, never release promotion."""

    sample_ids: list[int] = Field(min_length=1, max_length=200)
    pipeline_id: str | None = Field(default=None, min_length=1, max_length=128)
    run_label: str | None = Field(default=None, min_length=1, max_length=200)


class PipelineResponse(BaseModel):
    id: str
    label: str
    description: str
    retriever: str
    fusion: str
    reranker: str
    context_builder: str
    top_k: int
    fingerprint: str
    spec: dict


class PipelineValidateRequest(BaseModel):
    """Validate an inline developer/plugin pipeline without executing it."""

    spec: dict
    check_runtime: bool = True


class PipelineValidationResponse(BaseModel):
    valid: bool
    fingerprint: str | None = None
    errors: list[str] = Field(default_factory=list)
    spec: dict | None = None


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    status: RunStatus
    # 聚合指标（运行中为 None）
    hit_rate: float | None = None
    mrr: float | None = None
    recall: float | None = None
    precision: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    unsupported_claim_rate: float | None = None
    map_score: float | None = None
    ndcg: float | None = None
    pipeline_id: str | None = None
    pipeline_fingerprint: str | None = None
    experiment_key: str | None = None
    run_label: str | None = None
    comparison_role: str = "standalone"
    evaluation_scope: str = "full"
    sample_filter: str | None = None
    source_run_id: int | None = None
    baseline_run_id: int | None = None
    code_revision: str | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    generation_prompt_version: str | None = None
    judge_model: str | None = None
    judge_rubric_version: str | None = None
    environment_fingerprint: str | None = None
    latency_ms: float | None = None
    estimated_cost: float | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# ── 评估明细 ─────────────────────────────────────────────────────────────────

class ResultDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sample_id: int
    hit: int
    reciprocal_rank: float
    recall_at_k: float
    precision_at_k: float
    average_precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    faithfulness_score: float | None = None
    answer_relevancy_score: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    unsupported_claim_rate: float | None = None
    citation_report: str | None = None
    retrieved_chunk_ids: str | None = None
    retrieval_mode: str | None = None
    reranker_mode: str | None = None
    retrieval_trace: str | None = None
    generated_answer: str | None = None


class MetricResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    subject_type: str
    subject_id: int
    metric_name: str
    metric_version: str
    evaluator_kind: str
    score: float | None = None
    passed: bool | None = None
    reason: str | None = None
    details: str | None = None
    created_at: datetime


class RegressionGateCreateRequest(BaseModel):
    dataset_id: int
    name: str = Field(min_length=1, max_length=200)
    metric_name: str = Field(min_length=1, max_length=128)
    metric_version: str | None = Field(default=None, min_length=1, max_length=64)
    comparison: Literal[
        "absolute_min",
        "absolute_max",
        "max_drop",
        "max_increase",
    ]
    threshold: float = Field(ge=0.0)
    severity: Literal["error", "warning"] = "error"
    slice_filter: dict | None = None
    enabled: bool = True


class RegressionGateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    name: str
    metric_name: str
    metric_version: str | None = None
    comparison: str
    threshold: float
    severity: str
    slice_filter: str | None = None
    enabled: bool
    created_at: datetime


class RegressionResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    candidate_run_id: int
    baseline_run_id: int | None = None
    gate_id: int
    metric_name: str
    baseline_score: float | None = None
    candidate_score: float | None = None
    delta: float | None = None
    passed: bool
    reason: str
    created_at: datetime


class BadcaseResponse(BaseModel):
    result_id: int
    sample_id: int
    external_id: str | None = None
    question: str
    ground_truth_answer: str
    answerable: bool
    difficulty: str | None = None
    slice_tags: str | None = None
    categories: list[str] = Field(default_factory=list)
    diagnosis: list[dict] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    relevant_chunk_ids: list[int] = Field(default_factory=list)
    retrieved_chunk_ids: list[int] = Field(default_factory=list)
    generated_answer: str | None = None
    citation_report: dict | None = None
    retrieval_trace: list[dict] = Field(default_factory=list)
    source_links: list[dict] = Field(default_factory=list)
    rerun_request: dict | None = None


class BadcaseDiffResponse(BaseModel):
    baseline_run_id: int
    candidate_run_id: int
    newly_introduced: list[dict] = Field(default_factory=list)
    fixed: list[dict] = Field(default_factory=list)
    persistent: list[dict] = Field(default_factory=list)
    unchanged_passed_count: int = 0
