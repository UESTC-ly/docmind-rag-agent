"""评估模块的 Pydantic 请求/响应 Schema。"""

from datetime import datetime

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


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    document_id: int
    created_at: datetime
    sample_count: int = 0  # 由 service 层填入


# ── 评估运行 ─────────────────────────────────────────────────────────────────

class RunCreateRequest(BaseModel):
    """触发一次评估运行。"""

    dataset_id: int


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
    faithfulness_score: float | None = None
    answer_relevancy_score: float | None = None
    retrieved_chunk_ids: str | None = None
    retrieval_mode: str | None = None
    reranker_mode: str | None = None
    retrieval_trace: str | None = None
    generated_answer: str | None = None
