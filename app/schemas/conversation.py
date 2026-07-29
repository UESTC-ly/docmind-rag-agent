from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import MessageRole


class ChatRequest(BaseModel):
    """问答请求。conversation_id 为空则新建会话；document_id 限定只查某篇文档。"""

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    document_id: int | None = None


class Source(BaseModel):
    """RAG 来源引用。"""

    document_id: int
    chunk_index: int
    citation_id: str | None = None
    content: str
    score: float
    pipeline_id: str | None = None
    pipeline_fingerprint: str | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    dense_rank: int | None = None
    dense_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    fused_rank: int | None = None
    lexical_score: float | None = None
    local_rerank_score: float | None = None
    rerank_score: float | None = None
    reranker: str | None = None
    final_rank: int | None = None
    document_name: str | None = None
    source_uri: str | None = None
    source_version: str | None = None
    authority: str | None = None
    source_status: str = "unknown"
    effective_from: str | None = None
    effective_to: str | None = None
    jump_url: str | None = None
    document_policy_version: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    locator_version: str | None = None


class CitationClaim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    supported: bool
    lexical_overlap: float
    claim_type: str | None = None
    verdict: str | None = None
    evidence_state: str | None = None
    citation_verdicts: list[dict] = Field(default_factory=list)
    reason: str | None = None
    explicit_inference: bool | None = None
    semantically_supported: bool | None = None


class CitationReport(BaseModel):
    contract: str
    semantic_entailment_checked: bool
    claim_count: int
    supported_claim_count: int
    unsupported_claim_count: int
    citation_count: int
    valid_citation_count: int
    citation_precision: float
    citation_recall: float
    unsupported_claim_rate: float
    passed: bool
    claims: list[CitationClaim] = Field(default_factory=list)
    structural_contract: str | None = None
    structural_passed: bool | None = None
    judge_model: str | None = None
    judge_rubric_version: str | None = None
    judge_input_fingerprint: str | None = None
    judge_status: str | None = None
    judge_error: str | None = None
    groundedness: float | None = None
    faithfulness: float | None = None
    citation_correctness: float | None = None
    conflict_count: int = 0
    conflicts: list[dict] = Field(default_factory=list)
    conflicts_disclosed: bool = False
    claim_state_counts: dict[str, int] = Field(default_factory=dict)
    refused: bool = False
    delivery_action: str | None = None
    initial_verification: dict | None = None
    # Bounded policy trace: action metadata only, never query/source/model text.
    quality_interventions: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[Source] = Field(default_factory=list)
    citation_report: CitationReport


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    role: MessageRole
    content: str
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
