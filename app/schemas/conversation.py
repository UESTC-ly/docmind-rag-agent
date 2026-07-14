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
    content: str
    score: float
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


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[Source] = []


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
