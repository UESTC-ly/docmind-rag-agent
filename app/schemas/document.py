from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """文档元数据响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: DocumentStatus
    chunk_count: int
    error_message: str | None = None
    source_uri: str | None = None
    source_version: str | None = None
    authority: str | None = None
    content_fingerprint: str | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    source_status: str = "unknown"
    supersedes_document_id: int | None = None
    created_at: datetime


class DocumentMetadataUpdate(BaseModel):
    source_uri: str | None = Field(default=None, max_length=2000)
    source_version: str | None = Field(default=None, max_length=200)
    authority: str | None = Field(default=None, max_length=500)
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    source_status: Literal[
        "unknown",
        "current",
        "superseded",
        "expired",
    ] | None = None
    supersedes_document_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_effective_window(self):
        if not self.model_fields_set:
            raise ValueError("at least one document metadata field is required")
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError("effective_from must be earlier than effective_to")
        return self


class DocumentChunkResponse(BaseModel):
    document_id: int
    document_name: str
    chunk_index: int
    citation_id: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    locator_version: str | None = None
    source_excerpt: str
    highlight_start: int
    highlight_end: int
    source_uri: str | None = None
    source_version: str | None = None
    authority: str | None = None
    source_status: str
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    jump_url: str


class UploadResponse(BaseModel):
    """上传成功后返回文档 id 和当前状态，前端凭 id 轮询进度。"""

    document_id: int
    status: DocumentStatus
    message: str = "文档已上传，正在后台解析"
