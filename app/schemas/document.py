from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """文档元数据响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: DocumentStatus
    chunk_count: int
    error_message: str | None = None
    created_at: datetime


class UploadResponse(BaseModel):
    """上传成功后返回文档 id 和当前状态，前端凭 id 轮询进度。"""

    document_id: int
    status: DocumentStatus
    message: str = "文档已上传，正在后台解析"
