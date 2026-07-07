"""文档业务逻辑：保存文件、建记录、派发解析任务、查询、删除。"""

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document, DocumentStatus
from app.services import vector_store
from app.tasks.document_tasks import process_document

ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}


async def create_document(
    db: AsyncSession, user_id: int, file: UploadFile
) -> Document:
    """保存上传文件到磁盘，建记录，派发 Celery 解析任务。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型 {suffix}，支持：{', '.join(ALLOWED_SUFFIXES)}",
        )

    # 落盘，文件名用 uuid 防冲突，保留原扩展名
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    content = await file.read()
    saved_path.write_bytes(content)

    doc = Document(
        user_id=user_id,
        filename=file.filename or saved_path.name,
        file_path=str(saved_path),
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # 派发异步任务（.delay 把任务丢进 Redis 队列，立即返回，不阻塞请求）
    process_document.delay(doc.id)
    return doc


async def list_documents(db: AsyncSession, user_id: int) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document(
    db: AsyncSession, user_id: int, document_id: int
) -> Document:
    """查单个文档，校验归属权，防止越权访问别人的文档。"""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == user_id
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在"
        )
    return doc


async def delete_document(
    db: AsyncSession, user_id: int, document_id: int
) -> None:
    """删文档：PG 记录（级联删 chunks）+ Qdrant 向量。"""
    doc = await get_document(db, user_id, document_id)
    vector_store.delete_document(document_id)  # 先删向量
    await db.delete(doc)  # 再删 PG（chunks 级联删除）
