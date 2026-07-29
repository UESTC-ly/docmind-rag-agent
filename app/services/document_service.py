"""Document persistence, source navigation, and background parsing dispatch."""

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.schemas.document import DocumentMetadataUpdate
from app.services.document_policy import freshness_status
from app.services.evidence import citation_id
from app.services import vector_store
from app.services.task_dispatcher import dispatch_document
from app.utils.file_parser import extract_text_with_locations

ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}


async def create_document(
    db: AsyncSession, user_id: int, file: UploadFile
) -> Document:
    """保存上传文件、建记录，并派发 Celery/桌面本地解析任务。"""
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

    # The background worker owns a separate synchronous connection.  Commit
    # before dispatch so both Celery and the desktop thread executor can see
    # the row immediately; relying on the request dependency's later commit
    # creates a race where a fast local task observes "document not found".
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        saved_path.unlink(missing_ok=True)
        raise

    try:
        dispatch_document(doc.id)
    except Exception as exc:  # broker/local executor unavailable
        doc.status = DocumentStatus.FAILED
        doc.error_message = "文档处理后台任务派发失败"
        await db.commit()
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档处理后台任务暂不可用",
        ) from exc
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


async def update_document_metadata(
    db: AsyncSession,
    user_id: int,
    document_id: int,
    metadata: DocumentMetadataUpdate,
) -> Document:
    """Update source/version metadata and atomically supersede an older copy."""

    document = await get_document(db, user_id, document_id)
    previous = None
    if metadata.supersedes_document_id is not None:
        if metadata.supersedes_document_id == document.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="文档不能替代自身",
            )
        previous = await get_document(
            db,
            user_id,
            metadata.supersedes_document_id,
        )
        if previous.supersedes_document_id == document.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="文档版本关系不能形成环",
            )

    for field, value in metadata.model_dump(exclude_unset=True).items():
        setattr(document, field, value)
    if previous is not None:
        previous.source_status = "superseded"
    await db.flush()
    await db.refresh(document)
    return document


async def get_document_chunk(
    db: AsyncSession,
    user_id: int,
    document_id: int,
    chunk_index: int,
) -> dict:
    """Resolve one stable citation ID to its exact stored source chunk."""

    document = await get_document(db, user_id, document_id)
    result = await db.execute(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document.id,
            DocumentChunk.chunk_index == chunk_index,
        )
    )
    chunk = result.scalar_one_or_none()
    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档片段不存在",
        )
    source_excerpt, highlight_start, highlight_end = _source_excerpt(document, chunk)
    return {
        "document_id": document.id,
        "document_name": document.filename,
        "chunk_index": chunk.chunk_index,
        "citation_id": citation_id(document.id, chunk.chunk_index),
        "content": chunk.content,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "paragraph_start": chunk.paragraph_start,
        "paragraph_end": chunk.paragraph_end,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "locator_version": chunk.locator_version,
        "source_excerpt": source_excerpt,
        "highlight_start": highlight_start,
        "highlight_end": highlight_end,
        "source_uri": document.source_uri,
        "source_version": document.source_version,
        "authority": document.authority,
        "source_status": freshness_status(document),
        "effective_from": document.effective_from,
        "effective_to": document.effective_to,
        "jump_url": f"/documents/{document.id}/chunks/{chunk.chunk_index}",
    }


def _source_excerpt(document: Document, chunk: DocumentChunk) -> tuple[str, int, int]:
    """Return an excerpt whose offsets safely identify this chunk's text.

    The canonical extractor is deliberately rerun only when a source span is
    available.  It lets the browser place a marked range inside surrounding
    original text.  Old chunks, moved files, and extraction failures fall back
    to the stored chunk itself rather than presenting invented coordinates.
    """

    if chunk.char_start is None or chunk.char_end is None:
        return chunk.content, 0, len(chunk.content)
    try:
        parsed = extract_text_with_locations(document.file_path)
    except (OSError, ValueError, RuntimeError):
        return chunk.content, 0, len(chunk.content)

    start = max(0, min(chunk.char_start, len(parsed.text)))
    end = max(start, min(chunk.char_end, len(parsed.text)))
    if not parsed.text or start == end:
        return chunk.content, 0, len(chunk.content)
    if parsed.text[start:end] != chunk.content:
        # A parser upgrade or a modified source file must not make us highlight
        # a different span while claiming it is the persisted evidence chunk.
        return chunk.content, 0, len(chunk.content)

    # Keep enough surrounding text to make the highlighted span meaningful,
    # but do not return the entire original document through a chunk endpoint.
    padding = 360
    excerpt_start = max(0, start - padding)
    excerpt_end = min(len(parsed.text), end + padding)
    excerpt = parsed.text[excerpt_start:excerpt_end]
    highlight_start = start - excerpt_start
    highlight_end = end - excerpt_start
    return excerpt, highlight_start, highlight_end
