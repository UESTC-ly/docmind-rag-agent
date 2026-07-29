"""文档处理异步任务。

流程：读文件 → 提取文本 → 分块 → 存 PG → 向量化 → 写 Qdrant → 标记完成。
任何一步出错都把文档标记为 FAILED 并记录原因，绝不静默吞错。
"""

import hashlib
import json

from app.celery_app import celery_app
from app.config import settings
from app.database import SyncSessionLocal
from app.models.document import Document, DocumentChunk, DocumentStatus

# 必须导入 User 模型——即使这里用不到它，也要让 SQLAlchemy 把 users 表
# 注册进元数据，否则 Document.user_id 的外键解析会失败导致 worker 崩溃
from app.models.user import User  # noqa: F401
from app.services.embedding_service import embed_texts
from app.services.vector_store import upsert_chunks
from app.utils.file_parser import (
    extract_text_with_locations,
    split_text_with_locations,
)


@celery_app.task(name="process_document")
def process_document(document_id: int) -> dict:
    """解析并向量化一个文档。参数只传 id，任务内自己查库。"""
    db = SyncSessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            return {"status": "error", "reason": "document not found"}

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        # 1. Extract normalized text plus coordinates. New chunks retain their
        # exact source span; legacy chunks without coordinates remain readable.
        parsed = extract_text_with_locations(doc.file_path)
        if not parsed.text.strip():
            raise ValueError("文档为空或无法提取文本")

        # 2. Split without losing page/paragraph/character provenance.
        located_chunks = split_text_with_locations(
            parsed,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not located_chunks:
            raise ValueError("分块结果为空")
        chunks = [chunk.content for chunk in located_chunks]

        # 3. 存分块到 PG
        db.add_all(
            [
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    paragraph_start=chunk.paragraph_start,
                    paragraph_end=chunk.paragraph_end,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    locator_version=chunk.locator_version,
                )
                for idx, chunk in enumerate(located_chunks)
            ]
        )

        # 4. 向量化 + 写 Qdrant
        vectors = embed_texts(chunks)
        upsert_chunks(
            document_id=doc.id,
            user_id=doc.user_id,
            chunks=chunks,
            vectors=vectors,
        )

        # 5. 标记完成
        doc.status = DocumentStatus.COMPLETED
        doc.chunk_count = len(chunks)
        doc.content_fingerprint = hashlib.sha256(
            json.dumps(
                chunks,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        db.commit()
        return {"status": "ok", "chunks": len(chunks)}

    except Exception as exc:  # noqa: BLE001 - 顶层任务需兜底所有异常
        db.rollback()
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(exc)[:2000]
            db.commit()
        return {"status": "error", "reason": str(exc)}
    finally:
        db.close()
