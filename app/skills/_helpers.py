"""Skills 共用的同步数据访问 helper。

Skill 在同步上下文执行（Agent 用 to_thread 调用），所以用同步 DB session。
"""

from sqlalchemy import select

from app.database import SyncSessionLocal
from app.models.document import Document, DocumentChunk


def fetch_document_text(user_id: int, document_id: int) -> str:
    """取某文档所有分块拼成的全文，带归属校验。找不到返回空串。"""
    with SyncSessionLocal() as db:
        doc = db.get(Document, document_id)
        if doc is None or doc.user_id != user_id:
            return ""
        rows = db.execute(
            select(DocumentChunk.content)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).scalars()
        return "\n".join(rows)


def fetch_user_documents(user_id: int) -> list[dict]:
    """列出用户所有已完成的文档，供报告等技能选材。"""
    with SyncSessionLocal() as db:
        rows = db.execute(
            select(Document.id, Document.filename).where(
                Document.user_id == user_id
            )
        ).all()
        return [{"id": r[0], "filename": r[1]} for r in rows]
