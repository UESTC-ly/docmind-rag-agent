"""Skills 共用的同步数据访问 helper。

Skill 在同步上下文执行（Agent 用 to_thread 调用），所以用同步 DB session。
"""

from sqlalchemy import select

from app.database import SyncSessionLocal
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.skill_retrieval import retrieve_for_skill


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
                Document.user_id == user_id,
                Document.status == DocumentStatus.COMPLETED,
            )
        ).all()
        return [{"id": r[0], "filename": r[1]} for r in rows]


def fetch_material_text(
    user_id: int,
    document_id: int | None = None,
    max_chars: int = 12000,
) -> tuple[str, list[int]]:
    """取指定文档或用户全部文档材料，返回 (文本, 使用到的文档 id 列表)。

    文件产出类技能（周报/PPT）既支持用户显式指定 document_id，也支持在未指定时汇总
    当前用户已有文档作为材料来源。所有读取都复用 fetch_document_text 的归属校验。
    """
    if document_id is not None:
        text = fetch_document_text(user_id, document_id)
        return text[:max_chars], ([document_id] if text else [])

    parts: list[str] = []
    used_ids: list[int] = []
    for doc in fetch_user_documents(user_id):
        if sum(len(p) for p in parts) >= max_chars:
            break
        text = fetch_document_text(user_id, doc["id"])
        if not text:
            continue
        used_ids.append(doc["id"])
        remaining = max_chars - sum(len(p) for p in parts)
        parts.append(f"【文档 {doc['id']}：{doc['filename']}】\n{text[:remaining]}")

    return "\n\n".join(parts)[:max_chars], used_ids


def fetch_retrieved_material(
    user_id: int,
    query: str,
    document_id: int | None = None,
    max_chars: int = 12000,
) -> tuple[str, list[int], list[dict]]:
    """通过 DocMind 的混合 RAG 链路取写作材料。

    返回 ``(带来源标记的上下文, 文档 ID, 精简来源)``。与 ``fetch_material_text``
    的全文读取不同，此函数适合报告、周报、PPT 和通用 Skill 的目标式生成。
    """
    hits = retrieve_for_skill(
        user_id=user_id,
        query=query,
        document_id=document_id,
        top_k=max(8, min(20, max_chars // 800)),
    )
    parts: list[str] = []
    source_items: list[dict] = []
    used_ids: list[int] = []
    used_chars = 0
    for hit in hits:
        content = str(hit.get("content") or "")
        if not content or used_chars >= max_chars:
            continue
        doc_id = int(hit["document_id"])
        chunk_index = int(hit.get("chunk_index", 0))
        remaining = max_chars - used_chars
        excerpt = content[:remaining]
        parts.append(f"【文档 {doc_id} · 片段 {chunk_index}】\n{excerpt}")
        used_chars += len(excerpt)
        if doc_id not in used_ids:
            used_ids.append(doc_id)
        source_items.append(
            {
                "document_id": doc_id,
                "chunk_index": chunk_index,
                "score": hit.get("score"),
            }
        )
    return "\n\n".join(parts), used_ids, source_items
