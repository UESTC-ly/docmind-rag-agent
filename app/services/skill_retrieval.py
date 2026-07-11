"""同步 Skills 共用的 RAG 检索入口。

Agent 主循环在线程池里执行同步 Skill，因此不能直接复用需要 AsyncSession 的
``rag_service.retrieve``。本模块保持同一检索语义：embedding + Qdrant 稠密召回，
hybrid 模式下再走同步 PostgreSQL 关键词召回并用 RRF 融合。
"""

from app.config import settings
from app.services.embedding_service import embed_query
from app.services.retrieval import fuse_dense_and_keyword, keyword_search_sync
from app.services.vector_store import search


def retrieve_for_skill(
    user_id: int,
    query: str,
    top_k: int | None = None,
    document_id: int | None = None,
) -> list[dict]:
    """按当前 RAG 配置为同步 Skill 检索用户文档片段。"""
    limit = top_k or settings.retrieval_top_k
    query_vector = embed_query(query)
    dense_hits = search(query_vector, user_id, limit, document_id)
    if settings.retrieval_mode != "hybrid":
        return dense_hits

    keyword_hits = keyword_search_sync(
        query,
        user_id,
        document_id,
        settings.keyword_candidates,
    )
    return fuse_dense_and_keyword(dense_hits, keyword_hits, limit)
