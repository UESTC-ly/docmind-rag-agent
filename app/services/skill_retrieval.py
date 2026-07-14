"""同步 Skills 共用的 RAG 检索入口。

Agent 主循环在线程池里执行同步 Skill，因此不能直接复用需要 AsyncSession 的
``rag_service.retrieve``。本模块复用同一个候选融合与 reranker，只把数据库 I/O
换成同步 Session。Evaluation runner 也调用本入口，避免形成第三套检索逻辑。
"""

from sqlalchemy.orm import Session

from app.config import settings
from app.services.embedding_service import embed_query
from app.services.retrieval import finalize_retrieval, keyword_search_sync
from app.services.vector_store import search


def retrieve_for_skill(
    user_id: int,
    query: str,
    top_k: int | None = None,
    document_id: int | None = None,
    db: Session | None = None,
) -> list[dict]:
    """按当前 RAG 配置为同步 Skill 检索用户文档片段。"""
    limit = top_k or settings.retrieval_top_k
    query_vector = embed_query(query)
    dense_limit = max(limit, min(settings.dense_candidates, settings.reranker_candidate_limit))
    dense_hits = search(query_vector, user_id, dense_limit, document_id)

    keyword_hits: list[dict] = []
    if settings.retrieval_mode == "hybrid":
        keyword_hits = keyword_search_sync(
            query,
            user_id,
            document_id,
            min(settings.keyword_candidates, settings.reranker_candidate_limit),
            db=db,
        )
    return finalize_retrieval(query, dense_hits, keyword_hits, limit)
