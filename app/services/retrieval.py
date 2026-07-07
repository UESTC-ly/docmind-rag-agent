"""多路召回（Hybrid Retrieval）。

单一稠密向量检索抓不住精确关键词（如产品型号、专有名词）。这里再加一路
关键词检索，用 RRF（Reciprocal Rank Fusion，倒数排名融合）把两路结果融合：

    RRF_score(d) = Σ_i  1 / (k + rank_i(d))

k 是平滑常数（经验值 60）。RRF 只看排名不看各路原始分数量纲，天然免归一化，
是混合检索的业界标配。

关键词检索用命中词数打分：实现简单、免额外依赖，sqlite 也能跑（便于测试）。
生产可换成 PG 全文检索或 BM25，接口不变。

注意：chunk_index 只在单文档内唯一，跨文档检索时用 (document_id, chunk_index)
作为融合身份，避免不同文档的同序号块被误合并。
"""

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document, DocumentChunk

ChunkKey = tuple[int, int]  # (document_id, chunk_index)


def reciprocal_rank_fusion(
    ranked_lists: list[list[ChunkKey]], k: int = 60, top_k: int = 5
) -> list[ChunkKey]:
    """对多个「按相关性降序的 key 列表」做 RRF 融合。

    返回融合后按 RRF 分数降序的 key 列表（去重，取前 top_k）。
    key 可以是任意可哈希标识（这里用 (document_id, chunk_index)）。
    """
    scores: dict[ChunkKey, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return ordered[:top_k]


def _tokenize(text: str) -> list[str]:
    """极简分词：英文按词、中文按单字，兼顾中英。仅用于关键词召回打分。"""
    tokens = re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())
    return [t for t in tokens if t]


async def keyword_search(
    db: AsyncSession,
    query: str,
    user_id: int,
    document_id: int | None,
    limit: int,
) -> list[dict]:
    """关键词召回：在用户（可限定文档）的 chunk 里按命中词数排序。

    返回 [{chunk_index, content, document_id, keyword_score}, ...]，降序。
    小规模知识库直接内存打分；数据量大时应下推到 DB 全文索引。
    """
    terms = _tokenize(query)
    if not terms:
        return []

    stmt = (
        select(DocumentChunk)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.user_id == user_id)
    )
    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    rows = (await db.execute(stmt)).scalars().all()

    scored: list[dict] = []
    for chunk in rows:
        content_lower = chunk.content.lower()
        hit = sum(1 for t in terms if t in content_lower)
        if hit > 0:
            scored.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "keyword_score": hit,
                }
            )

    scored.sort(key=lambda x: x["keyword_score"], reverse=True)
    return scored[:limit]


def _key(hit: dict) -> ChunkKey:
    return (hit["document_id"], hit["chunk_index"])


def fuse_dense_and_keyword(
    dense_hits: list[dict],
    keyword_hits: list[dict],
    top_k: int,
) -> list[dict]:
    """把稠密检索与关键词检索结果 RRF 融合，返回融合后的命中列表。

    身份用 (document_id, chunk_index)；融合后优先保留稠密结果（它带 score）。
    """
    fused_keys = reciprocal_rank_fusion(
        [[_key(h) for h in dense_hits], [_key(h) for h in keyword_hits]],
        k=settings.rrf_k,
        top_k=top_k,
    )

    by_key: dict[ChunkKey, dict] = {}
    for h in keyword_hits:
        by_key.setdefault(_key(h), h)
    for h in dense_hits:
        by_key[_key(h)] = h  # 稠密覆盖，保留 score

    return [by_key[key] for key in fused_keys if key in by_key]
