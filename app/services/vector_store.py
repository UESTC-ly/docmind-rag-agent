"""Qdrant 向量库封装。

每个 chunk 在 Qdrant 里是一个 point：
  - id：全局唯一（用 document_id * 100000 + chunk_index 生成，保证不撞）
  - vector：chunk 文本的 embedding
  - payload：document_id / chunk_index / user_id / content，检索后可直接拿到原文
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings

_client = QdrantClient(url=settings.qdrant_url)

_ID_STRIDE = 100_000  # 每个文档最多 10 万个 chunk，够用


def _point_id(document_id: int, chunk_index: int) -> int:
    return document_id * _ID_STRIDE + chunk_index


def ensure_collection() -> None:
    """确保 collection 存在，不存在则按 embedding 维度创建。幂等。"""
    existing = {c.name for c in _client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        _client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


def upsert_chunks(
    document_id: int,
    user_id: int,
    chunks: list[str],
    vectors: list[list[float]],
) -> None:
    """把一个文档的所有分块向量写入 Qdrant。"""
    ensure_collection()
    points = [
        PointStruct(
            id=_point_id(document_id, idx),
            vector=vector,
            payload={
                "document_id": document_id,
                "user_id": user_id,
                "chunk_index": idx,
                "content": content,
            },
        )
        for idx, (content, vector) in enumerate(zip(chunks, vectors))
    ]
    _client.upsert(collection_name=settings.qdrant_collection, points=points)


def search(
    query_vector: list[float],
    user_id: int,
    top_k: int = 5,
    document_id: int | None = None,
) -> list[dict]:
    """向量检索。按 user_id 过滤（数据隔离），可选按 document_id 限定单文档。

    返回 [{score, content, document_id, chunk_index}, ...]
    """
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if document_id is not None:
        must.append(
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        )

    hits = _client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=Filter(must=must),
        limit=top_k,
    ).points
    return [
        {
            "score": h.score,
            "content": h.payload["content"],
            "document_id": h.payload["document_id"],
            "chunk_index": h.payload["chunk_index"],
        }
        for h in hits
    ]


def delete_document(document_id: int) -> None:
    """删除某文档的所有向量（文档被删时调用）。"""
    _client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id", match=MatchValue(value=document_id)
                )
            ]
        ),
    )
