"""Embedding 服务：把文本转成向量。

用独立的 embedding 客户端（可与对话 LLM 是不同服务，如对话用 gpt、
embedding 用 DashScope）。同步调用，异步场景用 asyncio.to_thread 包一层。

分批：部分服务单次批量有上限（DashScope 为 10），超限会报错，故按 batch 切分。
"""

from openai import OpenAI

from app.config import settings

_client = OpenAI(
    api_key=settings.resolved_embedding_key,
    base_url=settings.resolved_embedding_base_url,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化，自动分批。返回与输入等长、同序的向量列表。"""
    if not texts:
        return []

    batch_size = settings.embedding_batch_size
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = _client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        # resp.data 按输入顺序返回
        vectors.extend(item.embedding for item in resp.data)

    return vectors


def embed_query(query: str) -> list[float]:
    """单条查询向量化，供检索时用。"""
    return embed_texts([query])[0]
