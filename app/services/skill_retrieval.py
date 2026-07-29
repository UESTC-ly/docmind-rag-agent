"""同步 Skills 共用的 RAG 检索入口。

Agent 主循环在线程池里执行同步 Skill，因此不能直接复用需要 AsyncSession 的
``rag_service.retrieve``。本模块把同步 I/O 注入同一套可插拔 Pipeline executor。
Evaluation runner 也调用本入口，避免形成第三套检索逻辑。
"""

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings  # noqa: F401 - compatibility override surface
from app.services.document_policy import apply_document_policy_sync
from app.services.embedding_service import embed_query
from app.services.rag_pipeline import (
    PipelineExecution,
    PipelineSpec,
    SyncRetrievalRuntime,
    execute_sync_pipeline,
)
from app.services.retrieval import keyword_search_sync
from app.services.vector_store import search


def run_pipeline_for_skill(
    user_id: int,
    query: str,
    top_k: int | None = None,
    document_id: int | None = None,
    db: Session | None = None,
    pipeline: PipelineSpec | Mapping[str, Any] | str | None = None,
) -> PipelineExecution:
    """Execute an exact PipelineSpec and retain its reproducibility metadata."""

    def _keyword(
        keyword_query: str,
        keyword_user_id: int,
        keyword_document_id: int | None,
        limit: int,
    ) -> list[dict]:
        hits = keyword_search_sync(
            keyword_query,
            keyword_user_id,
            keyword_document_id,
            limit,
            db=db,
        )
        return apply_document_policy_sync(
            hits,
            user_id=keyword_user_id,
            db=db,
        )

    def _dense(
        vector: list[float],
        dense_user_id: int,
        limit: int,
        dense_document_id: int | None,
    ) -> list[dict]:
        hits = search(
            vector,
            dense_user_id,
            limit,
            dense_document_id,
        )
        return apply_document_policy_sync(
            hits,
            user_id=dense_user_id,
            db=db,
        )

    runtime = SyncRetrievalRuntime(
        embed_query=embed_query,
        dense_search=_dense,
        keyword_search=_keyword,
    )
    return execute_sync_pipeline(
        spec=pipeline,
        user_id=user_id,
        query=query,
        document_id=document_id,
        top_k=top_k,
        runtime=runtime,
    )


def retrieve_for_skill(
    user_id: int,
    query: str,
    top_k: int | None = None,
    document_id: int | None = None,
    db: Session | None = None,
    pipeline: PipelineSpec | Mapping[str, Any] | str | None = None,
) -> list[dict]:
    """Retrieve hits for a Skill; the default snapshots current RAG settings."""
    return run_pipeline_for_skill(
        user_id=user_id,
        query=query,
        top_k=top_k,
        document_id=document_id,
        db=db,
        pipeline=pipeline,
    ).hits
