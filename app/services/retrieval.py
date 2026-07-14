"""Shared hybrid retrieval primitives.

PostgreSQL keyword recall uses ``to_tsvector``/``ts_rank_cd`` and a GIN-backed
expression.  SQLite (desktop and tests) uses a bounded SQL ``LIKE`` score.  Both
paths apply ownership/document filters, ranking, and LIMIT in the database; no
path materializes a user's complete chunk collection in Python.

Dense and keyword candidates are fused with scored RRF, annotated with their raw
signals, then passed through the same reranker for online chat, Skills, and evals.
"""

from __future__ import annotations

import re
from collections.abc import Hashable

from sqlalchemy import case, func, literal, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SyncSessionLocal
from app.models.document import Document, DocumentChunk
from app.services.reranker import rerank

ChunkKey = tuple[int, int]


def reciprocal_rank_fusion_scores(
    ranked_lists: list[list[Hashable]], k: int = 60
) -> dict[Hashable, float]:
    """Return stable RRF scores without discarding diagnostic information."""
    scores: dict[Hashable, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def reciprocal_rank_fusion(
    ranked_lists: list[list[ChunkKey]], k: int = 60, top_k: int = 5
) -> list[ChunkKey]:
    """Fuse ranked identifiers with deterministic ordering for score ties."""
    scores = reciprocal_rank_fusion_scores(ranked_lists, k=k)
    first_seen: dict[Hashable, int] = {}
    best_rank: dict[Hashable, int] = {}
    position = 0
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            first_seen.setdefault(key, position)
            best_rank[key] = min(best_rank.get(key, rank), rank)
            position += 1
    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], best_rank[key], first_seen[key]),
    )
    return list(ordered[:top_k])


def _tokenize(text: str) -> list[str]:
    """Tokenize enough for the SQLite compatibility query and local reranker."""
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
        if token
    ]


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_keyword_statement(
    query: str,
    user_id: int,
    document_id: int | None,
    limit: int,
    dialect_name: str,
):
    """Build the dialect-aware, database-ranked keyword candidate query."""
    terms = list(dict.fromkeys(_tokenize(query)))
    if not terms or limit <= 0:
        return None

    if dialect_name == "postgresql":
        # Keep the regconfig literal identical to the migration's expression
        # index so PostgreSQL can choose the GIN index under prepared statements.
        simple_config = literal_column("'simple'::regconfig")
        vector = func.to_tsvector(simple_config, DocumentChunk.content)
        ts_query = func.websearch_to_tsquery(simple_config, query)
        score = func.ts_rank_cd(vector, ts_query)
        match_condition = vector.op("@@")(ts_query)
    else:
        # SQL-side compatibility scorer for desktop SQLite and unit tests.  Keep the
        # number of terms bounded so a hostile query cannot generate unbounded SQL.
        score = literal(0)
        for term in terms[:64]:
            pattern = f"%{_escape_like(term)}%"
            score = score + case(
                (func.lower(DocumentChunk.content).like(pattern, escape="\\"), 1),
                else_=0,
            )
        match_condition = score > 0

    keyword_score = score.label("keyword_score")
    stmt = (
        select(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            keyword_score,
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.user_id == user_id, match_condition)
    )
    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)
    return stmt.order_by(
        keyword_score.desc(),
        DocumentChunk.document_id.asc(),
        DocumentChunk.chunk_index.asc(),
    ).limit(limit)


def _keyword_rows(result) -> list[dict]:
    return [
        {
            "document_id": int(row["document_id"]),
            "chunk_index": int(row["chunk_index"]),
            "content": row["content"],
            "keyword_score": float(row["keyword_score"] or 0.0),
            "score": 0.0,
        }
        for row in result.mappings().all()
    ]


async def keyword_search(
    db: AsyncSession,
    query: str,
    user_id: int,
    document_id: int | None,
    limit: int,
) -> list[dict]:
    dialect = db.get_bind().dialect.name
    stmt = build_keyword_statement(query, user_id, document_id, limit, dialect)
    if stmt is None:
        return []
    return _keyword_rows(await db.execute(stmt))


def keyword_search_sync(
    query: str,
    user_id: int,
    document_id: int | None,
    limit: int,
    db: Session | None = None,
) -> list[dict]:
    """Synchronous database keyword recall for Skills, evals, and local tasks."""
    owns_session = db is None
    session = db or SyncSessionLocal()
    try:
        dialect = session.get_bind().dialect.name
        stmt = build_keyword_statement(query, user_id, document_id, limit, dialect)
        if stmt is None:
            return []
        return _keyword_rows(session.execute(stmt))
    finally:
        if owns_session:
            session.close()


def _key(hit: dict) -> ChunkKey:
    return (int(hit["document_id"]), int(hit["chunk_index"]))


def fuse_dense_and_keyword(
    dense_hits: list[dict],
    keyword_hits: list[dict],
    top_k: int,
) -> list[dict]:
    """Fuse candidates while preserving raw ranks/scores and recall provenance."""
    dense_keys = [_key(hit) for hit in dense_hits]
    keyword_keys = [_key(hit) for hit in keyword_hits]
    scores = reciprocal_rank_fusion_scores(
        [dense_keys, keyword_keys], k=settings.rrf_k
    )
    fused_keys = reciprocal_rank_fusion(
        [dense_keys, keyword_keys], k=settings.rrf_k, top_k=top_k
    )

    by_key: dict[ChunkKey, dict] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        key = _key(hit)
        enriched = dict(hit)
        enriched.update(
            dense_rank=rank,
            dense_score=float(hit.get("score", 0.0) or 0.0),
            retrieval_sources=["dense"],
        )
        by_key[key] = enriched

    for rank, hit in enumerate(keyword_hits, start=1):
        key = _key(hit)
        enriched = by_key.setdefault(key, dict(hit))
        enriched["keyword_rank"] = rank
        enriched["keyword_score"] = float(hit.get("keyword_score", 0.0) or 0.0)
        sources = enriched.setdefault("retrieval_sources", [])
        if "keyword" not in sources:
            sources.append("keyword")
        enriched.setdefault("score", 0.0)
        enriched.setdefault("dense_score", None)

    fused: list[dict] = []
    for rank, key in enumerate(fused_keys, start=1):
        hit = dict(by_key[key])
        hit["rrf_score"] = float(scores[key])
        hit["fused_rank"] = rank
        fused.append(hit)
    return fused


def finalize_retrieval(
    query: str,
    dense_hits: list[dict],
    keyword_hits: list[dict],
    top_k: int,
) -> list[dict]:
    """Shared candidate fusion + bounded reranking stage for every caller."""
    candidate_limit = max(
        top_k,
        min(
            settings.reranker_candidate_limit,
            max(len(dense_hits) + len(keyword_hits), top_k),
        ),
    )
    fused = fuse_dense_and_keyword(dense_hits, keyword_hits, candidate_limit)
    return rerank(query, fused, top_k)
