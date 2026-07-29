"""Shared document provenance, freshness, and citation-jump policy."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.database import SyncSessionLocal
from app.models.document import Document, DocumentChunk

DOCUMENT_POLICY_VERSION = "freshness_v1"
_EXCLUDED_STATUSES = {"expired", "superseded"}


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def freshness_status(
    document: Document,
    *,
    now: datetime | None = None,
) -> str:
    """Resolve effective source status without treating unknown as current."""

    moment = now or datetime.now(UTC)
    explicit = (document.source_status or "unknown").strip().lower()
    if explicit in _EXCLUDED_STATUSES:
        return explicit
    effective_from = _aware(document.effective_from)
    effective_to = _aware(document.effective_to)
    if effective_from is not None and moment < effective_from:
        return "not_yet_effective"
    if effective_to is not None and moment >= effective_to:
        return "expired"
    return "current" if explicit == "current" else "unknown"


def _enrich(
    hits: list[dict[str, Any]],
    documents: dict[int, Document],
    chunks: dict[tuple[int, int], DocumentChunk],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for raw in hits:
        document_id = int(raw["document_id"])
        document = documents.get(document_id)
        if document is None:
            # Missing/foreign metadata is not safe evidence.
            continue
        resolved_status = freshness_status(document, now=now)
        if resolved_status in _EXCLUDED_STATUSES | {"not_yet_effective"}:
            continue
        hit = dict(raw)
        chunk = chunks.get((document_id, int(raw["chunk_index"])))
        hit.update(
            {
                "document_name": document.filename,
                "source_uri": document.source_uri,
                "source_version": document.source_version,
                "authority": document.authority,
                "source_status": resolved_status,
                "effective_from": (
                    document.effective_from.isoformat()
                    if document.effective_from
                    else None
                ),
                "effective_to": (
                    document.effective_to.isoformat()
                    if document.effective_to
                    else None
                ),
                "jump_url": (
                    f"/documents/{document_id}/chunks/"
                    f"{int(raw['chunk_index'])}"
                ),
                "document_policy_version": DOCUMENT_POLICY_VERSION,
            }
        )
        if chunk is not None and (
            chunk.locator_version is not None
            or any(
                value is not None
                for value in (
                    chunk.page_start,
                    chunk.page_end,
                    chunk.paragraph_start,
                    chunk.paragraph_end,
                    chunk.char_start,
                    chunk.char_end,
                )
            )
        ):
            hit.update(
                {
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "paragraph_start": chunk.paragraph_start,
                    "paragraph_end": chunk.paragraph_end,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "locator_version": chunk.locator_version,
                }
            )
        filtered.append(hit)
    return filtered


def apply_document_policy_sync(
    hits: list[dict[str, Any]],
    *,
    user_id: int,
    db: Session | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Enrich current/unknown evidence and remove known stale documents."""

    if not hits:
        return []
    owns_session = db is None
    session = db or SyncSessionLocal()
    try:
        document_ids = {int(hit["document_id"]) for hit in hits}
        documents = {
            document.id: document
            for document in session.execute(
                select(Document).where(
                    Document.id.in_(document_ids),
                    Document.user_id == user_id,
                )
            ).scalars()
        }
        chunk_keys = {
            (int(hit["document_id"]), int(hit["chunk_index"])) for hit in hits
        }
        chunks = {
            (chunk.document_id, chunk.chunk_index): chunk
            for chunk in session.execute(
                select(DocumentChunk).where(
                    tuple_(
                        DocumentChunk.document_id,
                        DocumentChunk.chunk_index,
                    ).in_(chunk_keys)
                )
            ).scalars()
        }
        return _enrich(hits, documents, chunks, now=now)
    finally:
        if owns_session:
            session.close()


async def apply_document_policy_async(
    hits: list[dict[str, Any]],
    *,
    user_id: int,
    db: AsyncSession,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Async counterpart used by browser chat."""

    if not hits:
        return []
    document_ids = {int(hit["document_id"]) for hit in hits}
    result = await db.execute(
        select(Document).where(
            Document.id.in_(document_ids),
            Document.user_id == user_id,
        )
    )
    documents = {document.id: document for document in result.scalars()}
    chunk_keys = {
        (int(hit["document_id"]), int(hit["chunk_index"])) for hit in hits
    }
    chunk_result = await db.execute(
        select(DocumentChunk).where(
            tuple_(
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
            ).in_(chunk_keys)
        )
    )
    chunks = {
        (chunk.document_id, chunk.chunk_index): chunk
        for chunk in chunk_result.scalars()
    }
    return _enrich(hits, documents, chunks, now=now)


__all__ = [
    "DOCUMENT_POLICY_VERSION",
    "apply_document_policy_async",
    "apply_document_policy_sync",
    "freshness_status",
]
