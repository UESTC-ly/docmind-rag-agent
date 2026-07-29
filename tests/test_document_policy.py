"""Freshness filtering and source-jump enrichment."""

from datetime import UTC, datetime, timedelta

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.services.document_policy import (
    apply_document_policy_sync,
    freshness_status,
)


def _document(sync_db, user_id, name, **metadata):
    document = Document(
        user_id=user_id,
        filename=name,
        file_path=f"dataset://{name}",
        status=DocumentStatus.COMPLETED,
        **metadata,
    )
    sync_db.add(document)
    sync_db.flush()
    return document


def test_known_expired_and_superseded_documents_are_excluded(sync_db):
    user = User(email="freshness@test.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    current = _document(
        sync_db,
        user.id,
        "current",
        source_status="current",
        source_version="2",
    )
    expired = _document(
        sync_db,
        user.id,
        "expired",
        source_status="current",
        effective_to=datetime.now(UTC) - timedelta(days=1),
    )
    superseded = _document(
        sync_db,
        user.id,
        "superseded",
        source_status="superseded",
    )
    unknown = _document(sync_db, user.id, "unknown")
    sync_db.add_all(
        [
            DocumentChunk(
                document_id=current.id,
                chunk_index=4,
                content="current",
                page_start=2,
                page_end=2,
                paragraph_start=3,
                paragraph_end=3,
                char_start=12,
                char_end=19,
                locator_version="extracted_text_v1",
            ),
            DocumentChunk(
                document_id=unknown.id,
                chunk_index=4,
                content="unknown",
            ),
        ]
    )
    sync_db.flush()

    hits = [
        {
            "document_id": document.id,
            "chunk_index": 4,
            "content": document.filename,
            "score": 0.5,
        }
        for document in (current, expired, superseded, unknown)
    ]
    filtered = apply_document_policy_sync(
        hits,
        user_id=user.id,
        db=sync_db,
    )

    assert [hit["document_id"] for hit in filtered] == [
        current.id,
        unknown.id,
    ]
    assert filtered[0]["source_status"] == "current"
    assert filtered[0]["source_version"] == "2"
    assert filtered[0]["jump_url"] == f"/documents/{current.id}/chunks/4"
    assert filtered[0]["page_start"] == 2
    assert filtered[0]["paragraph_start"] == 3
    assert filtered[0]["char_end"] == 19
    assert filtered[1]["source_status"] == "unknown"
    assert "page_start" not in filtered[1]


def test_future_effective_document_is_not_yet_valid(sync_db):
    user = User(email="future@test.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    future = _document(
        sync_db,
        user.id,
        "future",
        source_status="current",
        effective_from=datetime.now(UTC) + timedelta(days=1),
    )
    assert freshness_status(future) == "not_yet_effective"
    assert (
        apply_document_policy_sync(
            [
                {
                    "document_id": future.id,
                    "chunk_index": 0,
                    "content": "future",
                    "score": 1.0,
                }
            ],
            user_id=user.id,
            db=sync_db,
        )
        == []
    )
