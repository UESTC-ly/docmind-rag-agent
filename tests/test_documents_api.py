"""文档路由集成测试：上传 / 列表 / 查询 / 删除。

mock 掉外部边界：
- Celery process_document.delay（避免连 Redis）
- vector_store.delete_document（避免连 Qdrant）
- 上传目录指向临时目录（避免污染真实 uploads/）
"""

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.services import document_service, task_dispatcher
from app.tasks import document_tasks
from app.utils.file_parser import extract_text_with_locations

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_boundaries(monkeypatch, tmp_path):
    """打桩所有外部副作用，返回被调用记录用于断言。"""
    calls = {"delay": [], "delete_vec": []}
    monkeypatch.setattr(
        document_service,
        "dispatch_document",
        lambda doc_id: calls["delay"].append(doc_id),
    )
    monkeypatch.setattr(
        document_service.vector_store, "delete_document",
        lambda doc_id: calls["delete_vec"].append(doc_id),
    )
    monkeypatch.setattr(document_service.settings, "upload_dir", str(tmp_path))
    return calls


class TestUpload:
    async def test_upload_txt_dispatches_task(self, client, registered_user, mock_boundaries):
        resp = await client.post(
            "/documents/upload",
            headers=registered_user["headers"],
            files={"file": ("note.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "pending"
        # Celery 任务应被派发一次，参数是新文档 id
        assert mock_boundaries["delay"] == [body["document_id"]]

    async def test_upload_unsupported_type_rejected(self, client, registered_user, mock_boundaries):
        resp = await client.post(
            "/documents/upload",
            headers=registered_user["headers"],
            files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert mock_boundaries["delay"] == []  # 不应派发任务

    async def test_upload_requires_auth(self, client, mock_boundaries):
        resp = await client.post(
            "/documents/upload",
            files={"file": ("note.txt", b"hi", "text/plain")},
        )
        assert resp.status_code == 401

    async def test_desktop_local_task_reads_committed_document(
        self, monkeypatch, tmp_path
    ):
        """The local worker's independent sync connection must see the row."""
        database = tmp_path / "desktop-local.db"
        async_engine = create_async_engine(
            f"sqlite+aiosqlite:///{database}",
            connect_args={"check_same_thread": False},
        )
        sync_engine = create_engine(
            f"sqlite:///{database}",
            connect_args={"check_same_thread": False},
        )
        async_sessions = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        sync_sessions = sessionmaker(sync_engine, expire_on_commit=False)

        async with async_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        task_dispatcher.shutdown_local_executor()
        monkeypatch.setattr(document_service.settings, "upload_dir", str(tmp_path / "uploads"))
        monkeypatch.setattr(task_dispatcher.settings, "task_execution_mode", "local")
        monkeypatch.setattr(document_tasks, "SyncSessionLocal", sync_sessions)
        monkeypatch.setattr(
            document_tasks,
            "embed_texts",
            lambda chunks: [[0.1] for _chunk in chunks],
        )
        monkeypatch.setattr(document_tasks, "upsert_chunks", lambda **_kwargs: None)

        try:
            async with async_sessions() as db:
                user = User(email="desktop@example.com", hashed_password="x")
                db.add(user)
                await db.commit()
                await db.refresh(user)
                uploaded = UploadFile(
                    file=BytesIO(b"local task text"),
                    filename="desktop.txt",
                )
                document = await document_service.create_document(db, user.id, uploaded)
                document_id = document.id

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 3
            while True:
                async with async_sessions() as db:
                    current = await db.get(Document, document_id)
                    assert current is not None
                    if current.status in {DocumentStatus.COMPLETED, DocumentStatus.FAILED}:
                        break
                if loop.time() >= deadline:
                    pytest.fail("desktop local document task did not reach a terminal state")
                await asyncio.sleep(0.02)

            assert current.status is DocumentStatus.COMPLETED
            assert current.chunk_count == 1
        finally:
            await asyncio.to_thread(task_dispatcher.shutdown_local_executor)
            await async_engine.dispose()
            sync_engine.dispose()


class TestListAndGet:
    async def _upload(self, client, headers, name="doc.txt"):
        r = await client.post(
            "/documents/upload", headers=headers,
            files={"file": (name, b"content", "text/plain")},
        )
        return r.json()["document_id"]

    async def test_list_returns_only_own_docs(self, client, registered_user, mock_boundaries):
        await self._upload(client, registered_user["headers"], "a.txt")
        await self._upload(client, registered_user["headers"], "b.txt")
        resp = await client.get("/documents/", headers=registered_user["headers"])
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_get_existing_doc(self, client, registered_user, mock_boundaries):
        doc_id = await self._upload(client, registered_user["headers"])
        resp = await client.get(f"/documents/{doc_id}", headers=registered_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["id"] == doc_id

    async def test_get_missing_doc_404(self, client, registered_user, mock_boundaries):
        resp = await client.get("/documents/9999", headers=registered_user["headers"])
        assert resp.status_code == 404

    async def test_version_metadata_supersedes_old_document(
        self,
        client,
        registered_user,
        mock_boundaries,
    ):
        old_id = await self._upload(
            client,
            registered_user["headers"],
            "policy-v1.txt",
        )
        new_id = await self._upload(
            client,
            registered_user["headers"],
            "policy-v2.txt",
        )
        response = await client.patch(
            f"/documents/{new_id}/metadata",
            headers=registered_user["headers"],
            json={
                "source_uri": "https://example.org/policy",
                "source_version": "2",
                "authority": "Example Authority",
                "source_status": "current",
                "supersedes_document_id": old_id,
            },
        )
        assert response.status_code == 200
        assert response.json()["source_version"] == "2"
        assert response.json()["supersedes_document_id"] == old_id
        old = await client.get(
            f"/documents/{old_id}",
            headers=registered_user["headers"],
        )
        assert old.json()["source_status"] == "superseded"

    async def test_citation_jump_returns_exact_owned_chunk(
        self,
        client,
        registered_user,
        mock_boundaries,
        db_session,
    ):
        document_id = await self._upload(
            client,
            registered_user["headers"],
            "source.txt",
        )
        db_session.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=3,
                content="可核验的原文段落。",
            )
        )
        await db_session.flush()

        response = await client.get(
            f"/documents/{document_id}/chunks/3",
            headers=registered_user["headers"],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["citation_id"] == f"D{document_id}:C3"
        assert body["content"] == "可核验的原文段落。"
        assert body["source_excerpt"] == "可核验的原文段落。"
        assert body["highlight_start"] == 0
        assert body["highlight_end"] == len("可核验的原文段落。")
        assert body["jump_url"] == f"/documents/{document_id}/chunks/3"

    async def test_citation_jump_returns_source_location_and_highlight_range(
        self,
        client,
        registered_user,
        mock_boundaries,
        db_session,
    ):
        document_id = await self._upload(
            client,
            registered_user["headers"],
            "located.txt",
        )
        document = await db_session.get(Document, document_id)
        assert document is not None
        Path(document.file_path).write_text(
            "前置上下文\n\n可核验的原文段落。\n\n后置上下文",
            encoding="utf-8",
        )
        parsed = extract_text_with_locations(document.file_path)
        content = "可核验的原文段落。"
        char_start = parsed.text.index(content)
        db_session.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=4,
                content=content,
                paragraph_start=2,
                paragraph_end=2,
                char_start=char_start,
                char_end=char_start + len(content),
                locator_version="extracted_text_v1",
            )
        )
        await db_session.flush()

        response = await client.get(
            f"/documents/{document_id}/chunks/4",
            headers=registered_user["headers"],
        )

        assert response.status_code == 200
        body = response.json()
        assert body["paragraph_start"] == 2
        assert body["char_start"] == char_start
        assert body["source_excerpt"] == parsed.text
        assert body["source_excerpt"][
            body["highlight_start"] : body["highlight_end"]
        ] == content

    async def test_metadata_patch_preserves_fields_that_were_not_sent(
        self,
        client,
        registered_user,
        mock_boundaries,
    ):
        document_id = await self._upload(
            client,
            registered_user["headers"],
            "versioned.txt",
        )
        first = await client.patch(
            f"/documents/{document_id}/metadata",
            headers=registered_user["headers"],
            json={
                "source_uri": "https://example.org/source",
                "source_version": "2026-01",
                "source_status": "current",
            },
        )
        assert first.status_code == 200

        second = await client.patch(
            f"/documents/{document_id}/metadata",
            headers=registered_user["headers"],
            json={"authority": "Example Authority"},
        )

        assert second.status_code == 200
        assert second.json()["authority"] == "Example Authority"
        assert second.json()["source_uri"] == "https://example.org/source"
        assert second.json()["source_version"] == "2026-01"
        assert second.json()["source_status"] == "current"

        empty = await client.patch(
            f"/documents/{document_id}/metadata",
            headers=registered_user["headers"],
            json={},
        )
        assert empty.status_code == 422


class TestDelete:
    async def test_delete_removes_doc_and_vectors(self, client, registered_user, mock_boundaries):
        r = await client.post(
            "/documents/upload", headers=registered_user["headers"],
            files={"file": ("del.txt", b"x", "text/plain")},
        )
        doc_id = r.json()["document_id"]

        resp = await client.delete(f"/documents/{doc_id}", headers=registered_user["headers"])
        assert resp.status_code == 204
        # 向量删除应被调用
        assert mock_boundaries["delete_vec"] == [doc_id]
        # 文档应查不到了
        assert (await client.get(f"/documents/{doc_id}", headers=registered_user["headers"])).status_code == 404

    async def test_delete_missing_doc_404(self, client, registered_user, mock_boundaries):
        resp = await client.delete("/documents/9999", headers=registered_user["headers"])
        assert resp.status_code == 404
        assert mock_boundaries["delete_vec"] == []
