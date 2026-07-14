"""文档路由集成测试：上传 / 列表 / 查询 / 删除。

mock 掉外部边界：
- Celery process_document.delay（避免连 Redis）
- vector_store.delete_document（避免连 Qdrant）
- 上传目录指向临时目录（避免污染真实 uploads/）
"""

import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.document import Document, DocumentStatus
from app.models.user import User
from app.services import document_service, task_dispatcher
from app.tasks import document_tasks

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
        monkeypatch.setattr(document_tasks, "extract_text", lambda _path: "local task text")
        monkeypatch.setattr(
            document_tasks,
            "split_text",
            lambda text, **_kwargs: [text],
        )
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
