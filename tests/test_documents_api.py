"""文档路由集成测试：上传 / 列表 / 查询 / 删除。

mock 掉外部边界：
- Celery process_document.delay（避免连 Redis）
- vector_store.delete_document（避免连 Qdrant）
- 上传目录指向临时目录（避免污染真实 uploads/）
"""

import pytest

from app.services import document_service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_boundaries(monkeypatch, tmp_path):
    """打桩所有外部副作用，返回被调用记录用于断言。"""
    calls = {"delay": [], "delete_vec": []}
    monkeypatch.setattr(
        document_service.process_document, "delay",
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
