"""RAG 问答路由集成测试（/chat/）。

mock 检索与生成边界：embed_query / search / chat_completion，
验证 HTTP 契约、会话创建、检索结果透传到 sources。
"""

import pytest

from app.services import rag_service

pytestmark = pytest.mark.asyncio


class _FakeLLMMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


@pytest.fixture
def mock_rag(monkeypatch):
    """打桩 RAG 三段：向量化、检索、生成。"""
    hits = [
        {"score": 0.9, "content": "存储层用 PostgreSQL 和 Qdrant。",
         "document_id": 1, "chunk_index": 0},
    ]
    monkeypatch.setattr(rag_service, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(rag_service, "search", lambda *a, **k: hits)
    monkeypatch.setattr(
        rag_service, "chat_completion",
        lambda messages: _FakeLLMMsg("存储层用了 PostgreSQL 和 Qdrant。"),
    )
    return hits


class TestChat:
    async def test_chat_returns_answer_and_sources(self, client, registered_user, mock_rag):
        resp = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "存储层用了什么？"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "PostgreSQL" in body["answer"]
        assert body["conversation_id"] > 0
        assert body["sources"][0]["document_id"] == 1
        assert body["sources"][0]["chunk_index"] == 0

    async def test_chat_creates_conversation(self, client, registered_user, mock_rag):
        resp = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "第一个问题"},
        )
        conv_id = resp.json()["conversation_id"]
        # 复用同一 conversation_id 应成功（会话存在）
        resp2 = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "追问", "conversation_id": conv_id},
        )
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == conv_id

    async def test_chat_unknown_conversation_404(self, client, registered_user, mock_rag):
        resp = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "q", "conversation_id": 9999},
        )
        assert resp.status_code == 404

    async def test_chat_requires_auth(self, client, mock_rag):
        resp = await client.post("/chat/", json={"question": "q"})
        assert resp.status_code == 401
