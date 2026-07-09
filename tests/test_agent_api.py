"""Agent 路由集成测试（/agent/chat, /agent/skills）。

mock run_agent（其循环逻辑已在 test_agent_orchestrator 单独覆盖），
这里验证 agent_service 的会话管理、历史落库、HTTP 契约。
"""

import pytest

from app.services import agent_service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_agent(monkeypatch):
    """打桩 run_agent 与 load_history（后者用 SyncSessionLocal 连真 PG，测试须隔离）。"""
    monkeypatch.setattr(
        agent_service, "run_agent",
        lambda user_id, message, history, document_id: {
            "answer": "Agent 的回答",
            "artifacts": [{"type": "mindmap", "nodes": ["a"]}],
            "trace": [{"step": 0, "skill": "search_knowledge_base", "args": {}}],
        },
    )
    # 复用会话时会调 load_history（同步、连真库），mock 掉保证测试无外部依赖
    monkeypatch.setattr(agent_service, "load_history", lambda conversation_id: [])


class TestAgentChat:
    async def test_chat_returns_answer_artifacts_trace(self, client, registered_user, mock_agent):
        resp = await client.post(
            "/agent/chat", headers=registered_user["headers"],
            json={"message": "帮我查一下"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Agent 的回答"
        assert body["artifacts"][0]["type"] == "mindmap"
        assert body["trace"][0]["skill"] == "search_knowledge_base"
        assert body["conversation_id"] > 0

    async def test_chat_reuses_conversation(self, client, registered_user, mock_agent):
        first = await client.post(
            "/agent/chat", headers=registered_user["headers"],
            json={"message": "第一句"},
        )
        conv_id = first.json()["conversation_id"]
        second = await client.post(
            "/agent/chat", headers=registered_user["headers"],
            json={"message": "第二句", "conversation_id": conv_id},
        )
        assert second.status_code == 200
        assert second.json()["conversation_id"] == conv_id

    async def test_chat_unknown_conversation_404(self, client, registered_user, mock_agent):
        resp = await client.post(
            "/agent/chat", headers=registered_user["headers"],
            json={"message": "q", "conversation_id": 8888},
        )
        assert resp.status_code == 404

    async def test_chat_requires_auth(self, client, mock_agent):
        resp = await client.post("/agent/chat", json={"message": "q"})
        assert resp.status_code == 401

    async def test_empty_message_rejected(self, client, registered_user, mock_agent):
        # schema 要求 message 至少 1 字符
        resp = await client.post(
            "/agent/chat", headers=registered_user["headers"], json={"message": ""}
        )
        assert resp.status_code == 422


class TestListSkills:
    async def test_skills_listed(self, client, registered_user):
        resp = await client.get("/agent/skills", headers=registered_user["headers"])
        assert resp.status_code == 200
        # 真实注册的技能应都在
        skills = resp.json()
        names = [s["name"] for s in skills] if isinstance(skills, list) else skills
        assert "search_knowledge_base" in str(names)
        assert "generate_weekly_report" in str(names)
        assert "generate_presentation" in str(names)

        packages = {
            s["name"]: s.get("package")
            for s in skills
            if isinstance(s, dict) and s.get("package")
        }
        assert packages["generate_weekly_report"]["slug"] == "weekly-report"
        assert "prompt.md" in packages["generate_presentation"]["templates"]
