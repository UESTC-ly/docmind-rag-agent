"""Agent 路由集成测试（/agent/chat, /agent/skills）。

mock run_agent（其循环逻辑已在 test_agent_orchestrator 单独覆盖），
这里验证 agent_service 的会话管理、历史落库、HTTP 契约。
"""

import pytest
from sqlalchemy import delete, func, select

from app.models.conversation import Conversation, Message, MessageRole
from app.services import agent_service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_agent(monkeypatch):
    """打桩 run_agent 与 load_history（后者用 SyncSessionLocal 连真 PG，测试须隔离）。"""
    captured = {}

    def _fake_run_agent(
        user_id,
        message,
        history,
        document_id,
        requested_skill=None,
        **kwargs,
    ):
        captured["run_calls"] = captured.get("run_calls", 0) + 1
        captured["requested_skill"] = requested_skill
        captured["thread_id"] = kwargs["thread_id"]
        captured["conversation_id"] = kwargs["conversation_id"]
        return {
            "run_id": kwargs["thread_id"],
            "thread_id": kwargs["thread_id"],
            "conversation_id": kwargs["conversation_id"],
            "status": "completed",
            "answer": "Agent 的回答",
            "artifacts": [{"type": "mindmap", "nodes": ["a"]}],
            "trace": [{"step": 0, "skill": "search_knowledge_base", "args": {}}],
            "approval": None,
        }

    monkeypatch.setattr(agent_service, "run_agent", _fake_run_agent)
    monkeypatch.setattr(
        agent_service,
        "resume_agent",
        lambda run_id, approved, **kwargs: {
            "run_id": run_id,
            "thread_id": run_id,
            "conversation_id": captured.get("conversation_id", 1),
            "status": "completed",
            "answer": "恢复后完成",
            "artifacts": [],
            "trace": [
                {
                    "step": 0,
                    "skill": "risk_tool",
                    "args": {},
                    "approval": {"required": True, "approved": approved},
                }
            ],
            "approval": None,
        },
    )

    def _fake_inspect(run_id, **kwargs):
        if run_id != captured.get("thread_id"):
            raise agent_service.AgentRunNotFoundError("missing")
        return {
            "run_id": run_id,
            "thread_id": run_id,
            "conversation_id": captured.get("conversation_id", 1),
            "status": "waiting_approval",
            "answer": "等待审批",
            "artifacts": [],
            "trace": [],
            "approval": {"tool": "risk_tool", "args": {}, "reason": "高风险"},
        }

    monkeypatch.setattr(agent_service, "inspect_agent_run", _fake_inspect)
    monkeypatch.setattr(
        agent_service,
        "recover_agent",
        lambda run_id, **kwargs: {
            "run_id": run_id,
            "thread_id": run_id,
            "conversation_id": captured.get("conversation_id", 1),
            "status": "completed",
            "recoverable": False,
            "answer": "checkpoint 恢复完成",
            "artifacts": [],
            "trace": [],
            "approval": None,
        },
    )
    # 复用会话时会调 load_history（同步、连真库），mock 掉保证测试无外部依赖
    monkeypatch.setattr(agent_service, "load_history", lambda conversation_id: [])
    return captured


class TestAgentChat:
    async def test_chat_returns_answer_artifacts_trace(
        self, client, registered_user, mock_agent
    ):
        resp = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "帮我查一下"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Agent 的回答"
        assert body["artifacts"][0]["type"] == "mindmap"
        assert body["trace"][0]["skill"] == "search_knowledge_base"
        assert body["conversation_id"] > 0
        assert body["status"] == "completed"
        assert body["run_id"]

    async def test_selected_skill_is_forwarded_to_orchestrator(
        self, client, registered_user, mock_agent
    ):
        resp = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={
                "message": "生成周报",
                "document_id": 1,
                "skill_name": "generate_weekly_report",
            },
        )
        assert resp.status_code == 200
        assert mock_agent["requested_skill"] == "generate_weekly_report"

    async def test_client_run_id_becomes_langgraph_thread_id(
        self, client, registered_user, mock_agent
    ):
        response = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行", "run_id": "client-generated-run"},
        )
        assert response.status_code == 200
        assert response.json()["run_id"] == "client-generated-run"
        assert mock_agent["thread_id"] == "client-generated-run"

    async def test_retried_client_run_id_returns_checkpoint_without_reexecution(
        self, client, registered_user, mock_agent
    ):
        payload = {"message": "执行", "run_id": "idempotent-run"}
        first = await client.post(
            "/agent/chat", headers=registered_user["headers"], json=payload
        )
        second = await client.post(
            "/agent/chat", headers=registered_user["headers"], json=payload
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == "waiting_approval"
        assert mock_agent["run_calls"] == 1

    async def test_completed_retry_repairs_history_without_duplicate_message(
        self, client, db_session, registered_user, monkeypatch, mock_agent
    ):
        payload = {"message": "执行", "run_id": "completed-idempotent-run"}
        first = await client.post(
            "/agent/chat", headers=registered_user["headers"], json=payload
        )
        conversation_id = first.json()["conversation_id"]
        await db_session.execute(
            delete(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.ASSISTANT,
            )
        )
        await db_session.commit()

        def _completed(run_id, **kwargs):
            return {
                "run_id": run_id,
                "thread_id": run_id,
                "conversation_id": conversation_id,
                "status": "completed",
                "answer": "Agent 的回答",
                "artifacts": [],
                "trace": [{"step": 0, "skill": "risk_tool", "args": {}}],
                "approval": None,
            }

        monkeypatch.setattr(agent_service, "inspect_agent_run", _completed)
        repaired = await client.post(
            "/agent/chat", headers=registered_user["headers"], json=payload
        )
        repeated = await client.post(
            "/agent/chat", headers=registered_user["headers"], json=payload
        )

        assert repaired.status_code == repeated.status_code == 200
        assistant_count = (
            await db_session.execute(
                select(func.count(Message.id)).where(
                    Message.conversation_id == conversation_id,
                    Message.role == MessageRole.ASSISTANT,
                )
            )
        ).scalar_one()
        assert assistant_count == 1
        assert mock_agent["run_calls"] == 1

    async def test_chat_reuses_conversation(self, client, registered_user, mock_agent):
        first = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "第一句"},
        )
        conv_id = first.json()["conversation_id"]
        second = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "第二句", "conversation_id": conv_id},
        )
        assert second.status_code == 200
        assert second.json()["conversation_id"] == conv_id

    async def test_chat_unknown_conversation_404(
        self, client, registered_user, mock_agent
    ):
        resp = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
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
        assert "codex_note" in str(names)

        packages = {
            s["name"]: s.get("package")
            for s in skills
            if isinstance(s, dict) and s.get("package")
        }
        assert packages["generate_weekly_report"]["slug"] == "weekly-report"
        assert "prompt.md" in packages["generate_presentation"]["templates"]
        assert packages["codex_note"]["source"] == "codex"

        modes = {
            s["name"]: s.get("execution_mode") for s in skills if isinstance(s, dict)
        }
        assert modes["codex_note"] == "generic_package"
        availability = {
            s["name"]: s.get("available") for s in skills if isinstance(s, dict)
        }
        assert availability["codex_note"] is True
        assert availability["gh-fix-ci"] is False


class TestAgentApprovalAndRecovery:
    async def test_busy_run_returns_423_without_graph_execution(
        self, client, registered_user, monkeypatch, mock_agent
    ):
        class BusyLease:
            def acquire(self):
                raise agent_service.AgentRunLeaseBusyError("held")

            def release(self):
                raise AssertionError("unacquired lease must not be released")

        monkeypatch.setattr(
            agent_service, "_new_agent_run_lease", lambda run_id: BusyLease()
        )
        response = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行", "run_id": "busy-run"},
        )

        assert response.status_code == 423
        assert response.headers["retry-after"] == "2"
        assert "另一个 worker" in response.json()["detail"]
        assert mock_agent.get("run_calls", 0) == 0

    async def test_lock_backend_failure_returns_503(
        self, client, registered_user, monkeypatch, mock_agent
    ):
        class FailedLease:
            def acquire(self):
                raise agent_service.AgentRunLeaseBackendError("redis down")

            def release(self):
                raise AssertionError("unacquired lease must not be released")

        monkeypatch.setattr(
            agent_service, "_new_agent_run_lease", lambda run_id: FailedLease()
        )
        response = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行", "run_id": "backend-down"},
        )

        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert mock_agent.get("run_calls", 0) == 0

    async def test_deleted_conversation_blocks_resume_before_graph_execution(
        self, client, db_session, registered_user, monkeypatch, mock_agent
    ):
        first = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行"},
        )
        run_id = first.json()["run_id"]
        conversation_id = first.json()["conversation_id"]
        await db_session.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        await db_session.commit()

        called = False

        def _must_not_resume(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("deleted conversation must block graph resume")

        monkeypatch.setattr(agent_service, "resume_agent", _must_not_resume)
        response = await client.post(
            "/agent/resume",
            headers=registered_user["headers"],
            json={"run_id": run_id, "approved": True},
        )

        assert response.status_code == 404
        assert called is False

    async def test_run_status_hides_checkpoint_ownership(
        self, client, registered_user, monkeypatch, mock_agent
    ):
        def _owned_by_another_user(*args, **kwargs):
            raise agent_service.AgentRunOwnershipError("wrong owner")

        monkeypatch.setattr(
            agent_service, "inspect_agent_run", _owned_by_another_user
        )
        response = await client.get(
            "/agent/runs/another-users-run", headers=registered_user["headers"]
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Agent run 不存在"

    async def test_recover_rejects_nonrecoverable_checkpoint(
        self, client, registered_user, monkeypatch, mock_agent
    ):
        first = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行"},
        )
        run_id = first.json()["run_id"]

        def _terminal_run(*args, **kwargs):
            raise agent_service.AgentRunStateError("Agent run 已结束")

        monkeypatch.setattr(agent_service, "recover_agent", _terminal_run)
        response = await client.post(
            f"/agent/runs/{run_id}/recover", headers=registered_user["headers"]
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Agent run 已结束"

    async def test_interrupt_returns_approval_without_assistant_message(
        self, client, db_session, registered_user, monkeypatch, mock_agent
    ):
        def _waiting(
            user_id, message, history, document_id, requested_skill=None, **kwargs
        ):
            return {
                "run_id": kwargs["thread_id"],
                "thread_id": kwargs["thread_id"],
                "conversation_id": kwargs["conversation_id"],
                "status": "waiting_approval",
                "answer": "检测到高风险工具调用，等待人工审批后继续。",
                "artifacts": [],
                "trace": [],
                "approval": {
                    "tool": "risk_tool",
                    "args": {"path": "x"},
                    "reason": "会修改外部状态",
                },
            }

        monkeypatch.setattr(agent_service, "run_agent", _waiting)
        response = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行高风险工具"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "waiting_approval"
        assert body["approval"]["tool"] == "risk_tool"
        assert body["run_id"]
        roles = (
            (
                await db_session.execute(
                    select(Message.role).where(
                        Message.conversation_id == body["conversation_id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert roles == [MessageRole.USER]

    async def test_resume_completes_run_and_persists_assistant(
        self, client, db_session, registered_user, mock_agent
    ):
        first = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行"},
        )
        run_id = first.json()["run_id"]
        conversation_id = first.json()["conversation_id"]

        response = await client.post(
            "/agent/resume",
            headers=registered_user["headers"],
            json={"run_id": run_id, "approved": True, "comment": "确认"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["answer"] == "恢复后完成"
        assistant = (
            await db_session.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.role == MessageRole.ASSISTANT,
                    Message.content == "恢复后完成",
                )
            )
        ).scalar_one()
        assert "risk_tool" in (assistant.sources or "")

    async def test_run_status_is_recoverable(self, client, registered_user, mock_agent):
        first = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行"},
        )
        run_id = first.json()["run_id"]
        response = await client.get(
            f"/agent/runs/{run_id}", headers=registered_user["headers"]
        )
        assert response.status_code == 200
        assert response.json()["status"] == "waiting_approval"
        assert response.json()["approval"]["tool"] == "risk_tool"

    async def test_run_status_rejects_oversized_id(
        self, client, registered_user, mock_agent
    ):
        response = await client.get(
            f"/agent/runs/{'x' * 129}", headers=registered_user["headers"]
        )
        assert response.status_code == 422

    async def test_incomplete_run_can_continue_from_checkpoint(
        self, client, registered_user, mock_agent
    ):
        first = await client.post(
            "/agent/chat",
            headers=registered_user["headers"],
            json={"message": "执行"},
        )
        run_id = first.json()["run_id"]
        response = await client.post(
            f"/agent/runs/{run_id}/recover", headers=registered_user["headers"]
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "checkpoint 恢复完成"
        assert response.json()["status"] == "completed"

    async def test_resume_requires_auth(self, client, mock_agent):
        response = await client.post(
            "/agent/resume", json={"run_id": "x", "approved": True}
        )
        assert response.status_code == 401
