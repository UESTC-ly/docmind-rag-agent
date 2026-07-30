"""v3 LangGraph Agent 的中断、审批与 checkpoint 恢复契约。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent import orchestrator
from app.services.provider_telemetry import (
    record_provider_attempt,
    record_provider_response,
)
from app.skills import registry
from app.skills.base import BaseSkill, SkillContext


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMsg:
    def __init__(self, content: str | None = None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _RiskSkill(BaseSkill):
    name = "risk_test_skill"
    description = "测试人工审批"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
    requires_approval = True
    executions: list[str] = []

    def run(self, context: SkillContext, **kwargs) -> dict:
        value = str(kwargs.get("value") or "")
        self.executions.append(value)
        return {"type": "echo", "value": value}


class _SafeSkill(BaseSkill):
    name = "safe_test_skill"
    description = "测试 checkpoint 后重放"
    parameters = {"type": "object", "properties": {}}
    executions = 0

    def run(self, context: SkillContext, **kwargs) -> dict:
        type(self).executions += 1
        return {"type": "echo", "value": "once"}


class _CapabilityRiskSkill(_SafeSkill):
    name = "capability_risk_skill"

    def apply_package_metadata(self) -> None:
        pass

    def load_package(self):
        return SimpleNamespace(required_capabilities=("browser", "workspace"))


class _ExplodingSkill(_SafeSkill):
    name = "exploding_test_skill"

    def run(self, context: SkillContext, **kwargs) -> dict:
        raise RuntimeError("boom")


@pytest.fixture
def risk_skill():
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_RiskSkill)
    registry.register_skill(_SafeSkill)
    registry.register_skill(_CapabilityRiskSkill)
    registry.register_skill(_ExplodingSkill)
    _RiskSkill.executions.clear()
    _SafeSkill.executions = 0
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    _RiskSkill.executions.clear()
    _SafeSkill.executions = 0


def _risk_call(value: str = "original") -> _FakeMsg:
    return _FakeMsg(
        tool_calls=[
            _FakeToolCall(
                "risk-call-1",
                "risk_test_skill",
                json.dumps({"value": value}),
            )
        ]
    )


def test_outer_orchestration_is_a_langgraph():
    graph = orchestrator.build_agent_graph()
    nodes = set(graph.get_graph().nodes)
    assert {
        "supervisor",
        "select_tool",
        "approval_gate",
        "execute_tool",
    } <= nodes


def test_generic_high_risk_capability_requires_outer_approval(risk_skill):
    skill = registry.get_skill("capability_risk_skill")
    required, capabilities, reason = orchestrator._risk_metadata(skill)
    assert required is True
    assert capabilities == ["browser"]
    assert "高风险宿主能力" in reason


def test_small_graph_helper_edges(monkeypatch, risk_skill):
    assert orchestrator._json_arguments({"x": 1}) == {"x": 1}
    assert orchestrator._json_arguments("not-json") == {}
    assert orchestrator._json_arguments("[]") == {}
    assert orchestrator._risk_metadata(None) == (False, [], "")
    assert orchestrator._select_tool({"pending_tool_calls": []}) == {
        "current_tool": None
    }
    assert orchestrator._approval_gate({"current_tool": None}) == {}

    monkeypatch.setattr(
        orchestrator.settings, "agent_high_risk_skills", "safe_test_skill"
    )
    required, capabilities, reason = orchestrator._risk_metadata(
        registry.get_skill("safe_test_skill")
    )
    assert required is True
    assert capabilities == []
    assert "部署策略" in reason


def test_skill_exception_becomes_tool_observation(monkeypatch, risk_skill):
    responses = iter(
        [
            _FakeMsg(
                tool_calls=[_FakeToolCall("explode-call", "exploding_test_skill", "{}")]
            ),
            _FakeMsg(content="已报告失败"),
        ]
    )
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )
    result = orchestrator.run_agent(user_id=1, question="explode")
    assert result["answer"] == "已报告失败"
    assert result["trace"][0]["ok"] is False


def test_invalid_selected_skill_and_history_paths(monkeypatch, risk_skill):
    invalid = orchestrator.run_agent(
        user_id=1,
        question="q",
        requested_skill="missing_selected_skill",
    )
    assert "不存在" in invalid["answer"]

    captured = {}

    def _llm(messages, tools=None, tool_choice="auto"):
        captured["messages"] = messages
        return _FakeMsg(content="history ok")

    monkeypatch.setattr(orchestrator, "chat_completion", _llm)
    result = orchestrator.run_agent(
        user_id=1,
        question="new",
        history=[{"role": "assistant", "content": "old"}],
    )
    assert result["answer"] == "history ok"
    assert captured["messages"][-2] == {"role": "assistant", "content": "old"}


def test_high_risk_tool_interrupts_before_side_effect_and_recovers_after_restart(
    monkeypatch, tmp_path, risk_skill
):
    responses = iter([_risk_call(), _FakeMsg(content="审批后完成")])
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"

    interrupted = orchestrator.run_agent(
        user_id=7,
        question="执行高风险操作",
        thread_id="run-risk-1",
        conversation_id=23,
        checkpoint_path=checkpoint_path,
    )

    assert interrupted["status"] == "waiting_approval"
    assert interrupted["run_id"] == "run-risk-1"
    assert interrupted["approval"]["tool"] == "risk_test_skill"
    assert interrupted["approval"]["args"] == {"value": "original"}
    assert _RiskSkill.executions == []

    # resume_agent 每次重新打开 SQLite saver 并重新 compile graph；这验证了不是
    # 依赖进程内 graph 对象的“伪恢复”。
    completed = orchestrator.resume_agent(
        run_id="run-risk-1",
        approved=True,
        comment="同意本次调用",
        checkpoint_path=checkpoint_path,
        expected_user_id=7,
    )

    assert completed["status"] == "completed"
    assert completed["answer"] == "审批后完成"
    assert _RiskSkill.executions == ["original"]
    assert completed["trace"][0]["approval"] == {
        "required": True,
        "approved": True,
        "comment": "同意本次调用",
    }


def test_provider_telemetry_preserves_interrupt_checkpoint_and_accumulates_on_resume(
    monkeypatch, tmp_path, risk_skill
):
    responses = iter([_risk_call(), _FakeMsg(content="审批后完成")])

    def _llm(messages, tools=None, tool_choice="auto"):
        record_provider_attempt(model="test-provider-model")
        record_provider_response(
            {
                "model": "test-provider-model",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        )
        return next(responses)

    monkeypatch.setattr(orchestrator, "chat_completion", _llm)
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"

    waiting = orchestrator.run_agent(
        user_id=71,
        question="执行高风险操作",
        thread_id="telemetry-interrupt",
        checkpoint_path=checkpoint_path,
    )
    inspected = orchestrator.inspect_agent_run(
        "telemetry-interrupt",
        checkpoint_path=checkpoint_path,
        expected_user_id=71,
    )

    assert waiting["status"] == "waiting_approval"
    assert inspected["status"] == "waiting_approval"
    assert inspected["provider_usage"]["total_tokens"] == 5.0
    assert inspected["provider_usage"]["request_count"] == 1

    completed = orchestrator.resume_agent(
        "telemetry-interrupt",
        approved=True,
        checkpoint_path=checkpoint_path,
        expected_user_id=71,
    )

    assert completed["status"] == "completed"
    assert completed["provider_usage"]["total_tokens"] == 10.0
    assert completed["provider_usage"]["request_count"] == 2


def test_rejection_skips_side_effect_and_returns_observation_to_supervisor(
    monkeypatch, tmp_path, risk_skill
):
    captured: dict = {}
    calls = {"count": 0}

    def _llm(messages, tools=None, tool_choice="auto"):
        calls["count"] += 1
        if calls["count"] == 1:
            return _risk_call()
        captured["tool_message"] = messages[-1]
        return _FakeMsg(content="操作已按你的要求取消")

    monkeypatch.setattr(orchestrator, "chat_completion", _llm)
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=8,
        question="尝试执行",
        thread_id="run-risk-reject",
        checkpoint_path=checkpoint_path,
    )

    result = orchestrator.resume_agent(
        run_id="run-risk-reject",
        approved=False,
        comment="权限范围不合适",
        checkpoint_path=checkpoint_path,
        expected_user_id=8,
    )

    assert result["status"] == "completed"
    assert result["answer"] == "操作已按你的要求取消"
    assert _RiskSkill.executions == []
    assert captured["tool_message"]["role"] == "tool"
    assert "用户拒绝" in captured["tool_message"]["content"]
    assert result["trace"][0]["approval"]["approved"] is False


def test_approver_can_edit_arguments_before_execution(
    monkeypatch, tmp_path, risk_skill
):
    responses = iter([_risk_call(), _FakeMsg(content="已按修订参数执行")])
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=9,
        question="执行",
        thread_id="run-risk-edit",
        checkpoint_path=checkpoint_path,
    )

    result = orchestrator.resume_agent(
        run_id="run-risk-edit",
        approved=True,
        edited_args={"value": "reviewed"},
        checkpoint_path=checkpoint_path,
        expected_user_id=9,
    )

    assert result["status"] == "completed"
    assert _RiskSkill.executions == ["reviewed"]
    assert result["trace"][0]["args"] == {"value": "reviewed"}


def test_checkpoint_status_enforces_owner_and_reports_interrupt(
    monkeypatch, tmp_path, risk_skill
):
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": _risk_call(),
    )
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=10,
        question="执行",
        thread_id="owned-run",
        conversation_id=99,
        checkpoint_path=checkpoint_path,
    )

    status = orchestrator.inspect_agent_run(
        "owned-run", checkpoint_path=checkpoint_path, expected_user_id=10
    )
    assert status["status"] == "waiting_approval"
    assert status["conversation_id"] == 99
    assert status["approval"]["tool"] == "risk_test_skill"

    with pytest.raises(orchestrator.AgentRunOwnershipError):
        orchestrator.inspect_agent_run(
            "owned-run", checkpoint_path=checkpoint_path, expected_user_id=11
        )


def test_unknown_checkpoint_cannot_be_resumed(tmp_path):
    with pytest.raises(orchestrator.AgentRunNotFoundError):
        orchestrator.resume_agent(
            run_id="does-not-exist",
            approved=True,
            checkpoint_path=tmp_path / "agent-checkpoints.sqlite3",
            expected_user_id=1,
        )


def test_recover_reuses_tool_receipt_instead_of_repeating_side_effect(
    monkeypatch, tmp_path, risk_skill
):
    responses = iter(
        [
            _FakeMsg(
                tool_calls=[_FakeToolCall("safe-call-1", "safe_test_skill", "{}")]
            ),
            _FakeMsg(content="恢复后完成"),
        ]
    )
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    original_record = orchestrator._record_tool_result
    crashed = {"value": False}

    def _crash_after_receipt(state, current, result):
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("simulated process exit after tool receipt")
        return original_record(state, current, result)

    monkeypatch.setattr(orchestrator, "_record_tool_result", _crash_after_receipt)
    with pytest.raises(RuntimeError, match="simulated process exit"):
        orchestrator.run_agent(
            user_id=12,
            question="只执行一次",
            thread_id="recover-once",
            conversation_id=55,
            checkpoint_path=checkpoint_path,
        )
    assert _SafeSkill.executions == 1

    status = orchestrator.inspect_agent_run(
        "recover-once", checkpoint_path=checkpoint_path, expected_user_id=12
    )
    assert status["status"] == "running"
    assert status["recoverable"] is True

    monkeypatch.setattr(orchestrator, "_record_tool_result", original_record)
    result = orchestrator.recover_agent(
        "recover-once",
        checkpoint_path=checkpoint_path,
        expected_user_id=12,
    )

    assert result["status"] == "completed"
    assert result["answer"] == "恢复后完成"
    assert _SafeSkill.executions == 1


def test_provider_tool_call_ids_can_repeat_across_supervisor_rounds(
    monkeypatch, tmp_path, risk_skill
):
    responses = iter(
        [
            _FakeMsg(
                tool_calls=[_FakeToolCall("reused-id", "safe_test_skill", "{}")]
            ),
            _FakeMsg(
                tool_calls=[_FakeToolCall("reused-id", "safe_test_skill", "{}")]
            ),
            _FakeMsg(content="两轮均已执行"),
        ]
    )
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )

    result = orchestrator.run_agent(
        user_id=12,
        question="执行两轮",
        thread_id="provider-reused-tool-id",
        checkpoint_path=tmp_path / "agent-checkpoints.sqlite3",
    )

    assert result["answer"] == "两轮均已执行"
    assert len(result["trace"]) == 2
    assert _SafeSkill.executions == 2


def test_duplicate_run_id_is_rejected(monkeypatch, tmp_path, risk_skill):
    responses = iter([_FakeMsg(content="first")])
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=13,
        question="first",
        thread_id="same-run",
        checkpoint_path=checkpoint_path,
    )

    with pytest.raises(orchestrator.AgentRunStateError, match="已存在"):
        orchestrator.run_agent(
            user_id=13,
            question="second",
            thread_id="same-run",
            checkpoint_path=checkpoint_path,
        )


def test_uncertain_inflight_tool_interrupts_instead_of_silent_retry(
    monkeypatch, tmp_path, risk_skill
):
    responses = iter(
        [
            _FakeMsg(
                tool_calls=[_FakeToolCall("uncertain-call", "safe_test_skill", "{}")]
            ),
            _FakeMsg(content="没有重复执行"),
        ]
    )
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": next(responses),
    )
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    original_complete = orchestrator._complete_tool_receipt

    def _crash_before_receipt_completion(state, current, result):
        raise RuntimeError("simulated uncertain side effect")

    monkeypatch.setattr(
        orchestrator, "_complete_tool_receipt", _crash_before_receipt_completion
    )
    with pytest.raises(RuntimeError, match="uncertain side effect"):
        orchestrator.run_agent(
            user_id=14,
            question="不要重复",
            thread_id="uncertain-run",
            conversation_id=77,
            checkpoint_path=checkpoint_path,
        )
    assert _SafeSkill.executions == 1

    monkeypatch.setattr(orchestrator, "_complete_tool_receipt", original_complete)
    waiting = orchestrator.recover_agent(
        "uncertain-run",
        checkpoint_path=checkpoint_path,
        expected_user_id=14,
    )
    assert waiting["status"] == "waiting_approval"
    assert waiting["approval"]["kind"] == "execution_recovery"
    assert _SafeSkill.executions == 1

    completed = orchestrator.resume_agent(
        "uncertain-run",
        approved=False,
        comment="不重试不确定副作用",
        checkpoint_path=checkpoint_path,
        expected_user_id=14,
    )
    assert completed["status"] == "completed"
    assert completed["answer"] == "没有重复执行"
    assert completed["trace"][0]["recovery"]["retried"] is False
    assert _SafeSkill.executions == 1


def test_terminal_and_interrupt_recovery_guards(monkeypatch, tmp_path, risk_skill):
    checkpoint_path = tmp_path / "agent-checkpoints.sqlite3"
    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": _FakeMsg(content="done"),
    )
    orchestrator.run_agent(
        user_id=15,
        question="done",
        thread_id="terminal-run",
        checkpoint_path=checkpoint_path,
    )
    with pytest.raises(orchestrator.AgentRunStateError, match="不在等待审批"):
        orchestrator.resume_agent(
            "terminal-run",
            approved=True,
            checkpoint_path=checkpoint_path,
            expected_user_id=15,
        )
    with pytest.raises(orchestrator.AgentRunStateError, match="已完成"):
        orchestrator.recover_agent(
            "terminal-run",
            checkpoint_path=checkpoint_path,
            expected_user_id=15,
        )

    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": _risk_call(),
    )
    orchestrator.run_agent(
        user_id=15,
        question="wait",
        thread_id="waiting-run",
        checkpoint_path=checkpoint_path,
    )
    with pytest.raises(orchestrator.AgentRunStateError, match="等待人工审批"):
        orchestrator.recover_agent(
            "waiting-run",
            checkpoint_path=checkpoint_path,
            expected_user_id=15,
        )
