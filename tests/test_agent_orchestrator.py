"""Agent 编排主循环测试（run_agent 的 ReAct 逻辑）。

不连 LLM：用脚本化的假 chat_completion 逐步返回 tool_calls / 最终答案，
验证：直接回答、单步工具调用+执行、artifacts 收集、未知技能兜底、最大步数保护、
以及 document_id 会注入系统提示。
"""

import pytest

from app.agent import orchestrator
from app.skills.base import BaseSkill, SkillContext
from app.skills import registry


# ── 构造假的 LLM 消息 / 工具调用对象 ──────────────────────────
class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def _scripted_llm(responses):
    """返回一个假的 chat_completion：按调用次序依次吐出 responses 里的消息。"""
    seq = iter(responses)

    def _fn(messages, tools=None):
        return next(seq)

    return _fn


# ── 假技能：一个普通 echo，一个产出 artifact ─────────────────
class _EchoSkill(BaseSkill):
    name = "echo_test_skill"
    description = "测试用回显"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, context: SkillContext, **kwargs) -> dict:
        return {"type": "echo", "text": kwargs.get("text", "")}


class _MindmapSkill(BaseSkill):
    name = "fake_mindmap_skill"
    description = "测试用思维导图"
    parameters = {"type": "object", "properties": {}}

    def run(self, context: SkillContext, **kwargs) -> dict:
        return {"type": "mindmap", "nodes": ["a", "b"]}


@pytest.fixture
def fake_skills():
    """临时注册假技能，测试后还原注册表，避免污染其他测试。"""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_EchoSkill)
    registry.register_skill(_MindmapSkill)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


class TestDirectAnswer:
    def test_no_tool_call_returns_answer(self, monkeypatch, fake_skills):
        monkeypatch.setattr(
            orchestrator, "chat_completion",
            _scripted_llm([_FakeMsg(content="直接回答，无需工具")]),
        )
        result = orchestrator.run_agent(user_id=1, question="你好")
        assert result["answer"] == "直接回答，无需工具"
        assert result["trace"] == []
        assert result["artifacts"] == []


class TestToolLoop:
    def test_single_tool_call_then_final_answer(self, monkeypatch, fake_skills):
        # 第一轮：调 echo 工具；第二轮：给最终答案
        script = [
            _FakeMsg(tool_calls=[_FakeToolCall("c1", "echo_test_skill", '{"text":"hi"}')]),
            _FakeMsg(content="工具执行完毕，结果是 hi"),
        ]
        monkeypatch.setattr(orchestrator, "chat_completion", _scripted_llm(script))
        result = orchestrator.run_agent(user_id=1, question="回显 hi")
        assert result["answer"] == "工具执行完毕，结果是 hi"
        assert len(result["trace"]) == 1
        assert result["trace"][0]["skill"] == "echo_test_skill"
        assert result["trace"][0]["args"] == {"text": "hi"}

    def test_artifact_skill_collected(self, monkeypatch, fake_skills):
        script = [
            _FakeMsg(tool_calls=[_FakeToolCall("c1", "fake_mindmap_skill", "{}")]),
            _FakeMsg(content="思维导图已生成"),
        ]
        monkeypatch.setattr(orchestrator, "chat_completion", _scripted_llm(script))
        result = orchestrator.run_agent(user_id=1, question="生成导图")
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["type"] == "mindmap"

    def test_unknown_skill_does_not_crash(self, monkeypatch, fake_skills):
        # LLM 幻觉出一个不存在的技能名，编排器应兜底而非抛异常
        script = [
            _FakeMsg(tool_calls=[_FakeToolCall("c1", "nonexistent_skill", "{}")]),
            _FakeMsg(content="已处理"),
        ]
        monkeypatch.setattr(orchestrator, "chat_completion", _scripted_llm(script))
        result = orchestrator.run_agent(user_id=1, question="调用不存在的技能")
        assert result["answer"] == "已处理"
        assert result["trace"][0]["skill"] == "nonexistent_skill"


class TestGuards:
    def test_max_steps_guard(self, monkeypatch, fake_skills):
        # LLM 每轮都返回 tool_call、永不收敛 → 应在 max_steps 后兜底返回
        always_tool = _FakeMsg(
            tool_calls=[_FakeToolCall("c1", "echo_test_skill", '{"text":"x"}')]
        )
        monkeypatch.setattr(orchestrator.settings, "agent_max_steps", 3)
        monkeypatch.setattr(
            orchestrator, "chat_completion", lambda messages, tools=None: always_tool
        )
        result = orchestrator.run_agent(user_id=1, question="死循环")
        assert "未能在限定步数内完成" in result["answer"]
        assert len(result["trace"]) == 3  # 恰好跑满 max_steps


class TestDocumentScope:
    def test_document_id_injected_into_system_prompt(self, monkeypatch, fake_skills):
        captured = {}

        def _capture_llm(messages, tools=None):
            captured["system"] = messages[0]["content"]
            return _FakeMsg(content="ok")

        monkeypatch.setattr(orchestrator, "chat_completion", _capture_llm)
        orchestrator.run_agent(user_id=1, question="q", document_id=42)
        assert "document_id=42" in captured["system"]
