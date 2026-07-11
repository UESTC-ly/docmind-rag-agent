"""Agent 编排主循环测试（run_agent 的 ReAct 逻辑）。

不连 LLM：用脚本化的假 chat_completion 逐步返回 tool_calls / 最终答案，
验证：直接回答、单步工具调用+执行、artifacts 收集、文件 artifact 精简回传、
未知技能兜底、最大步数保护、
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

    def _fn(messages, tools=None, tool_choice="auto"):
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


class _FileSkill(BaseSkill):
    name = "fake_file_skill"
    description = "测试用文件产出"
    parameters = {"type": "object", "properties": {}}

    def run(self, context: SkillContext, **kwargs) -> dict:
        return {
            "type": "presentation",
            "artifact_kind": "file",
            "download": {
                "filename": "x.pptx",
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "encoding": "base64",
                "content": "VERY-LARGE-BASE64",
            },
        }


class _UnavailableSkill(BaseSkill):
    name = "unavailable_test_skill"
    description = "测试用隔离技能"
    parameters = {"type": "object", "properties": {}}
    available = False
    unavailable_reason = "缺少测试 adapter"

    def run(self, context: SkillContext, **kwargs) -> dict:
        raise AssertionError("隔离技能不应执行")


@pytest.fixture
def fake_skills():
    """临时注册假技能，测试后还原注册表，避免污染其他测试。"""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_EchoSkill)
    registry.register_skill(_MindmapSkill)
    registry.register_skill(_FileSkill)
    registry.register_skill(_UnavailableSkill)
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

    def test_selected_skill_is_forced_on_first_model_turn(self, monkeypatch, fake_skills):
        captured = []

        def _llm(messages, tools=None, tool_choice="auto"):
            captured.append(tool_choice)
            if len(captured) == 1:
                return _FakeMsg(
                    tool_calls=[_FakeToolCall("c1", "echo_test_skill", '{"text":"hi"}')]
                )
            return _FakeMsg(content="已执行指定技能")

        monkeypatch.setattr(orchestrator, "chat_completion", _llm)
        result = orchestrator.run_agent(
            user_id=1,
            question="生成材料",
            requested_skill="echo_test_skill",
        )

        assert captured[0] == {
            "type": "function",
            "function": {"name": "echo_test_skill"},
        }
        assert captured[1] == "auto"
        assert result["trace"][0]["skill"] == "echo_test_skill"

    def test_document_scoped_agent_must_use_a_tool_before_answering(
        self, monkeypatch, fake_skills
    ):
        captured = []

        def _llm(messages, tools=None, tool_choice="auto"):
            captured.append(tool_choice)
            if len(captured) == 1:
                return _FakeMsg(
                    tool_calls=[_FakeToolCall("c1", "echo_test_skill", '{"text":"doc"}')]
                )
            return _FakeMsg(content="基于工具结果回答")

        monkeypatch.setattr(orchestrator, "chat_completion", _llm)
        orchestrator.run_agent(user_id=1, question="根据文档回答", document_id=42)

        assert captured[0] == {
            "type": "function",
            "function": {"name": "search_knowledge_base"},
        }

    def test_direct_answer_is_rejected_when_selected_skill_was_forced(
        self, monkeypatch, fake_skills
    ):
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda messages, tools=None, tool_choice="auto": _FakeMsg(
                content="我直接在聊天里生成了材料"
            ),
        )
        result = orchestrator.run_agent(
            user_id=1,
            question="生成材料",
            requested_skill="echo_test_skill",
        )
        assert "没有接受模型的直接回答" in result["answer"]
        assert "我直接在聊天里生成了材料" not in result["answer"]

    def test_unavailable_selected_skill_is_rejected_before_llm(
        self, monkeypatch, fake_skills
    ):
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("隔离技能不应进入 LLM")
            ),
        )
        result = orchestrator.run_agent(
            user_id=1,
            question="执行",
            requested_skill="unavailable_test_skill",
        )
        assert "缺少测试 adapter" in result["answer"]


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

    def test_file_artifact_collected_but_download_content_omitted_from_llm(
        self, monkeypatch, fake_skills
    ):
        captured = {}
        calls = {"n": 0}

        def _llm(messages, tools=None, tool_choice="auto"):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeMsg(tool_calls=[_FakeToolCall("c1", "fake_file_skill", "{}")])
            captured["tool_message"] = messages[-1]["content"]
            return _FakeMsg(content="文件已生成")

        monkeypatch.setattr(orchestrator, "chat_completion", _llm)
        result = orchestrator.run_agent(user_id=1, question="生成文件")

        assert result["artifacts"][0]["type"] == "presentation"
        assert result["artifacts"][0]["download"]["content"] == "VERY-LARGE-BASE64"
        assert "content_omitted" in captured["tool_message"]
        assert "VERY-LARGE-BASE64" not in captured["tool_message"]

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

    def test_model_cannot_bypass_quarantine_by_hallucinating_unavailable_skill(
        self, monkeypatch, fake_skills
    ):
        script = [
            _FakeMsg(
                tool_calls=[_FakeToolCall("c1", "unavailable_test_skill", "{}")]
            ),
            _FakeMsg(content="已拒绝不可用技能"),
        ]
        monkeypatch.setattr(orchestrator, "chat_completion", _scripted_llm(script))
        result = orchestrator.run_agent(user_id=1, question="调用隔离技能")
        assert result["trace"][0]["ok"] is False
        assert result["answer"] == "已拒绝不可用技能"


class TestGuards:
    def test_max_steps_guard(self, monkeypatch, fake_skills):
        # LLM 每轮都返回 tool_call、永不收敛 → 应在 max_steps 后兜底返回
        always_tool = _FakeMsg(
            tool_calls=[_FakeToolCall("c1", "echo_test_skill", '{"text":"x"}')]
        )
        monkeypatch.setattr(orchestrator.settings, "agent_max_steps", 3)
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda messages, tools=None, tool_choice="auto": always_tool,
        )
        result = orchestrator.run_agent(user_id=1, question="死循环")
        assert "未能在限定步数内完成" in result["answer"]
        assert len(result["trace"]) == 3  # 恰好跑满 max_steps


class TestDocumentScope:
    def test_document_id_injected_into_system_prompt(self, monkeypatch, fake_skills):
        captured = {}

        def _capture_llm(messages, tools=None, tool_choice="auto"):
            captured["system"] = messages[0]["content"]
            return _FakeMsg(content="ok")

        monkeypatch.setattr(orchestrator, "chat_completion", _capture_llm)
        orchestrator.run_agent(user_id=1, question="q", document_id=42)
        assert "document_id=42" in captured["system"]
