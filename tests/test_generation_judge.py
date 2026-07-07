"""LLM-as-judge 测试。

打桩 chat_completion，验证 JSON 解析、markdown 代码块剥离、score clamp、
异常兜底为 0.0，以及 prompt 拼装。
"""

import pytest

from app.services.evaluation import generation_judge


class _FakeMsg:
    def __init__(self, content):
        self.content = content


def _patch_llm(monkeypatch, content):
    monkeypatch.setattr(
        generation_judge, "chat_completion",
        lambda messages, temperature=0.0: _FakeMsg(content),
    )


class TestCallJudge:
    def test_parses_plain_json_score(self, monkeypatch):
        _patch_llm(monkeypatch, '{"score": 0.8, "reason": "OK"}')
        assert generation_judge.judge_answer_relevancy("q", "a") == 0.8

    def test_strips_markdown_fence(self, monkeypatch):
        _patch_llm(monkeypatch, '```json\n{"score": 0.5, "reason": "x"}\n```')
        assert generation_judge.judge_faithfulness(["c"], "a") == 0.5

    def test_score_clamped_above_1(self, monkeypatch):
        _patch_llm(monkeypatch, '{"score": 1.5}')
        assert generation_judge.judge_answer_relevancy("q", "a") == 1.0

    def test_score_clamped_below_0(self, monkeypatch):
        _patch_llm(monkeypatch, '{"score": -0.3}')
        assert generation_judge.judge_answer_relevancy("q", "a") == 0.0

    def test_invalid_json_returns_zero(self, monkeypatch):
        _patch_llm(monkeypatch, "这不是JSON")
        assert generation_judge.judge_faithfulness(["c"], "a") == 0.0

    def test_llm_exception_returns_zero(self, monkeypatch):
        def _boom(messages, temperature=0.0):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(generation_judge, "chat_completion", _boom)
        assert generation_judge.judge_answer_relevancy("q", "a") == 0.0


class TestPromptAssembly:
    def test_faithfulness_includes_context_and_answer(self, monkeypatch):
        captured = {}

        def _capture(messages, temperature=0.0):
            captured["prompt"] = messages[0]["content"]
            return _FakeMsg('{"score": 1.0}')

        monkeypatch.setattr(generation_judge, "chat_completion", _capture)
        generation_judge.judge_faithfulness(["片段A", "片段B"], "我的答案")
        assert "片段A" in captured["prompt"]
        assert "片段B" in captured["prompt"]
        assert "我的答案" in captured["prompt"]

    def test_relevancy_includes_question_and_answer(self, monkeypatch):
        captured = {}

        def _capture(messages, temperature=0.0):
            captured["prompt"] = messages[0]["content"]
            return _FakeMsg('{"score": 1.0}')

        monkeypatch.setattr(generation_judge, "chat_completion", _capture)
        generation_judge.judge_answer_relevancy("我的问题", "我的答案")
        assert "我的问题" in captured["prompt"]
        assert "我的答案" in captured["prompt"]
