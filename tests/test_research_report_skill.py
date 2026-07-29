"""Verified Agentic research-report workflow tests."""

from types import SimpleNamespace

import pytest

from app.services.evidence import attach_citation_ids, validate_citations
from app.skills.base import SkillContext
from app.skills import research_report


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None


@pytest.fixture(autouse=True)
def semantic_grounding(monkeypatch):
    def evaluate(answer, hits):
        report = validate_citations(answer, hits)
        for claim in report["claims"]:
            supported = bool(claim["valid_citations"]) and not claim[
                "invalid_citations"
            ]
            claim["semantically_supported"] = supported
            claim["reason"] = "fixture"
        count = report["claim_count"]
        grounded = (
            sum(
                bool(claim["semantically_supported"])
                for claim in report["claims"]
            )
            / count
            if count
            else None
        )
        return {
            **report,
            "contract": "claim_grounding_v1",
            "semantic_entailment_checked": bool(count),
            "judge_status": "completed" if count else "not_applicable",
            "groundedness": grounded,
            "faithfulness": grounded,
            "citation_correctness": (
                report["citation_precision"] if count else None
            ),
            "conflict_count": 0,
            "conflicts": [],
            "passed": report["passed"],
        }

    monkeypatch.setattr(research_report, "evaluate_grounding", evaluate)
    monkeypatch.setattr(
        research_report,
        "_automatic_pipeline_decision",
        lambda user_id: {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "no_eligible_pipeline",
            "reason": "fixture",
            "selected": None,
            "candidates": [],
            "dataset": None,
        },
    )


def _execution():
    hits = attach_citation_ids(
        [
            {
                "document_id": 3,
                "chunk_index": 1,
                "content": "DocMind 使用混合检索。",
                "score": 0.9,
            }
        ]
    )
    spec = SimpleNamespace(
        id="hybrid-rerank",
        to_dict=lambda: {
            "id": "hybrid-rerank",
            "retriever": "hybrid",
            "fusion": "rrf",
            "reranker": "local",
        },
    )
    return SimpleNamespace(
        hits=hits,
        spec=spec,
        fingerprint="abc123",
        trace=[{"stage": "retrieve", "component": "hybrid"}],
    )


def test_workflow_plans_retrieves_drafts_and_passes_verification(monkeypatch):
    responses = iter(
        [
            _FakeMessage(
                '{"sections":[{"title":"架构","query":"DocMind 检索架构"},'
                '{"title":"能力","query":"DocMind 主要能力"}]}'
            ),
            _FakeMessage(
                "# DocMind\n\n## 架构\n\nDocMind 使用混合检索。[D3:C1]"
            ),
        ]
    )
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return _execution()

    monkeypatch.setattr(research_report, "run_pipeline_for_skill", retrieve)
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )
    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7),
        topic="DocMind",
        section_count=2,
        pipeline_id="hybrid-rerank",
    )

    assert len(calls) == 2
    assert all(call["pipeline"] == "hybrid-rerank" for call in calls)
    assert result["workflow"]["status"] == "completed"
    assert result["workflow"]["repair_attempted"] is False
    assert result["verification"]["passed"] is True
    assert result["grounding"]["pipeline_fingerprint"] == "abc123"
    assert result["download"]["content"] == result["content"]


def test_workflow_repairs_an_uncited_draft(monkeypatch):
    responses = iter(
        [
            _FakeMessage("not-json"),
            _FakeMessage("# DocMind\n\nDocMind 使用混合检索。"),
            _FakeMessage("# DocMind\n\nDocMind 使用混合检索。[D3:C1]"),
        ]
    )
    monkeypatch.setattr(
        research_report,
        "run_pipeline_for_skill",
        lambda **kwargs: _execution(),
    )
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )

    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7, document_id=3),
        topic="DocMind",
        section_count=2,
    )
    assert result["workflow"]["repair_attempted"] is True
    assert result["workflow"]["fail_safe_applied"] is False
    assert result["verification"]["passed"] is True
    assert "[D3:C1]" in result["content"]


def test_workflow_accepts_registered_plugin_pipeline_id(monkeypatch):
    responses = iter(
        [
            _FakeMessage("not-json"),
            _FakeMessage("# DocMind\n\nDocMind 使用混合检索。[D3:C1]"),
        ]
    )
    selected = []

    def retrieve(**kwargs):
        selected.append(kwargs["pipeline"])
        return _execution()

    monkeypatch.setattr(research_report, "run_pipeline_for_skill", retrieve)
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )

    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7),
        topic="DocMind",
        section_count=2,
        pipeline_id="trusted-plugin-pipeline",
    )

    assert "enum" not in research_report.VerifiedResearchReportSkill.parameters[
        "properties"
    ]["pipeline_id"]
    assert selected == ["trusted-plugin-pipeline", "trusted-plugin-pipeline"]
    assert result["verification"]["passed"] is True


def test_workflow_removes_claims_that_still_fail_after_repair(monkeypatch):
    responses = iter(
        [
            _FakeMessage("not-json"),
            _FakeMessage("# 报告\n\n没有引用的断言。"),
            _FakeMessage("# 报告\n\n仍然没有引用的断言。"),
        ]
    )
    monkeypatch.setattr(
        research_report,
        "run_pipeline_for_skill",
        lambda **kwargs: _execution(),
    )
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )
    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7),
        topic="DocMind",
        section_count=2,
    )
    assert result["workflow"]["fail_safe_applied"] is True
    assert "仍然没有引用的断言" not in result["content"]
    assert "自动移除" in result["content"]
    assert result["verification"]["passed"] is False
    assert result["workflow"]["status"] == "failed"


def test_workflow_stops_when_no_evidence_is_retrieved(monkeypatch):
    responses = iter([_FakeMessage("not-json")])
    empty = _execution()
    empty.hits.clear()
    monkeypatch.setattr(
        research_report,
        "run_pipeline_for_skill",
        lambda **kwargs: empty,
    )
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )
    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7),
        topic="DocMind",
        section_count=2,
    )
    assert "error" in result
    assert result["workflow"]["status"] == "failed"


def test_workflow_uses_public_eval_selection_and_rewrites_weak_query(
    monkeypatch,
):
    monkeypatch.setattr(
        research_report,
        "_automatic_pipeline_decision",
        lambda user_id: {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "selected",
            "reason": "public regression winner",
            "dataset": {"id": 9},
            "selected": {
                "run_id": 11,
                "pipeline_id": "hybrid-rerank",
                "release_status": "approved",
            },
            "candidates": [
                {"pipeline_id": "hybrid-rerank"},
                {"pipeline_id": "dense"},
            ],
        },
    )
    responses = iter(
        [
            _FakeMessage(
                '{"sections":[{"title":"架构","query":"弱查询"},'
                '{"title":"能力","query":"稳定查询"}]}'
            ),
            _FakeMessage("DocMind 检索架构"),
            _FakeMessage(
                "# DocMind\n\n## 架构\n\nDocMind 使用混合检索。[D3:C1]"
            ),
        ]
    )
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        execution = _execution()
        if kwargs["query"] == "弱查询":
            execution.hits = []
        return execution

    monkeypatch.setattr(research_report, "run_pipeline_for_skill", retrieve)
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )

    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7),
        topic="DocMind",
        section_count=2,
    )

    assert result["workflow"]["status"] == "completed"
    assert result["workflow"]["pipeline_selection"]["dataset"]["id"] == 9
    assert all(call["pipeline"] == "hybrid-rerank" for call in calls)
    adapted = next(
        step
        for step in result["workflow"]["steps"]
        if step["step"] == "retrieve" and step["section"] == "架构"
    )
    assert adapted["adapted"] is True
    assert adapted["query"] == "DocMind 检索架构"
