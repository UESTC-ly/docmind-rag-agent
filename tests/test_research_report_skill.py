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


def test_default_workflow_bounds_sections_and_evidence_context(monkeypatch):
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
    captured = {}
    monkeypatch.setattr(
        research_report,
        "run_pipeline_for_skill",
        lambda **kwargs: _execution(),
    )

    def evidence_context(_hits, *, max_chars):
        captured["max_chars"] = max_chars
        return "[D3:C1] DocMind 使用混合检索。"

    monkeypatch.setattr(
        research_report,
        "build_evidence_context",
        evidence_context,
    )
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )

    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7),
        topic="DocMind",
        pipeline_id="hybrid-rerank",
    )

    assert result["workflow"]["steps"][1]["section_count"] == 2
    assert captured["max_chars"] == research_report.MAX_REPORT_EVIDENCE_CHARS
    assert research_report.MAX_REPORT_SECTION_COUNT == 3


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


def test_workflow_prunes_irrelevant_citation_links_after_repair(monkeypatch):
    responses = iter(
        [
            _FakeMessage(
                '{"sections":[{"title":"架构","query":"DocMind 检索架构"},'
                '{"title":"能力","query":"DocMind 主要能力"}]}'
            ),
            _FakeMessage(
                "# DocMind\n\nDocMind 使用混合检索。[D3:C1][D3:C2]"
            ),
            _FakeMessage(
                "# DocMind\n\nDocMind 使用混合检索。[D3:C1][D3:C2]"
            ),
        ]
    )
    execution = _execution()
    execution.hits = attach_citation_ids(
        [
            {
                "document_id": 3,
                "chunk_index": 1,
                "content": "DocMind 使用混合检索。",
                "score": 0.9,
            },
            {
                "document_id": 3,
                "chunk_index": 2,
                "content": "DocMind 也支持本地桌面应用。",
                "score": 0.8,
            },
        ]
    )

    def evaluate(answer, hits):
        report = validate_citations(answer, hits)
        for claim in report["claims"]:
            claim["semantically_supported"] = True
            claim["citation_verdicts"] = [
                {
                    "citation_id": citation_id,
                    "verdict": (
                        "supports" if citation_id == "D3:C1" else "irrelevant"
                    ),
                }
                for citation_id in claim["valid_citations"]
            ]
        contains_irrelevant_link = "[D3:C2]" in answer
        return {
            **report,
            "contract": "claim_grounding_v1",
            "semantic_entailment_checked": True,
            "judge_status": "completed",
            "groundedness": 1.0,
            "faithfulness": 1.0,
            "citation_correctness": 0.5 if contains_irrelevant_link else 1.0,
            "conflict_count": 0,
            "conflicts": [],
            "passed": report["passed"] and not contains_irrelevant_link,
        }

    monkeypatch.setattr(
        research_report,
        "run_pipeline_for_skill",
        lambda **kwargs: execution,
    )
    monkeypatch.setattr(research_report, "evaluate_grounding", evaluate)
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )

    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7, document_id=3),
        topic="DocMind",
        section_count=2,
        pipeline_id="hybrid-rerank",
    )

    assert result["verification"]["passed"] is True
    assert result["workflow"]["repair_attempted"] is True
    assert result["workflow"]["fail_safe_applied"] is False
    assert "[D3:C1]" in result["content"]
    assert "[D3:C2]" not in result["content"]
    assert any(
        step["step"] == "prune_irrelevant_citations"
        and step["status"] == "passed"
        and step["removed_link_count"] == 1
        for step in result["workflow"]["steps"]
    )


def test_workflow_marks_supported_inferences_after_repair(monkeypatch):
    responses = iter(
        [
            _FakeMessage(
                '{"sections":[{"title":"架构","query":"DocMind 检索架构"},'
                '{"title":"能力","query":"DocMind 主要能力"}]}'
            ),
            _FakeMessage(
                "# DocMind\n\n现有证据不能排除其他相反研究。[D3:C1]"
            ),
            _FakeMessage(
                "# DocMind\n\n现有证据不能排除其他相反研究。[D3:C1]"
            ),
        ]
    )

    def evaluate(answer, hits):
        report = validate_citations(answer, hits)
        for claim in report["claims"]:
            explicit_inference = claim["text"].startswith("推测：")
            claim.update(
                {
                    "verdict": "reasonable_inference",
                    "claim_type": "inference",
                    "citation_verdicts": [
                        {
                            "citation_id": citation_id,
                            "verdict": "supports",
                        }
                        for citation_id in claim["valid_citations"]
                    ],
                    "semantically_supported": explicit_inference,
                    "explicit_inference": explicit_inference,
                }
            )
        supported_count = sum(
            bool(claim["semantically_supported"])
            for claim in report["claims"]
        )
        claim_count = report["claim_count"]
        return {
            **report,
            "contract": "claim_grounding_v1",
            "semantic_entailment_checked": True,
            "judge_status": "completed",
            "groundedness": supported_count / claim_count if claim_count else 1.0,
            "faithfulness": supported_count / claim_count if claim_count else 1.0,
            "citation_correctness": 1.0,
            "conflict_count": 0,
            "conflicts": [],
            "supported_claim_count": supported_count,
            "unsupported_claim_count": claim_count - supported_count,
            "passed": report["passed"] and supported_count == claim_count,
        }

    monkeypatch.setattr(
        research_report,
        "run_pipeline_for_skill",
        lambda **kwargs: _execution(),
    )
    monkeypatch.setattr(research_report, "evaluate_grounding", evaluate)
    monkeypatch.setattr(
        research_report,
        "chat_completion",
        lambda messages, temperature=0.3: next(responses),
    )

    result = research_report.VerifiedResearchReportSkill().run(
        SkillContext(user_id=7, document_id=3),
        topic="DocMind",
        section_count=2,
        pipeline_id="hybrid-rerank",
    )

    assert result["verification"]["passed"] is True
    assert result["workflow"]["fail_safe_applied"] is False
    assert "推测：现有证据不能排除其他相反研究。[D3:C1]" in result[
        "content"
    ]
    assert any(
        step["step"] == "mark_supported_inferences"
        and step["status"] == "passed"
        and step["marked_inference_count"] == 1
        for step in result["workflow"]["steps"]
    )


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
