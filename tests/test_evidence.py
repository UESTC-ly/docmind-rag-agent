"""Sentence-level evidence marker and citation contract tests."""

from app.services.evidence import (
    attach_citation_ids,
    build_evidence_context,
    citation_marker,
    validate_citations,
)


HITS = [
    {
        "document_id": 7,
        "chunk_index": 2,
        "content": "DocMind 使用混合检索。",
        "score": 0.9,
    },
    {
        "document_id": 8,
        "chunk_index": 4,
        "content": "Agent 支持中断恢复和人工审批。",
        "score": 0.8,
    },
]


def test_evidence_ids_and_context_are_stable():
    enriched = attach_citation_ids(HITS)
    assert enriched[0]["citation_id"] == "D7:C2"
    assert citation_marker(8, 4) == "[D8:C4]"
    assert build_evidence_context(HITS).startswith(
        "[D7:C2] DocMind 使用混合检索。"
    )
    assert "citation_id" not in HITS[0]


def test_all_claims_with_available_citations_pass():
    report = validate_citations(
        "DocMind 使用混合检索。[D7:C2]\n"
        "Agent 支持人工审批，也支持恢复。[D8:C4]",
        HITS,
    )
    assert report["passed"] is True
    assert report["claim_count"] == 2
    assert report["citation_precision"] == 1.0
    assert report["citation_recall"] == 1.0
    assert report["semantic_entailment_checked"] is False


def test_missing_citation_is_reported_per_sentence():
    report = validate_citations(
        "DocMind 使用混合检索。[D7:C2]\nAgent 可以自动恢复。",
        HITS,
    )
    assert report["passed"] is False
    assert report["unsupported_claim_count"] == 1
    assert report["citation_recall"] == 0.5
    assert report["claims"][1]["supported"] is False


def test_fabricated_citation_reduces_precision_and_does_not_support_claim():
    report = validate_citations("系统使用知识图谱。[D99:C1]", HITS)
    assert report["passed"] is False
    assert report["valid_citation_count"] == 0
    assert report["citation_precision"] == 0.0
    assert report["claims"][0]["invalid_citations"] == ["D99:C1"]


def test_multiple_valid_citations_are_preserved():
    report = validate_citations(
        "系统同时具备检索和审批能力。[D7:C2][D8:C4]",
        HITS,
    )
    assert report["passed"] is True
    assert report["citation_count"] == 2
    assert report["claims"][0]["valid_citations"] == ["D7:C2", "D8:C4"]


def test_refusal_and_empty_answer_have_no_factual_claims():
    refusal = validate_citations("根据已有文档无法回答该问题。", HITS)
    empty = validate_citations("", HITS)
    assert refusal["claim_count"] == 0
    assert refusal["passed"] is True
    assert empty["claim_count"] == 0
    assert empty["passed"] is True


def test_markdown_heading_is_not_treated_as_claim():
    report = validate_citations("# 项目结论\n\n采用混合检索。[D7:C2]", HITS)
    assert report["claim_count"] == 1
    assert report["passed"] is True
