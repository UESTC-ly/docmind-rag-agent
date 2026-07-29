"""Unified quality gates for Agent-produced artifacts."""

from __future__ import annotations

from app.services.artifact_verification import (
    sanitize_presentation_slides,
    verify_cited_markdown,
    verify_mermaid_artifact,
    verify_presentation_artifact,
    verify_relation_graph_artifact,
)
from app.services.evidence import attach_citation_ids


def _hits():
    return attach_citation_ids(
        [
            {
                "document_id": 3,
                "chunk_index": 2,
                "content": "项目本周完成了登录模块。",
                "score": 0.9,
            }
        ]
    )


def test_markdown_quality_gate_combines_structure_and_citations():
    report = verify_cited_markdown(
        "# 周报\n\n## 本周完成\n\n登录模块已经完成。[D3:C2]",
        _hits(),
        artifact_type="weekly_report",
        required_headings=("本周完成",),
    )

    assert report["contract"] == "artifact_quality_v1"
    assert report["passed"] is True
    assert report["citation_recall"] == 1.0
    assert {check["id"] for check in report["checks"]} == {
        "non_empty",
        "required_headings",
        "citation_presence",
        "evidence_coverage",
    }


def test_markdown_quality_gate_rejects_heading_only_fail_safe():
    report = verify_cited_markdown(
        "# 周报\n\n## 本周完成\n\n材料未提及。",
        _hits(),
        artifact_type="weekly_report",
        required_headings=("本周完成",),
    )

    assert report["citation_recall"] == 1.0
    assert report["supported_claim_count"] == 0
    assert report["passed"] is False
    assert next(
        check for check in report["checks"] if check["id"] == "evidence_coverage"
    )["passed"] is False


def test_presentation_gate_removes_unsupported_bullets_before_delivery():
    slides = [
        {
            "title": "项目进展",
            "bullets": [
                "登录模块已经完成。[D3:C2]",
                "支付模块已经上线。",
            ],
        }
    ]
    sanitized = sanitize_presentation_slides(slides, _hits())
    report = verify_presentation_artifact(sanitized, _hits(), b"PK\x03\x04")

    assert sanitized == [
        {
            "title": "项目进展",
            "bullets": ["登录模块已经完成。[D3:C2]"],
        }
    ]
    assert report["passed"] is True
    assert report["citation_recall"] == 1.0


def test_structured_artifact_gates_reject_invalid_outputs():
    mermaid = verify_mermaid_artifact("not a diagram", artifact_type="mindmap")
    graph = verify_relation_graph_artifact(
        nodes=[{"id": "A"}],
        edges=[{"source": "A", "target": "missing", "relation": "依赖"}],
    )

    assert mermaid["passed"] is False
    assert graph["passed"] is False
