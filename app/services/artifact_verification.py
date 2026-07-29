"""Unified deterministic quality gates for Agent artifacts.

These checks validate observable delivery contracts: file/structure validity,
required sections, evidence identifiers, and graph consistency. Structural
checks remain distinct from optional ``claim_grounding_v1`` semantic evidence.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from app.services.evidence import validate_citations

_SAFE_MISSING_MARKERS = ("材料未提及", "证据不足", "无法确定")


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "detail": detail[:500]}


def _quality_report(
    artifact_type: str,
    checks: Iterable[dict[str, Any]],
    *,
    citation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(checks)
    passed_count = sum(1 for row in rows if row["passed"])
    report: dict[str, Any] = {
        "contract": "artifact_quality_v1",
        "artifact_type": artifact_type,
        "passed": bool(rows) and passed_count == len(rows),
        "score": round(passed_count / len(rows), 6) if rows else 0.0,
        "checks": rows,
        "semantic_entailment_checked": bool(
            citation and citation.get("semantic_entailment_checked")
        ),
    }
    if citation is not None:
        citation_dict = dict(citation)
        report["citation"] = citation_dict
        for key in (
            "claim_count",
            "supported_claim_count",
            "unsupported_claim_count",
            "citation_count",
            "valid_citation_count",
            "citation_precision",
            "citation_recall",
            "unsupported_claim_rate",
            "groundedness",
            "faithfulness",
            "citation_correctness",
            "conflict_count",
            "conflicts",
            "conflicts_disclosed",
            "judge_status",
            "judge_model",
            "judge_rubric_version",
            "judge_input_fingerprint",
            "claim_state_counts",
            "claims",
        ):
            if key in citation_dict:
                report[key] = citation_dict[key]
    return report


def verify_cited_markdown(
    text: str,
    hits: list[dict[str, Any]],
    *,
    artifact_type: str,
    required_headings: tuple[str, ...] = (),
) -> dict[str, Any]:
    content = str(text or "").strip()
    headings = {
        match.group(1).strip().lower()
        for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", content)
    }
    missing = [
        heading
        for heading in required_headings
        if not any(heading.lower() in actual for actual in headings)
    ]
    citation = validate_citations(content, hits)
    checks = [
        _check("non_empty", bool(content), "产物包含正文" if content else "产物为空"),
        _check(
            "required_headings",
            not missing,
            "所需标题齐全" if not missing else "缺少标题：" + "、".join(missing),
        ),
        _check(
            "citation_presence",
            bool(citation["passed"]),
            (
                f"{citation['supported_claim_count']}/{citation['claim_count']} "
                "条事实性内容具有本次检索的有效引用"
            ),
        ),
        _check(
            "evidence_coverage",
            citation["supported_claim_count"] > 0,
            (
                "至少一条实质内容具有本次检索的有效引用"
                if citation["supported_claim_count"] > 0
                else "产物没有任何经过引用支持的实质内容"
            ),
        ),
    ]
    return _quality_report(
        artifact_type,
        checks,
        citation=citation,
    )


def sanitize_cited_markdown(
    text: str,
    hits: list[dict[str, Any]],
) -> str:
    """Remove claims that fail the structural evidence contract."""

    report = validate_citations(text, hits)
    sanitized = str(text or "")
    for claim in report["claims"]:
        if not claim["valid_citations"]:
            sanitized = sanitized.replace(claim["text"], "")
            continue
        for invalid in claim["invalid_citations"]:
            sanitized = sanitized.replace(f"[{invalid}]", "")
    lines = [line.rstrip() for line in sanitized.splitlines()]
    compact: list[str] = []
    for line in lines:
        if not line and compact and not compact[-1]:
            continue
        compact.append(line)
    return "\n".join(compact).strip()


def _presentation_claim_text(slides: Sequence[Mapping[str, Any]]) -> str:
    claims: list[str] = []
    for slide in slides:
        for raw in slide.get("bullets") or []:
            bullet = str(raw).strip()
            if not bullet or any(marker in bullet for marker in _SAFE_MISSING_MARKERS):
                continue
            claims.append(bullet)
    return "\n".join(claims)


def sanitize_presentation_slides(
    slides: Sequence[Mapping[str, Any]],
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for raw_slide in slides:
        title = str(raw_slide.get("title") or "").strip()
        bullets: list[str] = []
        for raw_bullet in raw_slide.get("bullets") or []:
            bullet = str(raw_bullet).strip()
            if not bullet:
                continue
            if any(marker in bullet for marker in _SAFE_MISSING_MARKERS):
                bullets.append(bullet)
                continue
            if validate_citations(bullet, hits)["passed"]:
                bullets.append(bullet)
        sanitized.append(
            {
                "title": title,
                "bullets": bullets or ["材料未提及"],
            }
        )
    return sanitized


def verify_presentation_artifact(
    slides: Sequence[Mapping[str, Any]],
    hits: list[dict[str, Any]],
    pptx: bytes,
) -> dict[str, Any]:
    citation = validate_citations(_presentation_claim_text(slides), hits)
    structure_valid = bool(slides) and all(
        str(slide.get("title") or "").strip()
        and isinstance(slide.get("bullets"), list)
        and bool(slide.get("bullets"))
        for slide in slides
    )
    checks = [
        _check(
            "slide_structure",
            structure_valid,
            "每页均有标题和要点" if structure_valid else "存在空页面或无效页面结构",
        ),
        _check(
            "citation_presence",
            bool(citation["passed"]),
            (
                f"{citation['supported_claim_count']}/{citation['claim_count']} "
                "条 PPT 要点具有有效引用"
            ),
        ),
        _check(
            "evidence_coverage",
            not hits or citation["supported_claim_count"] > 0,
            (
                "至少一个实质要点关联了检索证据"
                if citation["supported_claim_count"] > 0
                else "没有实质要点关联检索证据"
            ),
        ),
        _check(
            "pptx_container",
            bool(pptx) and bytes(pptx).startswith(b"PK\x03\x04"),
            "PPTX ZIP 容器有效" if pptx else "PPTX 文件为空",
        ),
    ]
    return _quality_report("presentation", checks, citation=citation)


def verify_mermaid_artifact(
    source: str,
    *,
    artifact_type: str,
) -> dict[str, Any]:
    lines = [line.strip() for line in str(source or "").splitlines() if line.strip()]
    expected = "mindmap" if artifact_type == "mindmap" else "graph"
    directive_valid = bool(lines) and (
        lines[0] == expected
        if expected == "mindmap"
        else lines[0].startswith(("graph ", "flowchart "))
    )
    has_body = len(lines) >= 2
    return _quality_report(
        artifact_type,
        [
            _check(
                "mermaid_directive",
                directive_valid,
                f"首行必须是有效 {expected} 指令",
            ),
            _check(
                "mermaid_body",
                has_body,
                "图包含至少一个结构节点" if has_body else "图没有结构节点",
            ),
        ],
    )


def verify_relation_graph_artifact(
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
) -> dict[str, Any]:
    identifiers = [str(node.get("id") or "").strip() for node in nodes]
    valid_ids = {identifier for identifier in identifiers if identifier}
    nodes_valid = bool(valid_ids) and len(valid_ids) == len(identifiers)
    edges_valid = bool(edges) and all(
        str(edge.get("source") or "").strip() in valid_ids
        and str(edge.get("target") or "").strip() in valid_ids
        and bool(str(edge.get("relation") or "").strip())
        for edge in edges
    )
    return _quality_report(
        "relation_graph",
        [
            _check(
                "unique_nodes",
                nodes_valid,
                "节点 ID 非空且唯一" if nodes_valid else "节点为空、重复或缺少 ID",
            ),
            _check(
                "edge_references",
                edges_valid,
                "所有边均引用已知节点" if edges_valid else "边为空或引用未知节点",
            ),
        ],
    )


def merge_artifact_verification(
    artifact_type: str,
    citation: Mapping[str, Any],
    *,
    extra_checks: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    checks = list(extra_checks)
    semantic = bool(citation.get("semantic_entailment_checked"))
    checks.append(
        _check(
            "claim_grounding" if semantic else "citation_presence",
            bool(citation.get("passed")),
            (
                f"{citation.get('supported_claim_count', 0)}/"
                f"{citation.get('claim_count', 0)} 条事实性内容"
                + ("获得语义证据支持" if semantic else "具有有效引用")
            ),
        )
    )
    checks.append(
        _check(
            "evidence_coverage",
            int(citation.get("supported_claim_count") or 0) > 0,
            (
                "至少一条实质内容具有有效引用"
                if int(citation.get("supported_claim_count") or 0) > 0
                else "产物没有任何经过引用支持的实质内容"
            ),
        )
    )
    return _quality_report(artifact_type, checks, citation=citation)


__all__ = [
    "merge_artifact_verification",
    "sanitize_cited_markdown",
    "sanitize_presentation_slides",
    "verify_cited_markdown",
    "verify_mermaid_artifact",
    "verify_presentation_artifact",
    "verify_relation_graph_artifact",
]
