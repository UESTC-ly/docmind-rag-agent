"""Bounded Agentic workflow for a cited, automatically verified report.

The workflow is deliberately narrow and observable:
plan -> retrieve per section -> draft -> validate -> one repair -> fail-safe
removal of claims that still lack valid evidence.  The outer LangGraph Agent
provides durable execution, approval, run leases, and tool receipts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import settings
from app.database import SyncSessionLocal
from app.services.artifact_verification import merge_artifact_verification
from app.services.evaluation.grounding import evaluate_grounding
from app.services.evaluation.pipeline_selection import (
    recommend_evaluated_pipeline,
)
from app.services.evidence import build_evidence_context
from app.services.llm_service import chat_completion
from app.services.skill_retrieval import run_pipeline_for_skill
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill

_PLAN_PROMPT = """你正在规划一份基于知识库的研究报告。
主题：{topic}
请返回严格 JSON，不要 Markdown：
{{"sections":[{{"title":"小节标题","query":"用于检索证据的具体问题"}}]}}
要求：
- 恰好 {section_count} 个小节；
- 小节相互补充，不重复；
- query 必须能直接用于文档检索。"""

_DRAFT_PROMPT = """请根据给定证据撰写中文 Markdown 研究报告。
主题：{topic}
小节计划：
{outline}

证据：
{evidence}

规则：
1. 只能使用证据中的信息。
2. 每个事实性句子末尾必须带一个或多个证据编号，如 [D12:C3]。
3. 只能复制证据中真实存在的编号，不得创造编号。
4. 没有证据支持的内容不要写。
5. 多个证据冲突时明确写出冲突并分别引用，不得擅自选边。
6. 原文没有直接陈述的推断必须以「推测：」开头。
7. 保留 # 报告标题和 ## 小节标题。
8. 只保留最关键的 4 到 8 条结论；正文不超过 1200 个中文字符。
只输出报告正文。"""

_REPAIR_PROMPT = """下面的研究报告没有完全通过句子级引用校验。请只使用给定证据修订。
主题：{topic}

原报告：
{report}

未通过项目：
{problems}

证据：
{evidence}

规则：
1. 删除无法由证据支持的句子。
2. 每个保留的事实性句子末尾必须使用真实编号，如 [D12:C3]。
3. 不得创造编号。
4. 冲突证据必须并列说明；推断必须以「推测：」开头。
5. 保留 Markdown 标题结构。
只输出修订后的完整报告。"""

_REWRITE_QUERY_PROMPT = """知识库检索没有返回证据。请把下面的问题改写为一个更适合文档检索、
含义不变且更具体的查询。只输出改写后的查询，不要解释。

报告主题：{topic}
小节：{section}
原查询：{query}"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
DEFAULT_REPORT_SECTION_COUNT = 2
MAX_REPORT_SECTION_COUNT = 3
MAX_REPORT_EVIDENCE_CHARS = 8000


def _default_sections(topic: str, count: int) -> list[dict[str, str]]:
    labels = ["核心概述", "关键机制", "主要证据", "限制与边界", "结论"]
    return [
        {
            "title": labels[index],
            "query": f"{topic} {labels[index]}",
        }
        for index in range(count)
    ]


def _parse_sections(raw: str, topic: str, count: int) -> list[dict[str, str]]:
    cleaned = _FENCE_RE.sub("", raw.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return _default_sections(topic, count)
    rows = payload.get("sections") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return _default_sections(topic, count)
    sections: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        query = str(row.get("query") or "").strip()
        if not title or not query:
            continue
        sections.append({"title": title[:120], "query": query[:500]})
        if len(sections) == count:
            break
    if len(sections) != count:
        return _default_sections(topic, count)
    return sections


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    output: list[dict[str, Any]] = []
    for hit in hits:
        key = (int(hit["document_id"]), int(hit["chunk_index"]))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(hit))
    return output


def _problems(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for claim in report["claims"]:
        semantic_supported = claim.get("semantically_supported")
        if (
            claim["supported"]
            and not claim["invalid_citations"]
            and semantic_supported is not False
        ):
            continue
        reasons: list[str] = []
        if not claim["supported"]:
            reasons.append("缺少有效引用")
        if claim["invalid_citations"]:
            reasons.append("含不存在的引用 " + ",".join(claim["invalid_citations"]))
        if semantic_supported is False:
            reasons.append(
                "引用不支持该结论"
                + (
                    f"（{claim.get('reason')}）"
                    if claim.get("reason")
                    else ""
                )
            )
        rows.append(f"- {claim['text']}（{'；'.join(reasons)}）")
    return "\n".join(rows) or "- 引用结构不完整"


def _fail_safe_report(text: str, report: dict[str, Any]) -> str:
    """Remove unresolved claims instead of silently returning ungrounded prose."""
    sanitized = text
    reject_all = (
        not report.get("semantic_entailment_checked")
        or bool(report.get("conflict_count"))
    )
    for claim in report["claims"]:
        if (
            reject_all
            or not claim["valid_citations"]
            or claim.get("semantically_supported") is False
        ):
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
    sanitized = "\n".join(compact).strip()
    note = "> 证据不足或引用无效的句子已由验证器自动移除。"
    return f"{sanitized}\n\n{note}".strip()


def _prune_irrelevant_citations(
    text: str,
    report: dict[str, Any],
) -> tuple[str, int]:
    """Remove irrelevant links only when a claim retains supporting evidence."""

    sanitized = text
    removed_count = 0
    for claim in report.get("claims") or []:
        citation_verdicts = claim.get("citation_verdicts") or []
        supported_ids = {
            str(row.get("citation_id") or "")
            for row in citation_verdicts
            if row.get("verdict") == "supports"
        }
        irrelevant_ids = {
            str(row.get("citation_id") or "")
            for row in citation_verdicts
            if row.get("verdict") == "irrelevant"
        }
        if not supported_ids or not irrelevant_ids:
            continue
        original = str(claim.get("text") or "")
        if not original:
            continue
        corrected = original
        for citation_id in irrelevant_ids:
            marker = f"[{citation_id}]"
            if marker in corrected:
                corrected = corrected.replace(marker, "")
                removed_count += 1
        if corrected != original:
            sanitized = sanitized.replace(original, corrected, 1)
    return sanitized, removed_count


def _mark_supported_inferences(
    text: str,
    report: dict[str, Any],
) -> tuple[str, int]:
    """Label judge-approved inferences so they cannot masquerade as facts."""

    sanitized = text
    marked_count = 0
    for claim in report.get("claims") or []:
        if (
            claim.get("verdict") != "reasonable_inference"
            or claim.get("claim_type") != "inference"
            or claim.get("explicit_inference")
        ):
            continue
        citation_verdicts = claim.get("citation_verdicts") or []
        if not any(row.get("verdict") == "supports" for row in citation_verdicts):
            continue
        if any(row.get("verdict") == "contradicts" for row in citation_verdicts):
            continue
        original = str(claim.get("text") or "")
        if not original:
            continue
        corrected = f"推测：{original}"
        if corrected != original:
            sanitized = sanitized.replace(original, corrected, 1)
            marked_count += 1
    return sanitized, marked_count


def _automatic_pipeline_decision(user_id: int) -> dict[str, Any]:
    """Use public regression evidence when available; fail visibly to configured."""

    try:
        with SyncSessionLocal() as db:
            return recommend_evaluated_pipeline(
                db,
                user_id=user_id,
                target="retrieval",
            )
    except Exception as exc:
        return {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "unavailable",
            "reason": f"评测决策服务不可用：{type(exc).__name__}",
            "selected": None,
            "candidates": [],
            "dataset": None,
        }


def _rewrite_query(topic: str, section: dict[str, str]) -> str:
    try:
        message = chat_completion(
            [
                {
                    "role": "user",
                    "content": _REWRITE_QUERY_PROMPT.format(
                        topic=topic,
                        section=section["title"],
                        query=section["query"],
                    ),
                }
            ],
            temperature=0.1,
        )
        rewritten = str(message.content or "").strip()
    except Exception:
        return section["query"]
    return rewritten[:500] or section["query"]


@register_skill
class VerifiedResearchReportSkill(BaseSkill):
    name = "generate_verified_research_report"
    description = (
        "自动规划多个检索问题，跨文档收集证据，生成逐句引用的研究报告，"
        "并在返回前执行引用校验和一次自动修复。"
    )
    grounding_mode = "pipeline_rag_verified"
    produces_download = True
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "研究报告主题"},
            "document_id": {
                "type": "integer",
                "description": "可选；限定单篇文档。省略时允许跨用户知识库检索。",
            },
            "pipeline_id": {
                "type": "string",
                "description": (
                    "本次工作流使用的可复现知识管线 ID；可使用内置管线，"
                    "也可使用 /eval/pipelines 中已注册的受信任插件管线。"
                ),
            },
            "section_count": {
                "type": "integer",
                "minimum": 2,
                "maximum": MAX_REPORT_SECTION_COUNT,
                "description": "报告小节数，默认 2，最多 3。",
            },
        },
        "required": ["topic"],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        topic = str(kwargs["topic"]).strip()
        if not topic:
            return {"error": "报告主题不能为空"}
        document_id = kwargs.get("document_id") or context.document_id
        requested_pipeline = str(kwargs.get("pipeline_id") or "").strip()
        pipeline_decision = (
            {
                "contract": "evaluated_pipeline_selection_v1",
                "status": "explicit",
                "reason": "调用方显式指定 pipeline_id",
                "selected": {"pipeline_id": requested_pipeline},
                "candidates": [],
                "dataset": None,
            }
            if requested_pipeline
            else _automatic_pipeline_decision(context.user_id)
        )
        selected = pipeline_decision.get("selected") or {}
        pipeline_id = str(selected.get("pipeline_id") or "configured")
        alternate_pipeline_ids = [
            str(candidate["pipeline_id"])
            for candidate in pipeline_decision.get("candidates") or []
            if candidate.get("pipeline_id")
            and str(candidate["pipeline_id"]) != pipeline_id
        ]
        section_count = max(
            2,
            min(
                MAX_REPORT_SECTION_COUNT,
                int(kwargs.get("section_count") or DEFAULT_REPORT_SECTION_COUNT),
            ),
        )
        workflow_steps: list[dict[str, Any]] = [
            {
                "step": "select_pipeline",
                "status": pipeline_decision.get("status"),
                "pipeline_id": pipeline_id,
                "dataset_id": (
                    (pipeline_decision.get("dataset") or {}).get("id")
                ),
                "run_id": selected.get("run_id"),
                "release_status": selected.get("release_status"),
                "reason": pipeline_decision.get("reason"),
            }
        ]

        plan_message = chat_completion(
            [
                {
                    "role": "user",
                    "content": _PLAN_PROMPT.format(
                        topic=topic,
                        section_count=section_count,
                    ),
                }
            ],
            temperature=0.2,
        )
        sections = _parse_sections(
            plan_message.content or "",
            topic,
            section_count,
        )
        workflow_steps.append(
            {
                "step": "plan",
                "status": "completed",
                "section_count": len(sections),
            }
        )

        all_hits: list[dict[str, Any]] = []
        pipeline_spec = None
        pipeline_fingerprint = None
        effective_queries: dict[str, str] = {}
        for section in sections:
            execution = run_pipeline_for_skill(
                user_id=context.user_id,
                query=section["query"],
                top_k=settings.retrieval_top_k,
                document_id=document_id,
                pipeline=pipeline_id,
            )
            query = section["query"]
            adapted = False
            if not execution.hits:
                query = _rewrite_query(topic, section)
                execution = run_pipeline_for_skill(
                    user_id=context.user_id,
                    query=query,
                    top_k=min(max(settings.retrieval_top_k * 2, 1), 100),
                    document_id=document_id,
                    pipeline=pipeline_id,
                )
                adapted = True
            pipeline_spec = execution.spec
            pipeline_fingerprint = execution.fingerprint
            all_hits.extend(execution.hits)
            effective_queries[section["title"]] = query
            workflow_steps.append(
                {
                    "step": "retrieve",
                    "status": (
                        "completed" if execution.hits else "insufficient"
                    ),
                    "section": section["title"],
                    "query": query,
                    "original_query": section["query"],
                    "adapted": adapted,
                    "hit_count": len(execution.hits),
                    "pipeline_trace": execution.trace,
                }
            )

        hits = _dedupe_hits(all_hits)
        if not hits and alternate_pipeline_ids:
            pipeline_id = alternate_pipeline_ids[0]
            workflow_steps.append(
                {
                    "step": "switch_pipeline",
                    "status": "running",
                    "pipeline_id": pipeline_id,
                    "reason": "首选评测管线在扩展检索后仍无证据",
                }
            )
            for section in sections:
                query = effective_queries.get(section["title"], section["query"])
                execution = run_pipeline_for_skill(
                    user_id=context.user_id,
                    query=query,
                    top_k=min(max(settings.retrieval_top_k * 2, 1), 100),
                    document_id=document_id,
                    pipeline=pipeline_id,
                )
                pipeline_spec = execution.spec
                pipeline_fingerprint = execution.fingerprint
                all_hits.extend(execution.hits)
            hits = _dedupe_hits(all_hits)
            workflow_steps[-1]["status"] = (
                "completed" if hits else "insufficient"
            )
        if not hits or pipeline_spec is None:
            return {
                "error": "知识库中未检索到可用于报告的证据",
                "topic": topic,
                "workflow": {
                    "contract": "plan_select_retrieve_adapt_ground_repair_v3",
                    "status": "failed",
                    "steps": workflow_steps,
                },
            }

        evidence = build_evidence_context(
            hits,
            max_chars=MAX_REPORT_EVIDENCE_CHARS,
        )
        outline = "\n".join(
            f"- {section['title']}："
            f"{effective_queries.get(section['title'], section['query'])}"
            for section in sections
        )
        draft_message = chat_completion(
            [
                {
                    "role": "user",
                    "content": _DRAFT_PROMPT.format(
                        topic=topic,
                        outline=outline,
                        evidence=evidence,
                    ),
                }
            ],
            temperature=0.3,
        )
        report_text = (draft_message.content or "").strip()
        workflow_steps.append(
            {"step": "draft", "status": "completed", "chars": len(report_text)}
        )

        verification = evaluate_grounding(report_text, hits)
        workflow_steps.append(
            {
                "step": "verify",
                "status": "passed" if verification["passed"] else "failed",
                "unsupported_claim_count": verification[
                    "unsupported_claim_count"
                ],
                "semantic_entailment_checked": verification[
                    "semantic_entailment_checked"
                ],
                "groundedness": verification.get("groundedness"),
                "citation_correctness": verification.get(
                    "citation_correctness"
                ),
            }
        )

        repair_attempted = False
        fail_safe_applied = False
        if not verification["passed"]:
            repair_attempted = True
            repair_message = chat_completion(
                [
                    {
                        "role": "user",
                        "content": _REPAIR_PROMPT.format(
                            topic=topic,
                            report=report_text,
                            problems=_problems(verification),
                            evidence=evidence,
                        ),
                    }
                ],
                temperature=0.1,
            )
            report_text = (repair_message.content or "").strip()
            verification = evaluate_grounding(report_text, hits)
            workflow_steps.append(
                {
                    "step": "repair",
                    "status": (
                        "passed" if verification["passed"] else "failed"
                    ),
                    "unsupported_claim_count": verification[
                        "unsupported_claim_count"
                    ],
                }
            )

        if not verification["passed"]:
            report_text, marked_inference_count = _mark_supported_inferences(
                report_text,
                verification,
            )
            if marked_inference_count:
                verification = evaluate_grounding(report_text, hits)
                workflow_steps.append(
                    {
                        "step": "mark_supported_inferences",
                        "status": (
                            "passed" if verification["passed"] else "failed"
                        ),
                        "marked_inference_count": marked_inference_count,
                    }
                )

        if not verification["passed"]:
            report_text, pruned_link_count = _prune_irrelevant_citations(
                report_text,
                verification,
            )
            if pruned_link_count:
                verification = evaluate_grounding(report_text, hits)
                workflow_steps.append(
                    {
                        "step": "prune_irrelevant_citations",
                        "status": (
                            "passed" if verification["passed"] else "failed"
                        ),
                        "removed_link_count": pruned_link_count,
                    }
                )

        if not verification["passed"]:
            fail_safe_applied = True
            report_text = _fail_safe_report(report_text, verification)
            verification = evaluate_grounding(report_text, hits)
            workflow_steps.append(
                {
                    "step": "fail_safe",
                    "status": (
                        "passed" if verification["passed"] else "failed"
                    ),
                    "unsupported_claim_count": verification[
                        "unsupported_claim_count"
                    ],
                }
            )

        verification = merge_artifact_verification(
            "verified_research_report",
            verification,
        )
        verification["repair_attempted"] = repair_attempted
        verification["fail_safe_applied"] = fail_safe_applied
        source_items = [
            {
                "citation_id": hit["citation_id"],
                "document_id": int(hit["document_id"]),
                "chunk_index": int(hit["chunk_index"]),
                "score": hit.get("score"),
                "document_name": hit.get("document_name"),
                "source_uri": hit.get("source_uri"),
                "source_version": hit.get("source_version"),
                "source_status": hit.get("source_status", "unknown"),
                "jump_url": hit.get("jump_url"),
                "page_start": hit.get("page_start"),
                "page_end": hit.get("page_end"),
                "paragraph_start": hit.get("paragraph_start"),
                "paragraph_end": hit.get("paragraph_end"),
                "char_start": hit.get("char_start"),
                "char_end": hit.get("char_end"),
                "locator_version": hit.get("locator_version"),
            }
            for hit in hits
        ]
        return {
            "type": "verified_research_report",
            "artifact_kind": "file",
            "topic": topic,
            "document_id": document_id,
            "outline": sections,
            "content": report_text,
            "workflow": {
                "contract": "plan_select_retrieve_adapt_ground_repair_v3",
                "status": "completed" if verification["passed"] else "failed",
                "repair_attempted": repair_attempted,
                "fail_safe_applied": fail_safe_applied,
                "pipeline_selection": pipeline_decision,
                "steps": workflow_steps,
            },
            "verification": verification,
            "grounding": {
                "mode": "pipeline_rag_verified",
                "pipeline_id": pipeline_spec.id,
                "pipeline_fingerprint": pipeline_fingerprint,
                "pipeline_spec": pipeline_spec.to_dict(),
                "sources": source_items,
            },
            "download": {
                "filename": "verified-research-report.md",
                "mime_type": "text/markdown;charset=utf-8",
                "encoding": "text",
                "content": report_text,
            },
        }
