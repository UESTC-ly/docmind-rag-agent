"""Claim-level grounding, citation correctness, and conflict evaluation.

Structural citation presence and semantic support are intentionally separate.
If the semantic judge is unavailable, the result is ``unavailable`` rather
than a fabricated zero score, and evidence-closed delivery can fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from app.config import settings
from app.services.evidence import (
    citation_id,
    extract_claims,
    is_refusal,
    validate_citations,
)
from app.services.llm_service import chat_completion

GROUNDING_RUBRIC_VERSION = "claim_grounding_v1"
CANONICAL_REFUSAL = "根据已有文档无法回答该问题。"

_INFERENCE_RE = re.compile(
    r"(?:^|[：:。.!！？]\s*)(?:推测|推断|模型推测|可能|或许|据此可推)"
)
_CONFLICT_DISCLOSURE_RE = re.compile(
    r"(?:来源|文档|版本|证据).{0,20}(?:冲突|矛盾|不一致|说法不同)"
    r"|(?:冲突|矛盾|不一致).{0,20}(?:来源|文档|版本|证据)"
)
_CLAIM_VERDICTS = {
    "entailed",
    "reasonable_inference",
    "contradicted",
    "unsupported",
}
_CITATION_VERDICTS = {"supports", "contradicts", "irrelevant"}

GROUNDING_PROMPT = """你是严格的 RAG 依据性评估器。只判断给定证据是否支持结论，不补充外部知识。

判定规则：
1. entailed：证据直接蕴含结论。
2. reasonable_inference：证据只支持合理推测，但没有直接陈述结论。
3. contradicted：至少一条证据与结论冲突。
4. unsupported：证据既不蕴含也不明确冲突。
5. 每个引用还要分别判断 supports、contradicts 或 irrelevant。
6. claim_type 只能是 fact 或 inference。
7. conflicts 只报告证据之间针对同一事项的实质冲突。

输入 JSON：
{payload}

输出严格 JSON，不要 Markdown：
{{
  "claims": [
    {{
      "claim_index": 0,
      "claim_type": "fact",
      "verdict": "entailed",
      "citation_verdicts": [
        {{"citation_id": "D1:C2", "verdict": "supports"}}
      ],
      "reason": "简短理由"
    }}
  ],
  "conflicts": [
    {{
      "citation_ids": ["D1:C2", "D2:C4"],
      "reason": "简短说明"
    }}
  ]
}}"""


GroundingJudge = Callable[[dict[str, Any]], dict[str, Any]]
GroundingEvaluator = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


def _strip_json_fence(content: str) -> str:
    value = content.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return value


def _llm_judge(payload: dict[str, Any]) -> dict[str, Any]:
    response = chat_completion(
        [
            {
                "role": "user",
                "content": GROUNDING_PROMPT.format(
                    payload=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            }
        ],
        temperature=0.0,
    )
    decoded = json.loads(_strip_json_fence(response.content or ""))
    if not isinstance(decoded, dict):
        raise ValueError("grounding judge response must be a JSON object")
    return decoded


def _normalize_judgement(
    raw: dict[str, Any],
    claim_count: int,
    available_ids: set[str],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    rows = raw.get("claims")
    if not isinstance(rows, list):
        raise ValueError("grounding judge response requires claims")
    normalized: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("grounding claim verdict must be an object")
        index = int(row.get("claim_index", -1))
        if not 0 <= index < claim_count or index in normalized:
            raise ValueError("grounding judge returned invalid claim_index")
        verdict = str(row.get("verdict", "")).strip()
        claim_type = str(row.get("claim_type", "")).strip()
        if verdict not in _CLAIM_VERDICTS:
            raise ValueError("grounding judge returned invalid verdict")
        if claim_type not in {"fact", "inference"}:
            raise ValueError("grounding judge returned invalid claim_type")
        citation_rows: list[dict[str, str]] = []
        raw_citations = row.get("citation_verdicts") or []
        if not isinstance(raw_citations, list):
            raise ValueError("citation_verdicts must be a list")
        for citation in raw_citations:
            if not isinstance(citation, dict):
                raise ValueError("citation verdict must be an object")
            evidence_id = str(citation.get("citation_id", "")).strip()
            citation_verdict = str(citation.get("verdict", "")).strip()
            if (
                evidence_id not in available_ids
                or citation_verdict not in _CITATION_VERDICTS
            ):
                raise ValueError("grounding judge returned invalid citation verdict")
            citation_rows.append(
                {
                    "citation_id": evidence_id,
                    "verdict": citation_verdict,
                }
            )
        normalized[index] = {
            "claim_type": claim_type,
            "verdict": verdict,
            "citation_verdicts": citation_rows,
            "reason": str(row.get("reason") or "")[:1000],
        }
    if len(normalized) != claim_count:
        raise ValueError("grounding judge omitted one or more claims")

    conflicts: list[dict[str, Any]] = []
    for row in raw.get("conflicts") or []:
        if not isinstance(row, dict):
            continue
        evidence_ids = [
            str(value)
            for value in row.get("citation_ids") or []
            if str(value) in available_ids
        ]
        if len(set(evidence_ids)) < 2:
            continue
        conflicts.append(
            {
                "citation_ids": list(dict.fromkeys(evidence_ids)),
                "reason": str(row.get("reason") or "")[:1000],
            }
        )
    return normalized, conflicts


def evaluate_grounding(
    answer: str,
    hits: list[dict[str, Any]],
    *,
    judge: GroundingJudge | None = None,
) -> dict[str, Any]:
    """Evaluate claim support and per-citation correctness.

    The default judge uses the configured chat model at temperature zero. Tests
    and offline calibration can inject a deterministic judge implementing the
    same versioned contract.
    """

    structural = validate_citations(answer, hits)
    claims = structural["claims"]
    available = {
        citation_id(int(hit["document_id"]), int(hit["chunk_index"])): str(
            hit.get("content") or ""
        )
        for hit in hits
    }
    payload = {
        "contract": GROUNDING_RUBRIC_VERSION,
        "claims": [
            {
                "claim_index": index,
                "text": claim["text"],
                "cited_evidence": [
                    {
                        "citation_id": evidence_id,
                        "text": available[evidence_id],
                    }
                    for evidence_id in claim["valid_citations"]
                ],
            }
            for index, claim in enumerate(claims)
        ],
        "all_retrieved_evidence": [
            {"citation_id": evidence_id, "text": text}
            for evidence_id, text in available.items()
        ],
    }
    judge_input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "payload": payload,
                "model": settings.chat_model,
                "rubric_version": GROUNDING_RUBRIC_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    base: dict[str, Any] = {
        **structural,
        "contract": GROUNDING_RUBRIC_VERSION,
        "structural_contract": structural["contract"],
        "structural_passed": structural["passed"],
        "judge_model": settings.chat_model,
        "judge_rubric_version": GROUNDING_RUBRIC_VERSION,
        "judge_input_fingerprint": judge_input_fingerprint,
        "judge_status": "not_applicable" if not claims else "pending",
        "groundedness": None,
        "faithfulness": None,
        "citation_correctness": None,
        "conflict_count": 0,
        "conflicts": [],
        "conflicts_disclosed": False,
        "refused": is_refusal(answer),
    }
    if not claims:
        base["passed"] = bool(structural["passed"])
        return base

    try:
        raw = (judge or _llm_judge)(payload)
        judgements, conflicts = _normalize_judgement(
            raw,
            len(claims),
            set(available),
        )
    except Exception as exc:
        base["judge_status"] = "unavailable"
        base["judge_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        base["semantic_entailment_checked"] = False
        base["passed"] = False
        return base

    supported_claims = 0
    supporting_links = 0
    total_links = sum(len(claim["citations"]) for claim in claims)
    claim_state_counts = {
        "supported": 0,
        "contradicted": 0,
        "mixed": 0,
        "insufficient": 0,
        "inference": 0,
    }
    for index, claim in enumerate(claims):
        judgement = judgements[index]
        explicit_inference = bool(_INFERENCE_RE.search(claim["text"]))
        citation_states = {
            row["verdict"] for row in judgement["citation_verdicts"]
        }
        if {"supports", "contradicts"} <= citation_states:
            evidence_state = "mixed"
        elif judgement["verdict"] == "contradicted":
            evidence_state = "contradicted"
        elif judgement["verdict"] == "reasonable_inference":
            evidence_state = "inference"
        elif judgement["verdict"] == "entailed":
            evidence_state = "supported"
        else:
            evidence_state = "insufficient"
        claim_state_counts[evidence_state] += 1
        supported = judgement["verdict"] == "entailed" or (
            judgement["verdict"] == "reasonable_inference"
            and judgement["claim_type"] == "inference"
            and explicit_inference
        )
        claim.update(
            {
                **judgement,
                "evidence_state": evidence_state,
                "explicit_inference": explicit_inference,
                "semantically_supported": supported,
            }
        )
        supported_claims += int(supported)
        supporting_links += sum(
            row["verdict"] == "supports"
            for row in judgement["citation_verdicts"]
            if row["citation_id"] in claim["valid_citations"]
        )

    groundedness = supported_claims / len(claims)
    citation_correctness = (
        supporting_links / total_links if total_links else 0.0
    )
    conflicts_disclosed = bool(
        conflicts and _CONFLICT_DISCLOSURE_RE.search(answer)
    )
    base.update(
        {
            "semantic_entailment_checked": True,
            "judge_status": "completed",
            "supported_claim_count": supported_claims,
            "unsupported_claim_count": len(claims) - supported_claims,
            "unsupported_claim_rate": round(1.0 - groundedness, 6),
            "groundedness": round(groundedness, 6),
            "faithfulness": round(groundedness, 6),
            "citation_correctness": round(citation_correctness, 6),
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
            "conflicts_disclosed": conflicts_disclosed,
            "claim_state_counts": claim_state_counts,
            "claims": claims,
            "passed": bool(structural["passed"])
            and supported_claims == len(claims)
            and citation_correctness == 1.0
            and (not conflicts or conflicts_disclosed),
        }
    )
    return base


def enforce_grounded_answer(answer: str, report: dict[str, Any]) -> str:
    """Remove unsupported claims; fail closed if semantic checking was absent."""

    if not report.get("semantic_entailment_checked"):
        return CANONICAL_REFUSAL
    sanitized = str(answer or "")
    for claim in report.get("claims") or []:
        if not claim.get("semantically_supported"):
            sanitized = sanitized.replace(str(claim.get("text") or ""), "")
    compact: list[str] = []
    for line in (line.rstrip() for line in sanitized.splitlines()):
        if not line and compact and not compact[-1]:
            continue
        compact.append(line)
    value = "\n".join(compact).strip()
    return value or CANONICAL_REFUSAL


def prepare_grounded_delivery(
    answer: str,
    hits: list[dict[str, Any]],
    *,
    evaluator: GroundingEvaluator = evaluate_grounding,
) -> tuple[str, dict[str, Any]]:
    """Verify, repair once, then refuse if an evidence-closed answer is absent."""

    initial = evaluator(answer, hits)
    if initial.get("passed"):
        return answer, {**initial, "delivery_action": "accepted"}

    if (
        initial.get("semantic_entailment_checked")
        and not initial.get("conflict_count")
    ):
        repaired = enforce_grounded_answer(answer, initial)
        if repaired != CANONICAL_REFUSAL and repaired != answer:
            repaired_report = evaluator(repaired, hits)
            if repaired_report.get("passed"):
                return repaired, {
                    **repaired_report,
                    "delivery_action": "removed_unsupported_claims",
                    "initial_verification": {
                        "groundedness": initial.get("groundedness"),
                        "citation_correctness": initial.get(
                            "citation_correctness"
                        ),
                        "unsupported_claim_count": initial.get(
                            "unsupported_claim_count"
                        ),
                    },
                }

    refusal_report = evaluator(CANONICAL_REFUSAL, hits)
    conflict_count = int(initial.get("conflict_count") or 0)
    conflict_action = conflict_count > 0
    if conflict_action:
        refusal_report = {
            **refusal_report,
            "conflict_count": conflict_count,
            "conflicts": list(initial.get("conflicts") or []),
            "conflicts_disclosed": False,
        }
    return CANONICAL_REFUSAL, {
        **refusal_report,
        "delivery_action": (
            "refused_due_to_conflicting_evidence"
            if conflict_action
            else "refused_after_quality_gate"
        ),
        "initial_verification": {
            "judge_status": initial.get("judge_status"),
            "judge_error": initial.get("judge_error"),
            "groundedness": initial.get("groundedness"),
            "citation_correctness": initial.get("citation_correctness"),
            "conflict_count": initial.get("conflict_count"),
            "unsupported_claim_count": initial.get("unsupported_claim_count"),
        },
    }


__all__ = [
    "CANONICAL_REFUSAL",
    "GROUNDING_RUBRIC_VERSION",
    "enforce_grounded_answer",
    "evaluate_grounding",
    "prepare_grounded_delivery",
]
