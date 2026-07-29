"""Deterministic evidence identifiers and sentence-level citation validation.

The validator intentionally checks a structural contract, not semantic
entailment: every factual claim must cite one or more chunks that were actually
available to the model.  A lexical-overlap value is included only as a
diagnostic signal and is never treated as proof that a source entails a claim.
"""

from __future__ import annotations

import re
from typing import Any

_CITATION_RE = re.compile(r"\[D(?P<document_id>\d+):C(?P<chunk_index>\d+)\]")
_SENTENCE_RE = re.compile(
    r"[^。！？!?；;\n]+"
    r"(?:[。！？!?；;](?:\s*\[D\d+:C\d+\])*|$)"
)
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
_REFUSAL_MARKERS = (
    "根据已有文档无法回答",
    "根据提供的文档无法回答",
    "无法从已有文档",
    "无法从提供的文档",
    "未检索到相关",
    "材料未提及",
    "没有足够证据",
    "证据不足",
    "无法确定",
)


def citation_id(document_id: int, chunk_index: int) -> str:
    """Return the stable public identifier for one document chunk."""
    return f"D{int(document_id)}:C{int(chunk_index)}"


def citation_marker(document_id: int, chunk_index: int) -> str:
    """Return the marker the model must place after a factual claim."""
    return f"[{citation_id(document_id, chunk_index)}]"


def attach_citation_ids(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clone retrieval hits and attach stable evidence identifiers."""
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        item = dict(hit)
        item["citation_id"] = citation_id(
            int(item["document_id"]),
            int(item["chunk_index"]),
        )
        enriched.append(item)
    return enriched


def build_evidence_context(
    hits: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
) -> str:
    """Build model context with identifiers that can be cited verbatim."""
    parts: list[str] = []
    used = 0
    for hit in attach_citation_ids(hits):
        content = str(hit.get("content") or "").strip()
        if not content:
            continue
        if max_chars is not None:
            remaining = max_chars - used
            if remaining <= 0:
                break
            content = content[:remaining]
        parts.append(f"[{hit['citation_id']}] {content}")
        used += len(content)
    return "\n\n".join(parts)


def _claim_sentences(answer: str) -> list[str]:
    claims: list[str] = []
    in_code_fence = False
    for raw_line in answer.splitlines() or [answer]:
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or not line or line.startswith("#"):
            continue
        for raw_sentence in _SENTENCE_RE.findall(line):
            sentence = _MARKDOWN_PREFIX_RE.sub("", raw_sentence).strip()
            if not sentence or sentence in {"---", "***"}:
                continue
            plain = _CITATION_RE.sub("", sentence).strip()
            if not plain or plain.endswith(("：", ":")):
                continue
            if any(marker in plain for marker in _REFUSAL_MARKERS):
                continue
            claims.append(sentence)
    return claims


def extract_claims(answer: str) -> list[str]:
    """Return factual/inferential claim sentences in stable display order."""

    return _claim_sentences(answer)


def is_refusal(answer: str) -> bool:
    """Detect the canonical evidence-insufficient refusal contract."""

    normalized = str(answer or "").strip()
    return any(marker in normalized for marker in _REFUSAL_MARKERS)


def _lexical_overlap(claim: str, evidence_texts: list[str]) -> float:
    """Return token overlap for diagnostics; this is not an entailment score."""
    claim_tokens = set(_TOKEN_RE.findall(_CITATION_RE.sub("", claim).lower()))
    if not claim_tokens or not evidence_texts:
        return 0.0
    evidence_tokens = set(
        _TOKEN_RE.findall(" ".join(evidence_texts).lower())
    )
    return round(len(claim_tokens & evidence_tokens) / len(claim_tokens), 6)


def validate_citations(
    answer: str,
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate that each factual sentence cites evidence from this retrieval run.

    ``supported`` means "contains at least one valid citation marker".  It does
    not claim that the cited chunk semantically entails the sentence.
    """
    available: dict[str, str] = {}
    for hit in hits:
        key = citation_id(int(hit["document_id"]), int(hit["chunk_index"]))
        available[key] = str(hit.get("content") or "")

    claim_rows: list[dict[str, Any]] = []
    citation_count = 0
    valid_citation_count = 0
    for sentence in _claim_sentences(answer):
        references = [
            citation_id(
                int(match.group("document_id")),
                int(match.group("chunk_index")),
            )
            for match in _CITATION_RE.finditer(sentence)
        ]
        valid = [reference for reference in references if reference in available]
        invalid = [reference for reference in references if reference not in available]
        citation_count += len(references)
        valid_citation_count += len(valid)
        claim_rows.append(
            {
                "text": sentence,
                "citations": references,
                "valid_citations": valid,
                "invalid_citations": invalid,
                "supported": bool(valid),
                "lexical_overlap": _lexical_overlap(
                    sentence,
                    [available[reference] for reference in valid],
                ),
            }
        )

    claim_count = len(claim_rows)
    supported_claim_count = sum(1 for claim in claim_rows if claim["supported"])
    unsupported_claim_count = claim_count - supported_claim_count
    citation_precision = (
        valid_citation_count / citation_count if citation_count else 1.0
    )
    citation_recall = (
        supported_claim_count / claim_count if claim_count else 1.0
    )
    unsupported_claim_rate = (
        unsupported_claim_count / claim_count if claim_count else 0.0
    )
    return {
        "contract": "citation_presence_v1",
        "semantic_entailment_checked": False,
        "claim_count": claim_count,
        "supported_claim_count": supported_claim_count,
        "unsupported_claim_count": unsupported_claim_count,
        "citation_count": citation_count,
        "valid_citation_count": valid_citation_count,
        "citation_precision": round(citation_precision, 6),
        "citation_recall": round(citation_recall, 6),
        "unsupported_claim_rate": round(unsupported_claim_rate, 6),
        "passed": unsupported_claim_count == 0
        and valid_citation_count == citation_count,
        "claims": claim_rows,
    }
