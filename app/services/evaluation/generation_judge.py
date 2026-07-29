"""Versioned LLM judges for generation quality.

An unavailable or malformed judge result is represented as ``score=None``.
Infrastructure failures must never be converted into a fabricated quality
score of zero, because that makes regression diagnosis impossible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from app.config import settings
from app.services.llm_service import chat_completion

FAITHFULNESS_RUBRIC_VERSION = "faithfulness_v2"
ANSWER_RELEVANCE_RUBRIC_VERSION = "answer_relevance_v2"

FAITHFULNESS_PROMPT = """你是 RAG 质量评估专家。评估以下答案对检索到的文档片段的忠实程度。

【检索片段】
{context}

【生成答案】
{answer}

评估标准：
- 答案中每个事实性陈述，是否都能在检索片段中找到依据？
- 不能歪曲、扩大或缩小原文含义
- 不能出现超出检索片段范围的编造内容
- 忽略语气词、连接词等非事实内容

请输出严格 JSON：
{{
  "score": 0到1之间的小数（1=完全忠实，0=完全歪曲或编造）,
  "reason": "一句话说明依据"
}}"""

RELEVANCY_PROMPT = """你是 RAG 质量评估专家。评估以下答案对用户问题的相关性。

【用户问题】
{question}

【生成答案】
{answer}

评估标准：
- 答案是否直接回答了问题？
- 答案是否覆盖问题要求的关键点？
- 答案是否避免大量无关内容？

请输出严格 JSON：
{{
  "score": 0到1之间的小数（1=完整且直接，0=完全跑题）,
  "reason": "一句话说明依据"
}}"""

JudgeStatus = Literal["completed", "unavailable", "invalid"]


@dataclass(frozen=True)
class JudgeResult:
    score: float | None
    reason: str
    status: JudgeStatus
    model: str
    rubric_version: str
    input_fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_json_fence(content: str) -> str:
    value = content.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return value


def _fingerprint(prompt: str, rubric_version: str) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "rubric_version": rubric_version,
            "model": settings.chat_model,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _call_judge(prompt: str, rubric_version: str) -> JudgeResult:
    fingerprint = _fingerprint(prompt, rubric_version)
    try:
        response = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception as exc:
        return JudgeResult(
            score=None,
            reason=f"{type(exc).__name__}: {str(exc)[:500]}",
            status="unavailable",
            model=settings.chat_model,
            rubric_version=rubric_version,
            input_fingerprint=fingerprint,
        )

    try:
        data = json.loads(_strip_json_fence(response.content or ""))
        if not isinstance(data, dict) or "score" not in data:
            raise ValueError("judge response requires a score")
        score = float(data["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("judge score must be between 0 and 1")
        reason = str(data.get("reason") or "未提供理由").strip()[:1000]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return JudgeResult(
            score=None,
            reason=f"{type(exc).__name__}: {str(exc)[:500]}",
            status="invalid",
            model=settings.chat_model,
            rubric_version=rubric_version,
            input_fingerprint=fingerprint,
        )
    return JudgeResult(
        score=score,
        reason=reason,
        status="completed",
        model=settings.chat_model,
        rubric_version=rubric_version,
        input_fingerprint=fingerprint,
    )


def evaluate_faithfulness(
    context_chunks: list[str],
    answer: str,
) -> JudgeResult:
    context = "\n\n".join(
        f"[片段{i + 1}] {content}"
        for i, content in enumerate(context_chunks)
    )
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    return _call_judge(prompt, FAITHFULNESS_RUBRIC_VERSION)


def evaluate_answer_relevancy(question: str, answer: str) -> JudgeResult:
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    return _call_judge(prompt, ANSWER_RELEVANCE_RUBRIC_VERSION)


def judge_faithfulness(
    context_chunks: list[str],
    answer: str,
) -> float | None:
    """Compatibility score-only API. Prefer ``evaluate_faithfulness``."""

    return evaluate_faithfulness(context_chunks, answer).score


def judge_answer_relevancy(question: str, answer: str) -> float | None:
    """Compatibility score-only API. Prefer ``evaluate_answer_relevancy``."""

    return evaluate_answer_relevancy(question, answer).score


__all__ = [
    "ANSWER_RELEVANCE_RUBRIC_VERSION",
    "FAITHFULNESS_RUBRIC_VERSION",
    "JudgeResult",
    "evaluate_answer_relevancy",
    "evaluate_faithfulness",
    "judge_answer_relevancy",
    "judge_faithfulness",
]
