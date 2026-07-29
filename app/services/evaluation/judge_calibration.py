"""Calibrate DocMind's faithfulness judge on public RAGTruth labels.

The production judge is probabilistic, so its score must not become a release
gate merely because an API call succeeded.  This module compares a versioned
judge against the official RAGTruth response/source JSONL contract and retains
the confusion matrix, unavailable results, slice metrics, and compact
per-response evidence needed to audit the decision.

RAGTruth labels include ``implicit_true`` spans: the statement may be true in
the world but is absent from the supplied RAG context.  DocMind's strict
grounding contract still treats those spans as unsupported.  The report keeps
both the strict-grounding target and the narrower hallucination target so the
distinction is explicit rather than silently changing the public labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.services.evaluation.generation_judge import (
    JudgeResult,
    evaluate_faithfulness,
)

RAGTRUTH_CALIBRATION_CONTRACT = "ragtruth_faithfulness_calibration_v1"
RAGTRUTH_TRANSFORM_CONTRACT = "ragtruth_official_jsonl_strict_grounding_v1"

FaithfulnessEvaluator = Callable[
    [list[str], str],
    JudgeResult | float | int | None,
]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path.name}:{line_number} must contain a JSON object"
                )
            yield row


def _context_text(source_info: Any) -> str:
    if isinstance(source_info, str):
        return source_info.strip()
    if isinstance(source_info, dict):
        passages = source_info.get("passages")
        if isinstance(passages, str) and passages.strip():
            return passages.strip()
        return _canonical_json(source_info)
    return ""


def load_ragtruth_cases(
    source_info_path: str | Path,
    response_path: str | Path,
    *,
    split: str = "test",
    max_cases: int | None = None,
    task_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load deterministic, quality-clean RAGTruth cases from official JSONL.

    Responses marked ``incorrect_refusal`` or ``truncated`` are excluded from
    faithfulness calibration because they measure different failure modes.
    The file order is retained and ``max_cases`` selects a reproducible prefix.
    """

    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive")
    source_path = Path(source_info_path)
    answers_path = Path(response_path)
    if not source_path.is_file() or not answers_path.is_file():
        raise ValueError("RAGTruth source_info.jsonl and response.jsonl are required")

    sources: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(source_path):
        source_id = str(row.get("source_id") or "").strip()
        context = _context_text(row.get("source_info"))
        if not source_id or not context:
            continue
        sources[source_id] = {
            "source_id": source_id,
            "task_type": str(row.get("task_type") or "unknown"),
            "source": str(row.get("source") or "unknown"),
            "context": context,
            "prompt": str(row.get("prompt") or ""),
        }

    allowed_tasks = {value.casefold() for value in task_types or set()}
    cases: list[dict[str, Any]] = []
    for row in _iter_jsonl(answers_path):
        if str(row.get("split") or "") != split:
            continue
        if str(row.get("quality") or "good") != "good":
            continue
        source_id = str(row.get("source_id") or "").strip()
        source = sources.get(source_id)
        response = str(row.get("response") or "").strip()
        if source is None or not response:
            continue
        if (
            allowed_tasks
            and source["task_type"].casefold() not in allowed_tasks
        ):
            continue
        raw_labels = row.get("labels") or []
        labels = [
            label
            for label in raw_labels
            if isinstance(label, dict)
        ] if isinstance(raw_labels, list) else []
        strict_unsupported = bool(labels)
        hallucinated = any(
            not bool(label.get("implicit_true"))
            for label in labels
        )
        payload_fingerprint = _sha256_text(
            _canonical_json(
                {
                    "context": source["context"],
                    "response": response,
                }
            )
        )
        cases.append(
            {
                "response_id": str(row.get("id") or len(cases)),
                "source_id": source_id,
                "split": split,
                "task_type": source["task_type"],
                "source": source["source"],
                "generator_model": str(row.get("model") or "unknown"),
                "context": source["context"],
                "response": response,
                "gold_strict_unsupported": strict_unsupported,
                "gold_hallucinated": hallucinated,
                "label_count": len(labels),
                "implicit_true_count": sum(
                    bool(label.get("implicit_true")) for label in labels
                ),
                "input_fingerprint": payload_fingerprint,
            }
        )
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise ValueError("RAGTruth selection produced no calibration cases")
    return cases


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _judge_payload(
    result: JudgeResult | float | int | None,
) -> tuple[float | None, dict[str, Any]]:
    if isinstance(result, JudgeResult):
        return result.score, result.to_dict()
    if isinstance(result, (float, int)):
        score = float(result)
        if not 0.0 <= score <= 1.0:
            return None, {
                "status": "unavailable",
                "reason": "injected score was outside [0, 1]",
                "score": None,
            }
        return score, {
            "status": "completed",
            "reason": "injected evaluator result",
            "score": score,
            "model": "injected",
            "rubric_version": "injected",
            "input_fingerprint": None,
        }
    return None, {
        "status": "unavailable",
        "reason": "judge returned no result",
        "score": None,
    }


def _slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["score"] is not None]
    tp = sum(
        row["gold_strict_unsupported"] and row["predicted_unsupported"]
        for row in available
    )
    tn = sum(
        not row["gold_strict_unsupported"]
        and not row["predicted_unsupported"]
        for row in available
    )
    fp = sum(
        not row["gold_strict_unsupported"] and row["predicted_unsupported"]
        for row in available
    )
    fn = sum(
        row["gold_strict_unsupported"]
        and not row["predicted_unsupported"]
        for row in available
    )
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    balanced_accuracy = (
        round((recall + specificity) / 2, 6)
        if recall is not None and specificity is not None
        else None
    )
    precision = _ratio(tp, tp + fp)
    return {
        "case_count": len(rows),
        "available_count": len(available),
        "unavailable_count": len(rows) - len(available),
        "positive_count": sum(
            bool(row["gold_strict_unsupported"]) for row in rows
        ),
        "negative_count": sum(
            not bool(row["gold_strict_unsupported"]) for row in rows
        ),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "coverage": _ratio(len(available), len(rows)),
        "accuracy": _ratio(tp + tn, len(available)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": (
            round(2 * precision * recall / (precision + recall), 6)
            if precision is not None
            and recall is not None
            and precision + recall > 0
            else None
        ),
        "balanced_accuracy": balanced_accuracy,
    }


def calibrate_faithfulness_judge(
    cases: list[dict[str, Any]],
    *,
    evaluator: FaithfulnessEvaluator = evaluate_faithfulness,
    score_threshold: float = 0.8,
    minimum_cases: int = 100,
    minimum_balanced_accuracy: float = 0.8,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Compare the judge with held-out public labels and gate its own use."""

    if not cases:
        raise ValueError("calibration cases are required")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be within [0, 1]")
    if minimum_cases <= 0:
        raise ValueError("minimum_cases must be positive")
    if not 0.0 <= minimum_balanced_accuracy <= 1.0:
        raise ValueError("minimum_balanced_accuracy must be within [0, 1]")
    if not 1 <= max_workers <= 16:
        raise ValueError("max_workers must be between 1 and 16")

    def evaluate_case(case: dict[str, Any]):
        return _judge_payload(
            evaluator([str(case["context"])], str(case["response"]))
        )

    if max_workers == 1:
        judge_results = [evaluate_case(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            judge_results = list(pool.map(evaluate_case, cases))
    evidence: list[dict[str, Any]] = []
    for case, (score, judge) in zip(cases, judge_results, strict=True):
        predicted = None if score is None else score < score_threshold
        evidence.append(
            {
                "response_id": case["response_id"],
                "source_id": case["source_id"],
                "task_type": case["task_type"],
                "source": case["source"],
                "generator_model": case["generator_model"],
                "gold_strict_unsupported": bool(
                    case["gold_strict_unsupported"]
                ),
                "gold_hallucinated": bool(case["gold_hallucinated"]),
                "label_count": int(case["label_count"]),
                "implicit_true_count": int(case["implicit_true_count"]),
                "score": score,
                "predicted_unsupported": predicted,
                "correct": (
                    None
                    if predicted is None
                    else predicted == bool(case["gold_strict_unsupported"])
                ),
                "case_input_fingerprint": case["input_fingerprint"],
                "judge_status": judge.get("status"),
                "judge_reason": judge.get("reason"),
                "judge_model": judge.get("model"),
                "judge_rubric_version": judge.get("rubric_version"),
                "judge_input_fingerprint": judge.get("input_fingerprint"),
            }
        )

    overall = _slice_metrics(evidence)
    slices = {
        task_type: _slice_metrics(
            [row for row in evidence if row["task_type"] == task_type]
        )
        for task_type in sorted({row["task_type"] for row in evidence})
    }
    balanced_accuracy = overall["balanced_accuracy"]
    release_gate_eligible = bool(
        overall["case_count"] >= minimum_cases
        and overall["unavailable_count"] == 0
        and overall["positive_count"] > 0
        and overall["negative_count"] > 0
        and balanced_accuracy is not None
        and balanced_accuracy >= minimum_balanced_accuracy
    )
    return {
        "contract": RAGTRUTH_CALIBRATION_CONTRACT,
        "target": "faithfulness_v2_strict_grounding",
        "gold_target": "ragtruth_any_annotated_span_v1",
        "release_gate_eligible": release_gate_eligible,
        "decision_reason": (
            "public held-out labels meet coverage and balanced-accuracy gate"
            if release_gate_eligible
            else "judge calibration does not satisfy every release criterion"
        ),
        "criteria": {
            "score_threshold": score_threshold,
            "minimum_cases": minimum_cases,
            "minimum_balanced_accuracy": minimum_balanced_accuracy,
            "max_workers": max_workers,
            "require_full_coverage": True,
            "require_both_classes": True,
        },
        "metrics": overall,
        "slices": slices,
        "evidence": evidence,
    }


__all__ = [
    "RAGTRUTH_CALIBRATION_CONTRACT",
    "RAGTRUTH_TRANSFORM_CONTRACT",
    "calibrate_faithfulness_judge",
    "load_ragtruth_cases",
]
