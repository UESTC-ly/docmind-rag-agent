"""Meta-evaluate citation and answer-relevance judges on public human labels."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.services.evaluation.generation_judge import (
    JudgeResult,
    evaluate_answer_relevancy,
)
from app.services.evaluation.grounding import evaluate_grounding

PUBLIC_JUDGE_CALIBRATION_CONTRACT = "public_binary_judge_calibration_v1"
_NUMERIC_CITATION_RE = re.compile(r"\[\d+\]")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _balanced_prefix(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("max_cases must be positive")
    positive = [row for row in rows if row["gold_positive"]]
    negative = [row for row in rows if not row["gold_positive"]]
    selected = []
    while len(selected) < limit and (positive or negative):
        if positive:
            selected.append(positive.pop(0))
        if len(selected) < limit and negative:
            selected.append(negative.pop(0))
    return selected


def load_alce_citation_cases(
    path: str | Path,
    *,
    max_cases: int = 100,
) -> list[dict[str, Any]]:
    """Load sentence/citation pairs with ALCE's 0/1/2 human support labels."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ALCE human citation file must be a JSON object")
    rows: list[dict[str, Any]] = []
    for dataset_name, models in sorted(payload.items()):
        if not isinstance(models, dict):
            continue
        for model_name, examples in sorted(models.items()):
            if not isinstance(examples, dict):
                continue
            for question_id, example in sorted(examples.items()):
                if question_id == "overall_results" or not isinstance(example, dict):
                    continue
                for sentence_index, sentence in enumerate(
                    example.get("sentences") or []
                ):
                    if not isinstance(sentence, dict):
                        continue
                    claim = _NUMERIC_CITATION_RE.sub(
                        "",
                        str(sentence.get("text") or ""),
                    ).strip()
                    for citation_index, citation in enumerate(
                        sentence.get("citations") or []
                    ):
                        if not isinstance(citation, dict):
                            continue
                        evidence = str(citation.get("text") or "").strip()
                        try:
                            human_score = int(
                                citation["citation_precision_score"]
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                        if not claim or not evidence or human_score not in {0, 1, 2}:
                            continue
                        case_id = (
                            f"{dataset_name}:{model_name}:{question_id}:"
                            f"{sentence_index}:{citation_index}"
                        )
                        rows.append(
                            {
                                "case_id": case_id,
                                "slice": dataset_name,
                                "gold_positive": human_score == 2,
                                "gold_label": human_score,
                                "claim": claim,
                                "evidence": evidence,
                                "input_fingerprint": _fingerprint(
                                    {"claim": claim, "evidence": evidence}
                                ),
                            }
                        )
    selected = _balanced_prefix(rows, max_cases)
    if not selected:
        raise ValueError("ALCE selection produced no citation cases")
    return selected


def load_ares_relevance_cases(
    path: str | Path,
    *,
    max_cases: int = 100,
) -> list[dict[str, Any]]:
    """Load ARES NQ query/answer pairs with human answer-relevance labels."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            query = str(row.get("Query") or "").strip()
            answer = str(row.get("Answer") or "").strip()
            raw_label = str(row.get("Answer_Relevance_Label") or "").strip()
            if not query or not answer or raw_label not in {"0", "0.0", "1", "1.0"}:
                continue
            label = float(raw_label) == 1.0
            rows.append(
                {
                    "case_id": str(row.get("id") or index),
                    "slice": "natural_questions",
                    "gold_positive": label,
                    "gold_label": int(label),
                    "question": query,
                    "answer": answer,
                    "input_fingerprint": _fingerprint(
                        {"question": query, "answer": answer}
                    ),
                }
            )
    selected = _balanced_prefix(rows, max_cases)
    if not selected:
        raise ValueError("ARES selection produced no answer relevance cases")
    return selected


def evaluate_alce_citation(case: dict[str, Any]) -> float | None:
    report = evaluate_grounding(
        f"{case['claim']} [D1:C0]",
        [
            {
                "document_id": 1,
                "chunk_index": 0,
                "content": case["evidence"],
                "score": 1.0,
            }
        ],
    )
    if report.get("judge_status") != "completed":
        return None
    return float(report["citation_correctness"])


def evaluate_ares_relevance(case: dict[str, Any]) -> JudgeResult:
    return evaluate_answer_relevancy(case["question"], case["answer"])


def _result_payload(result: JudgeResult | float | int | None) -> dict[str, Any]:
    if isinstance(result, JudgeResult):
        return result.to_dict()
    if isinstance(result, (float, int)) and 0 <= float(result) <= 1:
        return {
            "score": float(result),
            "status": "completed",
            "reason": "injected evaluator result",
            "model": "injected",
            "rubric_version": "injected",
            "input_fingerprint": None,
        }
    return {
        "score": None,
        "status": "unavailable",
        "reason": "judge returned no usable result",
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def calibrate_public_binary_judge(
    cases: list[dict[str, Any]],
    *,
    target: str,
    evaluator: Callable[[dict[str, Any]], JudgeResult | float | int | None],
    score_threshold: float = 0.8,
    minimum_cases: int = 100,
    minimum_balanced_accuracy: float = 0.8,
    max_workers: int = 1,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("calibration cases are required")
    if not 1 <= max_workers <= 16:
        raise ValueError("max_workers must be between 1 and 16")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(evaluator, cases))
    evidence = []
    for case, result in zip(cases, results, strict=True):
        judge = _result_payload(result)
        score = judge.get("score")
        predicted = None if score is None else float(score) >= score_threshold
        evidence.append(
            {
                "case_id": case["case_id"],
                "slice": case["slice"],
                "gold_positive": bool(case["gold_positive"]),
                "gold_label": case["gold_label"],
                "score": score,
                "predicted_positive": predicted,
                "correct": (
                    None
                    if predicted is None
                    else predicted == bool(case["gold_positive"])
                ),
                "case_input_fingerprint": case["input_fingerprint"],
                "judge_status": judge.get("status"),
                "judge_reason": judge.get("reason"),
                "judge_model": judge.get("model"),
                "judge_rubric_version": judge.get("rubric_version"),
                "judge_input_fingerprint": judge.get("input_fingerprint"),
            }
        )
    available = [row for row in evidence if row["score"] is not None]
    tp = sum(row["gold_positive"] and row["predicted_positive"] for row in available)
    tn = sum(
        not row["gold_positive"] and not row["predicted_positive"]
        for row in available
    )
    fp = sum(
        not row["gold_positive"] and row["predicted_positive"]
        for row in available
    )
    fn = sum(
        row["gold_positive"] and not row["predicted_positive"]
        for row in available
    )
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    balanced_accuracy = (
        round((recall + specificity) / 2, 6)
        if recall is not None and specificity is not None
        else None
    )
    metrics = {
        "case_count": len(evidence),
        "available_count": len(available),
        "unavailable_count": len(evidence) - len(available),
        "positive_count": sum(row["gold_positive"] for row in evidence),
        "negative_count": sum(not row["gold_positive"] for row in evidence),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "coverage": _ratio(len(available), len(evidence)),
        "accuracy": _ratio(tp + tn, len(available)),
        "precision": _ratio(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }
    eligible = bool(
        len(evidence) >= minimum_cases
        and metrics["unavailable_count"] == 0
        and metrics["positive_count"] > 0
        and metrics["negative_count"] > 0
        and balanced_accuracy is not None
        and balanced_accuracy >= minimum_balanced_accuracy
    )
    return {
        "contract": PUBLIC_JUDGE_CALIBRATION_CONTRACT,
        "target": target,
        "release_gate_eligible": eligible,
        "criteria": {
            "score_threshold": score_threshold,
            "minimum_cases": minimum_cases,
            "minimum_balanced_accuracy": minimum_balanced_accuracy,
            "require_full_coverage": True,
            "require_both_classes": True,
        },
        "metrics": metrics,
        "evidence": evidence,
    }


__all__ = [
    "PUBLIC_JUDGE_CALIBRATION_CONTRACT",
    "calibrate_public_binary_judge",
    "evaluate_alce_citation",
    "evaluate_ares_relevance",
    "load_alce_citation_cases",
    "load_ares_relevance_cases",
]
