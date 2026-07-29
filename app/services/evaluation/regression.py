"""Deterministic regression gates for repeatable RAG evaluations.

The service evaluates persisted metric results rather than rerunning a model.
This keeps release decisions reproducible and lets a baseline finish before or
after its candidate without losing the comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.evaluation import (
    EvalMetricResult,
    EvalRegressionGate,
    EvalRegressionResult,
    EvalRun,
    EvalSample,
    RunStatus,
)


_RUN_METRIC_FIELDS = {
    "hit_rate": "hit_rate",
    "mrr": "mrr",
    "recall": "recall",
    "recall_at_k": "recall",
    "precision": "precision",
    "precision_at_k": "precision",
    "map": "map_score",
    "map_at_k": "map_score",
    "ndcg": "ndcg",
    "ndcg_at_k": "ndcg",
    "faithfulness": "faithfulness",
    "answer_relevance": "answer_relevancy",
    "answer_relevancy": "answer_relevancy",
    "citation_precision": "citation_precision",
    "citation_recall": "citation_recall",
    "unsupported_claim_rate": "unsupported_claim_rate",
}


@dataclass(frozen=True)
class MetricScore:
    score: float | None
    version: str | None
    sample_count: int | None = None


def _decode_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _decode_tags(raw: str | None) -> set[str]:
    if not raw:
        return set()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, str):
        return {value}
    return set()


def _sample_matches(sample: EvalSample, filters: dict) -> bool:
    """Apply the portable subset of badcase slice filters in Python.

    Supported keys are deliberately explicit so the same gate behaves on
    SQLite and PostgreSQL: ``tags``, ``difficulty``, and ``answerable``.
    """

    allowed = {"tags", "difficulty", "answerable"}
    if set(filters) - allowed:
        return False
    if "difficulty" in filters and sample.difficulty != filters["difficulty"]:
        return False
    if "answerable" in filters:
        requested_answerable = filters["answerable"]
        if (
            not isinstance(requested_answerable, bool)
            or sample.answerable != requested_answerable
        ):
            return False
    requested_tags = filters.get("tags")
    if requested_tags is not None:
        if not isinstance(requested_tags, list):
            return False
        if not set(map(str, requested_tags)) <= _decode_tags(sample.slice_tags):
            return False
    return True


def metric_score(
    db: Session,
    run_id: int,
    metric_name: str,
    *,
    metric_version: str | None = None,
    slice_filter: str | None = None,
) -> MetricScore:
    """Resolve an aggregate or slice score from versioned metric rows.

    Legacy fixed columns remain a read fallback for old completed runs. New
    runs always persist ``EvalMetricResult`` rows.
    """

    filters = _decode_json_object(slice_filter)
    query = select(EvalMetricResult).where(
        EvalMetricResult.run_id == run_id,
        EvalMetricResult.metric_name == metric_name,
    )
    if metric_version:
        query = query.where(EvalMetricResult.metric_version == metric_version)

    if filters:
        rows = db.execute(
            query.where(EvalMetricResult.subject_type == "sample").order_by(
                EvalMetricResult.metric_version.desc(),
                EvalMetricResult.id.asc(),
            )
        ).scalars()
        matching: list[EvalMetricResult] = []
        selected_version: str | None = metric_version
        for row in rows:
            if selected_version is None:
                selected_version = row.metric_version
            if row.metric_version != selected_version or row.score is None:
                continue
            sample = db.get(EvalSample, row.subject_id)
            if sample is not None and _sample_matches(sample, filters):
                matching.append(row)
        if not matching:
            return MetricScore(None, selected_version, 0)
        return MetricScore(
            sum(float(row.score) for row in matching) / len(matching),
            selected_version,
            len(matching),
        )

    metric = db.execute(
        query.where(
            EvalMetricResult.subject_type == "run",
            EvalMetricResult.subject_id == 0,
        ).order_by(
            EvalMetricResult.metric_version.desc(),
            EvalMetricResult.id.desc(),
        )
    ).scalars().first()
    if metric is not None:
        return MetricScore(metric.score, metric.metric_version)

    field = _RUN_METRIC_FIELDS.get(metric_name)
    run = db.get(EvalRun, run_id)
    if (
        run is not None
        and field is not None
        and metric_version in (None, "v1")
    ):
        value = getattr(run, field)
        return MetricScore(
            None if value is None else float(value),
            "v1",
        )
    return MetricScore(None, metric_version)


def _verdict(
    comparison: str,
    threshold: float,
    candidate: float,
    baseline: float | None,
) -> tuple[bool, float | None, str]:
    delta = None if baseline is None else candidate - baseline
    if comparison == "absolute_min":
        passed = candidate >= threshold
        return passed, delta, f"{candidate:.6f} >= {threshold:.6f}"
    if comparison == "absolute_max":
        passed = candidate <= threshold
        return passed, delta, f"{candidate:.6f} <= {threshold:.6f}"
    if baseline is None:
        return False, None, "baseline metric is unavailable"
    if comparison == "max_drop":
        passed = delta is not None and delta >= -threshold
        return passed, delta, f"delta {delta:.6f} >= -{threshold:.6f}"
    if comparison == "max_increase":
        passed = delta is not None and delta <= threshold
        return passed, delta, f"delta {delta:.6f} <= {threshold:.6f}"
    return False, delta, f"unsupported comparison: {comparison}"


def evaluate_candidate(
    db: Session,
    candidate_run_id: int,
) -> list[EvalRegressionResult]:
    """Evaluate all currently applicable gates for one completed candidate."""

    candidate = db.get(EvalRun, candidate_run_id)
    if (
        candidate is None
        or candidate.status != RunStatus.COMPLETED
        or candidate.evaluation_scope != "full"
    ):
        return []

    gates = list(
        db.execute(
            select(EvalRegressionGate).where(
                EvalRegressionGate.dataset_id == candidate.dataset_id,
                EvalRegressionGate.enabled.is_(True),
            )
        ).scalars()
    )
    baseline = (
        db.get(EvalRun, candidate.baseline_run_id)
        if candidate.baseline_run_id is not None
        else None
    )
    results: list[EvalRegressionResult] = []
    for gate in gates:
        relative = gate.comparison in {"max_drop", "max_increase"}
        if relative and (
            baseline is None
            or baseline.status != RunStatus.COMPLETED
            or baseline.dataset_id != candidate.dataset_id
        ):
            continue

        candidate_metric = metric_score(
            db,
            candidate.id,
            gate.metric_name,
            metric_version=gate.metric_version,
            slice_filter=gate.slice_filter,
        )
        baseline_metric = (
            metric_score(
                db,
                baseline.id,
                gate.metric_name,
                metric_version=gate.metric_version,
                slice_filter=gate.slice_filter,
            )
            if relative and baseline is not None
            else MetricScore(None, gate.metric_version)
        )

        if candidate_metric.score is None:
            passed, delta, reason = False, None, "candidate metric is unavailable"
        else:
            passed, delta, reason = _verdict(
                gate.comparison,
                gate.threshold,
                candidate_metric.score,
                baseline_metric.score,
            )
        if candidate_metric.sample_count is not None:
            reason = f"{reason}; slice_samples={candidate_metric.sample_count}"

        db.execute(
            delete(EvalRegressionResult).where(
                EvalRegressionResult.candidate_run_id == candidate.id,
                EvalRegressionResult.gate_id == gate.id,
            )
        )
        result = EvalRegressionResult(
            candidate_run_id=candidate.id,
            baseline_run_id=baseline.id if relative and baseline is not None else None,
            gate_id=gate.id,
            metric_name=gate.metric_name,
            baseline_score=baseline_metric.score,
            candidate_score=candidate_metric.score,
            delta=delta,
            passed=passed,
            reason=reason,
        )
        db.add(result)
        results.append(result)
    return results


def evaluate_related_regressions(
    db: Session,
    completed_run_id: int,
) -> list[EvalRegressionResult]:
    """Evaluate this run and candidates that were waiting for it as baseline."""

    results = evaluate_candidate(db, completed_run_id)
    waiting_candidates = db.execute(
        select(EvalRun.id).where(
            EvalRun.baseline_run_id == completed_run_id,
            EvalRun.status == RunStatus.COMPLETED,
        )
    ).scalars()
    for candidate_id in waiting_candidates:
        results.extend(evaluate_candidate(db, candidate_id))
    return results
