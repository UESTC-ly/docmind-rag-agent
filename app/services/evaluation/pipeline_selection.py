"""Select a RAG pipeline from comparable, public evaluation evidence.

The selector never compares scores across different datasets.  It first picks
one reproducible public dataset snapshot, then ranks current-compatible
pipelines inside that snapshot.  A failed or incomplete error-level regression
gate blocks automatic selection; warning gates remain visible evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.evaluation import (
    EvalDataset,
    EvalRegressionGate,
    EvalRegressionResult,
    EvalRun,
    RunStatus,
)
from app.services.evaluation.regression import MetricScore, metric_score
from app.services.rag_pipeline import PipelineSpec, pipeline_presets

SelectionTarget = Literal["retrieval", "grounded_generation"]

_RETRIEVAL_METRICS = (
    ("ndcg_at_k", "retrieval_v1"),
    ("map_at_k", "retrieval_v1"),
    ("recall_at_k", "retrieval_v1"),
    ("mrr", "retrieval_v1"),
)
_GENERATION_METRICS = (
    ("groundedness", "claim_grounding_v1"),
    ("citation_correctness", "claim_grounding_v1"),
    ("answer_relevance", "model_judge_v2"),
)


def _dataset_provenance_complete(dataset: EvalDataset) -> bool:
    return bool(
        dataset.release_eligible
        and dataset.source_name
        and dataset.source_uri
        and dataset.source_version
        and dataset.license_name
        and dataset.split
        and dataset.corpus_fingerprint
        and dataset.source_snapshot_fingerprint
    )


def _matches_target(
    dataset: EvalDataset,
    *,
    target: SelectionTarget,
    language: str | None,
) -> bool:
    if language and dataset.language != language:
        return False
    if target == "grounded_generation":
        return dataset.task_type == "rag_qa"
    return dataset.task_type in {
        "retrieval",
        "document_retrieval",
        "rag_qa",
    }


def _metric_bundle(
    db: Session,
    run_id: int,
    target: SelectionTarget,
) -> tuple[dict[str, float], dict[str, str]]:
    definitions = list(_RETRIEVAL_METRICS)
    if target == "grounded_generation":
        definitions.extend(_GENERATION_METRICS)
    scores: dict[str, float] = {}
    versions: dict[str, str] = {}
    for name, version in definitions:
        result: MetricScore = metric_score(
            db,
            run_id,
            name,
            metric_version=version,
        )
        if result.score is not None:
            scores[name] = float(result.score)
            versions[name] = result.version or version
    return scores, versions


def _gate_state(
    db: Session,
    run: EvalRun,
) -> tuple[str, list[dict[str, Any]]]:
    """Return approved/observational/blocked plus auditable gate evidence."""

    gates = list(
        db.execute(
            select(EvalRegressionGate)
            .where(
                EvalRegressionGate.dataset_id == run.dataset_id,
                EvalRegressionGate.enabled.is_(True),
            )
            .order_by(EvalRegressionGate.id)
        ).scalars()
    )
    results = {
        row.gate_id: row
        for row in db.execute(
            select(EvalRegressionResult).where(
                EvalRegressionResult.candidate_run_id == run.id
            )
        ).scalars()
    }
    evidence: list[dict[str, Any]] = []
    applicable_errors = 0
    blocked = False
    for gate in gates:
        relative = gate.comparison in {"max_drop", "max_increase"}
        applicable = not relative or run.baseline_run_id is not None
        result = results.get(gate.id)
        evidence.append(
            {
                "gate_id": gate.id,
                "name": gate.name,
                "severity": gate.severity,
                "comparison": gate.comparison,
                "metric_name": gate.metric_name,
                "metric_version": gate.metric_version,
                "applicable": applicable,
                "passed": result.passed if result is not None else None,
                "reason": (
                    result.reason
                    if result is not None
                    else "gate result unavailable"
                ),
            }
        )
        if gate.severity != "error" or not applicable:
            continue
        applicable_errors += 1
        if result is None or not result.passed:
            blocked = True
    if blocked:
        return "blocked", evidence
    if applicable_errors:
        return "approved", evidence
    return "observational", evidence


def _candidate(
    db: Session,
    run: EvalRun,
    dataset: EvalDataset,
    catalog: Mapping[str, PipelineSpec],
    target: SelectionTarget,
) -> dict[str, Any] | None:
    if not run.pipeline_id or not run.pipeline_fingerprint:
        return None
    current = catalog.get(run.pipeline_id)
    if current is None or current.fingerprint != run.pipeline_fingerprint:
        return None
    metrics, metric_versions = _metric_bundle(db, run.id, target)
    required = {"ndcg_at_k", "map_at_k", "recall_at_k", "mrr"}
    if target == "grounded_generation":
        required.update(
            {"groundedness", "citation_correctness", "answer_relevance"}
        )
    if not required <= metrics.keys():
        return None
    release_status, gates = _gate_state(db, run)
    return {
        "run_id": run.id,
        "dataset_id": dataset.id,
        "pipeline_id": run.pipeline_id,
        "pipeline_fingerprint": run.pipeline_fingerprint,
        "run_label": run.run_label,
        "comparison_role": run.comparison_role,
        "release_status": release_status,
        "metrics": metrics,
        "metric_versions": metric_versions,
        "latency_ms": run.latency_ms,
        "regression_gates": gates,
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    latency = candidate.get("latency_ms")
    return (
        candidate["release_status"] == "approved",
        metrics["ndcg_at_k"],
        metrics["map_at_k"],
        metrics["recall_at_k"],
        metrics["mrr"],
        -(float(latency) if latency is not None else float("inf")),
        candidate["run_id"],
    )


def recommend_evaluated_pipeline(
    db: Session,
    *,
    user_id: int,
    target: SelectionTarget = "retrieval",
    dataset_id: int | None = None,
    language: str | None = None,
    catalog: Mapping[str, PipelineSpec] | None = None,
) -> dict[str, Any]:
    """Recommend one current pipeline without crossing dataset boundaries."""

    if target not in {"retrieval", "grounded_generation"}:
        raise ValueError(f"unsupported selection target: {target}")
    if target == "grounded_generation":
        # The repository now provides RAGTruth calibration evidence for the
        # generation-level faithfulness judge, but claim/citation and answer
        # relevance judges do not yet share a complete persisted calibration
        # attestation.  Keep these metrics visible for diagnosis without
        # allowing them to promote a pipeline as release-approved.
        return {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "judge_calibration_required",
            "target": target,
            "selection_policy": (
                "public_calibrated_generation_judges_required_v1"
            ),
            "reason": (
                "Grounded-generation 自动选择需要先完成并持久化 "
                "faithfulness、claim/citation 与 answer-relevance judge 校准"
            ),
            "dataset": None,
            "selected": None,
            "candidates": [],
        }
    active_catalog = dict(catalog or pipeline_presets())
    query = select(EvalDataset).where(
        EvalDataset.user_id == user_id,
        EvalDataset.release_eligible.is_(True),
    )
    if dataset_id is not None:
        query = query.where(EvalDataset.id == dataset_id)
    datasets = [
        dataset
        for dataset in db.execute(query.order_by(EvalDataset.id.desc())).scalars()
        if _dataset_provenance_complete(dataset)
        and _matches_target(dataset, target=target, language=language)
    ]

    scopes: list[tuple[EvalDataset, list[dict[str, Any]]]] = []
    for dataset in datasets:
        runs = list(
            db.execute(
                select(EvalRun)
                .where(
                    EvalRun.dataset_id == dataset.id,
                    EvalRun.status == RunStatus.COMPLETED,
                    EvalRun.evaluation_scope == "full",
                )
                .order_by(EvalRun.id.desc())
            ).scalars()
        )
        latest_by_pipeline: dict[str, EvalRun] = {}
        for run in runs:
            if run.pipeline_id and run.pipeline_id not in latest_by_pipeline:
                latest_by_pipeline[run.pipeline_id] = run
        candidates = [
            row
            for run in latest_by_pipeline.values()
            if (row := _candidate(db, run, dataset, active_catalog, target))
            is not None
            and row["release_status"] != "blocked"
        ]
        if candidates:
            scopes.append((dataset, candidates))

    if not scopes:
        return {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "no_eligible_pipeline",
            "target": target,
            "selection_policy": (
                "latest_public_dataset_then_approved_lexicographic_v1"
            ),
            "reason": (
                "没有同时满足公开来源、完整指标、当前管线指纹兼容和回归门禁的运行"
            ),
            "dataset": None,
            "selected": None,
            "candidates": [],
        }

    # Dataset scores are never compared. Pick the scope with the newest
    # compatible evaluation evidence, then compare pipelines only within it.
    dataset, candidates = max(
        scopes,
        key=lambda item: (
            max(candidate["run_id"] for candidate in item[1]),
            item[0].id,
        ),
    )
    ranked = sorted(candidates, key=_candidate_rank, reverse=True)
    return {
        "contract": "evaluated_pipeline_selection_v1",
        "status": "selected",
        "target": target,
        "selection_policy": (
            "latest_public_dataset_then_approved_lexicographic_v1"
        ),
        "reason": (
            "先锁定同一公开语料快照，再按 release 状态、nDCG、MAP、"
            "Recall、MRR、延迟和运行 ID 依次排序"
        ),
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "source_name": dataset.source_name,
            "source_uri": dataset.source_uri,
            "source_version": dataset.source_version,
            "license_name": dataset.license_name,
            "split": dataset.split,
            "corpus_fingerprint": dataset.corpus_fingerprint,
            "source_snapshot_fingerprint": (
                dataset.source_snapshot_fingerprint
            ),
            "language": dataset.language,
            "task_type": dataset.task_type,
        },
        "selected": ranked[0],
        "candidates": ranked,
    }


__all__ = ["SelectionTarget", "recommend_evaluated_pipeline"]
