#!/usr/bin/env python3
"""Persist verified retrieval benchmark receipts for Agent pipeline selection.

``benchmark_retrieval.py`` intentionally runs without answer generation or an
LLM judge. This companion command imports only its deterministic, provenance
checked retrieval evidence after an independent release-gate artifact has
passed. It does not turn retrieval evidence into a grounded-generation claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SyncSessionLocal
from app.models.evaluation import (
    EvalDataset,
    EvalMetricResult,
    EvalRegressionGate,
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.services.evaluation.regression import evaluate_related_regressions
from app.services.rag_pipeline import pipeline_presets


_SAMPLE_METRICS = (
    ("hit", "hit_at_k"),
    ("reciprocal_rank", "reciprocal_rank"),
    ("recall_at_k", "recall_at_k"),
    ("precision_at_k", "precision_at_k"),
    ("average_precision_at_k", "average_precision_at_k"),
    ("ndcg_at_k", "ndcg_at_k"),
)
_SUMMARY_METRICS = (
    ("hit_rate", "hit_rate"),
    ("mrr", "mrr"),
    ("recall", "recall_at_k"),
    ("precision", "precision_at_k"),
    ("map_at_k", "map_at_k"),
    ("ndcg_at_k", "ndcg_at_k"),
)
_GATE_VERSIONS = {
    "hit_rate": "retrieval_v1",
    "recall": "retrieval_v1",
    "map_at_k": "retrieval_v1",
    "ndcg_at_k": "retrieval_v1",
    "badcase_count": "retrieval_badcase_v1",
}
_GATE_METRIC_NAMES = {
    "hit_rate": "hit_rate",
    "recall": "recall_at_k",
    "map_at_k": "map_at_k",
    "ndcg_at_k": "ndcg_at_k",
    "badcase_count": "badcase_count",
}
_PROVENANCE_FIELDS = (
    "source_name",
    "source_uri",
    "source_version",
    "license_name",
    "split",
    "language",
    "domain",
    "task_type",
    "label_source",
    "corpus_fingerprint",
    "source_snapshot_fingerprint",
    "transform_spec",
)


def _require_mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def _require_rows(value: Any, message: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(message)
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(message)
    return [dict(item) for item in value]


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative number")
    if value < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return float(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _expected_sample_ids(db, dataset_id: int) -> list[int]:
    return list(
        db.execute(
            select(EvalSample.id)
            .where(EvalSample.dataset_id == dataset_id)
            .order_by(EvalSample.id)
        ).scalars()
    )


def _validate_evidence(
    db,
    report: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    *,
    baseline_pipeline: str,
) -> tuple[EvalDataset, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if report.get("contract") != "docmind_retrieval_benchmark_v2":
        raise ValueError("unsupported retrieval benchmark contract")
    if release_gate.get("contract") != "docmind_retrieval_release_gate_v1":
        raise ValueError("unsupported retrieval release-gate contract")
    if release_gate.get("benchmark_fingerprint") != report.get(
        "benchmark_fingerprint"
    ):
        raise ValueError("release gate does not belong to this benchmark report")

    dataset_info = _require_mapping(report.get("dataset"), "report.dataset is required")
    dataset_id = dataset_info.get("id")
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int):
        raise ValueError("report.dataset.id must be an integer")
    dataset = db.get(EvalDataset, dataset_id)
    if dataset is None:
        raise ValueError(f"EvalDataset {dataset_id} does not exist")
    if not dataset.release_eligible or dataset.label_source != "public_ground_truth":
        raise ValueError("retrieval evidence requires a release-eligible public dataset")
    for field in _PROVENANCE_FIELDS:
        if dataset_info.get(field) != getattr(dataset, field):
            raise ValueError(f"report dataset {field} does not match stored provenance")

    samples = _expected_sample_ids(db, dataset.id)
    if int(dataset_info.get("sample_count") or 0) != len(samples):
        raise ValueError("report sample_count does not match the complete stored dataset")
    manifest = _require_mapping(report.get("manifest"), "report.manifest is required")
    expected_external_ids = list(
        db.execute(
            select(EvalSample.external_id)
            .where(EvalSample.dataset_id == dataset.id)
            .order_by(EvalSample.id)
        ).scalars()
    )
    if manifest.get("sample_ids") != expected_external_ids:
        raise ValueError("report manifest sample IDs do not match the stored dataset")

    pipelines = _require_rows(report.get("pipelines"), "report.pipelines is required")
    pipeline_ids = [str(row.get("id") or "") for row in pipelines]
    if len(set(pipeline_ids)) != len(pipeline_ids) or baseline_pipeline not in pipeline_ids:
        raise ValueError("report must contain distinct pipelines including the baseline")
    catalog = pipeline_presets()
    for row in pipelines:
        pipeline_id = str(row["id"])
        spec = catalog.get(pipeline_id)
        if spec is None or row.get("fingerprint") != spec.fingerprint:
            raise ValueError(
                f"report pipeline {pipeline_id} is not current-compatible"
            )
        verified = _require_mapping(
            row.get("aggregate_verification"),
            f"report pipeline {pipeline_id} lacks aggregate verification",
        )
        if verified.get("passed") is not True:
            raise ValueError(
                f"report pipeline {pipeline_id} aggregate verification failed"
            )
        for summary_field, _ in _SUMMARY_METRICS:
            _as_number(row.get(summary_field), f"{pipeline_id}.{summary_field}")
        _as_number(row.get("badcase_count"), f"{pipeline_id}.badcase_count")

    raw_results = _require_mapping(
        report.get("sample_results"),
        "report.sample_results is required",
    )
    sample_results: dict[str, list[dict[str, Any]]] = {}
    for pipeline_id in pipeline_ids:
        rows = _require_rows(
            raw_results.get(pipeline_id),
            f"report lacks sample results for {pipeline_id}",
        )
        ids = [row.get("sample_id") for row in rows]
        if ids != samples:
            raise ValueError(
                f"report sample results for {pipeline_id} do not match the "
                "complete stored dataset"
            )
        for row in rows:
            for sample_field, _ in _SAMPLE_METRICS:
                _as_number(row.get(sample_field), f"{pipeline_id}.{sample_field}")
            _as_number(row.get("latency_ms"), f"{pipeline_id}.latency_ms")
        sample_results[pipeline_id] = rows

    gate_candidates = _require_rows(
        release_gate.get("candidates"),
        "release gate must contain candidate outcomes",
    )
    expected_candidates = set(pipeline_ids) - {baseline_pipeline}
    actual_candidates = {str(row.get("pipeline_id") or "") for row in gate_candidates}
    if actual_candidates != expected_candidates:
        raise ValueError("release gate candidates do not match benchmark pipelines")
    if not all(row.get("passed") is True for row in gate_candidates):
        raise ValueError("release gate contains a blocked candidate")
    policy = _require_rows(
        release_gate.get("gate_policy"),
        "release gate must retain its policy",
    )
    policy_metrics = {str(row.get("metric") or "") for row in policy}
    if not policy_metrics <= set(_GATE_VERSIONS):
        raise ValueError("release gate contains unsupported persisted metrics")
    return dataset, pipelines, sample_results


def _ensure_gates(
    db,
    dataset_id: int,
    release_gate: Mapping[str, Any],
) -> list[EvalRegressionGate]:
    gates: list[EvalRegressionGate] = []
    for policy in _require_rows(
        release_gate.get("gate_policy"),
        "release gate must retain its policy",
    ):
        metric_name = str(policy["metric"])
        metric_version = _GATE_VERSIONS[metric_name]
        stored_metric_name = _GATE_METRIC_NAMES[metric_name]
        name = f"public-retrieval-v1:{metric_name}:{policy['comparison']}"
        existing = db.scalar(
            select(EvalRegressionGate).where(
                EvalRegressionGate.dataset_id == dataset_id,
                EvalRegressionGate.name == name,
            )
        )
        threshold = _as_number(policy.get("threshold"), f"gate {metric_name}")
        if existing is not None:
            if (
                existing.metric_name != stored_metric_name
                or existing.metric_version != metric_version
                or existing.comparison != policy["comparison"]
                or float(existing.threshold) != threshold
                or existing.severity != "error"
            ):
                raise ValueError(f"existing regression gate {name} conflicts")
            gates.append(existing)
            continue
        gate = EvalRegressionGate(
            dataset_id=dataset_id,
            name=name,
            metric_name=stored_metric_name,
            metric_version=metric_version,
            comparison=str(policy["comparison"]),
            threshold=threshold,
            severity="error",
            enabled=True,
        )
        db.add(gate)
        db.flush()
        gates.append(gate)
    return gates


def _metric_rows(
    run_id: int,
    sample_rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> list[EvalMetricResult]:
    rows: list[EvalMetricResult] = []
    for sample in sample_rows:
        for source_field, metric_name in _SAMPLE_METRICS:
            rows.append(
                EvalMetricResult(
                    run_id=run_id,
                    subject_type="sample",
                    subject_id=int(sample["sample_id"]),
                    metric_name=metric_name,
                    metric_version="retrieval_v1",
                    evaluator_kind="deterministic",
                    score=_as_number(
                        sample[source_field],
                        f"{metric_name} sample score",
                    ),
                )
            )
    for summary_field, metric_name in _SUMMARY_METRICS:
        rows.append(
            EvalMetricResult(
                run_id=run_id,
                subject_type="run",
                subject_id=0,
                metric_name=metric_name,
                metric_version="retrieval_v1",
                evaluator_kind="deterministic",
                score=_as_number(summary[summary_field], summary_field),
            )
        )
    rows.append(
        EvalMetricResult(
            run_id=run_id,
            subject_type="run",
            subject_id=0,
            metric_name="badcase_count",
            metric_version="retrieval_badcase_v1",
            evaluator_kind="deterministic",
            score=_as_number(summary["badcase_count"], "badcase_count"),
        )
    )
    return rows


def persist_evidence(
    db,
    report: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    *,
    baseline_pipeline: str = "dense",
    run_label: str | None = None,
) -> dict[str, Any]:
    """Persist one complete, independently gated benchmark bundle."""

    dataset, pipelines, sample_results = _validate_evidence(
        db,
        report,
        release_gate,
        baseline_pipeline=baseline_pipeline,
    )
    experiment_key = str(report["benchmark_fingerprint"])
    existing = list(
        db.execute(
            select(EvalRun).where(
                EvalRun.dataset_id == dataset.id,
                EvalRun.experiment_key == experiment_key,
            )
        ).scalars()
    )
    if existing:
        raise ValueError(
            "benchmark fingerprint is already persisted for this dataset"
        )

    gates = _ensure_gates(db, dataset.id, release_gate)
    catalog = pipeline_presets()
    ordered = sorted(
        pipelines,
        key=lambda item: (str(item["id"]) != baseline_pipeline, str(item["id"])),
    )
    baseline_run_id: int | None = None
    created: dict[str, EvalRun] = {}
    for summary in ordered:
        pipeline_id = str(summary["id"])
        spec = catalog[pipeline_id]
        rows = sample_results[pipeline_id]
        if pipeline_id == baseline_pipeline:
            comparison_role = "baseline"
            related_baseline_id = None
        else:
            comparison_role = "candidate"
            related_baseline_id = baseline_run_id
        if pipeline_id != baseline_pipeline and related_baseline_id is None:
            raise ValueError("baseline run must be persisted before candidates")
        total_latency_ms = sum(
            _as_number(row["latency_ms"], "sample latency") for row in rows
        )
        label_prefix = run_label or "public-retrieval"
        run = EvalRun(
            dataset_id=dataset.id,
            status=RunStatus.COMPLETED,
            hit_rate=_as_number(summary["hit_rate"], "hit_rate"),
            mrr=_as_number(summary["mrr"], "mrr"),
            recall=_as_number(summary["recall"], "recall"),
            precision=_as_number(summary["precision"], "precision"),
            map_score=_as_number(summary["map_at_k"], "map_at_k"),
            ndcg=_as_number(summary["ndcg_at_k"], "ndcg_at_k"),
            pipeline_id=pipeline_id,
            pipeline_spec=_canonical_json(spec.to_dict()),
            pipeline_fingerprint=spec.fingerprint,
            experiment_key=experiment_key,
            run_label=f"{label_prefix}:{pipeline_id}",
            comparison_role=comparison_role,
            evaluation_scope="full",
            baseline_run_id=related_baseline_id,
            code_revision=(
                _require_mapping(report["manifest"], "report.manifest is required").get(
                    "code_revision"
                )
            ),
            embedding_model=_require_mapping(
                report["manifest"], "report.manifest is required"
            ).get("embedding_model"),
            environment_fingerprint=experiment_key,
            latency_ms=total_latency_ms,
            completed_at=datetime.now(UTC),
            error_message=None,
        )
        db.add(run)
        db.flush()
        db.add_all(
            EvalResult(
                run_id=run.id,
                sample_id=int(row["sample_id"]),
                hit=int(_as_number(row["hit"], "hit")),
                reciprocal_rank=_as_number(
                    row["reciprocal_rank"], "reciprocal_rank"
                ),
                recall_at_k=_as_number(row["recall_at_k"], "recall_at_k"),
                precision_at_k=_as_number(
                    row["precision_at_k"], "precision_at_k"
                ),
                average_precision_at_k=_as_number(
                    row["average_precision_at_k"],
                    "average_precision_at_k",
                ),
                ndcg_at_k=_as_number(row["ndcg_at_k"], "ndcg_at_k"),
                retrieved_chunk_ids=_canonical_json(
                    row.get("retrieved_chunk_ids") or []
                ),
                retrieval_mode="public_retrieval_benchmark_v2",
                reranker_mode=spec.reranker,
                retrieval_trace=_canonical_json(
                    [
                        {
                            "benchmark_fingerprint": experiment_key,
                            "external_id": row.get("external_id"),
                            "badcase_reasons": row.get("badcase_reasons") or [],
                        }
                    ]
                ),
            )
            for row in rows
        )
        db.add_all(_metric_rows(run.id, rows, summary))
        db.flush()
        created[pipeline_id] = run
        if pipeline_id == baseline_pipeline:
            baseline_run_id = run.id

    for run in created.values():
        evaluate_related_regressions(db, run.id)
    db.flush()
    return {
        "contract": "persisted_retrieval_evidence_v1",
        "dataset_id": dataset.id,
        "benchmark_fingerprint": experiment_key,
        "release_gate_fingerprint": release_gate.get("gate_fingerprint"),
        "run_ids": {pipeline_id: run.id for pipeline_id, run in created.items()},
        "regression_gate_ids": [gate.id for gate in gates],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Persist an independently gated public retrieval report for "
            "Agent pipeline selection."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--run-label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    release_gate = json.loads(args.release_gate.read_text(encoding="utf-8"))
    with SyncSessionLocal() as db:
        receipt = persist_evidence(
            db,
            report,
            release_gate,
            baseline_pipeline=args.baseline,
            run_label=args.run_label,
        )
        db.commit()
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
