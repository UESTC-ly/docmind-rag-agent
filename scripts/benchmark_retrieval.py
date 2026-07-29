"""Run reproducible retrieval-only benchmarks over a stored EvalDataset.

The script deliberately excludes answer generation and LLM judging.  It
measures the retrieval behavior of each immutable PipelineSpec over exactly the
same questions, corpus, embedding index, and top-k contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

# Permit direct ``python scripts/benchmark_retrieval.py`` execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SyncSessionLocal
from app.config import settings
from app.models.evaluation import EvalDataset, EvalSample
from app.services.evaluation.retrieval_metrics import compute_retrieval_metrics
from app.services.rag_pipeline import pipeline_presets
from app.services.skill_retrieval import run_pipeline_for_skill


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return round(ordered[index], 3)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row[field]) for row in rows) / len(rows), 6)


def _sample_relevance(sample: EvalSample) -> set[int] | dict[int, float]:
    if sample.chunk_qrels:
        decoded = json.loads(sample.chunk_qrels)
        if isinstance(decoded, dict):
            return {
                int(chunk_id): float(score)
                for chunk_id, score in decoded.items()
            }
    return set(json.loads(sample.relevant_chunk_ids))


def _badcase_reasons(metrics: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if metrics["hit"] == 0:
        reasons.append("retrieval_miss")
    if metrics["recall_at_k"] < 1:
        reasons.append("incomplete_recall")
    if metrics["precision_at_k"] < 0.2:
        reasons.append("low_precision")
    if metrics["ndcg_at_k"] < 0.5:
        reasons.append("poor_ranking")
    return reasons


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recompute_aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Independently rebuild every published deterministic retrieval score."""

    return {
        "hit_rate": _mean(rows, "hit"),
        "mrr": _mean(rows, "reciprocal_rank"),
        "recall": _mean(rows, "recall_at_k"),
        "precision": _mean(rows, "precision_at_k"),
        "map_at_k": _mean(rows, "average_precision_at_k"),
        "ndcg_at_k": _mean(rows, "ndcg_at_k"),
    }


def run_benchmark(
    dataset_id: int,
    pipeline_ids: list[str],
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate requested RAG pipelines against one immutable stored dataset."""
    if not pipeline_ids:
        raise ValueError("at least one pipeline ID is required")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    presets = pipeline_presets()
    unknown = sorted(set(pipeline_ids) - set(presets))
    if unknown:
        raise ValueError(f"unknown pipeline IDs: {', '.join(unknown)}")

    with SyncSessionLocal() as db:
        dataset = db.get(EvalDataset, dataset_id)
        if dataset is None:
            raise ValueError(f"EvalDataset {dataset_id} does not exist")
        required_provenance = {
            "source_name": dataset.source_name,
            "source_uri": dataset.source_uri,
            "source_version": dataset.source_version,
            "license_name": dataset.license_name,
            "split": dataset.split,
            "corpus_fingerprint": dataset.corpus_fingerprint,
            "source_snapshot_fingerprint": (
                dataset.source_snapshot_fingerprint
            ),
        }
        missing_provenance = [
            name for name, value in required_provenance.items() if not value
        ]
        if not dataset.release_eligible or missing_provenance:
            missing = ", ".join(missing_provenance) or "release_eligible"
            raise ValueError(
                "retrieval benchmark requires an auditable public dataset; "
                f"missing: {missing}"
            )
        samples = list(
            db.execute(
                select(EvalSample)
                .where(EvalSample.dataset_id == dataset.id)
                .order_by(EvalSample.id)
            )
            .scalars()
            .all()
        )
        if max_samples is not None:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError(f"EvalDataset {dataset_id} has no samples")

        manifest = {
            "contract": "docmind_retrieval_benchmark_manifest_v2",
            "dataset_id": dataset.id,
            "corpus_fingerprint": dataset.corpus_fingerprint,
            "source_snapshot_fingerprint": (
                dataset.source_snapshot_fingerprint
            ),
            "source_version": dataset.source_version,
            "split": dataset.split,
            "sample_ids": [
                sample.external_id or f"local:{sample.id}" for sample in samples
            ],
            "pipelines": {
                pipeline_id: presets[pipeline_id].fingerprint
                for pipeline_id in pipeline_ids
            },
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "code_revision": os.getenv("DOCMIND_CODE_REVISION"),
        }
        result: dict[str, Any] = {
            "contract": "docmind_retrieval_benchmark_v2",
            "generated_at": datetime.now(UTC).isoformat(),
            "benchmark_fingerprint": _fingerprint(manifest),
            "manifest": manifest,
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "document_id": dataset.document_id,
                "sample_count": len(samples),
                **required_provenance,
                "language": dataset.language,
                "domain": dataset.domain,
                "task_type": dataset.task_type,
                "label_source": dataset.label_source,
                "transform_spec": dataset.transform_spec,
            },
            "metric_scope": "chunk_retrieval",
            "metric_scope_note": (
                "该报告只发布 passage/chunk 检索指标；单一虚拟文档语料"
                "不会被伪装成 document retrieval 指标。"
            ),
            "pipelines": [],
            "sample_results": {},
            "badcases": {},
        }

        for pipeline_id in pipeline_ids:
            rows: list[dict[str, Any]] = []
            for sample in samples:
                start = time.perf_counter()
                execution = run_pipeline_for_skill(
                    user_id=dataset.user_id,
                    query=sample.question,
                    document_id=dataset.document_id,
                    db=db,
                    pipeline=pipeline_id,
                )
                latency_ms = round((time.perf_counter() - start) * 1000, 3)
                retrieved_ids = [
                    int(hit["chunk_index"]) for hit in execution.hits
                ]
                relevance = _sample_relevance(sample)
                metrics = compute_retrieval_metrics(relevance, retrieved_ids)
                reasons = _badcase_reasons(metrics)
                rows.append(
                    {
                        "sample_id": sample.id,
                        "external_id": sample.external_id,
                        "question": sample.question,
                        "retrieved_chunk_ids": retrieved_ids,
                        "relevant_chunk_ids": json.loads(
                            sample.relevant_chunk_ids
                        ),
                        "chunk_qrels": (
                            json.loads(sample.chunk_qrels)
                            if sample.chunk_qrels
                            else None
                        ),
                        "hit": int(metrics["hit"]),
                        "reciprocal_rank": metrics["reciprocal_rank"],
                        "recall_at_k": metrics["recall_at_k"],
                        "precision_at_k": metrics["precision_at_k"],
                        "average_precision_at_k": metrics[
                            "average_precision_at_k"
                        ],
                        "ndcg_at_k": metrics["ndcg_at_k"],
                        "latency_ms": latency_ms,
                        "badcase_reasons": reasons,
                    }
                )

            latencies = [float(row["latency_ms"]) for row in rows]
            published = {
                "hit_rate": _mean(rows, "hit"),
                "mrr": _mean(rows, "reciprocal_rank"),
                "recall": _mean(rows, "recall_at_k"),
                "precision": _mean(rows, "precision_at_k"),
                "map_at_k": _mean(rows, "average_precision_at_k"),
                "ndcg_at_k": _mean(rows, "ndcg_at_k"),
            }
            recomputed = _recompute_aggregate(rows)
            verification_passed = all(
                published[name] == recomputed[name] for name in recomputed
            )
            if not verification_passed:
                raise RuntimeError(
                    f"aggregate verification failed for pipeline {pipeline_id}"
                )
            result["pipelines"].append(
                {
                    "id": pipeline_id,
                    "fingerprint": presets[pipeline_id].fingerprint,
                    "sample_count": len(rows),
                    "top_k": presets[pipeline_id].top_k,
                    **published,
                    "badcase_count": sum(
                        bool(row["badcase_reasons"]) for row in rows
                    ),
                    "latency_ms": {
                        "mean": round(sum(latencies) / len(latencies), 3),
                        "p50": _percentile(latencies, 0.5),
                        "p95": _percentile(latencies, 0.95),
                    },
                    "aggregate_verification": {
                        "contract": "retrieval_aggregate_recompute_v1",
                        "passed": verification_passed,
                        "recomputed": recomputed,
                    },
                }
            )
            result["sample_results"][pipeline_id] = rows
            result["badcases"][pipeline_id] = [
                row for row in rows if row["badcase_reasons"]
            ]
    return result


def render_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        "# DocMind 检索回归评测报告",
        "",
        f"- 数据集：{dataset['source_name']} {dataset['source_version']} "
        f"({dataset['split']})",
        f"- 语料指纹：`{dataset['corpus_fingerprint']}`",
        f"- 实验指纹：`{report['benchmark_fingerprint']}`",
        f"- 样本数：{dataset['sample_count']}",
        "",
        "## 汇总",
        "",
        "| Pipeline | Hit@K | MRR | Recall@K | Precision@K | MAP@K | nDCG@K | Badcase | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["pipelines"]:
        lines.append(
            f"| {row['id']} | {row['hit_rate']:.4f} | {row['mrr']:.4f} "
            f"| {row['recall']:.4f} | {row['precision']:.4f} "
            f"| {row['map_at_k']:.4f} | {row['ndcg_at_k']:.4f} "
            f"| {row['badcase_count']} | {row['latency_ms']['p95']:.1f} |"
        )
    lines.extend(["", "## Badcase 分类", ""])
    for pipeline_id, rows in report["badcases"].items():
        counts: dict[str, int] = {}
        for row in rows:
            for reason in row["badcase_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        summary = ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        ) or "无"
        lines.append(f"- **{pipeline_id}**：{summary}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only PipelineSpec comparisons."
    )
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument(
        "--pipelines",
        default="dense,hybrid,hybrid-rerank",
        help="Comma-separated PipelineSpec preset IDs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic prefix of dataset samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the full JSON evidence artifact.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional human-readable Markdown report.",
    )
    args = parser.parse_args()
    pipeline_ids = [
        value.strip() for value in args.pipelines.split(",") if value.strip()
    ]
    if not pipeline_ids:
        parser.error("--pipelines must include at least one pipeline ID")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    result = run_benchmark(
        args.dataset_id,
        pipeline_ids,
        max_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    for row in result["pipelines"]:
        print(
            f"{row['id']}: hit_rate={row['hit_rate']:.4f} "
            f"mrr={row['mrr']:.4f} recall={row['recall']:.4f} "
            f"precision={row['precision']:.4f} "
            f"map={row['map_at_k']:.4f} ndcg={row['ndcg_at_k']:.4f} "
            f"p95_ms={row['latency_ms']['p95']:.1f}"
        )
    print(f"evidence={args.output}")


if __name__ == "__main__":
    main()
