#!/usr/bin/env python3
"""Apply deterministic baseline gates to a retrieval benchmark artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_GATES = (
    ("hit_rate", "max_drop", 0.02),
    ("recall", "max_drop", 0.02),
    ("map_at_k", "max_drop", 0.02),
    ("ndcg_at_k", "max_drop", 0.02),
    ("badcase_count", "max_increase", 5.0),
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _badcase_map(report: dict, pipeline_id: str) -> dict[str, set[str]]:
    rows = report["sample_results"][pipeline_id]
    return {
        str(row.get("external_id") or row["sample_id"]): set(
            row.get("badcase_reasons") or []
        )
        for row in rows
    }


def _compare_badcases(
    baseline: dict[str, set[str]],
    candidate: dict[str, set[str]],
) -> dict[str, list[dict[str, Any]] | int]:
    introduced = []
    fixed = []
    persistent = []
    unchanged_passed = 0
    for sample_id in sorted(set(baseline) | set(candidate)):
        before = baseline.get(sample_id, set())
        after = candidate.get(sample_id, set())
        row = {
            "sample_id": sample_id,
            "baseline_reasons": sorted(before),
            "candidate_reasons": sorted(after),
            "introduced_reasons": sorted(after - before),
            "fixed_reasons": sorted(before - after),
        }
        if not before and after:
            introduced.append(row)
        elif before and not after:
            fixed.append(row)
        elif before and after:
            persistent.append(row)
        else:
            unchanged_passed += 1
    return {
        "newly_introduced": introduced,
        "fixed": fixed,
        "persistent": persistent,
        "unchanged_passed_count": unchanged_passed,
    }


def evaluate_report(
    report: dict[str, Any],
    *,
    baseline_pipeline: str = "dense",
    minimum_samples: int = 100,
) -> dict[str, Any]:
    if report.get("contract") != "docmind_retrieval_benchmark_v2":
        raise ValueError("unsupported retrieval benchmark contract")
    pipelines = {row["id"]: row for row in report.get("pipelines") or []}
    if baseline_pipeline not in pipelines:
        raise ValueError(f"baseline pipeline not found: {baseline_pipeline}")
    if int(report["dataset"]["sample_count"]) < minimum_samples:
        raise ValueError(
            f"benchmark requires at least {minimum_samples} samples"
        )
    if not all(
        row.get("aggregate_verification", {}).get("passed") is True
        for row in pipelines.values()
    ):
        raise ValueError("benchmark aggregate verification is incomplete")

    baseline = pipelines[baseline_pipeline]
    baseline_badcases = _badcase_map(report, baseline_pipeline)
    candidates = []
    for pipeline_id, candidate in pipelines.items():
        if pipeline_id == baseline_pipeline:
            continue
        gates = []
        for metric, comparison, threshold in DEFAULT_GATES:
            baseline_score = float(baseline[metric])
            candidate_score = float(candidate[metric])
            delta = candidate_score - baseline_score
            if comparison == "max_drop":
                passed = delta >= -threshold
            else:
                passed = delta <= threshold
            gates.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "threshold": threshold,
                    "baseline": baseline_score,
                    "candidate": candidate_score,
                    "delta": round(delta, 6),
                    "passed": passed,
                }
            )
        candidates.append(
            {
                "pipeline_id": pipeline_id,
                "pipeline_fingerprint": candidate["fingerprint"],
                "passed": all(row["passed"] for row in gates),
                "gates": gates,
                "badcase_diff": _compare_badcases(
                    baseline_badcases,
                    _badcase_map(report, pipeline_id),
                ),
            }
        )

    payload = {
        "contract": "docmind_retrieval_release_gate_v1",
        "benchmark_fingerprint": report["benchmark_fingerprint"],
        "dataset": {
            key: report["dataset"][key]
            for key in (
                "source_name",
                "source_version",
                "split",
                "corpus_fingerprint",
                "source_snapshot_fingerprint",
                "sample_count",
            )
        },
        "baseline_pipeline": baseline_pipeline,
        "baseline_fingerprint": baseline["fingerprint"],
        "minimum_samples": minimum_samples,
        "gate_policy": [
            {
                "metric": metric,
                "comparison": comparison,
                "threshold": threshold,
            }
            for metric, comparison, threshold in DEFAULT_GATES
        ],
        "candidates": candidates,
    }
    payload["gate_fingerprint"] = _fingerprint(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply deterministic release gates to retrieval evidence."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = evaluate_report(
        report,
        baseline_pipeline=args.baseline,
        minimum_samples=args.minimum_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for candidate in result["candidates"]:
        print(
            f"{candidate['pipeline_id']}: "
            f"{'PASS' if candidate['passed'] else 'BLOCKED'}"
        )
    print(f"evidence={args.output}")
    if args.fail_on_regression and not all(
        candidate["passed"] for candidate in result["candidates"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
