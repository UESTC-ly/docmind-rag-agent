import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "gate_retrieval_benchmark.py"
    spec = importlib.util.spec_from_file_location("retrieval_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report():
    rows = [
        {
            "sample_id": 1,
            "external_id": "q1",
            "badcase_reasons": [],
        },
        {
            "sample_id": 2,
            "external_id": "q2",
            "badcase_reasons": ["retrieval_miss"],
        },
    ]
    return {
        "contract": "docmind_retrieval_benchmark_v2",
        "benchmark_fingerprint": "a" * 64,
        "dataset": {
            "source_name": "Public",
            "source_version": "v1",
            "split": "test",
            "corpus_fingerprint": "b" * 64,
            "source_snapshot_fingerprint": "c" * 64,
            "sample_count": 100,
        },
        "pipelines": [
            {
                "id": "dense",
                "fingerprint": "d" * 64,
                "hit_rate": 0.9,
                "recall": 0.9,
                "map_at_k": 0.8,
                "ndcg_at_k": 0.8,
                "badcase_count": 1,
                "aggregate_verification": {"passed": True},
            },
            {
                "id": "candidate",
                "fingerprint": "e" * 64,
                "hit_rate": 0.9,
                "recall": 0.85,
                "map_at_k": 0.8,
                "ndcg_at_k": 0.8,
                "badcase_count": 2,
                "aggregate_verification": {"passed": True},
            },
        ],
        "sample_results": {
            "dense": rows,
            "candidate": [
                {**rows[0], "badcase_reasons": ["poor_ranking"]},
                rows[1],
            ],
        },
    }


def test_release_gate_blocks_metric_drop_and_retains_badcase_diff():
    report = _module().evaluate_report(_report())
    candidate = report["candidates"][0]

    assert candidate["passed"] is False
    assert next(
        row for row in candidate["gates"] if row["metric"] == "recall"
    )["passed"] is False
    assert candidate["badcase_diff"]["newly_introduced"][0]["sample_id"] == "q1"
    assert len(report["gate_fingerprint"]) == 64


def test_release_gate_rejects_unverified_or_too_small_report():
    module = _module()
    report = _report()
    report["dataset"]["sample_count"] = 10
    with pytest.raises(ValueError, match="at least"):
        module.evaluate_report(report)

    report = _report()
    report["pipelines"][0]["aggregate_verification"]["passed"] = False
    with pytest.raises(ValueError, match="aggregate verification"):
        module.evaluate_report(report)
