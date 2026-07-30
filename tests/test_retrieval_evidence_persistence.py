"""Public retrieval receipts become selectable Agent evidence only after gates."""

import importlib.util
from pathlib import Path

import pytest

from app.models.document import Document, DocumentStatus
from app.models.evaluation import EvalDataset, EvalRegressionResult, EvalSample
from app.models.user import User
from app.services.evaluation.pipeline_selection import recommend_evaluated_pipeline
from app.services.rag_pipeline import pipeline_presets


def _module():
    path = Path(__file__).parents[1] / "scripts" / "persist_retrieval_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "persist_retrieval_evidence",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _seed(sync_db):
    user = User(email="persisted-evidence@test.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    document = Document(
        user_id=user.id,
        filename="public",
        file_path="dataset://public",
        status=DocumentStatus.COMPLETED,
        chunk_count=2,
    )
    sync_db.add(document)
    sync_db.flush()
    dataset = EvalDataset(
        user_id=user.id,
        document_id=document.id,
        name="Public fixture",
        source_name="Public fixture",
        source_uri="https://example.test/dataset",
        source_version="v1",
        license_name="CC BY 4.0",
        split="test",
        corpus_fingerprint="b" * 64,
        source_snapshot_fingerprint="c" * 64,
        transform_spec='{"contract":"fixture_v1"}',
        language="en",
        task_type="retrieval",
        label_source="public_ground_truth",
        release_eligible=True,
    )
    sync_db.add(dataset)
    sync_db.flush()
    sync_db.add_all(
        [
            EvalSample(
                dataset_id=dataset.id,
                question="q1",
                ground_truth_answer="",
                relevant_chunk_ids="[0]",
                chunk_qrels='{"0":1}',
                external_id="public-q1",
            ),
            EvalSample(
                dataset_id=dataset.id,
                question="q2",
                ground_truth_answer="",
                relevant_chunk_ids="[1]",
                chunk_qrels='{"1":1}',
                external_id="public-q2",
            ),
        ]
    )
    sync_db.commit()
    return user, dataset


def _sample(sample_id, external_id, rank, *, latency=1.0):
    hit = 1 if rank else 0
    score = 1.0 if rank == 1 else 0.5 if rank == 2 else 0.0
    return {
        "sample_id": sample_id,
        "external_id": external_id,
        "retrieved_chunk_ids": [rank - 1] if rank else [],
        "hit": hit,
        "reciprocal_rank": score,
        "recall_at_k": float(hit),
        "precision_at_k": score / 5 if rank else 0.0,
        "average_precision_at_k": score,
        "ndcg_at_k": score,
        "latency_ms": latency,
        "badcase_reasons": [] if rank == 1 else ["poor_ranking"],
    }


def _summary(pipeline_id, fingerprint, rows):
    def average(key):
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "id": pipeline_id,
        "fingerprint": fingerprint,
        "hit_rate": average("hit"),
        "mrr": average("reciprocal_rank"),
        "recall": average("recall_at_k"),
        "precision": average("precision_at_k"),
        "map_at_k": average("average_precision_at_k"),
        "ndcg_at_k": average("ndcg_at_k"),
        "badcase_count": sum(bool(row["badcase_reasons"]) for row in rows),
        "aggregate_verification": {"passed": True},
    }


def _build_report(dataset, samples):
    catalog = pipeline_presets()
    dense_rows = [
        _sample(samples[0].id, samples[0].external_id, 1),
        _sample(samples[1].id, samples[1].external_id, 2),
    ]
    hybrid_rows = [
        _sample(samples[0].id, samples[0].external_id, 1),
        _sample(samples[1].id, samples[1].external_id, 1),
    ]
    return {
        "contract": "docmind_retrieval_benchmark_v2",
        "benchmark_fingerprint": "a" * 64,
        "manifest": {
            "sample_ids": [sample.external_id for sample in samples],
            "embedding_model": "test-embedding",
            "code_revision": "test-revision",
        },
        "dataset": {
            "id": dataset.id,
            "source_name": dataset.source_name,
            "source_uri": dataset.source_uri,
            "source_version": dataset.source_version,
            "license_name": dataset.license_name,
            "split": dataset.split,
            "language": dataset.language,
            "domain": dataset.domain,
            "task_type": dataset.task_type,
            "label_source": dataset.label_source,
            "corpus_fingerprint": dataset.corpus_fingerprint,
            "source_snapshot_fingerprint": dataset.source_snapshot_fingerprint,
            "transform_spec": dataset.transform_spec,
            "sample_count": len(samples),
        },
        "pipelines": [
            _summary("dense", catalog["dense"].fingerprint, dense_rows),
            _summary("hybrid", catalog["hybrid"].fingerprint, hybrid_rows),
        ],
        "sample_results": {"dense": dense_rows, "hybrid": hybrid_rows},
    }


def _release_gate(report):
    return {
        "contract": "docmind_retrieval_release_gate_v1",
        "benchmark_fingerprint": report["benchmark_fingerprint"],
        "gate_fingerprint": "d" * 64,
        "gate_policy": [
            {"metric": "hit_rate", "comparison": "max_drop", "threshold": 0.02},
            {"metric": "recall", "comparison": "max_drop", "threshold": 0.02},
            {"metric": "map_at_k", "comparison": "max_drop", "threshold": 0.02},
            {"metric": "ndcg_at_k", "comparison": "max_drop", "threshold": 0.02},
            {
                "metric": "badcase_count",
                "comparison": "max_increase",
                "threshold": 5.0,
            },
        ],
        "candidates": [{"pipeline_id": "hybrid", "passed": True}],
    }


def test_persisted_public_receipt_becomes_selectable_after_gates(sync_db):
    module = _module()
    user, dataset = _seed(sync_db)
    samples = list(
        sync_db.query(EvalSample)
        .filter(EvalSample.dataset_id == dataset.id)
        .order_by(EvalSample.id)
    )
    report = _build_report(dataset, samples)
    receipt = module.persist_evidence(
        sync_db,
        report,
        _release_gate(report),
        baseline_pipeline="dense",
    )
    sync_db.commit()

    selected = recommend_evaluated_pipeline(
        sync_db,
        user_id=user.id,
        dataset_id=dataset.id,
    )

    assert receipt["run_ids"]["dense"] != receipt["run_ids"]["hybrid"]
    assert len(receipt["regression_gate_ids"]) == 5
    assert selected["status"] == "selected"
    assert selected["selected"]["pipeline_id"] == "hybrid"
    gate_results = sync_db.query(EvalRegressionResult).all()
    assert len(gate_results) == 5
    assert all(row.passed for row in gate_results)


def test_persist_rejects_gate_from_a_different_benchmark(sync_db):
    module = _module()
    _, dataset = _seed(sync_db)
    samples = list(
        sync_db.query(EvalSample)
        .filter(EvalSample.dataset_id == dataset.id)
        .order_by(EvalSample.id)
    )
    report = _build_report(dataset, samples)
    release_gate = _release_gate(report)
    release_gate["benchmark_fingerprint"] = "x" * 64

    with pytest.raises(ValueError, match="does not belong"):
        module.persist_evidence(sync_db, report, release_gate)


def test_persist_rejects_changed_public_transform_spec(sync_db):
    module = _module()
    _, dataset = _seed(sync_db)
    samples = list(
        sync_db.query(EvalSample)
        .filter(EvalSample.dataset_id == dataset.id)
        .order_by(EvalSample.id)
    )
    report = _build_report(dataset, samples)
    report["dataset"]["transform_spec"] = '{"contract":"different_v1"}'

    with pytest.raises(ValueError, match="transform_spec"):
        module.persist_evidence(sync_db, report, _release_gate(report))
