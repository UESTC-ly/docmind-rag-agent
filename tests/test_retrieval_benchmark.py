"""Reproducible public retrieval benchmark evidence contract."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from app.models.document import Document, DocumentStatus
from app.models.evaluation import EvalDataset, EvalSample
from app.models.user import User
from app.services.rag_pipeline import pipeline_presets


def _module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_retrieval.py"
    spec = importlib.util.spec_from_file_location("benchmark_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _seed(sync_db, *, release_eligible=True):
    user = User(email="retrieval-benchmark@test.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    document = Document(
        user_id=user.id,
        filename="public",
        file_path="dataset://public",
        status=DocumentStatus.COMPLETED,
        chunk_count=3,
    )
    sync_db.add(document)
    sync_db.flush()
    dataset = EvalDataset(
        user_id=user.id,
        document_id=document.id,
        name="public benchmark",
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
        release_eligible=release_eligible,
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
                chunk_qrels='{"0":2}',
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
    return dataset


def test_benchmark_retains_raw_cases_and_recomputes_published_aggregates(
    sync_db,
    monkeypatch,
):
    benchmark = _module()
    dataset = _seed(sync_db)
    Session = sessionmaker(
        bind=sync_db.get_bind(),
        expire_on_commit=False,
    )
    monkeypatch.setattr(benchmark, "SyncSessionLocal", Session)
    presets = pipeline_presets()
    monkeypatch.setattr(
        benchmark,
        "run_pipeline_for_skill",
        lambda **kwargs: SimpleNamespace(
            hits=(
                [{"chunk_index": 0}]
                if kwargs["query"] == "q1"
                else [{"chunk_index": 2}, {"chunk_index": 1}]
            ),
        ),
    )

    report = benchmark.run_benchmark(dataset.id, ["dense"])

    assert report["contract"] == "docmind_retrieval_benchmark_v2"
    assert report["metric_scope"] == "chunk_retrieval"
    assert report["dataset"]["source_name"] == "Public fixture"
    assert report["manifest"]["pipelines"]["dense"] == presets["dense"].fingerprint
    assert report["manifest"]["sample_ids"] == ["public-q1", "public-q2"]
    assert report["sample_results"]["dense"][0]["chunk_qrels"] == {"0": 2}
    summary = report["pipelines"][0]
    assert summary["hit_rate"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["aggregate_verification"]["passed"] is True
    assert (
        summary["aggregate_verification"]["recomputed"]["ndcg_at_k"]
        == summary["ndcg_at_k"]
    )
    assert len(report["benchmark_fingerprint"]) == 64
    json.dumps(report)


def test_benchmark_rejects_non_public_or_incomplete_dataset(
    sync_db,
    monkeypatch,
):
    benchmark = _module()
    dataset = _seed(sync_db, release_eligible=False)
    Session = sessionmaker(
        bind=sync_db.get_bind(),
        expire_on_commit=False,
    )
    monkeypatch.setattr(benchmark, "SyncSessionLocal", Session)

    with pytest.raises(ValueError, match="auditable public dataset"):
        benchmark.run_benchmark(dataset.id, ["dense"])


def test_benchmark_rejects_empty_pipeline_list():
    benchmark = _module()
    with pytest.raises(ValueError, match="at least one"):
        benchmark.run_benchmark(1, [])
