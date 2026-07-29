"""Regression-gate evaluation over persisted, versioned metrics."""

import json

import pytest

from app.models.document import Document, DocumentStatus
from app.models.evaluation import (
    EvalDataset,
    EvalMetricResult,
    EvalRegressionGate,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.models.user import User
from app.services.evaluation.regression import (
    evaluate_candidate,
    evaluate_related_regressions,
    metric_score,
)


def _seed(sync_db):
    user = User(email="regression@test.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    document = Document(
        user_id=user.id,
        filename="public-corpus",
        file_path="dataset://public",
        status=DocumentStatus.COMPLETED,
        chunk_count=1,
    )
    sync_db.add(document)
    sync_db.flush()
    dataset = EvalDataset(
        user_id=user.id,
        document_id=document.id,
        name="public regression",
        label_source="public_ground_truth",
        release_eligible=True,
    )
    sync_db.add(dataset)
    sync_db.flush()
    easy = EvalSample(
        dataset_id=dataset.id,
        question="q1",
        ground_truth_answer="a1",
        relevant_chunk_ids="[0]",
        difficulty="easy",
        slice_tags=json.dumps(["zh"]),
    )
    hard = EvalSample(
        dataset_id=dataset.id,
        question="q2",
        ground_truth_answer="a2",
        relevant_chunk_ids="[0]",
        difficulty="hard",
        slice_tags=json.dumps(["zh", "conflict"]),
    )
    sync_db.add_all([easy, hard])
    sync_db.flush()
    baseline = EvalRun(
        dataset_id=dataset.id,
        status=RunStatus.COMPLETED,
        mrr=0.8,
        comparison_role="baseline",
    )
    candidate = EvalRun(
        dataset_id=dataset.id,
        status=RunStatus.COMPLETED,
        mrr=0.72,
        comparison_role="candidate",
    )
    sync_db.add_all([baseline, candidate])
    sync_db.flush()
    candidate.baseline_run_id = baseline.id
    return dataset, baseline, candidate, easy, hard


def _metric(run_id, subject_type, subject_id, name, score):
    return EvalMetricResult(
        run_id=run_id,
        subject_type=subject_type,
        subject_id=subject_id,
        metric_name=name,
        metric_version="v1",
        evaluator_kind="deterministic",
        score=score,
    )


def test_metric_score_prefers_versioned_aggregate_and_supports_slices(sync_db):
    _dataset, _baseline, candidate, easy, hard = _seed(sync_db)
    sync_db.add_all(
        [
            _metric(candidate.id, "run", 0, "mrr", 0.75),
            _metric(candidate.id, "sample", easy.id, "mrr", 1.0),
            _metric(candidate.id, "sample", hard.id, "mrr", 0.5),
        ]
    )
    sync_db.flush()

    assert metric_score(sync_db, candidate.id, "mrr").score == 0.75
    sliced = metric_score(
        sync_db,
        candidate.id,
        "mrr",
        slice_filter=json.dumps({"difficulty": "hard", "tags": ["conflict"]}),
    )
    assert sliced.score == 0.5
    assert sliced.sample_count == 1


def test_absolute_and_relative_gates_persist_explainable_verdicts(sync_db):
    dataset, baseline, candidate, _easy, _hard = _seed(sync_db)
    sync_db.add_all(
        [
            EvalRegressionGate(
                dataset_id=dataset.id,
                name="MRR floor",
                metric_name="mrr",
                comparison="absolute_min",
                threshold=0.7,
            ),
            EvalRegressionGate(
                dataset_id=dataset.id,
                name="MRR max drop",
                metric_name="mrr",
                comparison="max_drop",
                threshold=0.05,
            ),
        ]
    )
    sync_db.flush()

    results = evaluate_candidate(sync_db, candidate.id)
    assert len(results) == 2
    absolute = next(result for result in results if result.baseline_run_id is None)
    relative = next(result for result in results if result.baseline_run_id is not None)
    assert absolute.passed is True
    assert absolute.candidate_score == 0.72
    assert relative.baseline_run_id == baseline.id
    assert relative.delta == pytest.approx(-0.08)
    assert relative.passed is False
    assert "delta" in relative.reason


def test_relative_gate_waits_until_baseline_finishes(sync_db):
    dataset, baseline, candidate, _easy, _hard = _seed(sync_db)
    baseline.status = RunStatus.RUNNING
    sync_db.add(
        EvalRegressionGate(
            dataset_id=dataset.id,
            name="MRR max drop",
            metric_name="mrr",
            comparison="max_drop",
            threshold=0.1,
        )
    )
    sync_db.flush()

    assert evaluate_candidate(sync_db, candidate.id) == []

    baseline.status = RunStatus.COMPLETED
    results = evaluate_related_regressions(sync_db, baseline.id)
    assert len(results) == 1
    assert results[0].candidate_run_id == candidate.id
    assert results[0].passed is True


def test_missing_candidate_metric_fails_absolute_gate_closed(sync_db):
    dataset, _baseline, candidate, _easy, _hard = _seed(sync_db)
    candidate.mrr = None
    sync_db.add(
        EvalRegressionGate(
            dataset_id=dataset.id,
            name="Groundedness floor",
            metric_name="groundedness",
            comparison="absolute_min",
            threshold=0.9,
        )
    )
    sync_db.flush()

    [result] = evaluate_candidate(sync_db, candidate.id)
    assert result.passed is False
    assert result.candidate_score is None
    assert result.reason == "candidate metric is unavailable"


def test_invalid_answerable_slice_fails_closed_instead_of_matching_truthy_value(
    sync_db,
):
    _dataset, _baseline, candidate, easy, _hard = _seed(sync_db)
    sync_db.add(_metric(candidate.id, "sample", easy.id, "mrr", 1.0))
    sync_db.flush()

    score = metric_score(
        sync_db,
        candidate.id,
        "mrr",
        slice_filter=json.dumps({"answerable": 1}),
    )

    assert score.score is None
    assert score.sample_count == 0
