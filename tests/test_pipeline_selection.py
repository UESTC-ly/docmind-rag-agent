"""Public evaluation evidence drives automatic RAG pipeline selection."""

from app.models.document import Document, DocumentStatus
from app.models.evaluation import (
    EvalDataset,
    EvalMetricResult,
    EvalRegressionGate,
    EvalRegressionResult,
    EvalRun,
    RunStatus,
)
from app.models.user import User
from app.services.evaluation.pipeline_selection import (
    recommend_evaluated_pipeline,
)
from app.services.rag_pipeline import pipeline_presets
from app.skills.base import SkillContext
from app.skills.evaluation_advisor import _resolve_dataset_scope


def _seed_public_dataset(sync_db):
    user = User(email="selection@test.com", hashed_password="x")
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
        name="public retrieval snapshot",
        source_name="BEIR-fixture",
        source_uri="https://example.test/public-source",
        source_version="v1",
        license_name="CC BY 4.0",
        split="test",
        corpus_fingerprint="a" * 64,
        source_snapshot_fingerprint="b" * 64,
        language="en",
        task_type="retrieval",
        label_source="public_ground_truth",
        release_eligible=True,
    )
    sync_db.add(dataset)
    sync_db.flush()
    return user, dataset


def _add_run(sync_db, dataset, pipeline_id, scores):
    spec = pipeline_presets()[pipeline_id]
    run = EvalRun(
        dataset_id=dataset.id,
        status=RunStatus.COMPLETED,
        pipeline_id=spec.id,
        pipeline_fingerprint=spec.fingerprint,
        comparison_role="standalone",
        latency_ms=scores.pop("latency_ms", 100.0),
    )
    sync_db.add(run)
    sync_db.flush()
    for metric_name, score in scores.items():
        sync_db.add(
            EvalMetricResult(
                run_id=run.id,
                subject_type="run",
                subject_id=0,
                metric_name=metric_name,
                metric_version="retrieval_v1",
                evaluator_kind="deterministic",
                score=score,
            )
        )
    sync_db.flush()
    return run


def _scores(ndcg, map_score, recall, mrr):
    return {
        "ndcg_at_k": ndcg,
        "map_at_k": map_score,
        "recall_at_k": recall,
        "mrr": mrr,
    }


def test_selects_best_current_pipeline_inside_one_public_snapshot(sync_db):
    user, dataset = _seed_public_dataset(sync_db)
    _add_run(sync_db, dataset, "dense", _scores(0.6, 0.5, 0.7, 0.6))
    hybrid = _add_run(
        sync_db,
        dataset,
        "hybrid",
        _scores(0.8, 0.7, 0.9, 0.8),
    )

    decision = recommend_evaluated_pipeline(
        sync_db,
        user_id=user.id,
        target="retrieval",
    )

    assert decision["status"] == "selected"
    assert decision["dataset"]["id"] == dataset.id
    assert decision["selected"]["run_id"] == hybrid.id
    assert decision["selected"]["pipeline_id"] == "hybrid"
    assert decision["selected"]["release_status"] == "observational"
    assert [row["pipeline_id"] for row in decision["candidates"]] == [
        "hybrid",
        "dense",
    ]


def test_failed_error_gate_blocks_a_higher_scoring_pipeline(sync_db):
    user, dataset = _seed_public_dataset(sync_db)
    dense = _add_run(
        sync_db,
        dataset,
        "dense",
        _scores(0.7, 0.6, 0.8, 0.7),
    )
    hybrid = _add_run(
        sync_db,
        dataset,
        "hybrid",
        _scores(0.95, 0.95, 1.0, 0.95),
    )
    gate = EvalRegressionGate(
        dataset_id=dataset.id,
        name="nDCG release floor",
        metric_name="ndcg_at_k",
        metric_version="retrieval_v1",
        comparison="absolute_min",
        threshold=0.65,
        severity="error",
    )
    sync_db.add(gate)
    sync_db.flush()
    sync_db.add_all(
        [
            EvalRegressionResult(
                candidate_run_id=dense.id,
                gate_id=gate.id,
                metric_name=gate.metric_name,
                candidate_score=0.7,
                passed=True,
                reason="0.7 >= 0.65",
            ),
            EvalRegressionResult(
                candidate_run_id=hybrid.id,
                gate_id=gate.id,
                metric_name=gate.metric_name,
                candidate_score=0.95,
                passed=False,
                reason="new high-severity badcase",
            ),
        ]
    )
    sync_db.flush()

    decision = recommend_evaluated_pipeline(sync_db, user_id=user.id)

    assert decision["selected"]["pipeline_id"] == "dense"
    assert decision["selected"]["release_status"] == "approved"
    assert "hybrid" not in {
        candidate["pipeline_id"] for candidate in decision["candidates"]
    }


def test_stale_pipeline_fingerprint_is_not_automatically_selected(sync_db):
    user, dataset = _seed_public_dataset(sync_db)
    run = _add_run(
        sync_db,
        dataset,
        "dense",
        _scores(0.8, 0.8, 0.8, 0.8),
    )
    run.pipeline_fingerprint = "0" * 64
    sync_db.flush()

    decision = recommend_evaluated_pipeline(sync_db, user_id=user.id)

    assert decision["status"] == "no_eligible_pipeline"
    assert decision["selected"] is None


def test_incomplete_public_provenance_cannot_back_agent_selection(sync_db):
    user, dataset = _seed_public_dataset(sync_db)
    dataset.license_name = None
    _add_run(
        sync_db,
        dataset,
        "dense",
        _scores(0.8, 0.8, 0.8, 0.8),
    )
    sync_db.flush()

    decision = recommend_evaluated_pipeline(sync_db, user_id=user.id)

    assert decision["status"] == "no_eligible_pipeline"


def test_current_document_scope_recovers_from_unknown_dataset_id(sync_db):
    user, dataset = _seed_public_dataset(sync_db)

    resolved_id, strategy = _resolve_dataset_scope(
        sync_db,
        context=SkillContext(user_id=user.id, document_id=dataset.document_id),
        requested_dataset_id=999,
    )

    assert resolved_id == dataset.id
    assert strategy == "current_document_public_dataset"


def test_diagnostic_subset_run_never_replaces_full_release_evidence(sync_db):
    user, dataset = _seed_public_dataset(sync_db)
    full = _add_run(
        sync_db,
        dataset,
        "dense",
        _scores(0.7, 0.7, 0.7, 0.7),
    )
    subset = _add_run(
        sync_db,
        dataset,
        "dense",
        _scores(1.0, 1.0, 1.0, 1.0),
    )
    subset.evaluation_scope = "subset"
    subset.sample_filter = "[1]"
    subset.source_run_id = full.id
    sync_db.flush()

    decision = recommend_evaluated_pipeline(sync_db, user_id=user.id)

    assert decision["status"] == "selected"
    assert decision["selected"]["run_id"] == full.id


def test_grounded_generation_selection_fails_closed_until_all_judges_calibrated(
    sync_db,
):
    user, _dataset = _seed_public_dataset(sync_db)

    decision = recommend_evaluated_pipeline(
        sync_db,
        user_id=user.id,
        target="grounded_generation",
    )

    assert decision["status"] == "judge_calibration_required"
    assert decision["selected"] is None
