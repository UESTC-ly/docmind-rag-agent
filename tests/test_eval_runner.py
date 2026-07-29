"""评估运行编排器测试（run_evaluation）。

用同步内存库播种 dataset+samples+run，mock 掉 RAG 检索/生成与 judge，
验证：逐样本算指标、写 EvalResult、聚合均值、状态流转、异常兜底为 FAILED。
"""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.document import Document, DocumentStatus
from app.models.evaluation import (
    EvalDataset,
    EvalMetricResult,
    EvalRegressionGate,
    EvalRegressionResult,
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.models.user import User
from app.services.evaluation import runner
from app.services.evaluation.generation_judge import JudgeResult
from app.services.evidence import validate_citations


class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


def _seed(sync_db, relevant_ids_list):
    """播种一个用户+文档+数据集+若干样本+一个 run，返回 (run_id, user_id, doc_id)。"""
    user = User(email="e@t.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()

    doc = Document(
        user_id=user.id,
        filename="d",
        file_path="p",
        status=DocumentStatus.COMPLETED,
        chunk_count=3,
    )
    sync_db.add(doc)
    sync_db.flush()

    ds = EvalDataset(user_id=user.id, document_id=doc.id, name="ds")
    sync_db.add(ds)
    sync_db.flush()

    for i, rel in enumerate(relevant_ids_list):
        sync_db.add(
            EvalSample(
                dataset_id=ds.id,
                question=f"q{i}",
                ground_truth_answer=f"a{i}",
                relevant_chunk_ids=json.dumps(rel),
            )
        )
    run = EvalRun(dataset_id=ds.id)
    sync_db.add(run)
    sync_db.flush()
    return run.id, user.id, doc.id


@pytest.fixture
def patch_rag(monkeypatch):
    """默认：共享 hybrid 路径命中 [0]，生成与 judge 返回固定值。"""
    monkeypatch.setattr(
        runner,
        "retrieve_for_skill",
        lambda **kwargs: [
            {
                "score": 0.9,
                "content": "片段0",
                "document_id": kwargs["document_id"],
                "chunk_index": 0,
                "retrieval_sources": ["dense", "keyword"],
                "dense_rank": 1,
                "keyword_rank": 1,
                "rrf_score": 0.03,
                "fused_rank": 1,
                "rerank_score": 0.91,
                "reranker": "local",
                "final_rank": 1,
            }
        ],
    )
    monkeypatch.setattr(runner, "chat_completion", lambda messages: _FakeMsg("答案"))
    monkeypatch.setattr(runner, "evaluate_faithfulness", lambda ctx, ans: 1.0)
    monkeypatch.setattr(
        runner,
        "evaluate_answer_relevancy",
        lambda q, ans: 0.8,
    )

    def _grounding(answer, hits):
        report = validate_citations(answer, hits)
        groundedness = 1.0 if report["passed"] else 0.0
        return {
            **report,
            "contract": "claim_grounding_v1",
            "semantic_entailment_checked": True,
            "judge_status": "completed",
            "judge_model": "test-judge",
            "groundedness": groundedness,
            "faithfulness": groundedness,
            "citation_correctness": report["citation_precision"],
            "conflict_count": 0,
            "refused": False,
        }

    monkeypatch.setattr(runner, "evaluate_grounding", _grounding)


class TestRunEvaluation:
    def test_completes_and_aggregates(self, sync_db, patch_rag):
        # 两条样本都相关 [0]，检索命中 [0] → hit=1, recall=1
        run_id, uid, did = _seed(sync_db, [[0], [0]])
        runner.run_evaluation(sync_db, run_id, uid, did)

        run = sync_db.get(EvalRun, run_id)
        assert run.status == RunStatus.COMPLETED
        assert run.hit_rate == 1.0
        assert run.recall == 1.0
        assert run.faithfulness == 1.0
        assert run.answer_relevancy == 0.8
        assert run.citation_precision == 1.0
        assert run.citation_recall == 0.0
        assert run.unsupported_claim_rate == 1.0
        assert run.pipeline_id == "configured"
        assert len(run.pipeline_fingerprint) == 64
        assert run.completed_at is not None

    def test_writes_per_sample_results(self, sync_db, patch_rag):
        run_id, uid, did = _seed(sync_db, [[0], [0], [0]])
        runner.run_evaluation(sync_db, run_id, uid, did)
        results = sync_db.query(EvalResult).filter(EvalResult.run_id == run_id).all()
        assert len(results) == 3
        assert all(r.generated_answer == "答案" for r in results)
        assert all(r.retrieval_mode for r in results)
        assert all(r.reranker_mode == "local" for r in results)
        trace = json.loads(results[0].retrieval_trace)
        assert trace[0]["retrieval_sources"] == ["dense", "keyword"]
        assert "content" not in trace[0]
        citation = json.loads(results[0].citation_report)
        assert citation["contract"] == "claim_grounding_v1"
        assert results[0].unsupported_claim_rate == 1.0
        metric_rows = (
            sync_db.query(EvalMetricResult)
            .filter(EvalMetricResult.run_id == run_id)
            .all()
        )
        assert any(
            metric.subject_type == "run"
            and metric.metric_name == "map_at_k"
            and metric.score == 1.0
            for metric in metric_rows
        )

    def test_unavailable_generation_judge_persists_missing_score_and_reason(
        self,
        sync_db,
        monkeypatch,
        patch_rag,
    ):
        unavailable = JudgeResult(
            score=None,
            reason="RuntimeError: judge offline",
            status="unavailable",
            model="judge-model",
            rubric_version="faithfulness_v2",
            input_fingerprint="f" * 64,
        )
        monkeypatch.setattr(
            runner,
            "evaluate_faithfulness",
            lambda context, answer: unavailable,
        )
        monkeypatch.setattr(
            runner,
            "evaluate_answer_relevancy",
            lambda question, answer: unavailable,
        )
        run_id, uid, did = _seed(sync_db, [[0]])

        result = runner.run_evaluation(sync_db, run_id, uid, did)

        assert result["status"] == "completed"
        completed = sync_db.get(EvalRun, run_id)
        assert completed.faithfulness is None
        assert completed.answer_relevancy is None
        sample_metric = (
            sync_db.query(EvalMetricResult)
            .filter_by(
                run_id=run_id,
                subject_type="sample",
                metric_name="faithfulness",
            )
            .one()
        )
        assert sample_metric.score is None
        assert sample_metric.reason == "RuntimeError: judge offline"
        details = json.loads(sample_metric.details)
        assert details["status"] == "unavailable"
        assert details["input_fingerprint"] == "f" * 64

    def test_document_retrieval_metrics_are_separate_when_qrels_exist(
        self,
        sync_db,
        patch_rag,
    ):
        run_id, uid, did = _seed(sync_db, [[0]])
        run = sync_db.get(EvalRun, run_id)
        sample = (
            sync_db.query(EvalSample)
            .filter_by(dataset_id=run.dataset_id)
            .one()
        )
        sample.document_qrels = json.dumps({str(did): 2})
        sync_db.commit()

        runner.run_evaluation(sync_db, run_id, uid, did)

        document_metrics = {
            metric.metric_name: metric
            for metric in sync_db.query(EvalMetricResult)
            .filter_by(run_id=run_id, subject_type="run")
            .filter(
                EvalMetricResult.metric_version
                == "document_retrieval_v1"
            )
        }
        assert document_metrics["document_hit_at_k"].score == 1.0
        assert document_metrics["document_recall_at_k"].score == 1.0
        assert document_metrics["document_ndcg_at_k"].score == 1.0

    def test_missing_document_qrels_stays_unavailable_not_zero(
        self,
        sync_db,
        patch_rag,
    ):
        run_id, uid, did = _seed(sync_db, [[0]])

        runner.run_evaluation(sync_db, run_id, uid, did)

        metric = (
            sync_db.query(EvalMetricResult)
            .filter_by(
                run_id=run_id,
                subject_type="run",
                metric_name="document_ndcg_at_k",
            )
            .one()
        )
        assert metric.score is None
        assert metric.reason == "metric unavailable for every evaluated sample"

    def test_subset_rerun_evaluates_only_selected_samples_and_skips_gates(
        self,
        sync_db,
        monkeypatch,
        patch_rag,
    ):
        run_id, uid, did = _seed(sync_db, [[0], [0], [0]])
        run = sync_db.get(EvalRun, run_id)
        samples = (
            sync_db.query(EvalSample)
            .filter_by(dataset_id=run.dataset_id)
            .order_by(EvalSample.id)
            .all()
        )
        run.evaluation_scope = "subset"
        run.sample_filter = json.dumps([samples[1].id])
        gate = EvalRegressionGate(
            dataset_id=run.dataset_id,
            name="full-run-only",
            metric_name="hit_rate",
            comparison="absolute_min",
            threshold=1.0,
        )
        sync_db.add(gate)
        sync_db.commit()
        questions = []
        original = runner.retrieve_for_skill

        def _capture(**kwargs):
            questions.append(kwargs["query"])
            return original(**kwargs)

        monkeypatch.setattr(runner, "retrieve_for_skill", _capture)

        result = runner.run_evaluation(sync_db, run_id, uid, did)

        assert result["samples"] == 1
        assert questions == ["q1"]
        [detail] = (
            sync_db.query(EvalResult)
            .filter_by(run_id=run_id)
            .all()
        )
        assert detail.sample_id == samples[1].id
        assert (
            sync_db.query(EvalRegressionResult)
            .filter_by(candidate_run_id=run_id)
            .count()
            == 0
        )

    def test_subset_rerun_rejects_samples_outside_dataset(
        self,
        sync_db,
        patch_rag,
    ):
        run_id, uid, did = _seed(sync_db, [[0]])
        run = sync_db.get(EvalRun, run_id)
        run.evaluation_scope = "subset"
        run.sample_filter = "[999999]"
        sync_db.commit()

        result = runner.run_evaluation(sync_db, run_id, uid, did)

        assert result["status"] == "failed"
        assert "outside its dataset" in result["error"]

    def test_uses_persisted_pipeline_snapshot_and_scores_valid_citation(
        self, sync_db, monkeypatch, patch_rag
    ):
        from app.services.rag_pipeline import pipeline_presets

        run_id, uid, did = _seed(sync_db, [[0]])
        spec = pipeline_presets()["dense"]
        run = sync_db.get(EvalRun, run_id)
        run.pipeline_id = spec.id
        run.pipeline_spec = json.dumps(spec.to_dict())
        run.pipeline_fingerprint = spec.fingerprint
        sync_db.commit()
        captured = {}
        original = runner.retrieve_for_skill

        def _capture(**kwargs):
            captured["pipeline"] = kwargs["pipeline"]
            return original(**kwargs)

        monkeypatch.setattr(runner, "retrieve_for_skill", _capture)
        monkeypatch.setattr(
            runner,
            "chat_completion",
            lambda messages: _FakeMsg(f"答案。[D{did}:C0]"),
        )

        runner.run_evaluation(sync_db, run_id, uid, did)
        completed = sync_db.get(EvalRun, run_id)
        assert captured["pipeline"].id == "dense"
        assert completed.pipeline_fingerprint == spec.fingerprint
        assert completed.citation_precision == 1.0
        assert completed.citation_recall == 1.0
        assert completed.unsupported_claim_rate == 0.0

    def test_miss_gives_zero_hit(self, sync_db, monkeypatch, patch_rag):
        # 相关块是 [2]，但检索只返回 [0] → 未命中
        run_id, uid, did = _seed(sync_db, [[2]])
        runner.run_evaluation(sync_db, run_id, uid, did)
        run = sync_db.get(EvalRun, run_id)
        assert run.hit_rate == 0.0
        assert run.recall == 0.0

    def test_missing_run_returns_silently(self, sync_db, patch_rag):
        # run_id 不存在 → 直接返回，不抛异常
        result = runner.run_evaluation(sync_db, 99999, 1, 1)
        assert result["status"] == "missing"

    def test_exception_marks_failed(self, sync_db, monkeypatch):
        run_id, uid, did = _seed(sync_db, [[0]])
        # 让共享检索路径抛异常
        monkeypatch.setattr(
            runner,
            "retrieve_for_skill",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Qdrant down")),
        )
        runner.run_evaluation(sync_db, run_id, uid, did)
        run = sync_db.get(EvalRun, run_id)
        assert run.status == RunStatus.FAILED
        assert "Qdrant down" in run.error_message

    def test_retrieval_only_dataset_does_not_fabricate_generation_scores(
        self,
        sync_db,
        monkeypatch,
        patch_rag,
    ):
        run_id, uid, did = _seed(sync_db, [[0]])
        run = sync_db.get(EvalRun, run_id)
        dataset = sync_db.get(EvalDataset, run.dataset_id)
        dataset.task_type = "retrieval"
        sync_db.commit()
        monkeypatch.setattr(
            runner,
            "chat_completion",
            lambda messages: (_ for _ in ()).throw(
                AssertionError("retrieval benchmark must not call generation")
            ),
        )

        result = runner.run_evaluation(sync_db, run_id, uid, did)

        assert result["status"] == "completed"
        completed = sync_db.get(EvalRun, run_id)
        assert completed.hit_rate == 1.0
        assert completed.map_score == 1.0
        assert completed.faithfulness is None
        assert completed.answer_relevancy is None
        detail = sync_db.query(EvalResult).filter_by(run_id=run_id).one()
        assert detail.generated_answer == ""
        assert detail.citation_precision is None
        report = json.loads(detail.citation_report)
        assert report["contract"] == "not_evaluated"

    def test_duplicate_delivery_does_not_duplicate_results(self, sync_db, patch_rag):
        run_id, uid, did = _seed(sync_db, [[0], [0]])
        first = runner.run_evaluation(sync_db, run_id, uid, did)
        second = runner.run_evaluation(sync_db, run_id, uid, did)
        count = sync_db.query(EvalResult).filter(EvalResult.run_id == run_id).count()
        assert first["status"] == "completed"
        assert second == {"status": "completed", "run_id": run_id, "skipped": True}
        assert count == 2

    def test_failed_run_can_retry_cleanly(self, sync_db, monkeypatch, patch_rag):
        run_id, uid, did = _seed(sync_db, [[0]])
        original = runner.retrieve_for_skill
        monkeypatch.setattr(
            runner,
            "retrieve_for_skill",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("temporary")),
        )
        runner.run_evaluation(sync_db, run_id, uid, did)
        assert sync_db.get(EvalRun, run_id).status == RunStatus.FAILED

        monkeypatch.setattr(runner, "retrieve_for_skill", original)
        result = runner.run_evaluation(sync_db, run_id, uid, did)
        assert result["status"] == "completed"
        assert sync_db.query(EvalResult).filter_by(run_id=run_id).count() == 1

    def test_retryable_failure_returns_to_pending_until_next_attempt(
        self, sync_db, monkeypatch, patch_rag
    ):
        run_id, uid, did = _seed(sync_db, [[0]])
        original = runner.retrieve_for_skill
        monkeypatch.setattr(
            runner,
            "retrieve_for_skill",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("temporary")),
        )

        with pytest.raises(RuntimeError, match="temporary"):
            runner.run_evaluation(
                sync_db,
                run_id,
                uid,
                did,
                raise_on_error=True,
                retryable=True,
                task_id="celery-1",
            )

        pending = sync_db.get(EvalRun, run_id)
        assert pending.status == RunStatus.PENDING
        assert pending.completed_at is None
        assert "将重试" in pending.error_message

        monkeypatch.setattr(runner, "retrieve_for_skill", original)
        completed = runner.run_evaluation(
            sync_db,
            run_id,
            uid,
            did,
            task_id="celery-1",
        )
        assert completed["status"] == "completed"

    def test_redelivery_replaces_token_and_old_owner_cannot_heartbeat(self, sync_db):
        run_id, _uid, _did = _seed(sync_db, [[0]])
        first = runner._claim_run(
            sync_db,
            run_id,
            "celery-same-id",
            reclaim_running=False,
        )
        assert not isinstance(first, dict)
        first_token, _ = first

        second = runner._claim_run(
            sync_db,
            run_id,
            "celery-same-id",
            reclaim_running=True,
        )
        assert not isinstance(second, dict)
        second_token, _ = second
        assert second_token != first_token
        with pytest.raises(runner._LeaseLost):
            runner._refresh_lease(sync_db, run_id, first_token)

    def test_expired_running_lease_is_recovered(self, sync_db, patch_rag):
        run_id, uid, did = _seed(sync_db, [[0]])
        run = sync_db.get(EvalRun, run_id)
        run.status = RunStatus.RUNNING
        run.task_id = "dead-worker"
        run.lease_token = "dead-token"
        run.heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=2)
        sync_db.commit()

        result = runner.run_evaluation(sync_db, run_id, uid, did, task_id="replacement")
        assert result["status"] == "completed"
        assert sync_db.get(EvalRun, run_id).status == RunStatus.COMPLETED


def test_sqlite_concurrent_delivery_has_one_owner_and_one_result_set(
    tmp_path, monkeypatch, patch_rag
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent-eval.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as seed_db:
        run_id, uid, did = _seed(seed_db, [[0], [0]])
        seed_db.commit()

    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def _slow_retrieval(**kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        assert release.wait(timeout=5)
        return [
            {
                "content": "片段0",
                "document_id": kwargs["document_id"],
                "chunk_index": 0,
                "reranker": "local",
            }
        ]

    monkeypatch.setattr(runner, "retrieve_for_skill", _slow_retrieval)
    outcomes: list[dict] = []
    failures: list[BaseException] = []

    def _execute(task_id: str) -> None:
        try:
            with sessions() as db:
                outcomes.append(
                    runner.run_evaluation(db, run_id, uid, did, task_id=task_id)
                )
        except BaseException as exc:  # test thread must report failures to parent
            failures.append(exc)

    first = threading.Thread(target=_execute, args=("local-1",))
    first.start()
    assert started.wait(timeout=5)
    second = threading.Thread(target=_execute, args=("local-2",))
    second.start()
    second.join(timeout=5)
    assert not second.is_alive()
    release.set()
    first.join(timeout=5)

    assert failures == []
    assert sorted(item["status"] for item in outcomes) == ["completed", "running"]
    assert calls == 2  # one owner processes exactly two samples
    with sessions() as verify_db:
        assert verify_db.query(EvalResult).filter_by(run_id=run_id).count() == 2
        assert verify_db.get(EvalRun, run_id).status == RunStatus.COMPLETED
    engine.dispose()
