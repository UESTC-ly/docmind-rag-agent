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
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.models.user import User
from app.services.evaluation import runner


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
    monkeypatch.setattr(runner, "judge_faithfulness", lambda ctx, ans: 1.0)
    monkeypatch.setattr(runner, "judge_answer_relevancy", lambda q, ans: 0.8)


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
