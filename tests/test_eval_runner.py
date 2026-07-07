"""评估运行编排器测试（run_evaluation）。

用同步内存库播种 dataset+samples+run，mock 掉 RAG 检索/生成与 judge，
验证：逐样本算指标、写 EvalResult、聚合均值、状态流转、异常兜底为 FAILED。
"""

import json

import pytest

from app.models.document import Document, DocumentChunk, DocumentStatus
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

    doc = Document(user_id=user.id, filename="d", file_path="p",
                   status=DocumentStatus.COMPLETED, chunk_count=3)
    sync_db.add(doc)
    sync_db.flush()

    ds = EvalDataset(user_id=user.id, document_id=doc.id, name="ds")
    sync_db.add(ds)
    sync_db.flush()

    for i, rel in enumerate(relevant_ids_list):
        sync_db.add(EvalSample(
            dataset_id=ds.id, question=f"q{i}", ground_truth_answer=f"a{i}",
            relevant_chunk_ids=json.dumps(rel),
        ))
    run = EvalRun(dataset_id=ds.id)
    sync_db.add(run)
    sync_db.flush()
    return run.id, user.id, doc.id


@pytest.fixture
def patch_rag(monkeypatch):
    """默认：检索命中 [0]，生成固定答案，judge 都给 1.0 / 0.8。"""
    monkeypatch.setattr(runner, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(runner, "search", lambda *a, **k: [
        {"score": 0.9, "content": "片段0", "document_id": 1, "chunk_index": 0}
    ])
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
        results = (
            sync_db.query(EvalResult).filter(EvalResult.run_id == run_id).all()
        )
        assert len(results) == 3
        assert all(r.generated_answer == "答案" for r in results)

    def test_miss_gives_zero_hit(self, sync_db, monkeypatch, patch_rag):
        # 相关块是 [2]，但检索只返回 [0] → 未命中
        run_id, uid, did = _seed(sync_db, [[2]])
        runner.run_evaluation(sync_db, run_id, uid, did)
        run = sync_db.get(EvalRun, run_id)
        assert run.hit_rate == 0.0
        assert run.recall == 0.0

    def test_missing_run_returns_silently(self, sync_db, patch_rag):
        # run_id 不存在 → 直接返回，不抛异常
        runner.run_evaluation(sync_db, 99999, 1, 1)

    def test_exception_marks_failed(self, sync_db, monkeypatch):
        run_id, uid, did = _seed(sync_db, [[0]])
        # 让检索抛异常
        monkeypatch.setattr(runner, "embed_query", lambda q: [0.1])
        monkeypatch.setattr(
            runner, "search",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Qdrant down")),
        )
        runner.run_evaluation(sync_db, run_id, uid, did)
        run = sync_db.get(EvalRun, run_id)
        assert run.status == RunStatus.FAILED
        assert "Qdrant down" in run.error_message
