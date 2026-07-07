"""LLM 反向出题（generate_dataset）测试。

播种文档分块，mock chat_completion 返回 QA JSON，验证：
数据集创建、样本落库、markdown 剥离、生成失败跳过、无分块报错、抽样上限。
"""

import pytest

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.evaluation import EvalDataset, EvalSample
from app.models.user import User
from app.services.evaluation import dataset_gen


class _FakeMsg:
    def __init__(self, content):
        self.content = content


def _seed_doc(sync_db, n_chunks=3):
    user = User(email="g@t.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    doc = Document(user_id=user.id, filename="d", file_path="p",
                   status=DocumentStatus.COMPLETED, chunk_count=n_chunks)
    sync_db.add(doc)
    sync_db.flush()
    for i in range(n_chunks):
        sync_db.add(DocumentChunk(document_id=doc.id, chunk_index=i, content=f"内容{i}"))
    sync_db.flush()
    return user.id, doc.id


class TestGenerateDataset:
    def test_creates_dataset_with_samples(self, sync_db, monkeypatch):
        uid, did = _seed_doc(sync_db, 3)
        monkeypatch.setattr(
            dataset_gen, "chat_completion",
            lambda messages, temperature=0.7: _FakeMsg('{"question":"问?","answer":"答"}'),
        )
        ds = dataset_gen.generate_dataset(sync_db, uid, did, "测试集", sample_count=3)
        assert isinstance(ds, EvalDataset)
        samples = sync_db.query(EvalSample).filter(EvalSample.dataset_id == ds.id).all()
        assert len(samples) == 3
        assert samples[0].question == "问?"
        assert samples[0].ground_truth_answer == "答"

    def test_strips_markdown_fence(self, sync_db, monkeypatch):
        uid, did = _seed_doc(sync_db, 1)
        monkeypatch.setattr(
            dataset_gen, "chat_completion",
            lambda messages, temperature=0.7: _FakeMsg(
                '```json\n{"question":"q","answer":"a"}\n```'
            ),
        )
        ds = dataset_gen.generate_dataset(sync_db, uid, did, "集", sample_count=1)
        samples = sync_db.query(EvalSample).filter(EvalSample.dataset_id == ds.id).all()
        assert samples[0].question == "q"

    def test_failed_generation_skipped(self, sync_db, monkeypatch):
        # LLM 返回非法 JSON → 该样本跳过，数据集可能 0 样本
        uid, did = _seed_doc(sync_db, 2)
        monkeypatch.setattr(
            dataset_gen, "chat_completion",
            lambda messages, temperature=0.7: _FakeMsg("不是JSON"),
        )
        ds = dataset_gen.generate_dataset(sync_db, uid, did, "集", sample_count=2)
        samples = sync_db.query(EvalSample).filter(EvalSample.dataset_id == ds.id).all()
        assert len(samples) == 0

    def test_no_chunks_raises(self, sync_db, monkeypatch):
        # 文档无分块 → ValueError
        user = User(email="empty@t.com", hashed_password="x")
        sync_db.add(user)
        sync_db.flush()
        doc = Document(user_id=user.id, filename="d", file_path="p",
                       status=DocumentStatus.COMPLETED)
        sync_db.add(doc)
        sync_db.flush()
        with pytest.raises(ValueError):
            dataset_gen.generate_dataset(sync_db, user.id, doc.id, "集", sample_count=3)

    def test_sample_count_capped_by_chunks(self, sync_db, monkeypatch):
        # 只有 2 块但要 10 条 → 最多 2 条
        uid, did = _seed_doc(sync_db, 2)
        monkeypatch.setattr(
            dataset_gen, "chat_completion",
            lambda messages, temperature=0.7: _FakeMsg('{"question":"q","answer":"a"}'),
        )
        ds = dataset_gen.generate_dataset(sync_db, uid, did, "集", sample_count=10)
        samples = sync_db.query(EvalSample).filter(EvalSample.dataset_id == ds.id).all()
        assert len(samples) == 2
