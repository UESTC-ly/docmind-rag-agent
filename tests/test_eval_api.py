"""评估路由集成测试（/eval/*）。

难点：POST /datasets 与 POST /runs 走 async DB + sync SyncSessionLocal 双路径。
两条路径必须共享同一物理库才能互见数据——用文件型 sqlite（非 :memory:，那是
每连接独立），aiosqlite 与 pysqlite 指向同一文件即可看到彼此已提交的数据。

外部边界（LLM 出题 / RAG 检索 / judge）全部 mock。
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

import app.database as database
from app.database import Base, get_db
from app.main import app
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.evaluation import EvalDataset, EvalSample
from app.models.user import User
from app.services.evaluation import dataset_gen, runner
from app.utils.security import create_access_token, hash_password

pytestmark = pytest.mark.asyncio


class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


@pytest_asyncio.fixture
async def eval_env(tmp_path, monkeypatch):
    """共享文件 sqlite 的测试环境：async client + sync sessionmaker + 已登录用户。"""
    db_file = tmp_path / "eval_test.db"
    url = f"sqlite:///{db_file}"

    # 同步引擎（供 SyncSessionLocal 替身 + 播种用）
    sync_engine = create_engine(
        url, connect_args={"check_same_thread": False, "timeout": 30}
    )
    Base.metadata.create_all(sync_engine)
    SyncSession = sessionmaker(sync_engine, expire_on_commit=False)
    # 端点内 `from app.database import SyncSessionLocal` 是延迟导入，patch 模块属性即可
    monkeypatch.setattr(database, "SyncSessionLocal", SyncSession)

    # 异步引擎，指向同一文件
    async_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    AsyncSessionMaker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _override_get_db():
        async with AsyncSessionMaker() as session:
            yield session
            await session.commit()

    app.dependency_overrides[get_db] = _override_get_db

    # 播种一个用户（同步库直接写），生成 token
    with SyncSession() as s:
        user = User(email="ev@test.com", hashed_password=hash_password("test123"))
        s.add(user)
        s.commit()
        user_id = user.id
    headers = {"Authorization": f"Bearer {create_access_token('ev@test.com')}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield {
            "client": ac,
            "headers": headers,
            "user_id": user_id,
            "SyncSession": SyncSession,
        }

    app.dependency_overrides.clear()
    await async_engine.dispose()
    sync_engine.dispose()


def _seed_document(SyncSession, user_id, n_chunks=2):
    with SyncSession() as s:
        doc = Document(user_id=user_id, filename="d.txt", file_path="p",
                       status=DocumentStatus.COMPLETED, chunk_count=n_chunks)
        s.add(doc)
        s.flush()
        for i in range(n_chunks):
            s.add(DocumentChunk(document_id=doc.id, chunk_index=i, content=f"内容{i}"))
        s.commit()
        return doc.id


def _seed_dataset(SyncSession, user_id, doc_id, relevant_list):
    with SyncSession() as s:
        ds = EvalDataset(user_id=user_id, document_id=doc_id, name="ds")
        s.add(ds)
        s.flush()
        for i, rel in enumerate(relevant_list):
            s.add(EvalSample(
                dataset_id=ds.id, question=f"q{i}", ground_truth_answer=f"a{i}",
                relevant_chunk_ids=__import__("json").dumps(rel),
            ))
        s.commit()
        return ds.id


class TestGenerateDataset:
    async def test_generate_creates_dataset(self, eval_env, monkeypatch):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"], 3)
        monkeypatch.setattr(
            dataset_gen, "chat_completion",
            lambda messages, temperature=0.7: _FakeMsg('{"question":"q?","answer":"a"}'),
        )
        resp = await eval_env["client"].post(
            "/eval/datasets", headers=eval_env["headers"],
            json={"document_id": doc_id, "name": "我的评估集", "sample_count": 3},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "我的评估集"
        assert body["sample_count"] == 3

    async def test_generate_on_others_document_404(self, eval_env):
        resp = await eval_env["client"].post(
            "/eval/datasets", headers=eval_env["headers"],
            json={"document_id": 99999, "name": "x", "sample_count": 3},
        )
        assert resp.status_code == 404


class TestListDatasets:
    async def test_list_returns_datasets_with_count(self, eval_env):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        _seed_dataset(eval_env["SyncSession"], eval_env["user_id"], doc_id, [[0], [0]])
        resp = await eval_env["client"].get("/eval/datasets", headers=eval_env["headers"])
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["sample_count"] == 2


class TestCreateRun:
    async def test_run_completes_with_metrics(self, eval_env, monkeypatch):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(eval_env["SyncSession"], eval_env["user_id"], doc_id, [[0], [0]])
        # mock runner 的 RAG + judge 边界
        monkeypatch.setattr(runner, "embed_query", lambda q: [0.1])
        monkeypatch.setattr(runner, "search", lambda *a, **k: [
            {"score": 0.9, "content": "片段0", "document_id": doc_id, "chunk_index": 0}
        ])
        monkeypatch.setattr(runner, "chat_completion", lambda m: _FakeMsg("答案"))
        monkeypatch.setattr(runner, "judge_faithfulness", lambda c, a: 1.0)
        monkeypatch.setattr(runner, "judge_answer_relevancy", lambda q, a: 0.9)

        resp = await eval_env["client"].post(
            "/eval/runs", headers=eval_env["headers"], json={"dataset_id": ds_id},
        )
        assert resp.status_code == 201
        body = resp.json()
        # 双路径共享库生效：sync 跑完的结果 async 能读到
        assert body["status"] == "completed"
        assert body["hit_rate"] == 1.0
        assert body["faithfulness"] == 1.0

    async def test_run_on_others_dataset_404(self, eval_env):
        resp = await eval_env["client"].post(
            "/eval/runs", headers=eval_env["headers"], json={"dataset_id": 99999},
        )
        assert resp.status_code == 404


class TestGetRunAndDetails:
    async def _make_completed_run(self, eval_env, monkeypatch):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(eval_env["SyncSession"], eval_env["user_id"], doc_id, [[0]])
        monkeypatch.setattr(runner, "embed_query", lambda q: [0.1])
        monkeypatch.setattr(runner, "search", lambda *a, **k: [
            {"score": 0.9, "content": "片段0", "document_id": doc_id, "chunk_index": 0}
        ])
        monkeypatch.setattr(runner, "chat_completion", lambda m: _FakeMsg("答案"))
        monkeypatch.setattr(runner, "judge_faithfulness", lambda c, a: 1.0)
        monkeypatch.setattr(runner, "judge_answer_relevancy", lambda q, a: 0.9)
        resp = await eval_env["client"].post(
            "/eval/runs", headers=eval_env["headers"], json={"dataset_id": ds_id},
        )
        return resp.json()["id"]

    async def test_get_run(self, eval_env, monkeypatch):
        run_id = await self._make_completed_run(eval_env, monkeypatch)
        resp = await eval_env["client"].get(
            f"/eval/runs/{run_id}", headers=eval_env["headers"]
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    async def test_get_run_details(self, eval_env, monkeypatch):
        run_id = await self._make_completed_run(eval_env, monkeypatch)
        resp = await eval_env["client"].get(
            f"/eval/runs/{run_id}/details", headers=eval_env["headers"]
        )
        assert resp.status_code == 200
        details = resp.json()
        assert len(details) == 1
        assert details[0]["generated_answer"] == "答案"

    async def test_get_missing_run_404(self, eval_env):
        resp = await eval_env["client"].get("/eval/runs/99999", headers=eval_env["headers"])
        assert resp.status_code == 404
