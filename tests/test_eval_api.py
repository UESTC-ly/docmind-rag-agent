"""评估路由集成测试（/eval/*）。

难点：POST /datasets 与 POST /runs 走 async DB + sync SyncSessionLocal 双路径。
两条路径必须共享同一物理库才能互见数据——用文件型 sqlite（非 :memory:，那是
每连接独立），aiosqlite 与 pysqlite 指向同一文件即可看到彼此已提交的数据。

外部边界（LLM 出题 / RAG 检索 / judge）全部 mock。
"""

import json

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
from app.models.evaluation import (
    EvalDataset,
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.models.user import User
from app.services import task_dispatcher
from app.services.evaluation import dataset_gen, runner
from app.services.evidence import validate_citations
from app.services.task_dispatcher import DispatchReceipt
from app.services.rag_pipeline import pipeline_presets
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

    dispatched = []

    def _dispatch(run_id, user_id, document_id):
        dispatched.append((run_id, user_id, document_id))
        return DispatchReceipt(task_id=f"test-{run_id}", mode="celery")

    monkeypatch.setattr(task_dispatcher, "dispatch_evaluation", _dispatch)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield {
            "client": ac,
            "headers": headers,
            "user_id": user_id,
            "SyncSession": SyncSession,
            "dispatched": dispatched,
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


def _seed_dataset(
    SyncSession,
    user_id,
    doc_id,
    relevant_list,
    **dataset_kwargs,
):
    with SyncSession() as s:
        ds = EvalDataset(
            user_id=user_id,
            document_id=doc_id,
            name="ds",
            **dataset_kwargs,
        )
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
        _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0], [0]],
            source_name="CMRC 2018",
            source_version="2018",
            label_source="public_ground_truth",
            release_eligible=True,
        )
        resp = await eval_env["client"].get("/eval/datasets", headers=eval_env["headers"])
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["sample_count"] == 2
        assert body[0]["source_name"] == "CMRC 2018"
        assert body[0]["label_source"] == "public_ground_truth"
        assert body[0]["release_eligible"] is True


class TestCreateRun:
    async def test_run_returns_pending_without_calling_runner(self, eval_env, monkeypatch):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(eval_env["SyncSession"], eval_env["user_id"], doc_id, [[0], [0]])
        monkeypatch.setattr(
            runner,
            "run_evaluation",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("request must not execute the evaluation runner")
            ),
        )

        resp = await eval_env["client"].post(
            "/eval/runs", headers=eval_env["headers"], json={"dataset_id": ds_id},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "pending"
        assert body["hit_rate"] is None
        assert eval_env["dispatched"] == [(body["id"], eval_env["user_id"], doc_id)]
        assert resp.headers["X-Task-ID"] == f"test-{body['id']}"

    async def test_run_persists_requested_pipeline_snapshot(self, eval_env):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
        )
        resp = await eval_env["client"].post(
            "/eval/runs",
            headers=eval_env["headers"],
            json={"dataset_id": ds_id, "pipeline_id": "dense"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["pipeline_id"] == "dense"
        assert len(body["pipeline_fingerprint"]) == 64

    async def test_unknown_pipeline_is_rejected_without_dispatch(self, eval_env):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
        )
        resp = await eval_env["client"].post(
            "/eval/runs",
            headers=eval_env["headers"],
            json={"dataset_id": ds_id, "pipeline_id": "does-not-exist"},
        )
        assert resp.status_code == 422
        assert eval_env["dispatched"] == []

    async def test_run_on_others_dataset_404(self, eval_env):
        resp = await eval_env["client"].post(
            "/eval/runs", headers=eval_env["headers"], json={"dataset_id": 99999},
        )
        assert resp.status_code == 404

    async def test_candidate_requires_completed_baseline_on_same_dataset(
        self,
        eval_env,
    ):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
        )
        with eval_env["SyncSession"]() as session:
            baseline = EvalRun(dataset_id=ds_id, status=RunStatus.PENDING)
            session.add(baseline)
            session.commit()
            baseline_id = baseline.id

        pending = await eval_env["client"].post(
            "/eval/runs",
            headers=eval_env["headers"],
            json={"dataset_id": ds_id, "baseline_run_id": baseline_id},
        )
        assert pending.status_code == 409

        with eval_env["SyncSession"]() as session:
            baseline = session.get(EvalRun, baseline_id)
            baseline.status = RunStatus.COMPLETED
            session.commit()
        candidate = await eval_env["client"].post(
            "/eval/runs",
            headers=eval_env["headers"],
            json={
                "dataset_id": ds_id,
                "baseline_run_id": baseline_id,
                "run_label": "candidate-a",
            },
        )
        assert candidate.status_code == 202
        assert candidate.json()["baseline_run_id"] == baseline_id
        assert candidate.json()["comparison_role"] == "candidate"


class TestPipelineExperiments:
    async def test_lists_versioned_pipeline_presets(self, eval_env):
        resp = await eval_env["client"].get(
            "/eval/pipelines",
            headers=eval_env["headers"],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {"configured", "dense", "hybrid", "hybrid-rerank"} == {
            item["id"] for item in body
        }
        assert all(len(item["fingerprint"]) == 64 for item in body)
        assert all(item["spec"]["schema_version"] == 1 for item in body)

    async def test_validates_inline_pipeline_before_experiment(self, eval_env):
        client = eval_env["client"]
        headers = eval_env["headers"]
        listed = await client.get(
            "/eval/pipelines",
            headers=headers,
        )
        spec = next(
            item["spec"] for item in listed.json() if item["id"] == "dense"
        )

        valid = await client.post(
            "/eval/pipelines/validate",
            headers=headers,
            json={"spec": spec, "check_runtime": True},
        )
        assert valid.status_code == 200
        assert valid.json()["valid"] is True
        assert len(valid.json()["fingerprint"]) == 64

        spec["fusion"] = "missing-plugin-component"
        invalid = await client.post(
            "/eval/pipelines/validate",
            headers=headers,
            json={"spec": spec, "check_runtime": False},
        )
        assert invalid.status_code == 200
        assert invalid.json()["valid"] is False
        assert "missing-plugin-component" in invalid.json()["errors"][0]

    async def test_dispatches_paired_runs_for_same_dataset(self, eval_env):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
        )
        resp = await eval_env["client"].post(
            "/eval/experiments",
            headers=eval_env["headers"],
            json={
                "dataset_id": ds_id,
                "pipeline_ids": ["dense", "hybrid-rerank"],
            },
        )
        assert resp.status_code == 202
        runs = resp.json()
        assert [run["pipeline_id"] for run in runs] == [
            "dense",
            "hybrid-rerank",
        ]
        assert len({run["pipeline_fingerprint"] for run in runs}) == 2
        assert len({run["experiment_key"] for run in runs}) == 1
        baseline = next(run for run in runs if run["comparison_role"] == "baseline")
        candidate = next(run for run in runs if run["comparison_role"] == "candidate")
        assert candidate["baseline_run_id"] == baseline["id"]
        assert len(eval_env["dispatched"]) == 2

    async def test_rejects_duplicate_pipeline_ids(self, eval_env):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
        )
        resp = await eval_env["client"].post(
            "/eval/experiments",
            headers=eval_env["headers"],
            json={"dataset_id": ds_id, "pipeline_ids": ["dense", "dense"]},
        )
        assert resp.status_code == 422


class TestSelectiveBadcaseRerun:
    async def test_reuses_exact_pipeline_and_only_source_run_samples(
        self,
        eval_env,
    ):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0], [1]],
        )
        spec = pipeline_presets()["dense"]
        with eval_env["SyncSession"]() as session:
            samples = (
                session.query(EvalSample)
                .filter_by(dataset_id=ds_id)
                .order_by(EvalSample.id)
                .all()
            )
            source = EvalRun(
                dataset_id=ds_id,
                status=RunStatus.COMPLETED,
                pipeline_id=spec.id,
                pipeline_spec=json.dumps(spec.to_dict()),
                pipeline_fingerprint=spec.fingerprint,
            )
            session.add(source)
            session.flush()
            session.add_all(
                [
                    EvalResult(
                        run_id=source.id,
                        sample_id=sample.id,
                        hit=0,
                        reciprocal_rank=0,
                        recall_at_k=0,
                        precision_at_k=0,
                    )
                    for sample in samples
                ]
            )
            session.commit()
            source_id = source.id
            selected_id = samples[1].id

        response = await eval_env["client"].post(
            f"/eval/runs/{source_id}/rerun",
            headers=eval_env["headers"],
            json={"sample_ids": [selected_id], "run_label": "fix-check"},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["evaluation_scope"] == "subset"
        assert json.loads(body["sample_filter"]) == [selected_id]
        assert body["source_run_id"] == source_id
        assert body["comparison_role"] == "diagnostic"
        assert body["pipeline_id"] == "dense"
        assert body["pipeline_fingerprint"] == spec.fingerprint
        assert eval_env["dispatched"][-1] == (
            body["id"],
            eval_env["user_id"],
            doc_id,
        )

        invalid = await eval_env["client"].post(
            f"/eval/runs/{source_id}/rerun",
            headers=eval_env["headers"],
            json={"sample_ids": [999999]},
        )
        assert invalid.status_code == 422

    async def test_rejects_duplicate_or_unfinished_source(
        self,
        eval_env,
    ):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
        )
        with eval_env["SyncSession"]() as session:
            sample = session.query(EvalSample).filter_by(dataset_id=ds_id).one()
            source = EvalRun(dataset_id=ds_id, status=RunStatus.PENDING)
            session.add(source)
            session.flush()
            session.add(
                EvalResult(
                    run_id=source.id,
                    sample_id=sample.id,
                    hit=0,
                    reciprocal_rank=0,
                    recall_at_k=0,
                    precision_at_k=0,
                )
            )
            session.commit()
            source_id = source.id
            sample_id = sample.id

        pending = await eval_env["client"].post(
            f"/eval/runs/{source_id}/rerun",
            headers=eval_env["headers"],
            json={"sample_ids": [sample_id]},
        )
        assert pending.status_code == 409

        with eval_env["SyncSession"]() as session:
            session.get(EvalRun, source_id).status = RunStatus.COMPLETED
            session.commit()
        duplicate = await eval_env["client"].post(
            f"/eval/runs/{source_id}/rerun",
            headers=eval_env["headers"],
            json={"sample_ids": [sample_id, sample_id]},
        )
        assert duplicate.status_code == 422


class TestGetRunAndDetails:
    async def _make_completed_run(self, eval_env, monkeypatch):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(eval_env["SyncSession"], eval_env["user_id"], doc_id, [[0]])
        monkeypatch.setattr(runner, "retrieve_for_skill", lambda **kwargs: [
            {
                "score": 0.9,
                "content": "片段0",
                "document_id": doc_id,
                "chunk_index": 0,
                "retrieval_sources": ["dense", "keyword"],
                "rrf_score": 0.03,
                "fused_rank": 1,
                "rerank_score": 0.9,
                "reranker": "local",
                "final_rank": 1,
            }
        ])
        monkeypatch.setattr(runner, "chat_completion", lambda m: _FakeMsg("答案"))
        monkeypatch.setattr(
            runner,
            "evaluate_faithfulness",
            lambda c, a: 1.0,
        )
        monkeypatch.setattr(
            runner,
            "evaluate_answer_relevancy",
            lambda q, a: 0.9,
        )
        monkeypatch.setattr(
            runner,
            "evaluate_grounding",
            lambda answer, hits: {
                **validate_citations(answer, hits),
                "contract": "claim_grounding_v1",
                "semantic_entailment_checked": True,
                "judge_status": "completed",
                "judge_model": "test-judge",
                "groundedness": 0.0,
                "faithfulness": 0.0,
                "citation_correctness": 1.0,
                "conflict_count": 0,
                "refused": False,
            },
        )
        resp = await eval_env["client"].post(
            "/eval/runs", headers=eval_env["headers"], json={"dataset_id": ds_id},
        )
        assert resp.status_code == 202
        run_id = resp.json()["id"]
        with eval_env["SyncSession"]() as sync_db:
            runner.run_evaluation(
                sync_db,
                run_id,
                eval_env["user_id"],
                doc_id,
            )
        return run_id

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
        assert details[0]["retrieval_mode"]
        assert details[0]["reranker_mode"] == "local"

    async def test_get_versioned_metrics(self, eval_env, monkeypatch):
        run_id = await self._make_completed_run(eval_env, monkeypatch)
        resp = await eval_env["client"].get(
            f"/eval/runs/{run_id}/metrics",
            headers=eval_env["headers"],
        )
        assert resp.status_code == 200
        metrics = resp.json()
        aggregate = {
            item["metric_name"]: item
            for item in metrics
            if item["subject_type"] == "run"
        }
        assert aggregate["mrr"]["score"] == 1.0
        assert aggregate["map_at_k"]["metric_version"] == "retrieval_v1"
        assert aggregate["faithfulness"]["evaluator_kind"] == "model"

    async def test_get_badcases_with_root_causes_and_source_jumps(
        self,
        eval_env,
        monkeypatch,
    ):
        run_id = await self._make_completed_run(eval_env, monkeypatch)
        resp = await eval_env["client"].get(
            f"/eval/runs/{run_id}/badcases",
            headers=eval_env["headers"],
        )
        assert resp.status_code == 200
        [badcase] = resp.json()
        assert "unsupported_generation" in badcase["categories"]
        assert badcase["question"] == "q0"
        assert badcase["relevant_chunk_ids"] == [0]
        assert badcase["retrieved_chunk_ids"] == [0]
        assert badcase["source_links"][0]["citation_id"].endswith(":C0")
        assert badcase["source_links"][0]["jump_url"].endswith("/chunks/0")
        assert badcase["diagnosis"][0]["suggested_actions"]
        assert badcase["rerun_request"] == {
            "method": "POST",
            "path": f"/eval/runs/{run_id}/rerun",
            "body": {"sample_ids": [badcase["sample_id"]]},
        }

    async def test_badcase_diff_reports_new_regression_against_named_baseline(
        self,
        eval_env,
    ):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
            task_type="retrieval",
        )
        with eval_env["SyncSession"]() as session:
            sample = (
                session.query(EvalSample)
                .filter_by(dataset_id=ds_id)
                .one()
            )
            baseline = EvalRun(
                dataset_id=ds_id,
                status=RunStatus.COMPLETED,
                comparison_role="baseline",
            )
            session.add(baseline)
            session.flush()
            candidate = EvalRun(
                dataset_id=ds_id,
                status=RunStatus.COMPLETED,
                comparison_role="candidate",
                baseline_run_id=baseline.id,
            )
            session.add(candidate)
            session.flush()
            session.add_all(
                [
                    EvalResult(
                        run_id=baseline.id,
                        sample_id=sample.id,
                        hit=1,
                        reciprocal_rank=1.0,
                        recall_at_k=1.0,
                        precision_at_k=1.0,
                        average_precision_at_k=1.0,
                        ndcg_at_k=1.0,
                        retrieved_chunk_ids="[0]",
                        retrieval_trace=json.dumps(
                            [
                                {
                                    "document_id": doc_id,
                                    "chunk_index": 0,
                                    "final_rank": 1,
                                }
                            ]
                        ),
                        generated_answer="",
                    ),
                    EvalResult(
                        run_id=candidate.id,
                        sample_id=sample.id,
                        hit=0,
                        reciprocal_rank=0.0,
                        recall_at_k=0.0,
                        precision_at_k=0.0,
                        average_precision_at_k=0.0,
                        ndcg_at_k=0.0,
                        retrieved_chunk_ids="[1]",
                        retrieval_trace=json.dumps(
                            [
                                {
                                    "document_id": doc_id,
                                    "chunk_index": 1,
                                    "final_rank": 1,
                                }
                            ]
                        ),
                        generated_answer="",
                    ),
                ]
            )
            session.commit()
            candidate_id = candidate.id
            baseline_id = baseline.id

        response = await eval_env["client"].get(
            f"/eval/runs/{candidate_id}/badcase-diff",
            headers=eval_env["headers"],
        )

        assert response.status_code == 200
        body = response.json()
        assert body["baseline_run_id"] == baseline_id
        assert body["candidate_run_id"] == candidate_id
        assert body["fixed"] == []
        assert body["persistent"] == []
        assert body["newly_introduced"][0]["sample_id"]
        assert "retrieval_miss" in body["newly_introduced"][0][
            "candidate_categories"
        ]

    async def test_get_missing_run_404(self, eval_env):
        resp = await eval_env["client"].get("/eval/runs/99999", headers=eval_env["headers"])
        assert resp.status_code == 404


class TestRegressionGateApi:
    async def test_create_list_and_read_persisted_verdict(
        self,
        eval_env,
        monkeypatch,
    ):
        doc_id = _seed_document(eval_env["SyncSession"], eval_env["user_id"])
        ds_id = _seed_dataset(
            eval_env["SyncSession"],
            eval_env["user_id"],
            doc_id,
            [[0]],
            label_source="public_ground_truth",
            release_eligible=True,
        )
        created = await eval_env["client"].post(
            "/eval/regression-gates",
            headers=eval_env["headers"],
            json={
                "dataset_id": ds_id,
                "name": "Hit@K release floor",
                "metric_name": "hit_rate",
                "metric_version": "retrieval_v1",
                "comparison": "absolute_min",
                "threshold": 0.9,
            },
        )
        assert created.status_code == 201
        assert created.json()["threshold"] == 0.9

        listed = await eval_env["client"].get(
            f"/eval/regression-gates?dataset_id={ds_id}",
            headers=eval_env["headers"],
        )
        assert listed.status_code == 200
        assert [gate["name"] for gate in listed.json()] == [
            "Hit@K release floor"
        ]

        monkeypatch.setattr(
            runner,
            "retrieve_for_skill",
            lambda **kwargs: [
                {
                    "content": "片段0",
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "reranker": "local",
                }
            ],
        )
        monkeypatch.setattr(
            runner,
            "chat_completion",
            lambda messages: _FakeMsg("答案"),
        )
        monkeypatch.setattr(
            runner,
            "evaluate_faithfulness",
            lambda context, answer: 1.0,
        )
        monkeypatch.setattr(
            runner,
            "evaluate_answer_relevancy",
            lambda question, answer: 1.0,
        )
        monkeypatch.setattr(
            runner,
            "evaluate_grounding",
            lambda answer, hits: {
                **validate_citations(answer, hits),
                "contract": "claim_grounding_v1",
                "semantic_entailment_checked": True,
                "judge_status": "completed",
                "judge_model": "test-judge",
                "groundedness": 0.0,
                "faithfulness": 0.0,
                "citation_correctness": 1.0,
                "conflict_count": 0,
                "refused": False,
            },
        )
        pending = await eval_env["client"].post(
            "/eval/runs",
            headers=eval_env["headers"],
            json={"dataset_id": ds_id},
        )
        with eval_env["SyncSession"]() as session:
            runner.run_evaluation(
                session,
                pending.json()["id"],
                eval_env["user_id"],
                doc_id,
            )

        verdicts = await eval_env["client"].get(
            f"/eval/runs/{pending.json()['id']}/regression",
            headers=eval_env["headers"],
        )
        assert verdicts.status_code == 200
        assert verdicts.json()[0]["passed"] is True
