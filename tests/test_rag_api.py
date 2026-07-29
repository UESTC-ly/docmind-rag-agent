"""RAG 问答路由集成测试（/chat/）。

mock 检索与生成边界：embed_query / search / chat_completion，
验证 HTTP 契约、会话创建、检索结果透传到 sources。
"""

import pytest

from app.services import rag_service
from app.services.evaluation.grounding import CANONICAL_REFUSAL
from app.services.evidence import validate_citations
from app.services.quality_adaptation import (
    AdaptiveRetrievalResult,
    RetrievalQuality,
)

pytestmark = pytest.mark.asyncio


class _FakeLLMMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


@pytest.fixture
def mock_rag(monkeypatch):
    """打桩 RAG 三段：向量化、检索、生成。"""
    hits = [
        {"score": 0.9, "content": "存储层用 PostgreSQL 和 Qdrant。",
         "document_id": 1, "chunk_index": 0},
    ]
    monkeypatch.setattr(rag_service, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(rag_service, "search", lambda *a, **k: hits)

    async def _policy(rows, **kwargs):
        return rows

    monkeypatch.setattr(rag_service, "apply_document_policy_async", _policy)
    monkeypatch.setattr(
        rag_service, "chat_completion",
        lambda messages: _FakeLLMMsg("存储层用了 PostgreSQL 和 Qdrant。"),
    )

    def _grounding(answer, rows):
        structural = validate_citations(answer, rows)
        groundedness = 1.0 if structural["passed"] else 0.0
        return {
            **structural,
            "contract": "claim_grounding_v1",
            "semantic_entailment_checked": bool(structural["claim_count"]),
            "judge_status": (
                "completed"
                if structural["claim_count"]
                else "not_applicable"
            ),
            "groundedness": (
                groundedness if structural["claim_count"] else None
            ),
            "faithfulness": (
                groundedness if structural["claim_count"] else None
            ),
            "citation_correctness": (
                structural["citation_precision"]
                if structural["claim_count"]
                else None
            ),
            "conflict_count": 0,
            "refused": CANONICAL_REFUSAL in answer,
        }

    monkeypatch.setattr(rag_service, "evaluate_grounding", _grounding)
    return hits


class TestChat:
    async def test_chat_returns_answer_and_sources(self, client, registered_user, mock_rag):
        resp = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "存储层用了什么？"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == CANONICAL_REFUSAL
        assert body["conversation_id"] > 0
        assert body["sources"][0]["document_id"] == 1
        assert body["sources"][0]["chunk_index"] == 0
        assert body["sources"][0]["citation_id"] == "D1:C0"
        assert body["citation_report"]["passed"] is True
        assert (
            body["citation_report"]["delivery_action"]
            == "refused_after_quality_gate"
        )
        assert (
            body["citation_report"]["initial_verification"][
                "unsupported_claim_count"
            ]
            == 1
        )

    async def test_chat_accepts_only_citations_from_current_hits(
        self, client, registered_user, mock_rag, monkeypatch
    ):
        monkeypatch.setattr(
            rag_service,
            "chat_completion",
            lambda messages: _FakeLLMMsg(
                "存储层用了 PostgreSQL 和 Qdrant。[D1:C0]"
            ),
        )
        resp = await client.post(
            "/chat/",
            headers=registered_user["headers"],
            json={"question": "存储层用了什么？"},
        )
        assert resp.status_code == 200
        report = resp.json()["citation_report"]
        assert report["passed"] is True
        assert report["citation_precision"] == 1.0
        assert report["citation_recall"] == 1.0

    async def test_chat_retrieves_once_more_when_initial_claim_is_unsupported(
        self, client, registered_user, mock_rag, monkeypatch
    ):
        retrieval_calls = []

        async def _retrieval(db, user_id, question, document_id, **kwargs):
            retrieval_calls.append(
                {
                    "question": question,
                    "pipeline": kwargs.get("pipeline"),
                    "top_k": kwargs.get("top_k"),
                }
            )
            return type(
                "Execution",
                (),
                {
                    "fingerprint": "test-fingerprint",
                    "hits": [
                        {
                            "score": 0.9,
                            "content": "存储层用 PostgreSQL 和 Qdrant。",
                            "document_id": 1,
                            "chunk_index": 0,
                            "citation_id": "D1:C0",
                            "local_rerank_score": 0.9,
                            "source_status": "current",
                        }
                    ],
                },
            )()

        calls = iter(
            [
                _FakeLLMMsg("存储层使用了未知数据库。[D1:C0]"),
                _FakeLLMMsg("存储层用 PostgreSQL 和 Qdrant。[D1:C0]"),
            ]
        )

        def _grounding(answer, rows):
            structural = validate_citations(answer, rows)
            unsupported = "未知数据库" in answer
            return {
                **structural,
                "contract": "claim_grounding_v1",
                "semantic_entailment_checked": True,
                "judge_status": "completed",
                "groundedness": 0.0 if unsupported else 1.0,
                "faithfulness": 0.0 if unsupported else 1.0,
                "citation_correctness": 1.0,
                "conflict_count": 0,
                "unsupported_claim_count": int(unsupported),
                "supported_claim_count": int(not unsupported),
                "unsupported_claim_rate": float(unsupported),
                "claims": [
                    {
                        **structural["claims"][0],
                        "semantically_supported": not unsupported,
                    }
                ],
                "passed": not unsupported,
            }

        monkeypatch.setattr(rag_service, "run_retrieval_pipeline", _retrieval)
        monkeypatch.setattr(
            rag_service,
            "_evaluated_pipeline_ids",
            lambda user_id: (),
        )
        monkeypatch.setattr(
            rag_service,
            "chat_completion",
            lambda messages: next(calls),
        )
        monkeypatch.setattr(rag_service, "evaluate_grounding", _grounding)
        monkeypatch.setattr(
            rag_service,
            "rewrite_retrieval_query",
            lambda query, reason: query,
        )

        response = await client.post(
            "/chat/",
            headers=registered_user["headers"],
            json={"question": "存储层用了什么？"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "存储层用 PostgreSQL 和 Qdrant。[D1:C0]"
        assert len(retrieval_calls) == 2
        assert body["citation_report"]["quality_interventions"][0]["action"] == (
            "re_retrieve_after_unsupported_claims"
        )

    async def test_chat_refuses_when_retrieval_repair_finds_no_evidence(
        self, client, registered_user, mock_rag, monkeypatch
    ):
        retrieval_calls = 0
        supported_hit = {
            "score": 0.9,
            "content": "存储层用 PostgreSQL 和 Qdrant。",
            "document_id": 1,
            "chunk_index": 0,
            "citation_id": "D1:C0",
            "local_rerank_score": 0.9,
            "source_status": "current",
        }

        async def _retrieval(db, user_id, question, document_id, **kwargs):
            nonlocal retrieval_calls
            retrieval_calls += 1
            return type(
                "Execution",
                (),
                {
                    "fingerprint": "test-fingerprint",
                    "hits": [supported_hit] if retrieval_calls == 1 else [],
                },
            )()

        def _grounding(answer, rows):
            structural = validate_citations(answer, rows)
            unsupported = "未知数据库" in answer
            return {
                **structural,
                "contract": "claim_grounding_v1",
                "semantic_entailment_checked": bool(structural["claim_count"]),
                "judge_status": (
                    "completed"
                    if structural["claim_count"]
                    else "not_applicable"
                ),
                "groundedness": 0.0 if unsupported else None,
                "faithfulness": 0.0 if unsupported else None,
                "citation_correctness": (
                    structural["citation_precision"]
                    if structural["claim_count"]
                    else None
                ),
                "conflict_count": 0,
                "unsupported_claim_count": int(unsupported),
                "supported_claim_count": 0,
                "unsupported_claim_rate": float(unsupported),
                "passed": not unsupported,
            }

        monkeypatch.setattr(rag_service, "run_retrieval_pipeline", _retrieval)
        monkeypatch.setattr(rag_service, "_evaluated_pipeline_ids", lambda user_id: ())
        monkeypatch.setattr(
            rag_service,
            "chat_completion",
            lambda messages: _FakeLLMMsg("存储层使用了未知数据库。[D1:C0]"),
        )
        monkeypatch.setattr(rag_service, "evaluate_grounding", _grounding)
        monkeypatch.setattr(
            rag_service,
            "rewrite_retrieval_query",
            lambda query, reason: query,
        )

        response = await client.post(
            "/chat/",
            headers=registered_user["headers"],
            json={"question": "存储层用了什么？"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == CANONICAL_REFUSAL
        assert retrieval_calls == 2
        assert body["citation_report"]["delivery_action"] == (
            "refused_due_to_insufficient_evidence"
        )
        interventions = body["citation_report"]["quality_interventions"]
        assert [row["action"] for row in interventions] == [
            "re_retrieve_after_unsupported_claims",
            "refuse_insufficient_evidence",
        ]

    @pytest.mark.parametrize(
        ("failure_mode", "expected_action", "expected_policy_action"),
        [
            (
                "conflict",
                "refused_due_to_conflicting_evidence",
                "refuse_conflicting_evidence",
            ),
            (
                "judge_unavailable",
                "refused_due_to_judge_unavailable",
                "refuse_judge_unavailable",
            ),
        ],
    )
    async def test_chat_refuses_when_grounding_cannot_authorize_delivery(
        self,
        client,
        registered_user,
        mock_rag,
        monkeypatch,
        failure_mode,
        expected_action,
        expected_policy_action,
    ):
        monkeypatch.setattr(
            rag_service,
            "chat_completion",
            lambda messages: _FakeLLMMsg(
                "存储层用 PostgreSQL 和 Qdrant。[D1:C0]"
            ),
        )

        def _grounding(answer, rows):
            structural = validate_citations(answer, rows)
            if answer == CANONICAL_REFUSAL:
                return {
                    **structural,
                    "contract": "claim_grounding_v1",
                    "semantic_entailment_checked": False,
                    "judge_status": "not_applicable",
                    "passed": True,
                    "refused": True,
                }
            judge_unavailable = failure_mode == "judge_unavailable"
            conflict = failure_mode == "conflict"
            return {
                **structural,
                "contract": "claim_grounding_v1",
                "semantic_entailment_checked": not judge_unavailable,
                "judge_status": (
                    "unavailable" if judge_unavailable else "completed"
                ),
                "judge_error": (
                    "grounding judge unavailable" if judge_unavailable else None
                ),
                "conflict_count": int(conflict),
                "conflicts": (
                    [{"citation_ids": ["D1:C0"], "reason": "版本冲突"}]
                    if conflict
                    else []
                ),
                "passed": False,
            }

        monkeypatch.setattr(rag_service, "evaluate_grounding", _grounding)

        response = await client.post(
            "/chat/",
            headers=registered_user["headers"],
            json={"question": "存储层用了什么？"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == CANONICAL_REFUSAL
        report = body["citation_report"]
        assert report["delivery_action"] == expected_action
        assert report["quality_interventions"][-1]["action"] == (
            expected_policy_action
        )

    @pytest.mark.parametrize(
        ("quality_status", "expected_action"),
        [
            ("no_evidence", "refused_due_to_insufficient_evidence"),
            (
                "stale_or_no_current_evidence",
                "refused_due_to_stale_or_no_current_evidence",
            ),
        ],
    )
    async def test_chat_refuses_before_generation_when_retrieval_is_terminal(
        self,
        client,
        registered_user,
        mock_rag,
        monkeypatch,
        quality_status,
        expected_action,
    ):
        retrieval = AdaptiveRetrievalResult(
            execution=object(),
            hits=[],
            query="当前有效制度是什么？",
            pipeline_id="configured",
            top_k=5,
            quality=RetrievalQuality(
                status=quality_status,
                hit_count=0,
                unique_chunk_count=0,
                stale_hit_count=int(
                    quality_status == "stale_or_no_current_evidence"
                ),
                best_local_rerank_score=None,
            ),
            quality_interventions=[
                {
                    "contract": "quality_adaptive_retrieval_v1",
                    "action": "refuse_insufficient_evidence",
                    "reason": quality_status,
                    "status": "terminal",
                }
            ],
            terminal_reason=quality_status,
        )

        async def _terminal_retrieval(*args, **kwargs):
            return retrieval

        def _unexpected_generation(*args, **kwargs):
            raise AssertionError("terminal retrieval must not call the model")

        monkeypatch.setattr(
            rag_service,
            "_adaptive_chat_retrieval",
            _terminal_retrieval,
        )
        monkeypatch.setattr(
            rag_service,
            "chat_completion",
            _unexpected_generation,
        )

        response = await client.post(
            "/chat/",
            headers=registered_user["headers"],
            json={"question": "当前有效制度是什么？"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == CANONICAL_REFUSAL
        assert body["sources"] == []
        assert body["citation_report"]["delivery_action"] == expected_action
        assert body["citation_report"]["initial_verification"][
            "quality_observation"
        ]["status"] == quality_status

    async def test_chat_creates_conversation(self, client, registered_user, mock_rag):
        resp = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "第一个问题"},
        )
        conv_id = resp.json()["conversation_id"]
        # 复用同一 conversation_id 应成功（会话存在）
        resp2 = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "追问", "conversation_id": conv_id},
        )
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == conv_id

    async def test_chat_unknown_conversation_404(self, client, registered_user, mock_rag):
        resp = await client.post(
            "/chat/", headers=registered_user["headers"],
            json={"question": "q", "conversation_id": 9999},
        )
        assert resp.status_code == 404

    async def test_chat_requires_auth(self, client, mock_rag):
        resp = await client.post("/chat/", json={"question": "q"})
        assert resp.status_code == 401


class TestConversationHistory:
    async def test_list_conversations(self, client, registered_user, mock_rag):
        # 先产生两个会话
        await client.post("/chat/", headers=registered_user["headers"], json={"question": "q1"})
        await client.post("/chat/", headers=registered_user["headers"], json={"question": "q2"})
        resp = await client.get("/chat/conversations", headers=registered_user["headers"])
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_get_history_returns_messages(self, client, registered_user, mock_rag):
        chat = await client.post(
            "/chat/", headers=registered_user["headers"], json={"question": "存储层？"}
        )
        conv_id = chat.json()["conversation_id"]
        resp = await client.get(
            f"/chat/conversations/{conv_id}", headers=registered_user["headers"]
        )
        assert resp.status_code == 200
        msgs = resp.json()
        # 一问一答两条消息
        assert len(msgs) == 2

    async def test_get_history_unknown_conversation_empty(self, client, registered_user, mock_rag):
        resp = await client.get(
            "/chat/conversations/9999", headers=registered_user["headers"]
        )
        assert resp.status_code == 200
        assert resp.json() == []
