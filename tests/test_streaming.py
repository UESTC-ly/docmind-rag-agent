"""流式问答 SSE 端点测试（/chat/stream）。

mock 检索与流式生成，验证 SSE 事件序列 meta → token* → done，
以及答案在流结束后落库。
"""

import pytest

from app.services import rag_service
from app.services.evidence import validate_citations

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_stream(monkeypatch):
    """打桩检索（返回一个来源）与流式生成（逐 token）。"""
    hits = [
        {
            "content": "存储层用 Qdrant",
            "document_id": 1,
            "chunk_index": 0,
            "score": 0.9,
            "source_status": "current",
        }
    ]

    async def _retrieval(*args, **kwargs):
        return type(
            "Execution",
            (),
            {
                "fingerprint": "streaming-test-fingerprint",
                "hits": hits,
            },
        )()

    monkeypatch.setattr(
        rag_service,
        "run_retrieval_pipeline",
        _retrieval,
    )
    monkeypatch.setattr(rag_service, "_evaluated_pipeline_ids", lambda user_id: ())
    monkeypatch.setattr(
        rag_service, "chat_completion_stream",
        lambda messages, temperature=0.3: iter(
            ["存储层用", "Qdrant。", "[D1:C0]"]
        ),
    )

    def _grounding(answer, hits):
        report = validate_citations(answer, hits)
        groundedness = 1.0 if report["passed"] else 0.0
        return {
            **report,
            "contract": "claim_grounding_v1",
            "semantic_entailment_checked": True,
            "judge_status": "completed",
            "groundedness": groundedness,
            "faithfulness": groundedness,
            "citation_correctness": report["citation_precision"],
            "conflict_count": 0,
            "refused": False,
        }

    monkeypatch.setattr(rag_service, "evaluate_grounding", _grounding)


def _parse_sse(text):
    """把 SSE 原始文本解析成 [(event, data_str), ...]。"""
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        events.append((ev, data))
    return events


class TestChatStream:
    async def test_sse_event_sequence(self, client, registered_user, mock_stream):
        resp = await client.post(
            "/chat/stream", headers=registered_user["headers"],
            json={"question": "存储层用什么？"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)
        names = [e[0] for e in events]
        # meta 开头，done 结尾；token 后必须给出引用校验。
        assert names[0] == "meta"
        assert names[-1] == "done"
        assert names[-2] == "verification"
        # Candidate chunks are buffered; only verified output is emitted.
        assert names.count("token") == 1

    async def test_meta_carries_conversation_and_sources(self, client, registered_user, mock_stream):
        resp = await client.post(
            "/chat/stream", headers=registered_user["headers"],
            json={"question": "q"},
        )
        events = _parse_sse(resp.text)
        import json
        meta = json.loads(events[0][1])
        assert meta["conversation_id"] > 0
        assert meta["sources"][0]["document_id"] == 1

    async def test_tokens_reassemble_to_answer(self, client, registered_user, mock_stream):
        resp = await client.post(
            "/chat/stream", headers=registered_user["headers"],
            json={"question": "q"},
        )
        import json
        events = _parse_sse(resp.text)
        tokens = [json.loads(d)["text"] for e, d in events if e == "token"]
        assert "".join(tokens) == "存储层用Qdrant。[D1:C0]"

    async def test_verification_event_reports_grounded_claim(
        self, client, registered_user, mock_stream
    ):
        resp = await client.post(
            "/chat/stream",
            headers=registered_user["headers"],
            json={"question": "q"},
        )
        import json

        events = _parse_sse(resp.text)
        report = json.loads(
            next(data for event, data in events if event == "verification")
        )
        assert report["passed"] is True
        assert report["groundedness"] == 1.0
        assert report["delivery_action"] == "accepted"

    async def test_answer_persisted_after_stream(self, client, registered_user, mock_stream):
        stream_resp = await client.post(
            "/chat/stream", headers=registered_user["headers"],
            json={"question": "存储层？"},
        )
        import json
        conv_id = json.loads(_parse_sse(stream_resp.text)[0][1])["conversation_id"]
        # 流结束后，历史里应有一问一答
        hist = await client.get(
            f"/chat/conversations/{conv_id}", headers=registered_user["headers"]
        )
        msgs = hist.json()
        assert len(msgs) == 2
        assert msgs[1]["content"] == "存储层用Qdrant。[D1:C0]"

    async def test_stream_requires_auth(self, client, mock_stream):
        resp = await client.post("/chat/stream", json={"question": "q"})
        assert resp.status_code == 401
