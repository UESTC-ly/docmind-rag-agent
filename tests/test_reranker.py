"""Second-stage reranker ordering, bounds, and provider containment tests."""

import json
from urllib.error import URLError

import pytest

from app.services import reranker


def _hit(index: int, content: str, *, rrf: float = 0.01, dense: float = 0.5):
    return {
        "document_id": 1,
        "chunk_index": index,
        "content": content,
        "score": dense,
        "dense_score": dense,
        "keyword_score": 0.0,
        "rrf_score": rrf,
        "fused_rank": index + 1,
        "retrieval_sources": ["dense"],
    }


def test_local_reranker_uses_query_overlap(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_mode", "local")
    hits = reranker.rerank(
        "精确 型号",
        [_hit(0, "普通说明"), _hit(1, "精确 型号 的技术说明")],
        top_k=2,
    )
    assert hits[0]["chunk_index"] == 1
    assert hits[0]["reranker"] == "local"
    assert hits[0]["score"] == hits[0]["rerank_score"]
    assert hits[0]["dense_score"] == 0.5


def test_ties_preserve_fused_order(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_mode", "local")
    hits = reranker.rerank(
        "zzz",
        [_hit(0, "a"), _hit(1, "b")],
        top_k=2,
    )
    assert [hit["chunk_index"] for hit in hits] == [0, 1]


def test_candidate_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_mode", "local")
    monkeypatch.setattr(reranker.settings, "reranker_candidate_limit", 3)
    hits = reranker.rerank("q", [_hit(i, f"q {i}") for i in range(10)], top_k=9)
    assert len(hits) == 3


def test_http_provider_scores_preserve_local_diagnostics(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_mode", "http")
    monkeypatch.setattr(
        reranker,
        "_http_scores",
        lambda query, candidates: {0: 0.1, 1: 0.9},
    )
    hits = reranker.rerank("q", [_hit(0, "a"), _hit(1, "b")], top_k=2)
    assert [hit["chunk_index"] for hit in hits] == [1, 0]
    assert hits[0]["reranker"] == "http"
    assert "local_rerank_score" in hits[0]


def test_http_failure_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_mode", "http")
    monkeypatch.setattr(
        reranker,
        "http_rerank",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    hits = reranker.rerank("q", [_hit(0, "q")], top_k=1)
    assert hits[0]["reranker"] == "local_fallback"
    assert "URLError" in hits[0]["reranker_error"]


def test_http_request_contract_and_response_parsing(monkeypatch):
    captured = {}
    monkeypatch.setattr(reranker.settings, "reranker_http_url", "https://rank.test/v1")
    monkeypatch.setattr(reranker.settings, "reranker_http_api_key", "secret")
    monkeypatch.setattr(reranker.settings, "reranker_http_model", "cross-encoder")
    monkeypatch.setattr(reranker.settings, "reranker_http_timeout_seconds", 4)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.25},
                        {"index": 1, "score": 0.75},
                    ]
                }
            ).encode()

    def _urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return _Response()

    monkeypatch.setattr(reranker, "urlopen", _urlopen)
    candidates = [_hit(0, "a"), _hit(1, "b")]
    scores = reranker._http_scores("query", candidates)

    assert scores == {0: 0.25, 1: 0.75}
    assert captured["timeout"] == 4
    assert captured["request"].headers["Authorization"] == "Bearer secret"
    payload = json.loads(captured["request"].data)
    assert payload == {
        "query": "query",
        "documents": ["a", "b"],
        "top_n": 2,
        "model": "cross-encoder",
    }


def test_http_response_must_score_every_candidate(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_http_url", "https://rank.test/v1")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"results":[{"index":0,"score":0.5}]}'

    monkeypatch.setattr(reranker, "urlopen", lambda *args, **kwargs: _Response())
    with pytest.raises(ValueError, match="every candidate"):
        reranker._http_scores("q", [_hit(0, "a"), _hit(1, "b")])


def test_missing_http_url_and_unknown_mode_are_contained(monkeypatch):
    monkeypatch.setattr(reranker.settings, "reranker_http_url", None)
    with pytest.raises(ValueError, match="RERANKER_HTTP_URL"):
        reranker._http_scores("q", [_hit(0, "a")])

    monkeypatch.setattr(reranker.settings, "reranker_mode", "future-mode")
    hits = reranker.rerank("q", [_hit(0, "q")], top_k=1)
    assert hits[0]["reranker"] == "local"
