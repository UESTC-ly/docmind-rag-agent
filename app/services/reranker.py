"""Bounded second-stage reranking for every DocMind retrieval surface.

The default reranker is deliberately deterministic and dependency-free: it combines
RRF, dense similarity, database keyword rank, and direct query/chunk token overlap.
An optional HTTP cross-encoder can replace the final ordering.  Provider failures are
contained and fall back to the local scorer so document Q&A remains available.
"""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import settings
from app.utils.logging import logger


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _lexical_score(query: str, content: str) -> float:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0.0
    content_lower = content.lower()
    content_tokens = set(_tokens(content_lower))
    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    phrase_bonus = 0.15 if query.strip().lower() in content_lower else 0.0
    return min(1.0, overlap + phrase_bonus)


def _stable_order(hits: list[dict], score_field: str) -> list[dict]:
    return sorted(
        hits,
        key=lambda hit: (
            -float(hit.get(score_field, 0.0) or 0.0),
            int(hit.get("fused_rank", 10**9)),
            int(hit.get("document_id", 0)),
            int(hit.get("chunk_index", 0)),
        ),
    )


def local_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Score candidates with deterministic local signals and stable tie-breaking."""
    if not candidates:
        return []

    rrf = _normalize([float(hit.get("rrf_score", 0.0) or 0.0) for hit in candidates])
    dense = _normalize(
        [float(hit.get("dense_score", 0.0) or 0.0) for hit in candidates]
    )
    keyword = _normalize(
        [float(hit.get("keyword_score", 0.0) or 0.0) for hit in candidates]
    )

    scored: list[dict] = []
    for index, hit in enumerate(candidates):
        lexical = _lexical_score(query, str(hit.get("content", "")))
        final = (
            settings.reranker_rrf_weight * rrf[index]
            + settings.reranker_dense_weight * dense[index]
            + settings.reranker_keyword_weight * keyword[index]
            + settings.reranker_lexical_weight * lexical
        )
        enriched = dict(hit)
        enriched.update(
            lexical_score=round(lexical, 8),
            local_rerank_score=round(final, 8),
            rerank_score=round(final, 8),
            reranker="local",
        )
        scored.append(enriched)
    return _stable_order(scored, "rerank_score")


def _http_scores(query: str, candidates: list[dict]) -> dict[int, float]:
    if not settings.reranker_http_url:
        raise ValueError("RERANKER_HTTP_URL is required when reranker_mode=http")

    payload: dict = {
        "query": query,
        "documents": [str(hit.get("content", "")) for hit in candidates],
        "top_n": len(candidates),
    }
    if settings.reranker_http_model:
        payload["model"] = settings.reranker_http_model

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.reranker_http_api_key:
        headers["Authorization"] = f"Bearer {settings.reranker_http_api_key}"

    request = Request(
        settings.reranker_http_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=settings.reranker_http_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))

    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        raise ValueError("reranker response must contain a results array")

    scores: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if isinstance(index, int) and 0 <= index < len(candidates) and isinstance(
            score, (int, float)
        ):
            scores[index] = float(score)
    if len(scores) != len(candidates):
        raise ValueError("reranker response did not score every candidate")
    return scores


def http_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Use a configured cross-encoder API while retaining every original signal."""
    local = local_rerank(query, candidates)
    # The provider indexes the original fused candidate order, not local order.
    local_by_key = {
        (hit["document_id"], hit["chunk_index"]): hit for hit in local
    }
    scores = _http_scores(query, candidates)
    reranked: list[dict] = []
    for index, candidate in enumerate(candidates):
        key = (candidate["document_id"], candidate["chunk_index"])
        enriched = dict(local_by_key[key])
        enriched.update(rerank_score=scores[index], reranker="http")
        reranked.append(enriched)
    return _stable_order(reranked, "rerank_score")


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank a bounded candidate set and annotate the reproducible final order."""
    bounded = [dict(hit) for hit in candidates[: settings.reranker_candidate_limit]]
    mode = settings.reranker_mode.lower().strip()

    if mode == "off":
        ranked = _stable_order(bounded, "rrf_score")
        for hit in ranked:
            hit["rerank_score"] = float(hit.get("rrf_score", 0.0) or 0.0)
            hit["reranker"] = "off"
    elif mode == "http":
        try:
            ranked = http_rerank(query, bounded)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.bind(error=type(exc).__name__).warning(
                "HTTP reranker unavailable; using deterministic local fallback"
            )
            ranked = local_rerank(query, bounded)
            for hit in ranked:
                hit["reranker"] = "local_fallback"
                hit["reranker_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    else:
        if mode != "local":
            logger.bind(mode=mode).warning("unknown reranker mode; using local")
        ranked = local_rerank(query, bounded)

    output = ranked[:top_k]
    for final_rank, hit in enumerate(output, start=1):
        hit["final_rank"] = final_rank
        # `score` is the public Source score; raw dense similarity remains available
        # as `dense_score` for diagnostics and evaluation reproduction.
        hit["score"] = float(hit.get("rerank_score", 0.0) or 0.0)
    return output
