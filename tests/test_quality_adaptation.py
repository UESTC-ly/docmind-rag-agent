from types import SimpleNamespace

import pytest

from app.services.quality_adaptation import (
    QualityPolicyConfig,
    adaptive_retrieve_async,
    adaptive_retrieve_sync,
    assess_retrieval_quality,
    delivery_quality_decision,
    evaluated_pipeline_allowlist,
)


def _execution(hits):
    return SimpleNamespace(hits=hits)


def _hit(
    *,
    document_id=1,
    chunk_index=0,
    score=0.8,
    source_status="current",
):
    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": "可核验的原文",
        "local_rerank_score": score,
        "source_status": source_status,
    }


def test_intervention_cap_stops_before_unbounded_retrieval():
    calls = []
    config = QualityPolicyConfig(
        max_interventions=2,
        min_evidence_hits=1,
        max_retrieval_top_k=8,
    )

    def retrieve(query, pipeline_id, top_k):
        calls.append((query, pipeline_id, top_k))
        return _execution([])

    result = adaptive_retrieve_sync(
        retrieve=retrieve,
        query="原问题",
        pipeline_id="configured",
        top_k=2,
        rewrite_query=lambda query, reason: "改写后的问题",
        allowed_pipeline_ids=("approved-pipeline",),
        config=config,
    )

    applied = [
        row for row in result.quality_interventions
        if row["status"] == "applied"
    ]
    assert [row["action"] for row in applied] == [
        "rewrite_query",
        "expand_retrieval",
    ]
    assert result.quality_interventions[-1]["action"] == (
        "refuse_insufficient_evidence"
    )
    assert result.terminal_reason == "no_evidence"
    assert len(calls) == 3  # initial retrieval plus two bounded interventions


def test_only_evaluated_allowlisted_pipeline_can_be_switched_to():
    calls = []
    config = QualityPolicyConfig(
        max_interventions=3,
        min_evidence_hits=1,
        max_retrieval_top_k=1,
    )

    def retrieve(query, pipeline_id, top_k):
        calls.append(pipeline_id)
        return _execution([_hit()] if pipeline_id == "approved" else [])

    result = adaptive_retrieve_sync(
        retrieve=retrieve,
        query="问题",
        pipeline_id="configured",
        top_k=1,
        rewrite_query=lambda query, reason: "改写问题",
        allowed_pipeline_ids=("approved",),
        config=config,
    )

    assert result.deliverable is True
    assert calls == ["configured", "configured", "approved"]
    switch = next(
        row
        for row in result.quality_interventions
        if row["action"] == "switch_pipeline"
    )
    assert switch["to_pipeline_id"] == "approved"
    assert evaluated_pipeline_allowlist(
        {
            "status": "selected",
            "candidates": [
                {"pipeline_id": "approved", "release_status": "approved"},
                {"pipeline_id": "blocked", "release_status": "blocked"},
                {
                    "pipeline_id": "observational",
                    "release_status": "observational",
                },
            ],
        }
    ) == ("approved", "observational")


def test_stale_and_empty_evidence_are_terminal_and_never_rewritten():
    config = QualityPolicyConfig(max_interventions=3)
    rewrites = []

    def stale_retrieve(query, pipeline_id, top_k):
        return _execution([_hit(source_status="expired")])

    stale = adaptive_retrieve_sync(
        retrieve=stale_retrieve,
        query="当前制度是什么",
        pipeline_id="configured",
        top_k=1,
        rewrite_query=lambda query, reason: rewrites.append((query, reason)) or query,
        config=config,
    )
    assert stale.terminal_reason == "stale_or_no_current_evidence"
    assert stale.hits == []
    assert rewrites == []
    assert stale.quality_interventions[-1]["action"] == "refuse_stale_evidence"

    empty = adaptive_retrieve_sync(
        retrieve=lambda *args: _execution([]),
        query="不存在的问题",
        pipeline_id="configured",
        top_k=1,
        rewrite_query=lambda query, reason: query,
        config=QualityPolicyConfig(max_interventions=0),
    )
    assert empty.terminal_reason == "no_evidence"
    assert empty.quality_interventions[-1]["action"] == (
        "refuse_insufficient_evidence"
    )


def test_weak_local_rerank_score_drives_recovery():
    quality = assess_retrieval_quality(
        [_hit(score=0.01)],
        config=QualityPolicyConfig(min_local_rerank_score=0.15),
    )
    assert quality.status == "weak_retrieval"


def test_delivery_policy_retries_unsupported_once_then_fails_closed():
    report = {
        "judge_status": "completed",
        "semantic_entailment_checked": True,
        "unsupported_claim_count": 1,
        "conflict_count": 0,
    }
    retry = delivery_quality_decision(
        report,
        intervention_count=1,
        config=QualityPolicyConfig(max_interventions=2),
    )
    exhausted = delivery_quality_decision(
        report,
        intervention_count=2,
        config=QualityPolicyConfig(max_interventions=2),
    )
    assert retry["action"] == "re_retrieve_after_unsupported_claims"
    assert retry["consumes_budget"] is True
    assert exhausted["action"] == "remove_unsupported_or_refuse"
    assert exhausted["terminal"] is True


def test_delivery_policy_refuses_conflicts_and_unavailable_judges():
    conflict = delivery_quality_decision(
        {
            "judge_status": "completed",
            "semantic_entailment_checked": True,
            "unsupported_claim_count": 0,
            "conflict_count": 1,
        },
        intervention_count=0,
    )
    unavailable = delivery_quality_decision(
        {
            "judge_status": "unavailable",
            "semantic_entailment_checked": False,
            "unsupported_claim_count": 0,
            "conflict_count": 0,
        },
        intervention_count=0,
    )
    assert conflict["action"] == "refuse_conflicting_evidence"
    assert unavailable["action"] == "refuse_judge_unavailable"


@pytest.mark.asyncio
async def test_async_policy_keeps_the_same_intervention_boundaries():
    calls = []

    async def retrieve(query, pipeline_id, top_k):
        calls.append((query, pipeline_id, top_k))
        return _execution([_hit()] if pipeline_id == "approved" else [])

    async def rewrite(query, reason):
        return "改写问题"

    result = await adaptive_retrieve_async(
        retrieve=retrieve,
        query="问题",
        pipeline_id="configured",
        top_k=1,
        rewrite_query=rewrite,
        allowed_pipeline_ids=("approved",),
        config=QualityPolicyConfig(
            max_interventions=3,
            max_retrieval_top_k=1,
        ),
    )

    assert result.deliverable is True
    assert [row["action"] for row in result.quality_interventions] == [
        "rewrite_query",
        "switch_pipeline",
    ]
    assert calls[-1][1] == "approved"
