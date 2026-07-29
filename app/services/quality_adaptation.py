"""Bounded quality-driven retrieval adaptation for Agentic RAG.

The policy owns control-flow only. Retrieval I/O and pipeline selection are
injected by callers so browser chat, synchronous Agent Skills, and offline
tests share the same bounded behavior without introducing another RAG stack.
Every emitted intervention is deliberately compact: it contains no query,
retrieved text, or model output and is safe to retain in an Agent audit trace.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.llm_service import chat_completion

QUALITY_ADAPTATION_CONTRACT = "quality_adaptive_retrieval_v1"
_STALE_SOURCE_STATUSES = {
    "expired",
    "superseded",
    "not_yet_effective",
}

_REWRITE_PROMPT = """将下面的问题改写为更适合在文档知识库中检索的短查询。
不得改变问题含义，不得补充外部事实；只输出查询本身，不要解释。

问题：{query}
触发原因：{reason}
"""


@dataclass(frozen=True)
class QualityPolicyConfig:
    """Limits that make automatic quality recovery observable and bounded."""

    max_interventions: int = 4
    min_evidence_hits: int = 1
    min_local_rerank_score: float = 0.15
    max_retrieval_top_k: int = 20

    @classmethod
    def from_settings(cls) -> "QualityPolicyConfig":
        return cls(
            max_interventions=settings.quality_max_interventions,
            min_evidence_hits=settings.quality_min_evidence_hits,
            min_local_rerank_score=settings.quality_min_local_rerank_score,
            max_retrieval_top_k=settings.quality_max_retrieval_top_k,
        )


@dataclass(frozen=True)
class RetrievalQuality:
    """A deterministic observation, not a claim that evidence is relevant."""

    status: str
    hit_count: int
    unique_chunk_count: int
    stale_hit_count: int
    best_local_rerank_score: float | None

    def compact(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hit_count": self.hit_count,
            "unique_chunk_count": self.unique_chunk_count,
            "stale_hit_count": self.stale_hit_count,
            "best_local_rerank_score": self.best_local_rerank_score,
        }


@dataclass
class AdaptiveRetrievalResult:
    """The final retrieval attempt and the policy trace that produced it."""

    execution: Any
    hits: list[dict[str, Any]]
    query: str
    pipeline_id: str
    top_k: int
    quality: RetrievalQuality
    quality_interventions: list[dict[str, Any]] = field(default_factory=list)
    terminal_reason: str | None = None
    intervention_count: int = 0

    @property
    def deliverable(self) -> bool:
        return self.terminal_reason is None and self.quality.status == "sufficient"


def assess_retrieval_quality(
    hits: Sequence[Mapping[str, Any]],
    *,
    config: QualityPolicyConfig | None = None,
) -> RetrievalQuality:
    """Classify evidence availability using only stable retrieval metadata.

    Raw reranker scores are not comparable across pipeline types.  A low-score
    observation is therefore used only when the deterministic local reranker
    explicitly produced its normalized ``local_rerank_score`` field.
    """

    policy = config or QualityPolicyConfig.from_settings()
    meaningful = [
        hit
        for hit in hits
        if str(hit.get("content") or "").strip()
    ]
    stale = [
        hit
        for hit in meaningful
        if str(hit.get("source_status") or "unknown").lower()
        in _STALE_SOURCE_STATUSES
    ]
    eligible = [
        hit
        for hit in meaningful
        if str(hit.get("source_status") or "unknown").lower()
        not in _STALE_SOURCE_STATUSES
    ]
    unique_chunks = {
        (
            str(hit.get("document_id") or ""),
            str(hit.get("chunk_index") or ""),
        )
        for hit in eligible
    }
    local_scores = [
        float(hit["local_rerank_score"])
        for hit in eligible
        if isinstance(hit.get("local_rerank_score"), (int, float))
    ]
    best_score = round(max(local_scores), 6) if local_scores else None

    if stale and not eligible:
        status = "stale_or_no_current_evidence"
    elif not eligible:
        status = "no_evidence"
    elif len(unique_chunks) < policy.min_evidence_hits:
        status = "weak_retrieval"
    elif (
        best_score is not None
        and best_score < policy.min_local_rerank_score
    ):
        status = "weak_retrieval"
    else:
        status = "sufficient"
    return RetrievalQuality(
        status=status,
        hit_count=len(eligible),
        unique_chunk_count=len(unique_chunks),
        stale_hit_count=len(stale),
        best_local_rerank_score=best_score,
    )


def evaluated_pipeline_allowlist(
    decision: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Extract only current, evaluated, non-blocked candidate pipeline IDs."""

    if not isinstance(decision, Mapping) or decision.get("status") != "selected":
        return ()
    rows = decision.get("candidates")
    if not isinstance(rows, list):
        selected = decision.get("selected")
        rows = [selected] if isinstance(selected, Mapping) else []
    allowed: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pipeline_id = str(row.get("pipeline_id") or "").strip()
        release_status = str(row.get("release_status") or "").strip().lower()
        if (
            pipeline_id
            and release_status in {"approved", "observational"}
            and pipeline_id not in allowed
        ):
            allowed.append(pipeline_id)
    return tuple(allowed)


def rewrite_retrieval_query(query: str, reason: str) -> str:
    """Use the configured LLM for one constrained retrieval-query rewrite."""

    original = str(query or "").strip()
    if not original:
        return original
    try:
        response = chat_completion(
            [
                {
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(
                        query=original,
                        reason=str(reason or "weak_retrieval")[:80],
                    ),
                }
            ],
            temperature=0.0,
        )
        rewritten = " ".join(str(response.content or "").split())
    except Exception:  # noqa: BLE001 - policy must preserve its bounded fallback
        return original
    if not rewritten or len(rewritten) > 500:
        return original
    return rewritten


def delivery_quality_decision(
    verification: Mapping[str, Any],
    *,
    intervention_count: int,
    config: QualityPolicyConfig | None = None,
    verified_delivery: bool = True,
) -> dict[str, Any]:
    """Choose the next safe action after claim-level grounding verification."""

    policy = config or QualityPolicyConfig.from_settings()
    judge_status = str(verification.get("judge_status") or "")
    if verified_delivery and (
        judge_status == "unavailable"
        or verification.get("semantic_entailment_checked") is False
    ):
        return {
            "action": "refuse_judge_unavailable",
            "reason": "judge_unavailable",
            "terminal": True,
            "consumes_budget": False,
        }
    if int(verification.get("conflict_count") or 0) > 0:
        return {
            "action": "refuse_conflicting_evidence",
            "reason": "conflicting_evidence",
            "terminal": True,
            "consumes_budget": False,
        }
    if (
        verification.get("semantic_entailment_checked")
        and int(verification.get("unsupported_claim_count") or 0) > 0
    ):
        if intervention_count < policy.max_interventions:
            return {
                "action": "re_retrieve_after_unsupported_claims",
                "reason": "unsupported_claims",
                "terminal": False,
                "consumes_budget": True,
            }
        return {
            "action": "remove_unsupported_or_refuse",
            "reason": "unsupported_claims_budget_exhausted",
            "terminal": True,
            "consumes_budget": False,
        }
    return {
        "action": "accept",
        "reason": "quality_gate_passed",
        "terminal": False,
        "consumes_budget": False,
    }


def delivery_intervention_record(
    decision: Mapping[str, Any],
    *,
    intervention_count: int,
    config: QualityPolicyConfig | None = None,
    before_hit_count: int,
    after_hit_count: int | None = None,
) -> dict[str, Any]:
    """Create a compact audit record for a post-generation policy decision."""

    policy = config or QualityPolicyConfig.from_settings()
    consumes_budget = bool(decision.get("consumes_budget"))
    return {
        "contract": QUALITY_ADAPTATION_CONTRACT,
        "action": str(decision.get("action") or "unknown"),
        "reason": str(decision.get("reason") or "unknown"),
        "status": "applied" if consumes_budget else "terminal",
        "attempt": intervention_count + int(consumes_budget),
        "max_interventions": policy.max_interventions,
        "before_hit_count": before_hit_count,
        "after_hit_count": after_hit_count,
    }


class _AdaptiveController:
    def __init__(
        self,
        *,
        query: str,
        pipeline_id: str,
        top_k: int,
        allowed_pipeline_ids: Sequence[str],
        config: QualityPolicyConfig,
    ) -> None:
        self.query = query
        self.pipeline_id = pipeline_id
        self.top_k = min(max(1, top_k), config.max_retrieval_top_k)
        self.allowed_pipeline_ids = tuple(
            pipeline
            for pipeline in dict.fromkeys(allowed_pipeline_ids)
            if pipeline and pipeline != pipeline_id
        )
        self.config = config
        self.interventions: list[dict[str, Any]] = []
        self.intervention_count = 0
        self.attempted_actions: set[str] = set()
        self.attempted_pipelines: set[str] = {pipeline_id}

    def next_action(
        self, quality: RetrievalQuality
    ) -> tuple[str, str | None]:
        if quality.status == "sufficient":
            return "accept", None
        if quality.status == "stale_or_no_current_evidence":
            return "refuse_stale_evidence", None
        if self.intervention_count >= self.config.max_interventions:
            return "refuse_insufficient_evidence", None
        if "rewrite_query" not in self.attempted_actions:
            return "rewrite_query", None
        if (
            "expand_retrieval" not in self.attempted_actions
            and self.top_k < self.config.max_retrieval_top_k
        ):
            return "expand_retrieval", None
        for pipeline_id in self.allowed_pipeline_ids:
            if pipeline_id not in self.attempted_pipelines:
                return "switch_pipeline", pipeline_id
        return "refuse_insufficient_evidence", None

    def begin(
        self,
        action: str,
        quality: RetrievalQuality,
        *,
        target_pipeline_id: str | None = None,
    ) -> dict[str, Any]:
        self.intervention_count += 1
        self.attempted_actions.add(action)
        event = {
            "contract": QUALITY_ADAPTATION_CONTRACT,
            "action": action,
            "reason": quality.status,
            "status": "applied",
            "attempt": self.intervention_count,
            "max_interventions": self.config.max_interventions,
            "before_hit_count": quality.hit_count,
            "after_hit_count": None,
            "from_pipeline_id": self.pipeline_id,
            "to_pipeline_id": target_pipeline_id or self.pipeline_id,
            "quality_before": quality.compact(),
        }
        self.interventions.append(event)
        return event

    def terminal(
        self,
        action: str,
        quality: RetrievalQuality,
    ) -> None:
        self.interventions.append(
            {
                "contract": QUALITY_ADAPTATION_CONTRACT,
                "action": action,
                "reason": quality.status,
                "status": "terminal",
                "attempt": self.intervention_count,
                "max_interventions": self.config.max_interventions,
                "before_hit_count": quality.hit_count,
                "after_hit_count": quality.hit_count,
                "from_pipeline_id": self.pipeline_id,
                "to_pipeline_id": self.pipeline_id,
                "quality_before": quality.compact(),
            }
        )


def _finish(
    controller: _AdaptiveController,
    execution: Any,
    quality: RetrievalQuality,
    *,
    terminal_reason: str | None = None,
) -> AdaptiveRetrievalResult:
    hits = list(getattr(execution, "hits", []) or [])
    return AdaptiveRetrievalResult(
        execution=execution,
        hits=hits if terminal_reason is None else [],
        query=controller.query,
        pipeline_id=controller.pipeline_id,
        top_k=controller.top_k,
        quality=quality,
        quality_interventions=list(controller.interventions),
        terminal_reason=terminal_reason,
        intervention_count=controller.intervention_count,
    )


def adaptive_retrieve_sync(
    *,
    retrieve: Callable[[str, str, int], Any],
    query: str,
    pipeline_id: str,
    top_k: int,
    rewrite_query: Callable[[str, str], str] = rewrite_retrieval_query,
    allowed_pipeline_ids: Sequence[str] = (),
    config: QualityPolicyConfig | None = None,
    initial_execution: Any | None = None,
) -> AdaptiveRetrievalResult:
    """Execute bounded adaptive retrieval in synchronous Agent Skill contexts."""

    policy = config or QualityPolicyConfig.from_settings()
    controller = _AdaptiveController(
        query=query,
        pipeline_id=pipeline_id,
        top_k=top_k,
        allowed_pipeline_ids=allowed_pipeline_ids,
        config=policy,
    )
    execution = initial_execution or retrieve(
        controller.query,
        controller.pipeline_id,
        controller.top_k,
    )
    while True:
        quality = assess_retrieval_quality(execution.hits, config=policy)
        action, target_pipeline_id = controller.next_action(quality)
        if action == "accept":
            return _finish(controller, execution, quality)
        if action.startswith("refuse_"):
            controller.terminal(action, quality)
            return _finish(
                controller,
                execution,
                quality,
                terminal_reason=quality.status,
            )

        event = controller.begin(
            action,
            quality,
            target_pipeline_id=target_pipeline_id,
        )
        if action == "rewrite_query":
            rewritten = rewrite_query(controller.query, quality.status).strip()
            event["query_changed"] = rewritten != controller.query
            if not event["query_changed"]:
                event["status"] = "skipped"
                event["after_hit_count"] = quality.hit_count
                event["quality_after"] = quality.compact()
                continue
            controller.query = rewritten
        elif action == "expand_retrieval":
            controller.top_k = min(
                controller.top_k * 2,
                controller.config.max_retrieval_top_k,
            )
        elif action == "switch_pipeline":
            controller.pipeline_id = str(target_pipeline_id)
            controller.attempted_pipelines.add(controller.pipeline_id)

        execution = retrieve(
            controller.query,
            controller.pipeline_id,
            controller.top_k,
        )
        after = assess_retrieval_quality(execution.hits, config=policy)
        event["after_hit_count"] = after.hit_count
        event["quality_after"] = after.compact()


async def adaptive_retrieve_async(
    *,
    retrieve: Callable[[str, str, int], Awaitable[Any]],
    query: str,
    pipeline_id: str,
    top_k: int,
    rewrite_query: Callable[[str, str], Awaitable[str]],
    allowed_pipeline_ids: Sequence[str] = (),
    config: QualityPolicyConfig | None = None,
    initial_execution: Any | None = None,
) -> AdaptiveRetrievalResult:
    """Async counterpart used by the browser chat retrieval path."""

    policy = config or QualityPolicyConfig.from_settings()
    controller = _AdaptiveController(
        query=query,
        pipeline_id=pipeline_id,
        top_k=top_k,
        allowed_pipeline_ids=allowed_pipeline_ids,
        config=policy,
    )
    execution = initial_execution or await retrieve(
        controller.query,
        controller.pipeline_id,
        controller.top_k,
    )
    while True:
        quality = assess_retrieval_quality(execution.hits, config=policy)
        action, target_pipeline_id = controller.next_action(quality)
        if action == "accept":
            return _finish(controller, execution, quality)
        if action.startswith("refuse_"):
            controller.terminal(action, quality)
            return _finish(
                controller,
                execution,
                quality,
                terminal_reason=quality.status,
            )

        event = controller.begin(
            action,
            quality,
            target_pipeline_id=target_pipeline_id,
        )
        if action == "rewrite_query":
            rewritten = (await rewrite_query(controller.query, quality.status)).strip()
            event["query_changed"] = rewritten != controller.query
            if not event["query_changed"]:
                event["status"] = "skipped"
                event["after_hit_count"] = quality.hit_count
                event["quality_after"] = quality.compact()
                continue
            controller.query = rewritten
        elif action == "expand_retrieval":
            controller.top_k = min(
                controller.top_k * 2,
                controller.config.max_retrieval_top_k,
            )
        elif action == "switch_pipeline":
            controller.pipeline_id = str(target_pipeline_id)
            controller.attempted_pipelines.add(controller.pipeline_id)

        execution = await retrieve(
            controller.query,
            controller.pipeline_id,
            controller.top_k,
        )
        after = assess_retrieval_quality(execution.hits, config=policy)
        event["after_hit_count"] = after.hit_count
        event["quality_after"] = after.compact()


__all__ = [
    "AdaptiveRetrievalResult",
    "QUALITY_ADAPTATION_CONTRACT",
    "QualityPolicyConfig",
    "RetrievalQuality",
    "adaptive_retrieve_async",
    "adaptive_retrieve_sync",
    "assess_retrieval_quality",
    "delivery_intervention_record",
    "delivery_quality_decision",
    "evaluated_pipeline_allowlist",
    "rewrite_retrieval_query",
]
