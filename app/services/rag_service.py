"""RAG 问答服务：检索相关分块 → 拼进 Prompt → LLM 生成答案 → 存历史。"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings  # noqa: F401 - compatibility override surface
from app.database import SyncSessionLocal
from app.models.conversation import Conversation, Message, MessageRole
from app.schemas.conversation import ChatResponse, CitationReport, Source
from app.services.document_policy import apply_document_policy_async
from app.services.evaluation.grounding import (
    CANONICAL_REFUSAL,
    evaluate_grounding,
    prepare_grounded_delivery,
)
from app.services.evaluation.pipeline_selection import (
    recommend_evaluated_pipeline,
)
from app.services.embedding_service import embed_query
from app.services.evidence import validate_citations
from app.services.llm_service import chat_completion, chat_completion_stream
from app.services.quality_adaptation import (
    AdaptiveRetrievalResult,
    QUALITY_ADAPTATION_CONTRACT,
    QualityPolicyConfig,
    adaptive_retrieve_async,
    assess_retrieval_quality,
    delivery_intervention_record,
    delivery_quality_decision,
    evaluated_pipeline_allowlist,
    rewrite_retrieval_query,
)
from app.services.rag_pipeline import (
    AsyncRetrievalRuntime,
    PipelineExecution,
    PipelineSpec,
    execute_async_pipeline,
)
from app.services.retrieval import keyword_search
from app.services.vector_store import search
from app.utils.logging import logger


async def run_retrieval_pipeline(
    db: AsyncSession,
    user_id: int,
    question: str,
    document_id: int | None,
    *,
    top_k: int | None = None,
    pipeline: PipelineSpec | Mapping[str, Any] | str | None = None,
) -> PipelineExecution:
    """Execute the shared async PipelineSpec with request-local DB access.

    向量检索是同步 I/O（放线程池）；关键词检索用当前 async session。
    """
    async def _embed(query: str) -> list[float]:
        return await asyncio.to_thread(embed_query, query)

    async def _dense(
        vector: list[float],
        dense_user_id: int,
        limit: int,
        dense_document_id: int | None,
    ) -> list[dict]:
        hits = await asyncio.to_thread(
            search,
            vector,
            dense_user_id,
            limit,
            dense_document_id,
        )
        return await apply_document_policy_async(
            hits,
            user_id=dense_user_id,
            db=db,
        )

    async def _keyword(
        query: str,
        keyword_user_id: int,
        keyword_document_id: int | None,
        limit: int,
    ) -> list[dict]:
        hits = await keyword_search(
            db,
            query,
            keyword_user_id,
            keyword_document_id,
            limit,
        )
        return await apply_document_policy_async(
            hits,
            user_id=keyword_user_id,
            db=db,
        )

    runtime = AsyncRetrievalRuntime(
        embed_query=_embed,
        dense_search=_dense,
        keyword_search=_keyword,
    )
    execution = await execute_async_pipeline(
        spec=pipeline,
        user_id=user_id,
        query=question,
        document_id=document_id,
        top_k=top_k,
        runtime=runtime,
    )
    logger.bind(
        pipeline=execution.spec.id,
        fingerprint=execution.fingerprint[:12],
        hits=len(execution.hits),
    ).info("retrieval completed")
    return execution


async def retrieve(
    db: AsyncSession,
    user_id: int,
    question: str,
    document_id: int | None,
    pipeline: PipelineSpec | Mapping[str, Any] | str | None = None,
) -> list[dict]:
    """Retrieve document chunks through the versioned shared pipeline."""
    execution = await run_retrieval_pipeline(
        db,
        user_id,
        question,
        document_id,
        pipeline=pipeline,
    )
    return execution.hits


def _evaluated_pipeline_ids(user_id: int) -> tuple[str, ...]:
    """Read only the current public-evaluation allowlist for one user."""

    try:
        with SyncSessionLocal() as sync_db:
            decision = recommend_evaluated_pipeline(
                sync_db,
                user_id=user_id,
                target="retrieval",
            )
    except Exception as exc:  # noqa: BLE001 - optional optimization only
        logger.bind(error=type(exc).__name__).warning(
            "evaluated pipeline selection unavailable"
        )
        return ()
    return evaluated_pipeline_allowlist(decision)


async def _adaptive_chat_retrieval(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_id: int | None,
) -> AdaptiveRetrievalResult:
    """Run shared retrieval with one bounded, audit-ready recovery policy."""

    policy = QualityPolicyConfig.from_settings()
    allowed_pipeline_ids = await asyncio.to_thread(
        _evaluated_pipeline_ids,
        user_id,
    )

    async def _retrieve(
        query: str,
        pipeline_id: str,
        top_k: int,
    ) -> PipelineExecution:
        return await run_retrieval_pipeline(
            db,
            user_id,
            query,
            document_id,
            top_k=top_k,
            pipeline=pipeline_id,
        )

    async def _rewrite(query: str, reason: str) -> str:
        return await asyncio.to_thread(
            rewrite_retrieval_query,
            query,
            reason,
        )

    return await adaptive_retrieve_async(
        retrieve=_retrieve,
        query=question,
        pipeline_id="configured",
        top_k=settings.retrieval_top_k,
        rewrite_query=_rewrite,
        allowed_pipeline_ids=allowed_pipeline_ids,
        config=policy,
    )


SYSTEM_PROMPT = """你是 DocMind 的文档问答助手。请严格根据下面提供的「文档片段」回答用户问题。
规则：
1. 只用文档片段里的信息作答，不要编造。
2. 如果文档片段里没有相关信息，明确说「根据已有文档无法回答该问题」。
3. 每个事实性句子末尾必须标注一个或多个实际支持它的证据编号，例如 [D12:C3]。
4. 只能使用文档片段中出现的证据编号，不得创造编号。
5. 发现资料冲突时，不得擅自选边；明确说明冲突并分别引用来源。
6. 原文未直接陈述、只能推断的内容必须以「推测：」开头，不得写成事实。
7. 回答用中文，简洁准确。"""


def _build_context(sources: list[dict]) -> str:
    """Build context through the configured evidence-marker contract."""
    from app.services.evidence import build_evidence_context

    return build_evidence_context(sources)


def _prepare_delivery(
    answer: str,
    hits: list[dict],
) -> tuple[str, dict]:
    if settings.grounding_verification_mode == "off":
        return answer, validate_citations(answer, hits)
    if settings.grounding_fail_closed:
        return prepare_grounded_delivery(
            answer,
            hits,
            evaluator=evaluate_grounding,
        )
    return answer, evaluate_grounding(answer, hits)


async def _complete_chat_candidate(
    hits: list[dict],
    question: str,
) -> str:
    llm_msg = await asyncio.to_thread(
        chat_completion,
        _build_messages(hits, question),
    )
    return llm_msg.content or ""


async def _streamed_chat_candidate(
    hits: list[dict],
    question: str,
) -> str:
    """Buffer the model stream until its evidence gate has approved output."""

    parts: list[str] = []
    generator = chat_completion_stream(_build_messages(hits, question))

    def _next(iterator):
        return next(iterator, None)

    while True:
        piece = await asyncio.to_thread(_next, generator)
        if piece is None:
            break
        parts.append(piece)
    return "".join(parts)


async def _resolve_evidence_closed_delivery(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_id: int | None,
    retrieval: AdaptiveRetrievalResult,
    generate_candidate: Callable[[list[dict]], Awaitable[str]],
) -> tuple[str, list[dict], dict]:
    """Generate only after retrieval is sufficient, then apply one bounded repair."""

    policy = QualityPolicyConfig.from_settings()
    interventions = list(retrieval.quality_interventions)
    hits = list(retrieval.hits)
    if not retrieval.deliverable:
        answer, report = await asyncio.to_thread(
            _prepare_delivery,
            CANONICAL_REFUSAL,
            [],
        )
        report = dict(report)
        report["delivery_action"] = (
            "refused_due_to_stale_or_no_current_evidence"
            if retrieval.terminal_reason == "stale_or_no_current_evidence"
            else "refused_due_to_insufficient_evidence"
        )
        report["initial_verification"] = {
            "quality_observation": retrieval.quality.compact(),
        }
        report["quality_interventions"] = interventions
        return answer, [], report

    candidate = await generate_candidate(hits)
    forced_action: str | None = None
    initial: dict[str, Any] | None = None
    if settings.grounding_verification_mode != "off":
        initial = await asyncio.to_thread(evaluate_grounding, candidate, hits)
        decision = delivery_quality_decision(
            initial,
            intervention_count=retrieval.intervention_count,
            config=policy,
            verified_delivery=settings.grounding_fail_closed,
        )
        action = decision["action"]
        if action == "re_retrieve_after_unsupported_claims":
            event = delivery_intervention_record(
                decision,
                intervention_count=retrieval.intervention_count,
                config=policy,
                before_hit_count=len(hits),
            )
            interventions.append(event)
            recovery_query = await asyncio.to_thread(
                rewrite_retrieval_query,
                retrieval.query,
                "unsupported_claims",
            )
            event["query_changed"] = recovery_query != retrieval.query
            recovery_execution = await run_retrieval_pipeline(
                db,
                user_id,
                recovery_query,
                document_id,
                top_k=min(
                    max(retrieval.top_k * 2, settings.retrieval_top_k),
                    policy.max_retrieval_top_k,
                ),
                pipeline=retrieval.pipeline_id,
            )
            recovery_quality = assess_retrieval_quality(
                recovery_execution.hits,
                config=policy,
            )
            event["after_hit_count"] = recovery_quality.hit_count
            event["quality_after"] = recovery_quality.compact()
            if recovery_quality.status == "sufficient":
                hits = list(recovery_execution.hits)
                candidate = await generate_candidate(hits)
            else:
                candidate = CANONICAL_REFUSAL
                forced_action = (
                    "refused_due_to_stale_or_no_current_evidence"
                    if recovery_quality.status
                    == "stale_or_no_current_evidence"
                    else "refused_due_to_insufficient_evidence"
                )
                terminal_action = (
                    "refuse_stale_evidence"
                    if recovery_quality.status
                    == "stale_or_no_current_evidence"
                    else "refuse_insufficient_evidence"
                )
                interventions.append(
                    {
                        "contract": event["contract"],
                        "action": terminal_action,
                        "reason": recovery_quality.status,
                        "status": "terminal",
                        "attempt": event["attempt"],
                        "max_interventions": policy.max_interventions,
                        "before_hit_count": recovery_quality.hit_count,
                        "after_hit_count": recovery_quality.hit_count,
                        "quality_before": recovery_quality.compact(),
                    }
                )
        elif action != "accept":
            interventions.append(
                delivery_intervention_record(
                    decision,
                    intervention_count=retrieval.intervention_count,
                    config=policy,
                    before_hit_count=len(hits),
                )
            )
            forced_action = (
                "refused_due_to_conflicting_evidence"
                if action == "refuse_conflicting_evidence"
                else (
                    "refused_due_to_judge_unavailable"
                    if action == "refuse_judge_unavailable"
                    else "refused_after_quality_gate"
                )
            )
            candidate = CANONICAL_REFUSAL

    answer, report = await asyncio.to_thread(
        _prepare_delivery,
        candidate,
        hits,
    )
    report = dict(report)
    if forced_action is not None:
        report["delivery_action"] = forced_action
        report["initial_verification"] = {
            "judge_status": (initial or {}).get("judge_status"),
            "judge_error": (initial or {}).get("judge_error"),
            "conflict_count": (initial or {}).get("conflict_count"),
            "unsupported_claim_count": (initial or {}).get(
                "unsupported_claim_count"
            ),
        }
    if (
        initial is not None
        and initial.get("unsupported_claim_count")
        and report.get("delivery_action")
        in {"removed_unsupported_claims", "refused_after_quality_gate"}
    ):
        interventions.append(
            {
                "contract": QUALITY_ADAPTATION_CONTRACT,
                "action": "remove_unsupported_or_refuse",
                "reason": report["delivery_action"],
                "status": "terminal",
                "attempt": retrieval.intervention_count + 1,
                "max_interventions": policy.max_interventions,
                "before_hit_count": len(hits),
                "after_hit_count": len(hits),
            }
        )
    report["quality_interventions"] = interventions
    return answer, hits, report


async def _get_or_create_conversation(
    db: AsyncSession, user_id: int, conversation_id: int | None, question: str
) -> Conversation:
    if conversation_id is not None:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conv = result.scalar_one_or_none()
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="对话不存在"
            )
        return conv

    # 新建会话，标题取问题前 20 字
    conv = Conversation(user_id=user_id, title=question[:20])
    db.add(conv)
    await db.flush()
    await db.refresh(conv)
    return conv


async def answer_question(
    db: AsyncSession,
    user_id: int,
    question: str,
    conversation_id: int | None = None,
    document_id: int | None = None,
) -> ChatResponse:
    conv = await _get_or_create_conversation(
        db, user_id, conversation_id, question
    )

    retrieval = await _adaptive_chat_retrieval(
        db,
        user_id=user_id,
        question=question,
        document_id=document_id,
    )
    answer, hits, citation_report = await _resolve_evidence_closed_delivery(
        db,
        user_id=user_id,
        question=question,
        document_id=document_id,
        retrieval=retrieval,
        generate_candidate=lambda rows: _complete_chat_candidate(rows, question),
    )

    # 4. 存历史（用户问题 + 助手回答）
    sources = [Source(**h) for h in hits]
    db.add(
        Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=question,
        )
    )
    db.add(
        Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            sources=json.dumps([s.model_dump() for s in sources]),
        )
    )

    return ChatResponse(
        conversation_id=conv.id,
        answer=answer,
        sources=sources,
        citation_report=CitationReport.model_validate(citation_report),
    )


def _build_messages(hits: list[dict], question: str) -> list[dict]:
    context = _build_context(hits) if hits else "（未检索到相关文档片段）"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"文档片段：\n{context}\n\n用户问题：{question}"},
    ]


async def stream_answer(
    db: AsyncSession,
    user_id: int,
    question: str,
    conversation_id: int | None = None,
    document_id: int | None = None,
):
    """流式问答：先 yield 会话与来源，再逐 token yield 答案，最后落库。

    产出一系列 SSE 事件字典：
      {"event": "meta",  "data": {"conversation_id", "sources"}}
      {"event": "token", "data": {"text": "..."}}   （多次）
      {"event": "verification", "data": {citation report}}
      {"event": "done",  "data": {}}
    """
    conv = await _get_or_create_conversation(db, user_id, conversation_id, question)
    retrieval = await _adaptive_chat_retrieval(
        db,
        user_id=user_id,
        question=question,
        document_id=document_id,
    )
    sources = [Source(**h) for h in retrieval.hits]

    # 先把元信息（会话 id + 来源）推给前端
    yield {
        "event": "meta",
        "data": {
            "conversation_id": conv.id,
            "sources": [s.model_dump() for s in sources],
        },
    }

    answer, hits, citation_report = await _resolve_evidence_closed_delivery(
        db,
        user_id=user_id,
        question=question,
        document_id=document_id,
        retrieval=retrieval,
        generate_candidate=lambda rows: _streamed_chat_candidate(rows, question),
    )
    final_sources = [Source(**h) for h in hits]
    if [source.citation_id for source in final_sources] != [
        source.citation_id for source in sources
    ]:
        sources = final_sources
        yield {
            "event": "sources",
            "data": {"sources": [source.model_dump() for source in sources]},
        }
    for offset in range(0, len(answer), 200):
        yield {"event": "token", "data": {"text": answer[offset: offset + 200]}}
    yield {"event": "verification", "data": citation_report}

    # 全部生成完再落库（用户问题 + 助手回答）
    db.add(Message(conversation_id=conv.id, role=MessageRole.USER, content=question))
    db.add(
        Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            sources=json.dumps([s.model_dump() for s in sources]),
        )
    )
    yield {"event": "done", "data": {}}
