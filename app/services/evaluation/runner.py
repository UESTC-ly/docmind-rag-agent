"""评估运行编排器——遍历样本、调用 RAG、算指标、存结果。"""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models.evaluation import (
    EvalDataset,
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.services.evaluation.generation_judge import (
    judge_answer_relevancy,
    judge_faithfulness,
)
from app.services.evaluation.retrieval_metrics import compute_retrieval_metrics
from app.services.llm_service import chat_completion
from app.services.skill_retrieval import retrieve_for_skill

RAG_SYSTEM_PROMPT = """你是 DocMind 的文档问答助手。请严格根据下面提供的「文档片段」回答用户问题。
规则：
1. 只用文档片段里的信息作答，不要编造。
2. 如果文档片段里没有相关信息，明确说「根据已有文档无法回答该问题」。
3. 回答用中文，简洁准确。"""


def _run_rag_once(
    db: Session,
    user_id: int,
    document_id: int,
    question: str,
    top_k: int,
) -> tuple[str, list[int], list[str], list[dict]]:
    """执行一次 RAG 问答，不写对话历史但复用完整在线检索策略。

    返回：(生成答案, chunk_index 列表, chunk 内容列表, 检索轨迹命中)
    """
    hits = retrieve_for_skill(
        user_id=user_id,
        query=question,
        top_k=top_k,
        document_id=document_id,
        db=db,
    )

    retrieved_ids = [h["chunk_index"] for h in hits]
    retrieved_contents = [h["content"] for h in hits]

    # 2. 拼 Prompt
    context = (
        "\n\n".join(f"[片段{i + 1}] {c}" for i, c in enumerate(retrieved_contents))
        if retrieved_contents
        else "（未检索到相关文档片段）"
    )
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"文档片段：\n{context}\n\n用户问题：{question}"},
    ]

    # 3. 生成
    response = chat_completion(messages)
    answer = response.content or ""

    return answer, retrieved_ids, retrieved_contents, hits


_TRACE_FIELDS = (
    "document_id",
    "chunk_index",
    "retrieval_sources",
    "dense_rank",
    "dense_score",
    "keyword_rank",
    "keyword_score",
    "rrf_score",
    "fused_rank",
    "lexical_score",
    "local_rerank_score",
    "rerank_score",
    "reranker",
    "final_rank",
)


def _retrieval_trace(hits: list[dict]) -> str:
    """Serialize ranking evidence without duplicating document contents."""
    trace = [
        {field: hit[field] for field in _TRACE_FIELDS if field in hit} for hit in hits
    ]
    return json.dumps(trace, ensure_ascii=False, separators=(",", ":"))


class _LeaseLost(RuntimeError):
    """The run was reclaimed by another worker while this attempt was active."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _claim_run(
    db: Session,
    run_id: int,
    task_id: str,
    *,
    reclaim_running: bool,
) -> tuple[str, int] | dict:
    """Atomically claim one run on PostgreSQL *and* SQLite.

    SQLite ignores ``SELECT ... FOR UPDATE``.  A conditional UPDATE with a
    row-count check is the portable compare-and-swap primitive we need here.
    """

    now = _utcnow()
    stale_before = now - timedelta(seconds=settings.evaluation_lease_seconds)
    claimable = [EvalRun.status.in_((RunStatus.PENDING, RunStatus.FAILED))]
    claimable.append(
        and_(
            EvalRun.status == RunStatus.RUNNING,
            or_(
                EvalRun.heartbeat_at.is_(None),
                EvalRun.heartbeat_at < stale_before,
            ),
        )
    )
    if reclaim_running:
        # Celery keeps its task id when Redis redelivers an unacknowledged
        # message.  Reclaim immediately after a worker crash instead of waiting
        # for the lease TTL; a fresh random token invalidates the old worker.
        claimable.append(
            and_(
                EvalRun.status == RunStatus.RUNNING,
                EvalRun.task_id == task_id,
            )
        )

    lease_token = uuid4().hex
    claimed = db.execute(
        update(EvalRun)
        .where(EvalRun.id == run_id, or_(*claimable))
        .values(
            status=RunStatus.RUNNING,
            task_id=task_id,
            lease_token=lease_token,
            heartbeat_at=now,
            error_message=None,
            completed_at=None,
        )
    )
    if claimed.rowcount == 1:
        dataset_id = db.execute(
            select(EvalRun.dataset_id).where(EvalRun.id == run_id)
        ).scalar_one()
        db.commit()
        return lease_token, dataset_id

    db.rollback()
    existing = db.get(EvalRun, run_id)
    if existing is None:
        return {"status": "missing", "run_id": run_id}
    status = existing.status.value
    return {"status": status, "run_id": run_id, "skipped": True}


def _refresh_lease(db: Session, run_id: int, lease_token: str) -> None:
    """Refresh a lease in a short independent transaction.

    The evaluation session is rolled back before this is called, so SQLite has
    no lingering read transaction that could block the heartbeat writer.
    """

    with Session(bind=db.get_bind(), expire_on_commit=False) as lease_db:
        refreshed = lease_db.execute(
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.status == RunStatus.RUNNING,
                EvalRun.lease_token == lease_token,
            )
            .values(heartbeat_at=_utcnow())
        )
        if refreshed.rowcount != 1:
            lease_db.rollback()
            raise _LeaseLost(f"evaluation run {run_id} lease was replaced")
        lease_db.commit()


def _average(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[field] or 0.0) for row in rows) / len(rows)


def run_evaluation(
    db: Session,
    run_id: int,
    user_id: int,
    document_id: int,
    top_k: int | None = None,
    *,
    raise_on_error: bool = False,
    retryable: bool = False,
    task_id: str | None = None,
    reclaim_running: bool = False,
) -> dict:
    """Execute a lease-owned, idempotent evaluation attempt.

    A retryable error returns the row to PENDING before raising, so the frontend
    keeps polling and only the final exhausted attempt becomes FAILED.
    """

    delivery_id = task_id or f"direct-{uuid4().hex}"
    claim = _claim_run(
        db,
        run_id,
        delivery_id,
        reclaim_running=reclaim_running,
    )
    if isinstance(claim, dict):
        return claim
    lease_token, dataset_id = claim

    try:
        retrieval_limit = top_k or settings.retrieval_top_k
        ownership = db.execute(
            select(EvalDataset.user_id, EvalDataset.document_id).where(
                EvalDataset.id == dataset_id
            )
        ).one_or_none()
        if ownership is None:
            raise ValueError("evaluation dataset does not exist")
        if ownership.user_id != user_id or ownership.document_id != document_id:
            raise PermissionError("evaluation task ownership does not match dataset")

        # Copy primitives out of ORM objects, then close the read transaction.
        samples = [
            (sample.id, sample.question, sample.relevant_chunk_ids)
            for sample in (
                db.execute(
                    select(EvalSample).where(EvalSample.dataset_id == dataset_id)
                )
                .scalars()
                .all()
            )
        ]
        db.rollback()
        result_rows: list[dict] = []

        for sample_id, question, relevant_chunk_ids in samples:
            relevant_ids = set(json.loads(relevant_chunk_ids))

            # 跑一次 RAG
            answer, retrieved_ids, retrieved_contents, retrieval_hits = _run_rag_once(
                db, user_id, document_id, question, retrieval_limit
            )
            # Retrieval is read-only. End its transaction before the independent
            # heartbeat writer, which is required for SQLite local mode.
            db.rollback()

            # 计算检索指标
            metrics = compute_retrieval_metrics(relevant_ids, retrieved_ids)

            # 计算生成指标
            faith = judge_faithfulness(retrieved_contents, answer)
            relevancy = judge_answer_relevancy(question, answer)

            result_rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "hit": int(metrics["hit"]),
                    "reciprocal_rank": metrics["reciprocal_rank"],
                    "recall_at_k": metrics["recall_at_k"],
                    "precision_at_k": metrics["precision_at_k"],
                    "faithfulness_score": faith,
                    "answer_relevancy_score": relevancy,
                    "retrieved_chunk_ids": json.dumps(retrieved_ids),
                    "retrieval_mode": settings.retrieval_mode,
                    "reranker_mode": (
                        ",".join(
                            dict.fromkeys(
                                str(hit.get("reranker", settings.reranker_mode))
                                for hit in retrieval_hits
                            )
                        )
                        if retrieval_hits
                        else settings.reranker_mode
                    ),
                    "retrieval_trace": _retrieval_trace(retrieval_hits),
                    "generated_answer": answer,
                }
            )
            _refresh_lease(db, run_id, lease_token)

        # The terminal CAS happens before result insertion.  Both operations are
        # in one transaction; if this worker lost the lease, no rows are written.
        completed = db.execute(
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.status == RunStatus.RUNNING,
                EvalRun.lease_token == lease_token,
            )
            .values(
                hit_rate=_average(result_rows, "hit"),
                mrr=_average(result_rows, "reciprocal_rank"),
                recall=_average(result_rows, "recall_at_k"),
                precision=_average(result_rows, "precision_at_k"),
                faithfulness=_average(result_rows, "faithfulness_score"),
                answer_relevancy=_average(result_rows, "answer_relevancy_score"),
                status=RunStatus.COMPLETED,
                completed_at=_utcnow(),
                error_message=None,
                task_id=None,
                lease_token=None,
                heartbeat_at=None,
            )
        )
        if completed.rowcount != 1:
            db.rollback()
            return {"status": "lease_lost", "run_id": run_id, "skipped": True}
        db.execute(delete(EvalResult).where(EvalResult.run_id == run_id))
        db.add_all(EvalResult(**row) for row in result_rows)
        db.commit()
        return {"status": "completed", "run_id": run_id, "samples": len(samples)}

    except _LeaseLost:
        db.rollback()
        return {"status": "lease_lost", "run_id": run_id, "skipped": True}
    except Exception as exc:
        db.rollback()
        next_status = RunStatus.PENDING if retryable else RunStatus.FAILED
        transitioned = db.execute(
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.status == RunStatus.RUNNING,
                EvalRun.lease_token == lease_token,
            )
            .values(
                status=next_status,
                error_message=(
                    f"后台任务将重试: {str(exc)[:1950]}"
                    if retryable
                    else str(exc)[:2000]
                ),
                completed_at=None if retryable else _utcnow(),
                task_id=None,
                lease_token=None,
                heartbeat_at=None,
            )
        )
        db.commit()
        if transitioned.rowcount != 1:
            return {"status": "lease_lost", "run_id": run_id, "skipped": True}
        if raise_on_error:
            raise
        return {"status": next_status.value, "run_id": run_id, "error": str(exc)}
