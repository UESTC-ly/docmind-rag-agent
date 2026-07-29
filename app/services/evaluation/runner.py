"""评估运行编排器——遍历样本、调用 RAG、算指标、存结果。"""

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models.evaluation import (
    EvalDataset,
    EvalMetricResult,
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.services.evaluation.generation_judge import (
    JudgeResult,
    evaluate_answer_relevancy,
    evaluate_faithfulness,
)
from app.services.evaluation.grounding import evaluate_grounding
from app.services.evaluation.retrieval_metrics import compute_retrieval_metrics
from app.services.evaluation.regression import evaluate_related_regressions
from app.services.evidence import build_evidence_context
from app.services.llm_service import chat_completion
from app.services.rag_pipeline import PipelineSpec, resolve_pipeline_spec
from app.services.skill_retrieval import retrieve_for_skill

RAG_SYSTEM_PROMPT = """你是 DocMind 的文档问答助手。请严格根据下面提供的「文档片段」回答用户问题。
规则：
1. 只用文档片段里的信息作答，不要编造。
2. 如果文档片段里没有相关信息，明确说「根据已有文档无法回答该问题」。
3. 每个事实性句子末尾必须标注一个或多个实际支持它的证据编号，例如 [D12:C3]。
4. 只能使用文档片段中出现的证据编号，不得创造编号。
5. 回答用中文，简洁准确。"""
RAG_PROMPT_VERSION = "rag_answer_v2"
JUDGE_RUBRIC_VERSION = "faithfulness_v2+answer_relevance_v2"


_SAMPLE_METRICS = (
    ("hit", "hit_at_k", "retrieval_v1", "deterministic"),
    ("reciprocal_rank", "reciprocal_rank", "retrieval_v1", "deterministic"),
    ("recall_at_k", "recall_at_k", "retrieval_v1", "deterministic"),
    ("precision_at_k", "precision_at_k", "retrieval_v1", "deterministic"),
    (
        "average_precision_at_k",
        "average_precision_at_k",
        "retrieval_v1",
        "deterministic",
    ),
    ("ndcg_at_k", "ndcg_at_k", "retrieval_v1", "deterministic"),
    ("faithfulness_score", "faithfulness", "model_judge_v2", "model"),
    (
        "answer_relevancy_score",
        "answer_relevance",
        "model_judge_v2",
        "model",
    ),
    (
        "citation_precision",
        "citation_precision",
        "citation_presence_v1",
        "structural",
    ),
    (
        "citation_recall",
        "citation_recall",
        "citation_presence_v1",
        "structural",
    ),
    (
        "unsupported_claim_rate",
        "unsupported_claim_rate",
        "citation_presence_v1",
        "structural",
    ),
)

_AGGREGATE_METRICS = (
    ("hit", "hit_rate", "retrieval_v1", "deterministic"),
    ("reciprocal_rank", "mrr", "retrieval_v1", "deterministic"),
    ("recall_at_k", "recall_at_k", "retrieval_v1", "deterministic"),
    ("precision_at_k", "precision_at_k", "retrieval_v1", "deterministic"),
    (
        "average_precision_at_k",
        "map_at_k",
        "retrieval_v1",
        "deterministic",
    ),
    ("ndcg_at_k", "ndcg_at_k", "retrieval_v1", "deterministic"),
    ("faithfulness_score", "faithfulness", "model_judge_v2", "model"),
    (
        "answer_relevancy_score",
        "answer_relevance",
        "model_judge_v2",
        "model",
    ),
    (
        "citation_precision",
        "citation_precision",
        "citation_presence_v1",
        "structural",
    ),
    (
        "citation_recall",
        "citation_recall",
        "citation_presence_v1",
        "structural",
    ),
    (
        "unsupported_claim_rate",
        "unsupported_claim_rate",
        "citation_presence_v1",
        "structural",
    ),
)


def _run_rag_once(
    db: Session,
    user_id: int,
    document_id: int,
    question: str,
    top_k: int,
    pipeline: PipelineSpec,
    *,
    generate_answer: bool = True,
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
        pipeline=pipeline,
    )

    retrieved_ids = [h["chunk_index"] for h in hits]
    retrieved_contents = [h["content"] for h in hits]

    if not generate_answer:
        return "", retrieved_ids, retrieved_contents, hits

    # 2. 拼 Prompt
    context = (
        build_evidence_context(hits)
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
    "citation_id",
    "document_name",
    "source_uri",
    "source_version",
    "authority",
    "source_status",
    "effective_from",
    "effective_to",
    "jump_url",
    "document_policy_version",
    "page_start",
    "page_end",
    "paragraph_start",
    "paragraph_end",
    "char_start",
    "char_end",
    "locator_version",
    "pipeline_id",
    "pipeline_fingerprint",
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


def _average(rows: list[dict], field: str) -> float | None:
    if not rows:
        return None
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _judge_score_and_metadata(
    result: JudgeResult | float | int | None,
) -> tuple[float | None, dict[str, Any]]:
    """Normalize deterministic test doubles and production JudgeResult values."""

    if isinstance(result, JudgeResult):
        return result.score, result.to_dict()
    if isinstance(result, (float, int)):
        score = float(result)
        return score, {
            "score": score,
            "reason": "injected evaluator result",
            "status": "completed",
            "model": "injected",
            "rubric_version": "injected",
            "input_fingerprint": None,
        }
    return None, {
        "score": None,
        "reason": "judge returned no result",
        "status": "unavailable",
        "model": settings.chat_model,
        "rubric_version": JUDGE_RUBRIC_VERSION,
        "input_fingerprint": None,
    }


def _environment_snapshot(pipeline: PipelineSpec) -> tuple[dict, str]:
    snapshot = {
        "schema_version": 1,
        "code_revision": os.getenv("DOCMIND_CODE_REVISION") or None,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "generation_model": settings.chat_model,
        "generation_prompt_version": RAG_PROMPT_VERSION,
        "judge_model": settings.chat_model,
        "judge_rubric_version": JUDGE_RUBRIC_VERSION,
        "pipeline_fingerprint": pipeline.fingerprint,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return snapshot, hashlib.sha256(encoded).hexdigest()


def _metric_rows(run_id: int, result_rows: list[dict]) -> list[EvalMetricResult]:
    metrics: list[EvalMetricResult] = []
    extra_groups: dict[tuple[str, str, str], list[dict]] = {}
    for result in result_rows:
        for field, name, version, kind in _SAMPLE_METRICS:
            value = result.get(field)
            metadata = (result.get("_metric_metadata") or {}).get(field, {})
            metrics.append(
                EvalMetricResult(
                    run_id=run_id,
                    subject_type="sample",
                    subject_id=result["sample_id"],
                    metric_name=name,
                    metric_version=version,
                    evaluator_kind=kind,
                    score=None if value is None else float(value),
                    reason=metadata.get("reason"),
                    details=(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if metadata
                        else None
                    ),
                )
            )
        for metric in result.get("_metric_results") or []:
            key = (
                metric["metric_name"],
                metric["metric_version"],
                metric["evaluator_kind"],
            )
            extra_groups.setdefault(key, []).append(metric)
            metrics.append(
                EvalMetricResult(
                    run_id=run_id,
                    subject_type="sample",
                    subject_id=result["sample_id"],
                    metric_name=metric["metric_name"],
                    metric_version=metric["metric_version"],
                    evaluator_kind=metric["evaluator_kind"],
                    score=metric.get("score"),
                    passed=metric.get("passed"),
                    reason=metric.get("reason"),
                    details=metric.get("details"),
                )
            )
    for field, name, version, kind in _AGGREGATE_METRICS:
        score = _average(result_rows, field)
        metrics.append(
            EvalMetricResult(
                run_id=run_id,
                subject_type="run",
                subject_id=0,
                metric_name=name,
                metric_version=version,
                evaluator_kind=kind,
                score=score,
                reason=(
                    None
                    if score is not None
                    else "metric unavailable for every evaluated sample"
                ),
            )
        )
    for (name, version, kind), rows in extra_groups.items():
        scores = [
            float(row["score"])
            for row in rows
            if row.get("score") is not None
        ]
        passed_values = [
            bool(row["passed"])
            for row in rows
            if row.get("passed") is not None
        ]
        metrics.append(
            EvalMetricResult(
                run_id=run_id,
                subject_type="run",
                subject_id=0,
                metric_name=name,
                metric_version=version,
                evaluator_kind=kind,
                score=sum(scores) / len(scores) if scores else None,
                passed=(
                    all(passed_values)
                    if len(passed_values) == len(rows)
                    else None
                ),
                reason=(
                    None
                    if scores
                    else "metric unavailable for every evaluated sample"
                ),
            )
        )
    return metrics


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
    started_at = perf_counter()

    try:
        ownership = db.execute(
            select(
                EvalDataset.user_id,
                EvalDataset.document_id,
                EvalDataset.task_type,
            ).where(
                EvalDataset.id == dataset_id
            )
        ).one_or_none()
        if ownership is None:
            raise ValueError("evaluation dataset does not exist")
        if ownership.user_id != user_id or ownership.document_id != document_id:
            raise PermissionError("evaluation task ownership does not match dataset")
        retrieval_only = ownership.task_type in {
            "retrieval",
            "document_retrieval",
        }

        run_config = db.execute(
            select(
                EvalRun.pipeline_spec,
                EvalRun.pipeline_fingerprint,
                EvalRun.evaluation_scope,
                EvalRun.sample_filter,
            ).where(EvalRun.id == run_id)
        ).one()
        if run_config.pipeline_spec:
            decoded = json.loads(run_config.pipeline_spec)
            if not isinstance(decoded, dict):
                raise ValueError("evaluation pipeline_spec must be a JSON object")
            pipeline = resolve_pipeline_spec(decoded)
        else:
            pipeline = resolve_pipeline_spec(None)
        if top_k is not None:
            pipeline = replace(pipeline, top_k=top_k)
        retrieval_limit = pipeline.top_k
        if (
            run_config.pipeline_fingerprint
            and run_config.pipeline_fingerprint != pipeline.fingerprint
        ):
            raise ValueError("evaluation pipeline fingerprint does not match its spec")
        pipeline_json = json.dumps(
            pipeline.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        sample_ids: list[int] | None = None
        if run_config.evaluation_scope == "subset":
            try:
                decoded_filter = json.loads(run_config.sample_filter or "")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "subset evaluation requires a valid sample_filter"
                ) from exc
            if (
                not isinstance(decoded_filter, list)
                or not decoded_filter
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in decoded_filter
                )
                or len(set(decoded_filter)) != len(decoded_filter)
            ):
                raise ValueError(
                    "subset evaluation sample_filter must be unique integer IDs"
                )
            sample_ids = decoded_filter
        elif run_config.evaluation_scope != "full":
            raise ValueError(
                f"unsupported evaluation_scope: {run_config.evaluation_scope}"
            )

        sample_query = select(EvalSample).where(
            EvalSample.dataset_id == dataset_id
        )
        if sample_ids is not None:
            sample_query = sample_query.where(EvalSample.id.in_(sample_ids))

        # Copy primitives out of ORM objects, then close the read transaction.
        samples = [
            (
                sample.id,
                sample.question,
                sample.relevant_chunk_ids,
                sample.document_qrels,
                sample.chunk_qrels,
                sample.answerable,
                sample.conflict_labels,
            )
            for sample in (
                db.execute(sample_query.order_by(EvalSample.id))
                .scalars()
                .all()
            )
        ]
        if sample_ids is not None and {
            sample[0] for sample in samples
        } != set(sample_ids):
            raise ValueError(
                "subset evaluation references samples outside its dataset"
            )
        if not samples:
            raise ValueError("evaluation run has no selected samples")
        db.rollback()
        result_rows: list[dict] = []

        for (
            sample_id,
            question,
            relevant_chunk_ids,
            document_qrels,
            chunk_qrels,
            answerable,
            conflict_labels,
        ) in samples:
            relevant_ids = set(json.loads(relevant_chunk_ids))
            relevance = relevant_ids
            if chunk_qrels:
                decoded_qrels = json.loads(chunk_qrels)
                if isinstance(decoded_qrels, dict):
                    relevance = {
                        int(chunk_id): float(score)
                        for chunk_id, score in decoded_qrels.items()
                    }

            # 跑一次 RAG
            answer, retrieved_ids, retrieved_contents, retrieval_hits = _run_rag_once(
                db,
                user_id,
                document_id,
                question,
                retrieval_limit,
                pipeline,
                generate_answer=not retrieval_only,
            )
            # Retrieval is read-only. End its transaction before the independent
            # heartbeat writer, which is required for SQLite local mode.
            db.rollback()

            # 计算检索指标
            metrics = compute_retrieval_metrics(relevance, retrieved_ids)
            document_metric_rows: list[dict[str, Any]] = []
            document_relevance: dict[int, float] | None = None
            if document_qrels:
                decoded_document_qrels = json.loads(document_qrels)
                if isinstance(decoded_document_qrels, dict):
                    document_relevance = {
                        int(document_id): float(score)
                        for document_id, score in decoded_document_qrels.items()
                        if float(score) > 0
                    }
            retrieved_document_ids = list(
                dict.fromkeys(
                    int(hit["document_id"])
                    for hit in retrieval_hits
                    if hit.get("document_id") is not None
                )
            )
            if document_relevance:
                document_metrics = compute_retrieval_metrics(
                    document_relevance,
                    retrieved_document_ids,
                )
                for source_name, metric_name in (
                    ("hit", "document_hit_at_k"),
                    ("reciprocal_rank", "document_mrr"),
                    ("recall_at_k", "document_recall_at_k"),
                    ("precision_at_k", "document_precision_at_k"),
                    ("average_precision_at_k", "document_map_at_k"),
                    ("ndcg_at_k", "document_ndcg_at_k"),
                ):
                    document_metric_rows.append(
                        {
                            "metric_name": metric_name,
                            "metric_version": "document_retrieval_v1",
                            "evaluator_kind": "deterministic",
                            "score": document_metrics[source_name],
                            "passed": None,
                        }
                    )
            else:
                for metric_name in (
                    "document_hit_at_k",
                    "document_mrr",
                    "document_recall_at_k",
                    "document_precision_at_k",
                    "document_map_at_k",
                    "document_ndcg_at_k",
                ):
                    document_metric_rows.append(
                        {
                            "metric_name": metric_name,
                            "metric_version": "document_retrieval_v1",
                            "evaluator_kind": "deterministic",
                            "score": None,
                            "passed": None,
                            "reason": "document qrels unavailable",
                        }
                    )

            # 计算生成指标
            if retrieval_only:
                faith = None
                relevancy = None
                citation = {
                    "contract": "not_evaluated",
                    "reason": "retrieval-only public benchmark",
                    "citation_precision": None,
                    "citation_recall": None,
                    "unsupported_claim_rate": None,
                    "semantic_entailment_checked": False,
                }
                extra_metrics: list[dict] = document_metric_rows
            else:
                faith, faith_metadata = _judge_score_and_metadata(
                    evaluate_faithfulness(retrieved_contents, answer)
                )
                relevancy, relevance_metadata = _judge_score_and_metadata(
                    evaluate_answer_relevancy(question, answer)
                )
                citation = evaluate_grounding(answer, retrieval_hits)
                extra_metrics = [
                    *document_metric_rows,
                    {
                        "metric_name": "groundedness",
                        "metric_version": "claim_grounding_v1",
                        "evaluator_kind": "model",
                        "score": citation.get("groundedness"),
                        "passed": (
                            None
                            if citation.get("groundedness") is None
                            else citation.get("groundedness") == 1.0
                        ),
                        "reason": citation.get("judge_error"),
                        "details": json.dumps(
                            {
                                "judge_status": citation.get("judge_status"),
                                "judge_model": citation.get("judge_model"),
                                "judge_rubric_version": citation.get(
                                    "judge_rubric_version"
                                ),
                                "judge_input_fingerprint": citation.get(
                                    "judge_input_fingerprint"
                                ),
                                "conflict_count": citation.get(
                                    "conflict_count",
                                    0,
                                ),
                                "claim_state_counts": citation.get(
                                    "claim_state_counts",
                                    {},
                                ),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "metric_name": "citation_correctness",
                        "metric_version": "claim_grounding_v1",
                        "evaluator_kind": "model",
                        "score": citation.get("citation_correctness"),
                        "passed": (
                            None
                            if citation.get("citation_correctness") is None
                            else citation.get("citation_correctness") == 1.0
                        ),
                        "reason": citation.get("judge_error"),
                    },
                    {
                        "metric_name": "refusal_correctness",
                        "metric_version": "refusal_v1",
                        "evaluator_kind": "deterministic",
                        "score": float(
                            bool(citation.get("refused")) is (not answerable)
                        ),
                        "passed": bool(citation.get("refused"))
                        is (not answerable),
                    },
                ]
                if conflict_labels:
                    labels = json.loads(conflict_labels)
                    expected_conflict = bool(
                        labels.get("has_conflict")
                        if isinstance(labels, dict)
                        else labels
                    )
                    detected_conflict = bool(
                        citation.get("conflict_count")
                        and citation.get("conflicts_disclosed")
                    )
                    extra_metrics.append(
                        {
                            "metric_name": "conflict_detection_accuracy",
                            "metric_version": "conflict_v1",
                            "evaluator_kind": "model",
                            "score": float(
                                expected_conflict == detected_conflict
                            ),
                            "passed": expected_conflict == detected_conflict,
                        }
                    )

            result_rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "hit": int(metrics["hit"]),
                    "reciprocal_rank": metrics["reciprocal_rank"],
                    "recall_at_k": metrics["recall_at_k"],
                    "precision_at_k": metrics["precision_at_k"],
                    "average_precision_at_k": metrics[
                        "average_precision_at_k"
                    ],
                    "ndcg_at_k": metrics["ndcg_at_k"],
                    "faithfulness_score": faith,
                    "answer_relevancy_score": relevancy,
                    "citation_precision": citation["citation_precision"],
                    "citation_recall": citation["citation_recall"],
                    "unsupported_claim_rate": citation[
                        "unsupported_claim_rate"
                    ],
                    "citation_report": json.dumps(
                        citation,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "retrieved_chunk_ids": json.dumps(retrieved_ids),
                    "retrieval_mode": pipeline.retriever,
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
                    "_metric_metadata": (
                        {}
                        if retrieval_only
                        else {
                            "faithfulness_score": faith_metadata,
                            "answer_relevancy_score": relevance_metadata,
                        }
                    ),
                    "_metric_results": extra_metrics,
                }
            )
            _refresh_lease(db, run_id, lease_token)

        # The terminal CAS happens before result insertion.  Both operations are
        # in one transaction; if this worker lost the lease, no rows are written.
        environment, environment_fingerprint = _environment_snapshot(pipeline)
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
                map_score=_average(result_rows, "average_precision_at_k"),
                ndcg=_average(result_rows, "ndcg_at_k"),
                faithfulness=_average(result_rows, "faithfulness_score"),
                answer_relevancy=_average(result_rows, "answer_relevancy_score"),
                citation_precision=_average(result_rows, "citation_precision"),
                citation_recall=_average(result_rows, "citation_recall"),
                unsupported_claim_rate=_average(
                    result_rows,
                    "unsupported_claim_rate",
                ),
                pipeline_id=pipeline.id,
                pipeline_spec=pipeline_json,
                pipeline_fingerprint=pipeline.fingerprint,
                code_revision=environment["code_revision"],
                embedding_model=environment["embedding_model"],
                generation_model=environment["generation_model"],
                generation_prompt_version=environment[
                    "generation_prompt_version"
                ],
                judge_model=environment["judge_model"],
                judge_rubric_version=environment["judge_rubric_version"],
                environment_fingerprint=environment_fingerprint,
                latency_ms=(perf_counter() - started_at) * 1000,
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
        db.execute(
            delete(EvalMetricResult).where(EvalMetricResult.run_id == run_id)
        )
        db.add_all(
            EvalResult(
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"_metric_results", "_metric_metadata"}
                }
            )
            for row in result_rows
        )
        db.add_all(_metric_rows(run_id, result_rows))
        db.flush()
        if run_config.evaluation_scope == "full":
            evaluate_related_regressions(db, run_id)
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
