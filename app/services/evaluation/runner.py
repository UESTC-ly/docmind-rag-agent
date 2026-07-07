"""评估运行编排器——遍历样本、调用 RAG、算指标、存结果。"""

import json
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import DocumentChunk
from app.models.evaluation import EvalResult, EvalRun, EvalSample, RunStatus
from app.services.embedding_service import embed_query
from app.services.evaluation.generation_judge import (
    judge_answer_relevancy,
    judge_faithfulness,
)
from app.services.evaluation.retrieval_metrics import compute_retrieval_metrics
from app.services.llm_service import chat_completion
from app.services.vector_store import search

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
) -> tuple[str, list[int], list[str]]:
    """执行一次 RAG 问答（简化版，不走 service 层以避免存历史）。

    返回：(生成答案, 检索到的 chunk_index 列表, 检索到的 chunk 内容列表)
    """
    # 1. 检索
    query_vector = embed_query(question)
    hits = search(query_vector, user_id, top_k, document_id)

    retrieved_ids = [h["chunk_index"] for h in hits]
    retrieved_contents = [h["content"] for h in hits]

    # 2. 拼 Prompt
    context = (
        "\n\n".join(f"[片段{i+1}] {c}" for i, c in enumerate(retrieved_contents))
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

    return answer, retrieved_ids, retrieved_contents


def run_evaluation(
    db: Session,
    run_id: int,
    user_id: int,
    document_id: int,
    top_k: int = 5,
) -> None:
    """执行一次完整评估，结果写入数据库。由 Celery task 或 API 直接调用。"""
    run = db.get(EvalRun, run_id)
    if run is None:
        return

    run.status = RunStatus.RUNNING
    db.commit()

    try:
        # 加载该数据集的所有样本
        samples = (
            db.execute(
                select(EvalSample).where(EvalSample.dataset_id == run.dataset_id)
            )
            .scalars()
            .all()
        )

        for sample in samples:
            relevant_ids = set(json.loads(sample.relevant_chunk_ids))

            # 跑一次 RAG
            answer, retrieved_ids, retrieved_contents = _run_rag_once(
                db, user_id, document_id, sample.question, top_k
            )

            # 计算检索指标
            metrics = compute_retrieval_metrics(relevant_ids, retrieved_ids)

            # 计算生成指标
            faith = judge_faithfulness(retrieved_contents, answer)
            relevancy = judge_answer_relevancy(sample.question, answer)

            result = EvalResult(
                run_id=run_id,
                sample_id=sample.id,
                hit=int(metrics["hit"]),
                reciprocal_rank=metrics["reciprocal_rank"],
                recall_at_k=metrics["recall_at_k"],
                precision_at_k=metrics["precision_at_k"],
                faithfulness_score=faith,
                answer_relevancy_score=relevancy,
                retrieved_chunk_ids=json.dumps(retrieved_ids),
                generated_answer=answer,
            )
            db.add(result)

        db.flush()

        # 计算聚合分数（所有样本取均值）
        agg = db.execute(
            select(
                func.avg(EvalResult.hit),
                func.avg(EvalResult.reciprocal_rank),
                func.avg(EvalResult.recall_at_k),
                func.avg(EvalResult.precision_at_k),
                func.avg(EvalResult.faithfulness_score),
                func.avg(EvalResult.answer_relevancy_score),
            ).where(EvalResult.run_id == run_id)
        ).one()

        run.hit_rate = float(agg[0] or 0)
        run.mrr = float(agg[1] or 0)
        run.recall = float(agg[2] or 0)
        run.precision = float(agg[3] or 0)
        run.faithfulness = float(agg[4] or 0)
        run.answer_relevancy = float(agg[5] or 0)
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.utcnow()

    except Exception as e:
        run.status = RunStatus.FAILED
        run.error_message = str(e)

    db.commit()
