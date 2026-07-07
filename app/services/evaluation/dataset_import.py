"""把外部标准数据集导入为 DocMind 的评估数据集。

与 dataset_gen.py（LLM 反向出题）互补：这里导入业界基准的 ground truth，
简历上"用公开基准评估检索质量"比"自生成"更有说服力。

落地方式（关键）：评估 runner 的检索指标依赖 Qdrant 里真实存在的向量，
所以导入必须走完整链路：
    passages → Document + DocumentChunk(PG) → embed → upsert Qdrant
              → EvalDataset + EvalSample(relevant_chunk_ids 指向相关块)

支持的数据集：
    - MS MARCO v2.1 : 每题 passages 带 is_selected 标记 → 语料级检索评估
    - CMRC 2018     : 多题共享 context → 去重段落，每题指向自己的段落

RAGTruth 不在此列：它是"给定 context + 已有回答 + 幻觉标注"，属于验证
faithfulness judge 的独立任务，不走检索流程。见 validate_faithfulness_judge()。
"""

from __future__ import annotations

import json

from datasets import load_dataset
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.evaluation import EvalDataset, EvalSample
from app.services.embedding_service import embed_texts
from app.services.vector_store import upsert_chunks


def parse_ms_marco_rows(rows, limit: int) -> tuple[list[str], list[dict]]:
    """纯解析：MS MARCO 原始行 → (corpus 语料池, pending 问题列表)。

    无 DB / embedding 副作用，可独立单测。规则见 import_ms_marco 文档。
    """
    corpus: list[str] = []      # 全局 passage 池，下标即 chunk_index
    pending: list[dict] = []    # 暂存问题，等语料池建好再落库

    for row in rows:
        if len(pending) >= limit:
            break
        answers = [a for a in (row.get("answers") or []) if a.strip()]
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or []
        flags = passages.get("is_selected") or []
        if not answers or not texts:
            continue

        # 该题相关 passage 的全局下标；无 selected 则跳过
        relevant = [len(corpus) + i for i, sel in enumerate(flags) if sel == 1]
        if not relevant:
            continue

        corpus.extend(texts)  # 所有 passage 都进池（含干扰项，才是真检索）
        pending.append(
            {
                "question": row["query"].strip(),
                "answer": answers[0],
                "relevant": relevant,
            }
        )

    return corpus, pending


def parse_cmrc_rows(rows, limit: int) -> tuple[list[str], list[dict]]:
    """纯解析：CMRC 原始行 → (corpus 去重段落池, pending 问题列表)。

    多题共享 context 时按文本去重，每题 relevant 指向自己的段落下标。
    """
    corpus: list[str] = []
    ctx_index: dict[str, int] = {}  # context 文本 → chunk_index，去重
    pending: list[dict] = []

    for row in rows:
        if len(pending) >= limit:
            break
        context = (row.get("context") or "").strip()
        answers = (row.get("answers") or {}).get("text") or []
        answers = [a for a in answers if a.strip()]
        if not context or not answers:
            continue

        if context not in ctx_index:
            ctx_index[context] = len(corpus)
            corpus.append(context)

        pending.append(
            {
                "question": row["question"].strip(),
                "answer": answers[0],
                "relevant": [ctx_index[context]],
            }
        )

    return corpus, pending


def _materialize_corpus(
    db: Session,
    user_id: int,
    filename: str,
    chunk_texts: list[str],
) -> Document:
    """把一批文本落成 Document + DocumentChunk，向量化后写入 Qdrant。

    chunk_index 就是 chunk_texts 的下标（0-based），与后续 EvalSample
    的 relevant_chunk_ids 保持同一套编号。返回已 flush、带 id 的 Document。
    """
    if not chunk_texts:
        raise ValueError("语料为空，无法建文档")

    # 1. 建 Document（虚拟来源，file_path 标注数据集出处，便于溯源）
    document = Document(
        user_id=user_id,
        filename=filename,
        file_path=f"dataset://{filename}",
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunk_texts),
    )
    db.add(document)
    db.flush()  # 拿到 document.id

    # 2. 建 DocumentChunk（文本存 PG）
    for idx, content in enumerate(chunk_texts):
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=idx,
                content=content,
            )
        )
    db.flush()

    # 3. 向量化 + 写 Qdrant（chunks 与 vectors 同序，chunk_index 即下标）
    vectors = embed_texts(chunk_texts)
    upsert_chunks(document.id, user_id, chunk_texts, vectors)

    return document


def import_ms_marco(
    db: Session,
    user_id: int,
    parquet_path: str,
    dataset_name: str = "MS MARCO v2.1",
    limit: int = 30,
) -> EvalDataset:
    """导入 MS MARCO：所有选中题的 passages 汇成一个语料池做检索评估。

    只保留「有答案 且 至少一个 passage 被标记 is_selected」的题——否则
    relevant_chunk_ids 为空，recall 恒 0，样本无意义。
    """
    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    corpus, pending = parse_ms_marco_rows(raw, limit)

    if not pending:
        raise ValueError("MS MARCO 没有可用样本（缺答案或缺 selected passage）")

    document = _materialize_corpus(db, user_id, dataset_name, corpus)
    dataset = _build_dataset(db, user_id, document.id, dataset_name, pending)
    db.commit()
    db.refresh(dataset)
    return dataset


def import_cmrc2018(
    db: Session,
    user_id: int,
    parquet_path: str,
    dataset_name: str = "CMRC 2018",
    limit: int = 30,
) -> EvalDataset:
    """导入 CMRC 2018：去重 context 建段落池，每题指向自己的段落。"""
    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    corpus, pending = parse_cmrc_rows(raw, limit)

    if not pending:
        raise ValueError("CMRC 2018 没有可用样本")

    document = _materialize_corpus(db, user_id, dataset_name, corpus)
    dataset = _build_dataset(db, user_id, document.id, dataset_name, pending)
    db.commit()
    db.refresh(dataset)
    return dataset


def _build_dataset(
    db: Session,
    user_id: int,
    document_id: int,
    name: str,
    pending: list[dict],
) -> EvalDataset:
    """用暂存的问题列表建 EvalDataset + EvalSample。"""
    dataset = EvalDataset(user_id=user_id, document_id=document_id, name=name)
    db.add(dataset)
    db.flush()

    for item in pending:
        db.add(
            EvalSample(
                dataset_id=dataset.id,
                question=item["question"],
                ground_truth_answer=item["answer"],
                relevant_chunk_ids=json.dumps(item["relevant"]),
            )
        )
    return dataset
