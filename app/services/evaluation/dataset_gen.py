"""从文档自动生成评估数据集（LLM 反向出题）。

策略：从文档的分块中随机抽样，让 LLM 读分块内容生成问题+答案。
生成的三元组：(question, ground_truth_answer, [relevant_chunk_index])
"""

import json
import random

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import DocumentChunk
from app.models.evaluation import EvalDataset, EvalSample
from app.services.llm_service import chat_completion

GENERATION_PROMPT = """你是评估数据生成专家。我会给你一段文档片段，请基于它生成一个问答对。

要求：
1. 问题必须能用这段文档片段回答，不要问片段外的内容
2. 答案必须准确、完整，直接从片段中提取或总结
3. 问题要自然、有价值，像真实用户会问的
4. 用中文输出

文档片段：
{content}

请用 JSON 格式输出，格式如下：
{{
  "question": "基于片段的问题",
  "answer": "基于片段的答案"
}}
"""


def _generate_qa_from_chunk(chunk_content: str) -> dict[str, str] | None:
    """让 LLM 从一个分块生成一个 QA 对。失败返回 None。"""
    try:
        messages = [
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(content=chunk_content),
            }
        ]
        response = chat_completion(messages, temperature=0.7)
        content = response.content or ""

        # 尝试解析 JSON（LLM 可能在前后加 markdown 代码块）
        content = content.strip()
        if content.startswith("```"):
            # 去掉代码块包裹
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        data = json.loads(content)
        if "question" in data and "answer" in data:
            return {"question": data["question"], "answer": data["answer"]}
        return None
    except Exception:
        return None


def generate_dataset(
    db: Session,
    user_id: int,
    document_id: int,
    dataset_name: str,
    sample_count: int,
) -> EvalDataset:
    """从指定文档生成评估数据集。

    流程：
    1. 随机抽 sample_count 个分块
    2. 每个分块让 LLM 生成一个 QA 对
    3. 存入数据库
    """
    # 1. 获取文档的所有分块
    chunks = (
        db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        .scalars()
        .all()
    )

    if not chunks:
        raise ValueError("文档无分块，无法生成数据集")

    # 2. 随机抽样（如果分块数不足，就全用）
    selected = random.sample(chunks, min(sample_count, len(chunks)))

    # 3. 创建数据集
    dataset = EvalDataset(
        user_id=user_id,
        document_id=document_id,
        name=dataset_name,
    )
    db.add(dataset)
    db.flush()

    # 4. 逐个生成样本
    for chunk in selected:
        qa = _generate_qa_from_chunk(chunk.content)
        if qa is None:
            continue  # 生成失败跳过

        sample = EvalSample(
            dataset_id=dataset.id,
            question=qa["question"],
            ground_truth_answer=qa["answer"],
            relevant_chunk_ids=json.dumps([chunk.chunk_index]),  # 单个来源块
        )
        db.add(sample)

    db.commit()
    db.refresh(dataset)
    return dataset
