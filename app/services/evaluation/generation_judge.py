"""LLM-as-judge：评估 RAG 生成答案的质量。

两个指标：
  faithfulness     — 忠实度：答案里的每个陈述是否都能从检索片段中找到依据（0.0~1.0）
  answer_relevancy — 答案相关性：答案是否真正回答了问题（0.0~1.0）
"""

import json

from app.services.llm_service import chat_completion

# ── Faithfulness（忠实度）────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """你是 RAG 质量评估专家。评估以下答案对检索到的文档片段的忠实程度。

【检索片段】
{context}

【生成答案】
{answer}

评估标准：
- 答案中每个事实性陈述，是否都能在检索片段中找到依据？
- 不能出现超出检索片段范围的编造内容
- 忽略语气词、连接词等非事实内容

请输出 JSON：
{{
  "score": 0到1之间的小数（1=完全忠实，0=全是幻觉）,
  "reason": "一句话说明扣分原因，如没扣分写OK"
}}"""

# ── Answer Relevancy（答案相关性）────────────────────────────────────────────

RELEVANCY_PROMPT = """你是 RAG 质量评估专家。评估以下答案对用户问题的相关性。

【用户问题】
{question}

【生成答案】
{answer}

评估标准：
- 答案是否直接回答了问题？
- 答案是否完整（没有遗漏关键信息）？
- 答案是否简洁，没有大量无关内容？

请输出 JSON：
{{
  "score": 0到1之间的小数（1=完美作答，0=完全跑题）,
  "reason": "一句话说明扣分原因，如没扣分写OK"
}}"""


def _call_judge(prompt: str) -> float:
    """调用 LLM 打分，解析 JSON，返回 score（失败返回 0.0）。"""
    try:
        response = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,  # 打分用 0 温度，结果更稳定
        )
        content = (response.content or "").strip()

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        data = json.loads(content)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))  # 保险 clamp 到 [0, 1]
    except Exception:
        return 0.0


def judge_faithfulness(context_chunks: list[str], answer: str) -> float:
    """评估答案忠实度。context_chunks 是检索到的片段列表。"""
    context = "\n\n".join(
        f"[片段{i+1}] {c}" for i, c in enumerate(context_chunks)
    )
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    return _call_judge(prompt)


def judge_answer_relevancy(question: str, answer: str) -> float:
    """评估答案与问题的相关性。"""
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    return _call_judge(prompt)
