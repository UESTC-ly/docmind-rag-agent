"""RAG 问答服务：检索相关分块 → 拼进 Prompt → LLM 生成答案 → 存历史。"""

import asyncio
import json

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.conversation import Conversation, Message, MessageRole
from app.schemas.conversation import ChatResponse, Source
from app.services.embedding_service import embed_query
from app.services.llm_service import chat_completion
from app.services.vector_store import search

SYSTEM_PROMPT = """你是 DocMind 的文档问答助手。请严格根据下面提供的「文档片段」回答用户问题。
规则：
1. 只用文档片段里的信息作答，不要编造。
2. 如果文档片段里没有相关信息，明确说「根据已有文档无法回答该问题」。
3. 回答用中文，简洁准确。"""


def _build_context(sources: list[dict]) -> str:
    """把检索到的片段拼成带编号的上下文。"""
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"[片段{i}] {s['content']}")
    return "\n\n".join(lines)


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

    # 1. 检索（OpenAI 调用是同步的，用 to_thread 避免阻塞事件循环）
    query_vector = await asyncio.to_thread(embed_query, question)
    hits = await asyncio.to_thread(
        search,
        query_vector,
        user_id,
        settings.retrieval_top_k,
        document_id,
    )

    # 2. 拼 Prompt
    context = _build_context(hits) if hits else "（未检索到相关文档片段）"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"文档片段：\n{context}\n\n用户问题：{question}",
        },
    ]

    # 3. 生成
    llm_msg = await asyncio.to_thread(chat_completion, messages)
    answer = llm_msg.content or ""

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
        conversation_id=conv.id, answer=answer, sources=sources
    )
