"""Agent 业务层：衔接异步 FastAPI 与同步 orchestrator，管理会话与历史落库。"""

import asyncio
import json

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory import load_history
from app.agent.orchestrator import run_agent
from app.models.conversation import Conversation, Message, MessageRole
from app.schemas.agent import AgentResponse


async def _get_or_create_conversation(
    db: AsyncSession, user_id: int, conversation_id: int | None, first_msg: str
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

    conv = Conversation(user_id=user_id, title=first_msg[:20])
    db.add(conv)
    await db.flush()
    await db.refresh(conv)
    return conv


async def chat_with_agent(
    db: AsyncSession,
    user_id: int,
    message: str,
    conversation_id: int | None = None,
    document_id: int | None = None,
    requested_skill: str | None = None,
) -> AgentResponse:
    conv = await _get_or_create_conversation(
        db, user_id, conversation_id, message
    )

    # 加载历史（同步 DB 操作放线程池）
    history = (
        await asyncio.to_thread(load_history, conv.id)
        if conversation_id is not None
        else []
    )

    # 跑 Agent 主循环（含多次 LLM 调用和技能执行，都是同步阻塞，放线程池）
    result = await asyncio.to_thread(
        run_agent,
        user_id,
        message,
        history,
        document_id,
        requested_skill,
    )

    # 落库
    db.add(
        Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=message,
        )
    )
    db.add(
        Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=result["answer"],
            sources=json.dumps(result["trace"], ensure_ascii=False),
        )
    )

    return AgentResponse(
        conversation_id=conv.id,
        answer=result["answer"],
        artifacts=result["artifacts"],
        trace=result["trace"],
    )
