import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageResponse,
)
from app.services.rag_service import answer_question, stream_answer
from app.utils.deps import get_current_user

router = APIRouter(prefix="/chat", tags=["问答"])


@router.post("/", response_model=ChatResponse, summary="RAG 问答")
async def chat(
    data: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await answer_question(
        db,
        user_id=current_user.id,
        question=data.question,
        conversation_id=data.conversation_id,
        document_id=data.document_id,
    )


@router.post("/stream", summary="RAG 流式问答（SSE）")
async def chat_stream(
    data: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """流式问答：以 Server-Sent Events 逐 token 推送答案。

    事件流：event: meta（会话+来源）→ 多个 event: token → event: done。
    """

    async def _event_source():
        async for evt in stream_answer(
            db,
            user_id=current_user.id,
            question=data.question,
            conversation_id=data.conversation_id,
            document_id=data.document_id,
        ):
            payload = json.dumps(evt["data"], ensure_ascii=False)
            yield f"event: {evt['event']}\ndata: {payload}\n\n"
        await db.commit()  # 流结束后提交落库的消息

    return StreamingResponse(
        _event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
    summary="我的对话列表",
)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
    )
    return list(result.scalars().all())


@router.get(
    "/conversations/{conversation_id}",
    response_model=list[MessageResponse],
    summary="某对话的消息历史",
)
async def get_history(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 先确认会话属于当前用户，防越权
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        return []

    msgs = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list(msgs.scalars().all())
