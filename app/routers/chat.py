from fastapi import APIRouter, Depends
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
from app.services.rag_service import answer_question
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
