"""对话记忆管理。

从数据库加载最近 N 轮对话，转成 LLM messages 格式。
只保留窗口内的历史，控制 Token 消耗（对应 ConversationBufferWindowMemory 思路）。
"""

from sqlalchemy import select

from app.database import SyncSessionLocal
from app.models.conversation import Message

WINDOW_SIZE = 10  # 最多带入最近 10 条历史消息


def load_history(conversation_id: int) -> list[dict]:
    """加载某会话最近的历史消息，转成 OpenAI messages 格式。"""
    with SyncSessionLocal() as db:
        rows = db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(WINDOW_SIZE)
        ).scalars().all()

    # 上面按 id 倒序取最近的，这里翻回正序
    rows = list(reversed(rows))
    return [{"role": m.role.value, "content": m.content} for m in rows]
