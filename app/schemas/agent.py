from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Agent 交互请求。"""

    message: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    document_id: int | None = None  # 可选，限定作用于某文档
    skill_name: str | None = Field(
        default=None,
        max_length=100,
        description="可选：技能面板明确选择的技能；提供后首轮强制调用该技能。",
    )
    run_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "可选客户端幂等 ID；也作为 LangGraph thread_id。已有 ID 返回其 checkpoint 状态。"
        ),
    )


class AgentResponse(BaseModel):
    conversation_id: int
    run_id: str
    thread_id: str
    status: Literal["running", "waiting_approval", "completed", "failed"] = (
        "completed"
    )
    recoverable: bool = False
    answer: str
    artifacts: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None


class AgentResumeRequest(BaseModel):
    """恢复一个因 LangGraph interrupt 暂停的 Agent run。"""

    run_id: str = Field(min_length=1, max_length=128)
    approved: bool
    comment: str | None = Field(default=None, max_length=1000)
    edited_args: dict[str, Any] | None = None
