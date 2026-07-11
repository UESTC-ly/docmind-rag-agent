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


class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    artifacts: list[dict] = []  # 思维导图/图谱/报告等结构化产出
    trace: list[dict] = []  # 执行轨迹（调了哪些技能）
