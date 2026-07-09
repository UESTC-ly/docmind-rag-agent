from fastapi import APIRouter, Depends

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent_service import chat_with_agent
from app.skills.registry import all_skills
from app.utils.deps import get_current_user

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentResponse, summary="与 Agent 对话")
async def agent_chat(
    data: AgentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await chat_with_agent(
        db,
        user_id=current_user.id,
        message=data.message,
        conversation_id=data.conversation_id,
        document_id=data.document_id,
    )


@router.get("/skills", summary="列出所有可用技能")
async def list_skills(current_user: User = Depends(get_current_user)):
    """展示 Agent 当前拥有的技能，体现可插拔能力。"""
    out = []
    for skill in all_skills():
        skill.apply_package_metadata()
        item = {"name": skill.name, "description": skill.description}
        package = skill.load_package()
        if package is not None:
            item["package"] = {
                "slug": package.slug,
                "has_skill_md": bool(package.instructions.strip()),
                "templates": package.template_names,
                "references": package.reference_names,
            }
        out.append(item)
    return out
