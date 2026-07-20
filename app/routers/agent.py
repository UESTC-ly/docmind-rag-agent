from typing import Annotated

from fastapi import APIRouter, Depends, Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.agent import AgentRequest, AgentResponse, AgentResumeRequest
from app.services.agent_service import (
    chat_with_agent,
    get_agent_run,
    recover_agent_run,
    resume_agent_run,
)
from app.skills.registry import all_skills
from app.utils.deps import get_current_user

router = APIRouter(prefix="/agent", tags=["Agent"])
AgentRunId = Annotated[str, Path(min_length=1, max_length=128)]


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
        requested_skill=data.skill_name,
        run_id=data.run_id,
    )


@router.post(
    "/resume",
    response_model=AgentResponse,
    summary="审批并恢复暂停的 Agent",
)
async def agent_resume(
    data: AgentResumeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await resume_agent_run(db, current_user.id, data)


@router.get(
    "/runs/{run_id}",
    response_model=AgentResponse,
    summary="读取 Agent checkpoint 状态",
)
async def agent_run_status(
    run_id: AgentRunId,
    current_user: User = Depends(get_current_user),
):
    return await get_agent_run(current_user.id, run_id)


@router.post(
    "/runs/{run_id}/recover",
    response_model=AgentResponse,
    summary="从未完成 checkpoint 继续 Agent",
)
async def agent_run_recover(
    run_id: AgentRunId,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await recover_agent_run(db, current_user.id, run_id)


@router.get("/skills", summary="列出所有可用技能")
async def list_skills(current_user: User = Depends(get_current_user)):
    """展示 Agent 当前拥有的技能，体现可插拔能力。"""
    out = []
    for skill in all_skills():
        skill.apply_package_metadata()
        item = {
            "name": skill.name,
            "description": skill.description,
            "execution_mode": skill.execution_mode,
            "grounding_mode": skill.grounding_mode,
            "produces_download": skill.produces_download,
            "available": skill.available,
            "unavailable_reason": skill.unavailable_reason,
        }
        package = skill.load_package()
        if package is not None:
            item["package"] = {
                "slug": package.slug,
                "source": package.source,
                "has_skill_md": bool(package.instructions.strip()),
                "templates": package.template_names,
                "references": package.reference_names,
                "scripts": package.script_names,
                "assets": package.asset_names,
                "runtime_status": package.runtime_status,
                "runtime_reason": package.runtime_reason,
                "required_capabilities": list(package.required_capabilities),
            }
        out.append(item)
    return out
