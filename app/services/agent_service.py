"""Agent 业务层：衔接异步 FastAPI 与同步 orchestrator，管理会话与历史落库。"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory import load_history
from app.agent.orchestrator import (
    AgentRunNotFoundError,
    AgentRunOwnershipError,
    AgentRunStateError,
    compact_agent_trace_for_audit,
    compact_artifact_for_audit,
    inspect_agent_run,
    recover_agent,
    resume_agent,
    run_agent,
)
from app.agent.run_lock import (
    AgentRunLeaseBackendError,
    AgentRunLeaseBusyError,
    AgentRunLeaseLostError,
    RunLockConfig,
    create_agent_run_lease,
)
from app.config import settings
from app.models.conversation import Conversation, Message, MessageRole
from app.schemas.agent import AgentResponse, AgentResumeRequest


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


def _new_agent_run_lease(run_id: str):
    config = RunLockConfig(
        backend=settings.resolved_agent_run_lock_backend,
        checkpoint_path=settings.agent_checkpoint_path,
        redis_url=settings.redis_url,
        ttl_seconds=settings.agent_run_lock_ttl_seconds,
        heartbeat_seconds=settings.agent_run_lock_heartbeat_seconds,
        namespace=settings.agent_run_lock_namespace,
    )
    return create_agent_run_lease(run_id, config)


def _lease_http_error(error: Exception) -> HTTPException:
    if isinstance(error, AgentRunLeaseBusyError):
        return HTTPException(
            status_code=423,
            detail="Agent run 正由另一个 worker 执行，请稍后重试",
            headers={"Retry-After": "2"},
        )
    if isinstance(error, AgentRunLeaseLostError):
        detail = "Agent run 执行期间失去租约，请读取 checkpoint 状态后重试"
    else:
        detail = "Agent run 锁服务暂不可用，请稍后重试"
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
        headers={"Retry-After": "2"},
    )


@asynccontextmanager
async def _agent_run_guard(run_id: str):
    lease = _new_agent_run_lease(run_id)
    try:
        await asyncio.to_thread(lease.acquire)
    except (AgentRunLeaseBusyError, AgentRunLeaseBackendError) as exc:
        raise _lease_http_error(exc) from exc
    try:
        yield
    except BaseException:
        try:
            await asyncio.to_thread(lease.release)
        except (AgentRunLeaseBackendError, AgentRunLeaseLostError):
            # Preserve the original business/cancellation exception. The lease
            # has a TTL, so a crashed or partitioned owner cannot deadlock a run.
            pass
        raise
    else:
        try:
            await asyncio.to_thread(lease.release)
        except (AgentRunLeaseBackendError, AgentRunLeaseLostError) as exc:
            raise _lease_http_error(exc) from exc


async def chat_with_agent(
    db: AsyncSession,
    user_id: int,
    message: str,
    conversation_id: int | None = None,
    document_id: int | None = None,
    requested_skill: str | None = None,
    run_id: str | None = None,
) -> AgentResponse:
    effective_run_id = run_id or uuid.uuid4().hex
    async with _agent_run_guard(effective_run_id):
        return await _chat_with_agent_locked(
            db,
            user_id,
            message,
            conversation_id,
            document_id,
            requested_skill,
            effective_run_id,
        )


async def _chat_with_agent_locked(
    db: AsyncSession,
    user_id: int,
    message: str,
    conversation_id: int | None,
    document_id: int | None,
    requested_skill: str | None,
    run_id: str,
) -> AgentResponse:
    # 客户端可在发请求前持久化随机 run_id。若响应丢失后重试同一 ID，直接返回
    # checkpoint，而不是创建第二个会话或重复执行工具。
    try:
        existing = await asyncio.to_thread(
            inspect_agent_run,
            run_id,
            checkpoint_path=settings.agent_checkpoint_path,
            expected_user_id=user_id,
        )
    except AgentRunNotFoundError:
        existing = None
    except AgentRunOwnershipError as exc:
        raise _checkpoint_http_error(exc) from exc
    if existing is not None:
        existing_conversation_id = existing.get("conversation_id")
        if not isinstance(existing_conversation_id, int):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Agent checkpoint 缺少 conversation_id",
            )
        await _get_or_create_conversation(db, user_id, existing_conversation_id, "")
        if existing["status"] in {"completed", "failed"}:
            await _persist_assistant_message(db, existing_conversation_id, existing)
        return _agent_response(existing, existing_conversation_id)

    conv = await _get_or_create_conversation(db, user_id, conversation_id, message)

    # 加载历史（同步 DB 操作放线程池）
    history = (
        await asyncio.to_thread(load_history, conv.id)
        if conversation_id is not None
        else []
    )

    # 用户消息先形成一个 durable transaction。若进程在图执行中退出，客户端已知的
    # run_id 可以恢复到仍然存在的 conversation，而不会指向已回滚的临时主键。
    db.add(
        Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=message,
        )
    )
    await db.commit()

    # 跑 Agent 状态图（含同步 LLM/Skill 调用，放线程池）
    try:
        result = await asyncio.to_thread(
            run_agent,
            user_id,
            message,
            history,
            document_id,
            requested_skill,
            thread_id=run_id,
            conversation_id=conv.id,
            checkpoint_path=settings.agent_checkpoint_path,
        )
    except AgentRunStateError as exc:
        raise _checkpoint_http_error(exc) from exc

    if result["status"] in {"completed", "failed"}:
        await _persist_assistant_message(db, conv.id, result)

    return _agent_response(result, conv.id)


async def _persist_assistant_message(
    db: AsyncSession, conversation_id: int, result: dict
) -> None:
    # 将 run_id 放进 Agent 专用 trace envelope，既保留调试信息，也提供无需
    # schema migration 的幂等键。若图已完成但进程在业务事务提交前退出，客户端
    # 重试同一 run_id 会补写消息；若第一次提交已成功，则不会重复插入。
    sources = json.dumps(
        {
            "agent_run_id": str(result["run_id"]),
            "plan": result.get("plan"),
            "trace": compact_agent_trace_for_audit(result.get("trace", [])),
            "artifacts": [
                compact
                for artifact in result.get("artifacts", [])
                if (compact := compact_artifact_for_audit(artifact)) is not None
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    existing = await db.execute(
        select(Message.id).where(
            Message.conversation_id == conversation_id,
            Message.role == MessageRole.ASSISTANT,
            Message.sources == sources,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return
    db.add(
        Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=result["answer"],
            sources=sources,
        )
    )


def _agent_response(result: dict, conversation_id: int) -> AgentResponse:
    run_id = str(result["run_id"])
    return AgentResponse(
        conversation_id=conversation_id,
        run_id=run_id,
        thread_id=str(result.get("thread_id") or run_id),
        status=result.get("status", "completed"),
        recoverable=bool(result.get("recoverable", False)),
        answer=result.get("answer", ""),
        artifacts=result.get("artifacts", []),
        trace=result.get("trace", []),
        plan=result.get("plan"),
        approval=result.get("approval"),
    )


def _checkpoint_http_error(error: Exception) -> HTTPException:
    if isinstance(error, (AgentRunNotFoundError, AgentRunOwnershipError)):
        # 不区分“不存在”和“不属于当前用户”，避免泄露其他用户的 run_id。
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent run 不存在",
        )
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=str(error),
    )


async def _validate_checkpoint_conversation(
    db: AsyncSession, user_id: int, run_id: str
) -> int:
    """Validate ownership and the durable business row before graph mutation."""
    try:
        checkpoint = await asyncio.to_thread(
            inspect_agent_run,
            run_id,
            checkpoint_path=settings.agent_checkpoint_path,
            expected_user_id=user_id,
        )
    except (AgentRunNotFoundError, AgentRunOwnershipError) as exc:
        raise _checkpoint_http_error(exc) from exc
    conversation_id = checkpoint.get("conversation_id")
    if not isinstance(conversation_id, int):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent checkpoint 缺少 conversation_id",
        )
    # 必须在恢复图、尤其是重新进入高风险 execute_tool 之前验证。否则会先产生
    # 副作用，再发现 checkpoint 指向的业务会话已被删除。
    await _get_or_create_conversation(db, user_id, conversation_id, "")
    return conversation_id


async def resume_agent_run(
    db: AsyncSession,
    user_id: int,
    data: AgentResumeRequest,
) -> AgentResponse:
    async with _agent_run_guard(data.run_id):
        return await _resume_agent_run_locked(db, user_id, data)


async def _resume_agent_run_locked(
    db: AsyncSession,
    user_id: int,
    data: AgentResumeRequest,
) -> AgentResponse:
    conversation_id = await _validate_checkpoint_conversation(
        db, user_id, data.run_id
    )
    try:
        result = await asyncio.to_thread(
            resume_agent,
            data.run_id,
            data.approved,
            checkpoint_path=settings.agent_checkpoint_path,
            expected_user_id=user_id,
            comment=data.comment,
            edited_args=data.edited_args,
        )
    except (AgentRunNotFoundError, AgentRunOwnershipError, AgentRunStateError) as exc:
        raise _checkpoint_http_error(exc) from exc

    if result.get("conversation_id") != conversation_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent checkpoint conversation_id 在恢复期间发生变化",
        )
    if result["status"] in {"completed", "failed"}:
        await _persist_assistant_message(db, conversation_id, result)
    return _agent_response(result, conversation_id)


async def get_agent_run(user_id: int, run_id: str) -> AgentResponse:
    try:
        result = await asyncio.to_thread(
            inspect_agent_run,
            run_id,
            checkpoint_path=settings.agent_checkpoint_path,
            expected_user_id=user_id,
        )
    except (AgentRunNotFoundError, AgentRunOwnershipError) as exc:
        raise _checkpoint_http_error(exc) from exc

    conversation_id = result.get("conversation_id")
    if not isinstance(conversation_id, int):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent checkpoint 缺少 conversation_id",
        )
    return _agent_response(result, conversation_id)


async def recover_agent_run(
    db: AsyncSession,
    user_id: int,
    run_id: str,
) -> AgentResponse:
    async with _agent_run_guard(run_id):
        return await _recover_agent_run_locked(db, user_id, run_id)


async def _recover_agent_run_locked(
    db: AsyncSession,
    user_id: int,
    run_id: str,
) -> AgentResponse:
    conversation_id = await _validate_checkpoint_conversation(db, user_id, run_id)
    try:
        result = await asyncio.to_thread(
            recover_agent,
            run_id,
            checkpoint_path=settings.agent_checkpoint_path,
            expected_user_id=user_id,
        )
    except (
        AgentRunNotFoundError,
        AgentRunOwnershipError,
        AgentRunStateError,
    ) as exc:
        raise _checkpoint_http_error(exc) from exc

    if result.get("conversation_id") != conversation_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent checkpoint conversation_id 在恢复期间发生变化",
        )
    if result["status"] in {"completed", "failed"}:
        await _persist_assistant_message(db, conversation_id, result)
    return _agent_response(result, conversation_id)
