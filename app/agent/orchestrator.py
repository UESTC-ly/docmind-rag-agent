"""LangGraph 外层 Agent 编排、人工审批与 checkpoint 恢复。

v3.0 把旧的手写 ``for`` 循环迁移为显式状态图：Supervisor 负责调用
LLM，图一次只选择并执行一个外层 Skill，因此每个已经完成的工具调用都会先写入
checkpoint，再进入下一步。高风险调用在任何副作用发生前通过 ``interrupt()`` 暂停，
使用相同 ``thread_id`` 和 ``Command(resume=...)`` 可在进程重启后继续。

Generic Skill 的内部 ReAct runner 暂时仍是一个外层 Skill；它若声明脚本、MCP、
浏览器或 App 等高风险能力，会在进入整个 Skill 前审批。把内部每个动作迁移为
LangGraph 子图、以及完整 Multi-Agent 编排，是后续版本的独立迁移边界。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import app.skills  # noqa: F401  触发所有技能注册
from app.agent.checkpoint_store import (
    prepare_checkpoint_connection,
    record_agent_run,
    unindexed_checkpoint_ids,
)
from app.agent.planning import (
    claim_plan_step,
    create_task_plan,
    fail_plan,
    finalize_plan,
    finish_plan_step,
    plan_context,
)
from app.config import settings
from app.services.llm_service import chat_completion
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import all_tools, get_skill

SYSTEM_PROMPT = """你是 DocMind 的智能文档助手。你可以调用工具来完成任务：
- 回答文档相关问题：用 search_knowledge_base 检索知识库
- 生成思维导图：用 generate_mindmap
- 生成关系图谱：用 generate_relation_graph
- 写报告：用 generate_report
- 生成经过逐句引用校验的多文档研究报告：用 generate_verified_research_report
- 在证据型交付前依据公开回归评测选择 RAG 管线：用 select_evaluated_rag_pipeline
- 写周报/进度周总结：用 generate_weekly_report
- 制作 PPT/演示文稿/答辩材料：用 generate_presentation
- 知识库答不了或需要外部信息：用 web_search
- 目录化通用技能：可调用 Codex-style SKILL.md package（如 codex_note），由技能内部按 Markdown 指令规划并使用受控工具执行

规则：
1. 根据用户意图自主选择合适的工具，可以多步调用。
2. 拿到工具结果后，用中文给用户清晰的最终回复。
3. 如果生成了思维导图/图谱/报告/周报/PPT/通用技能文件，在回复里说明已生成，正文或下载文件在产出物里。
4. 不要编造工具没返回的信息。
5. 生成逐句证据校验报告前优先调用 select_evaluated_rag_pipeline；若它返回
   selected，把 selected.pipeline_id 原样传给 generate_verified_research_report。
   该交付路径的 target 必须为 retrieval，不要传 dataset_id 或 language；
   宿主会把当前文档映射到匹配的公开评测集。若没有 selected，不得自行编造
   pipeline_id 或声称获得评测背书。
6. 高风险能力由宿主在执行前暂停并请求用户审批；不要声称绕过或代替用户审批。"""

ARTIFACT_TYPES = {
    "mindmap",
    "relation_graph",
    "report",
    "weekly_report",
    "presentation",
}
_QUALITY_ADVISOR_SKILL = "select_evaluated_rag_pipeline"
_EVIDENCE_DELIVERY_SKILL = "generate_verified_research_report"


class AgentState(TypedDict, total=False):
    """只保存可序列化值，保证 SQLite checkpoint 可跨进程恢复。"""

    run_id: str
    user_id: int
    conversation_id: int | None
    document_id: int | None
    requested_skill: str | None
    question: str
    checkpoint_path: str | None
    plan: dict[str, Any] | None
    messages: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    current_tool: dict[str, Any] | None
    evaluated_pipeline_selection: dict[str, Any] | None
    step: int
    answer: str
    status: str


class AgentRunError(RuntimeError):
    """Agent checkpoint 操作的公共异常基类。"""


class AgentRunNotFoundError(AgentRunError):
    """指定 ``thread_id`` 没有 checkpoint。"""


class AgentRunOwnershipError(AgentRunError):
    """当前用户不是 checkpoint 的所有者。"""


class AgentRunStateError(AgentRunError):
    """当前 checkpoint 状态不允许执行请求的恢复操作。"""


def _tool_message_content(result: dict) -> str:
    """压缩给 LLM 看的工具结果，避免把 base64 文件内容塞回上下文。"""
    if result.get("artifact_kind") != "file":
        return json.dumps(result, ensure_ascii=False)[:4000]

    compact = {key: value for key, value in result.items() if key != "download"}
    if "download" in result:
        compact["download"] = {
            "filename": result["download"].get("filename"),
            "mime_type": result["download"].get("mime_type"),
            "encoding": result["download"].get("encoding"),
            "content_omitted": True,
        }
    return json.dumps(compact, ensure_ascii=False)[:4000]


def _audit_scalar(value: Any, *, max_chars: int = 240) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_chars]
    return f"<{type(value).__name__}>"


def compact_quality_interventions(value: Any) -> list[dict[str, Any]]:
    """Keep bounded quality-policy actions while excluding query and source text."""

    rows: list[dict[str, Any]] = []
    for raw in value[:20] if isinstance(value, list) else []:
        if not isinstance(raw, Mapping):
            continue
        row = {
            key: _audit_scalar(raw.get(key), max_chars=120)
            for key in (
                "contract",
                "action",
                "reason",
                "status",
                "attempt",
                "max_interventions",
                "before_hit_count",
                "after_hit_count",
                "from_pipeline_id",
                "to_pipeline_id",
            )
            if key in raw
        }
        for source_key, target_key in (
            ("quality_before", "quality_before"),
            ("quality_after", "quality_after"),
        ):
            quality_detail = raw.get(source_key)
            if not isinstance(quality_detail, Mapping):
                continue
            row[target_key] = {
                key: _audit_scalar(quality_detail.get(key), max_chars=80)
                for key in (
                    "status",
                    "hit_count",
                    "unique_chunk_count",
                    "stale_hit_count",
                    "best_local_rerank_score",
                )
                if key in quality_detail
            }
        if row:
            rows.append(row)
    return rows


def compact_verification_for_audit(
    value: Any,
) -> dict[str, Any] | None:
    """Keep quality counts and decisions, never claim/artifact prose."""

    if not isinstance(value, Mapping):
        return None
    keys = (
        "contract",
        "artifact_type",
        "passed",
        "score",
        "claim_count",
        "supported_claim_count",
        "unsupported_claim_count",
        "citation_count",
        "valid_citation_count",
        "citation_precision",
        "citation_recall",
        "unsupported_claim_rate",
        "semantic_entailment_checked",
        "groundedness",
        "faithfulness",
        "citation_correctness",
        "conflict_count",
        "judge_status",
        "judge_model",
        "judge_rubric_version",
        "judge_input_fingerprint",
        "repair_attempted",
        "fail_safe_applied",
    )
    compact = {
        key: _audit_scalar(value.get(key))
        for key in keys
        if key in value
    }
    checks: list[dict[str, Any]] = []
    raw_checks = value.get("checks")
    for raw in raw_checks[:50] if isinstance(raw_checks, list) else []:
        if not isinstance(raw, Mapping):
            continue
        checks.append(
            {
                "id": _audit_scalar(raw.get("id"), max_chars=120),
                "passed": bool(raw.get("passed")),
            }
        )
    if checks:
        compact["checks"] = checks
    state_counts = value.get("claim_state_counts")
    if isinstance(state_counts, Mapping):
        compact["claim_state_counts"] = {
            str(key)[:40]: int(count)
            for key, count in state_counts.items()
            if isinstance(count, int)
        }
    return compact or None


def compact_grounding_for_audit(value: Any) -> dict[str, Any] | None:
    """Retain evidence coordinates while dropping retrieved source text."""

    if not isinstance(value, Mapping):
        return None
    compact: dict[str, Any] = {}
    for key in (
        "mode",
        "pipeline_id",
        "pipeline_fingerprint",
        "max_chars",
    ):
        if key in value:
            compact[key] = _audit_scalar(value.get(key))
    if isinstance(value.get("document_ids"), list):
        compact["document_ids"] = [
            _audit_scalar(item, max_chars=80)
            for item in value["document_ids"][:100]
        ]
    sources: list[dict[str, Any]] = []
    raw_sources = value.get("sources")
    for raw in raw_sources[:100] if isinstance(raw_sources, list) else []:
        if not isinstance(raw, Mapping):
            continue
        source = {
            key: _audit_scalar(raw.get(key), max_chars=120)
            for key in ("citation_id", "document_id", "chunk_index", "score")
            if key in raw
        }
        if source:
            sources.append(source)
    if sources:
        compact["sources"] = sources
    interventions = compact_quality_interventions(
        value.get("quality_interventions")
    )
    if interventions:
        compact["quality_interventions"] = interventions
    return compact or None


def compact_artifact_for_audit(artifact: Any) -> dict[str, Any] | None:
    """Return bounded artifact metadata suitable for Message.sources."""

    if not isinstance(artifact, Mapping):
        return None
    compact: dict[str, Any] = {}
    if "type" in artifact:
        compact["type"] = _audit_scalar(artifact.get("type"), max_chars=120)
    download = artifact.get("download")
    if isinstance(download, Mapping):
        for source_key, target_key in (
            ("filename", "filename"),
            ("mime_type", "mime_type"),
            ("encoding", "encoding"),
        ):
            if source_key in download:
                compact[target_key] = _audit_scalar(
                    download.get(source_key),
                    max_chars=255,
                )
    grounding = compact_grounding_for_audit(artifact.get("grounding"))
    if grounding:
        compact["grounding"] = grounding
    verification = compact_verification_for_audit(
        artifact.get("verification")
    )
    if verification:
        compact["verification"] = verification
    workflow = artifact.get("workflow")
    if isinstance(workflow, Mapping):
        compact["workflow"] = {
            key: _audit_scalar(workflow.get(key))
            for key in (
                "contract",
                "status",
                "repair_attempted",
                "fail_safe_applied",
            )
            if key in workflow
        }
    return compact or None


def compact_agent_trace_for_audit(value: Any) -> list[dict[str, Any]]:
    """Drop arguments and free-form observations from persisted trace rows."""

    rows: list[dict[str, Any]] = []
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, Mapping):
            continue
        row = {
            key: _audit_scalar(raw.get(key))
            for key in (
                "step",
                "skill",
                "ok",
                "graph_node",
                "plan_step_id",
                "receipt_reused",
                "grounding_mode",
            )
            if key in raw
        }
        args = raw.get("args")
        if isinstance(args, Mapping):
            row["arg_keys"] = sorted(
                str(key)[:120] for key in list(args)[:100]
            )
        approval = raw.get("approval")
        if isinstance(approval, Mapping):
            row["approval"] = {
                key: bool(approval.get(key))
                for key in ("required", "approved")
                if key in approval
            }
        recovery = raw.get("recovery")
        if isinstance(recovery, Mapping):
            row["recovery"] = {
                key: bool(recovery.get(key))
                for key in ("required", "retried")
                if key in recovery
            }
        grounding = compact_grounding_for_audit(raw.get("grounding"))
        if grounding:
            row["grounding"] = grounding
        verification = compact_verification_for_audit(raw.get("verification"))
        if verification:
            row["verification"] = verification
        workflow = raw.get("workflow")
        if isinstance(workflow, Mapping):
            row["workflow"] = {
                key: _audit_scalar(workflow.get(key))
                for key in (
                    "contract",
                    "status",
                    "repair_attempted",
                    "fail_safe_applied",
                )
                if key in workflow
            }
        rows.append(row)
    return rows


def _system_prompt(document_id: int | None, requested_skill: str | None) -> str:
    prompt = SYSTEM_PROMPT
    if document_id is not None:
        prompt += (
            f"\n\n当前用户已选定文档（document_id={document_id}），"
            "检索类工具会自动作用于该文档，不要向用户索要文档 ID，直接调用工具。"
        )
    if requested_skill is not None:
        prompt += (
            f"\n\n用户已在技能面板明确选择 {requested_skill}。"
            "第一步必须调用该技能，不允许只在聊天正文中模拟技能产出。"
        )
    return prompt


def _assistant_tool_calls(llm_message: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        for tool_call in (llm_message.tool_calls or [])
    ]


def _planner(state: AgentState) -> dict[str, Any]:
    """Create one inspectable plan before any tool side effect can run."""

    if state.get("status") in {"completed", "failed"} or state.get("plan") is not None:
        return {}
    if settings.agent_planning_mode == "off":
        return {"plan": None}

    tools = all_tools()
    available_skills = [
        {
            "name": item["function"]["name"],
            "description": item["function"].get("description", ""),
        }
        for item in tools
    ]
    try:
        plan = create_task_plan(
            question=str(state.get("question") or ""),
            available_skills=available_skills,
            llm=chat_completion,
            max_steps=settings.agent_max_steps,
            document_id=state.get("document_id"),
            requested_skill=state.get("requested_skill"),
        )
    except Exception:  # noqa: BLE001 - provider boundary must fail closed
        return _model_provider_failure(state, stage="planner")
    return {"plan": plan}


def _model_provider_failure(
    state: AgentState,
    *,
    stage: str,
) -> dict[str, Any]:
    """End a run safely when the model provider cannot make a decision."""

    summary = "模型服务暂不可用，未执行或伪造后续 Agent 操作。"
    trace = list(state.get("trace", []))
    trace.append(
        {
            "step": int(state.get("step", 0)),
            "skill": "model_provider",
            "ok": False,
            "graph_node": stage,
            "failure_code": "provider_unavailable",
        }
    )
    return {
        "answer": "模型服务暂不可用，本次任务未完成。请稍后重试。",
        "status": "failed",
        "plan": fail_plan(state.get("plan"), summary=summary),
        "trace": trace,
        "pending_tool_calls": [],
        "current_tool": None,
    }


def _next_planned_skill(plan: Mapping[str, Any] | None) -> str | None:
    for step in (plan or {}).get("steps") or []:
        if step.get("status") != "pending" or not step.get("skill"):
            continue
        skill = get_skill(str(step["skill"]))
        if skill is not None and skill.available:
            return skill.name
    return None


def _plan_requires_verified_delivery(plan: Mapping[str, Any] | None) -> bool:
    return any(
        step.get("skill") == _EVIDENCE_DELIVERY_SKILL
        for step in (plan or {}).get("steps") or []
        if isinstance(step, Mapping)
    )


def _normalize_tool_args(
    state: AgentState,
    *,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Keep the eval-selected evidence path independent of model ID guesses."""

    normalized = dict(args)
    if (
        name == _QUALITY_ADVISOR_SKILL
        and _plan_requires_verified_delivery(state.get("plan"))
    ):
        return {"target": "retrieval"}
    if name == _EVIDENCE_DELIVERY_SKILL:
        selection = state.get("evaluated_pipeline_selection")
        selected_pipeline_id = (
            str(selection.get("pipeline_id") or "").strip()
            if isinstance(selection, Mapping)
            else ""
        )
        if selected_pipeline_id:
            normalized["pipeline_id"] = selected_pipeline_id
        elif isinstance(selection, Mapping):
            normalized.pop("pipeline_id", None)
    return normalized


def _evidence_delivery_block_reason(
    state: AgentState,
    *,
    name: str,
) -> str | None:
    """Require a successful same-workflow evaluation decision before delivery."""

    if name != _EVIDENCE_DELIVERY_SKILL:
        return None
    selection = state.get("evaluated_pipeline_selection")
    if not isinstance(selection, Mapping):
        return "生成证据报告前必须先选择经过公开回归评测的 RAG 管线。"
    if (
        selection.get("status") != "selected"
        or not str(selection.get("pipeline_id") or "").strip()
    ):
        return "当前文档没有通过公开回归门禁的 RAG 管线，已拒绝生成证据报告。"
    return None


def _supervisor(state: AgentState) -> dict[str, Any]:
    """执行一轮 LLM 决策；不在该节点内执行任何工具副作用。"""
    if state.get("status") in {"completed", "failed"}:
        return {}

    step = int(state.get("step", 0))
    if step >= settings.agent_max_steps:
        return {
            "answer": "任务较复杂，未能在限定步数内完成，请尝试拆分问题。",
            "status": "completed",
            "pending_tool_calls": [],
            "current_tool": None,
        }

    tool_choice: str | dict[str, Any] = "auto"
    forced_skill_name: str | None = None
    requested_skill = state.get("requested_skill")
    document_id = state.get("document_id")
    if step == 0 and requested_skill is not None:
        forced_skill_name = requested_skill
        tool_choice = {
            "type": "function",
            "function": {"name": requested_skill},
        }
    elif step == 0 and document_id is not None:
        forced_skill_name = "search_knowledge_base"
        tool_choice = {
            "type": "function",
            "function": {"name": forced_skill_name},
        }
    else:
        forced_skill_name = _next_planned_skill(state.get("plan"))
        if forced_skill_name is not None:
            tool_choice = {
                "type": "function",
                "function": {"name": forced_skill_name},
            }

    messages = [dict(message) for message in state.get("messages", [])]
    progress = plan_context(state.get("plan"))
    if progress and messages and messages[0].get("role") == "system":
        messages[0]["content"] = (
            f"{messages[0].get('content')}\n\n当前任务计划与进度：\n{progress}\n"
            "优先完成 pending 步骤；工具失败时可以调整后续行动，但不得伪造完成状态。"
        )
    try:
        llm_message = chat_completion(
            messages,
            tools=all_tools(),
            tool_choice=tool_choice,
        )
    except Exception:  # noqa: BLE001 - provider boundary must fail closed
        return _model_provider_failure(state, stage="supervisor")
    tool_calls = _assistant_tool_calls(llm_message)
    if not tool_calls:
        if forced_skill_name is not None:
            answer = (
                f"模型未按要求调用技能 {forced_skill_name}，"
                "为避免绕过 RAG/文件产出，本次没有接受模型的直接回答。"
            )
        else:
            answer = llm_message.content or ""
        return {
            "answer": answer,
            "status": "completed",
            "step": step + 1,
            "plan": finalize_plan(state.get("plan")),
            "pending_tool_calls": [],
            "current_tool": None,
        }

    messages.append(
        {
            "role": "assistant",
            "content": llm_message.content,
            "tool_calls": tool_calls,
        }
    )
    return {
        "messages": messages,
        "pending_tool_calls": tool_calls,
        "current_tool": None,
        "step": step + 1,
        "answer": "",
        "status": "running",
    }


def _route_after_supervisor(state: AgentState) -> str:
    if state.get("status") in {"completed", "failed"}:
        return "end"
    return "select_tool"


def _json_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _risk_metadata(skill: BaseSkill | None) -> tuple[bool, list[str], str]:
    """把宿主策略映射到一个外层 Skill 调用。

    Generic package 仍是单一外层调用，因此当前审批粒度是“本次 package 调用可
    使用哪些高风险能力”，而不是内部每个 ReAct action。这个边界会明确返回给 UI。
    """
    if skill is None:
        return False, [], ""

    configured_name = skill.name in settings.agent_high_risk_skill_set
    explicitly_required = bool(getattr(skill, "requires_approval", False))
    package = skill.load_package()
    required_capabilities = {
        str(capability).strip().lower()
        for capability in (
            package.required_capabilities if package is not None else ()
        )
        if str(capability).strip()
    }
    risk_capabilities = sorted(
        required_capabilities & settings.agent_high_risk_capability_set
    )
    required = configured_name or explicitly_required or bool(risk_capabilities)
    if not required:
        return False, [], ""

    if risk_capabilities:
        reason = "该 Generic Skill 在本次调用中可能使用高风险宿主能力：" + "、".join(
            risk_capabilities
        )
    elif configured_name:
        reason = "该工具被部署策略列入高风险外层 Skill。"
    else:
        reason = "该工具声明执行前必须由用户审批。"
    return True, risk_capabilities, reason


def _select_tool(state: AgentState) -> dict[str, Any]:
    pending = list(state.get("pending_tool_calls", []))
    if not pending:
        return {"current_tool": None}
    raw_call = pending.pop(0)
    function = raw_call.get("function") or {}
    name = str(function.get("name") or "")
    skill = get_skill(name)
    approval_required, risk_capabilities, risk_reason = _risk_metadata(skill)
    updated_plan, plan_step_id = claim_plan_step(
        state.get("plan"),
        skill=name,
        waiting_approval=approval_required,
    )
    call_id = str(raw_call.get("id") or uuid.uuid4().hex)
    step = max(0, int(state.get("step", 1)) - 1)
    args = _normalize_tool_args(
        state,
        name=name,
        args=_json_arguments(function.get("arguments")),
    )
    current = {
        "id": call_id,
        # 一些兼容 OpenAI 的 provider 会在后续轮次复用 tool_call.id。
        # Tool 消息仍使用 provider ID；执行回执额外加 supervisor 轮次，避免把
        # 后一轮调用误判成前一轮已完成的副作用。
        "receipt_id": f"{step}:{call_id}",
        "name": name,
        "args": args,
        "step": step,
        "approval_required": approval_required,
        "risk_capabilities": risk_capabilities,
        "risk_reason": risk_reason,
        "plan_step_id": plan_step_id,
    }
    return {
        "pending_tool_calls": pending,
        "current_tool": current,
        "plan": updated_plan,
    }


def _route_selected_tool(state: AgentState) -> str:
    current = state.get("current_tool") or {}
    return "approval_gate" if current.get("approval_required") else "execute_tool"


def _record_tool_result(
    state: AgentState,
    current: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    trace = list(state.get("trace", []))
    verification = result.get("verification")
    workflow = result.get("workflow")
    succeeded = (
        "error" not in result
        and result.get("ok") is not False
        and not (
            isinstance(verification, Mapping)
            and verification.get("passed") is False
        )
        and not (
            isinstance(workflow, Mapping)
            and workflow.get("status") == "failed"
        )
    )
    trace_item: dict[str, Any] = {
        "step": int(current.get("step", 0)),
        "skill": str(current.get("name") or ""),
        "args": dict(current.get("args") or {}),
        "ok": succeeded,
        "graph_node": "execute_tool",
    }
    if current.get("plan_step_id"):
        trace_item["plan_step_id"] = current["plan_step_id"]
    if current.get("receipt_reused"):
        trace_item["receipt_reused"] = True
    skill = get_skill(trace_item["skill"])
    if skill is not None:
        trace_item["grounding_mode"] = skill.grounding_mode
    if current.get("approval") is not None:
        trace_item["approval"] = current["approval"]
    if current.get("recovery") is not None:
        trace_item["recovery"] = current["recovery"]
    grounding = compact_grounding_for_audit(result.get("grounding"))
    if grounding:
        trace_item["grounding"] = grounding
    if isinstance(workflow, dict):
        trace_item["workflow"] = {
            key: workflow.get(key)
            for key in (
                "contract",
                "status",
                "repair_attempted",
                "fail_safe_applied",
            )
            if key in workflow
        }
    compact_verification = compact_verification_for_audit(verification)
    if compact_verification:
        trace_item["verification"] = compact_verification
    trace.append(trace_item)

    artifacts = list(state.get("artifacts", []))
    if result.get("type") in ARTIFACT_TYPES or result.get("artifact_kind") == "file":
        artifacts.append(result)

    messages = list(state.get("messages", []))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": current["id"],
            "content": _tool_message_content(result),
        }
    )
    summary = (
        result.get("summary")
        or result.get("error")
        or result.get("type")
        or ("成功" if succeeded else "失败")
    )
    plan = finish_plan_step(
        state.get("plan"),
        step_id=current.get("plan_step_id"),
        succeeded=succeeded,
        summary=str(summary),
    )
    selected_pipeline = state.get("evaluated_pipeline_selection")
    if trace_item["skill"] == _QUALITY_ADVISOR_SKILL:
        selected = result.get("selected")
        selected_pipeline = {
            "status": str(result.get("status") or ""),
            "pipeline_id": (
                str(selected.get("pipeline_id") or "").strip()
                if isinstance(selected, Mapping)
                else ""
            ),
        }
        scope_resolution = result.get("scope_resolution")
        if isinstance(scope_resolution, Mapping):
            trace_item["scope_resolution"] = {
                key: scope_resolution.get(key)
                for key in (
                    "requested_dataset_id",
                    "resolved_dataset_id",
                    "strategy",
                )
                if key in scope_resolution
            }
    return {
        "messages": messages,
        "artifacts": artifacts,
        "trace": trace,
        "plan": plan,
        "current_tool": None,
        "evaluated_pipeline_selection": selected_pipeline,
        "status": "running",
    }


def _approval_gate(state: AgentState) -> dict[str, Any]:
    current = dict(state.get("current_tool") or {})
    if not current:
        return {}

    decision = interrupt(
        {
            "kind": "tool_approval",
            "scope": "outer_skill",
            "run_id": state.get("run_id"),
            "tool": current.get("name"),
            "args": current.get("args") or {},
            "reason": current.get("risk_reason"),
            "risk_capabilities": current.get("risk_capabilities") or [],
        }
    )
    decision = decision if isinstance(decision, dict) else {}
    approved = bool(decision.get("approved"))
    comment = decision.get("comment")
    approval = {
        "required": True,
        "approved": approved,
        "comment": str(comment) if comment is not None else None,
    }
    if approved:
        edited_args = decision.get("edited_args")
        if isinstance(edited_args, dict):
            current["args"] = edited_args
        current["approval"] = approval
        return {"current_tool": current, "status": "running"}

    current["approval"] = approval
    result = {
        "error": f"用户拒绝执行高风险工具 {current.get('name')}。",
        "approval": approval,
    }
    return _record_tool_result(state, current, result)


def _route_after_approval(state: AgentState) -> str:
    return "execute_tool" if state.get("current_tool") else _route_after_tool(state)


def _execute_tool(state: AgentState) -> dict[str, Any]:
    current = dict(state.get("current_tool") or {})
    name = str(current.get("name") or "")
    args = dict(current.get("args") or {})
    result: dict[str, Any]
    receipt_state, cached_result = _claim_tool_receipt(state, current)
    if receipt_state == "completed" and cached_result is not None:
        current["receipt_reused"] = True
        return _record_tool_result(state, current, cached_result)
    if receipt_state == "uncertain":
        decision = interrupt(
            {
                "kind": "execution_recovery",
                "scope": "outer_skill",
                "run_id": state.get("run_id"),
                "tool": name,
                "args": args,
                "reason": (
                    "上次进程在工具执行期间退出，无法确认外部副作用是否已经完成。"
                    "为避免重复执行，必须人工决定是否重试。"
                ),
                "risk_capabilities": current.get("risk_capabilities") or [],
            }
        )
        decision = decision if isinstance(decision, dict) else {}
        retry = bool(decision.get("approved"))
        current["recovery"] = {
            "required": True,
            "retried": retry,
            "comment": decision.get("comment"),
        }
        if not retry:
            result = {
                "error": f"用户拒绝重试结果不确定的工具 {name}。",
                "recovery": current["recovery"],
            }
            _complete_tool_receipt(state, current, result)
            return _record_tool_result(state, current, result)

    skill = get_skill(name)
    if skill is None:
        result = {"error": f"未知技能: {name}"}
    elif not skill.available:
        result = {"error": f"技能 {name} 当前不可调用：{skill.unavailable_reason}"}
    elif delivery_error := _evidence_delivery_block_reason(state, name=name):
        result = {
            "error": delivery_error,
            "workflow": {
                "contract": "evaluated_pipeline_delivery_gate_v1",
                "status": "failed",
                "reason": "no_evaluated_pipeline",
            },
        }
    else:
        context = SkillContext(
            user_id=int(state["user_id"]),
            document_id=state.get("document_id"),
        )
        try:
            raw_result = skill.run(context, **args)
            result = (
                raw_result
                if isinstance(raw_result, dict)
                else {"error": f"技能 {name} 返回了无效结果"}
            )
        except Exception as exc:  # noqa: BLE001 - Skill 是隔离边界
            result = {"error": f"技能执行失败: {exc}"}
    _complete_tool_receipt(state, current, result)
    return _record_tool_result(state, current, result)


def _route_after_tool(state: AgentState) -> str:
    return "select_tool" if state.get("pending_tool_calls") else "supervisor"


def _receipt_connection(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    prepare_checkpoint_connection(connection)
    return connection


def _claim_tool_receipt(
    state: AgentState, current: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """原子声明一次外层工具执行，防止 checkpoint 重放静默重复副作用。"""
    path = state.get("checkpoint_path")
    if not path:
        return "execute", None
    connection = _receipt_connection(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT status, result_json
            FROM docmind_tool_receipts
            WHERE run_id = ? AND tool_call_id = ?
            """,
            (state["run_id"], current.get("receipt_id", current["id"])),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO docmind_tool_receipts
                    (run_id, tool_call_id, status, result_json)
                VALUES (?, ?, 'started', NULL)
                """,
                (state["run_id"], current.get("receipt_id", current["id"])),
            )
            connection.commit()
            return "execute", None
        connection.commit()
        status, result_json = row
        if status == "completed" and result_json:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                return "completed", parsed
        return "uncertain", None
    finally:
        connection.close()


def _complete_tool_receipt(
    state: AgentState,
    current: dict[str, Any],
    result: dict[str, Any],
) -> None:
    path = state.get("checkpoint_path")
    if not path:
        return
    payload = json.dumps(result, ensure_ascii=False, default=str)
    connection = _receipt_connection(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO docmind_tool_receipts
                (run_id, tool_call_id, status, result_json, updated_at)
            VALUES (?, ?, 'completed', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, tool_call_id) DO UPDATE SET
                status = 'completed',
                result_json = excluded.result_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                state["run_id"],
                current.get("receipt_id", current["id"]),
                payload,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def build_agent_graph(checkpointer: Any = None):
    """构建 v3 外层状态图；传入 saver 后启用 durable execution。"""
    builder = StateGraph(AgentState)
    builder.add_node("planner", _planner)
    builder.add_node("supervisor", _supervisor)
    builder.add_node("select_tool", _select_tool)
    builder.add_node("approval_gate", _approval_gate)
    builder.add_node("execute_tool", _execute_tool)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"select_tool": "select_tool", "end": END},
    )
    builder.add_conditional_edges(
        "select_tool",
        _route_selected_tool,
        {"approval_gate": "approval_gate", "execute_tool": "execute_tool"},
    )
    builder.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {
            "execute_tool": "execute_tool",
            "select_tool": "select_tool",
            "supervisor": "supervisor",
        },
    )
    builder.add_conditional_edges(
        "execute_tool",
        _route_after_tool,
        {"select_tool": "select_tool", "supervisor": "supervisor"},
    )
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def _open_checkpointer(
    checkpoint_path: str | Path | None,
) -> Iterator[InMemorySaver | SqliteSaver]:
    if checkpoint_path is None:
        yield InMemorySaver()
        return

    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    try:
        prepare_checkpoint_connection(connection)
        saver = SqliteSaver(connection)
        saver.setup()
        yield saver
    finally:
        connection.close()


def _graph_config(run_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": run_id},
        "recursion_limit": max(50, settings.agent_max_steps * 10),
    }


def _interrupt_value(source: Any) -> dict[str, Any] | None:
    raw_interrupts = None
    if isinstance(source, dict):
        raw_interrupts = source.get("__interrupt__")
    else:
        tasks = getattr(source, "tasks", ()) or ()
        raw_interrupts = [
            item for task in tasks for item in (getattr(task, "interrupts", ()) or ())
        ]
    if not raw_interrupts:
        return None
    first = raw_interrupts[0]
    value = getattr(first, "value", first)
    return dict(value) if isinstance(value, dict) else {"value": value}


def _public_result(
    state: dict[str, Any],
    run_id: str,
    approval: dict[str, Any] | None = None,
    *,
    recoverable: bool = False,
) -> dict[str, Any]:
    waiting = approval is not None
    answer = str(state.get("answer") or "")
    if waiting and not answer:
        answer = "检测到高风险工具调用，等待人工审批后继续。"
    return {
        "run_id": run_id,
        "thread_id": run_id,
        "conversation_id": state.get("conversation_id"),
        "status": "waiting_approval" if waiting else state.get("status", "completed"),
        "recoverable": recoverable,
        "answer": answer,
        "artifacts": list(state.get("artifacts", [])),
        "trace": list(state.get("trace", [])),
        "plan": state.get("plan"),
        "approval": approval,
    }


def _checkpoint_state(graph: Any, run_id: str, expected_user_id: int | None):
    snapshot = graph.get_state(_graph_config(run_id))
    values = dict(snapshot.values or {})
    if not values or "user_id" not in values:
        raise AgentRunNotFoundError(f"Agent run 不存在: {run_id}")
    if expected_user_id is not None and int(values["user_id"]) != expected_user_id:
        raise AgentRunOwnershipError("无权访问该 Agent run")
    return snapshot, values


def _record_persistent_run(
    checkpoint_path: str | Path | None,
    state: Mapping[str, Any],
    run_id: str,
    status: str,
    *,
    only_if_missing: bool = False,
    now: float | None = None,
) -> None:
    """Index run lifecycle metadata without copying private graph messages."""
    if checkpoint_path is None:
        return
    user_id = state.get("user_id")
    if not isinstance(user_id, int):
        return
    conversation_id = state.get("conversation_id")
    record_agent_run(
        checkpoint_path,
        run_id=run_id,
        user_id=user_id,
        conversation_id=(conversation_id if isinstance(conversation_id, int) else None),
        status=status,
        only_if_missing=only_if_missing,
        now=now,
    )


def backfill_agent_run_index(
    checkpoint_path: str | Path,
    *,
    batch_size: int,
    now: float | None = None,
) -> int:
    """Index pre-v3.1 checkpoints without immediately expiring user state."""
    run_ids = unindexed_checkpoint_ids(checkpoint_path, batch_size=batch_size)
    recovered: list[tuple[str, dict[str, Any], str]] = []
    with _open_checkpointer(checkpoint_path) as saver:
        graph = build_agent_graph(saver)
        for run_id in run_ids:
            try:
                snapshot, values = _checkpoint_state(graph, run_id, None)
            except AgentRunNotFoundError:
                continue
            approval = _interrupt_value(snapshot)
            public = _public_result(
                values,
                run_id,
                approval,
                recoverable=bool(snapshot.next) and approval is None,
            )
            recovered.append((run_id, values, public["status"]))
    for run_id, values, status_value in recovered:
        _record_persistent_run(
            checkpoint_path,
            values,
            run_id,
            status_value,
            only_if_missing=True,
            now=now,
        )
    return len(recovered)


def run_agent(
    user_id: int,
    question: str,
    history: list[dict] | None = None,
    document_id: int | None = None,
    requested_skill: str | None = None,
    *,
    thread_id: str | None = None,
    conversation_id: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """开始一次外层图执行。

    兼容旧的五个位置参数。HTTP 业务层会显式传入持久化路径；直接单元调用默认
    使用内存 saver，避免测试和一次性脚本在仓库中留下 checkpoint 文件。
    """
    run_id = thread_id or uuid.uuid4().hex
    selected = get_skill(requested_skill) if requested_skill is not None else None
    preflight_error = ""
    if requested_skill is not None and selected is None:
        preflight_error = f"指定技能不存在或未加载：{requested_skill}"
    elif selected is not None and not selected.available:
        preflight_error = (
            f"技能 {requested_skill} 当前未通过 DocMind 运行时兼容性审计："
            f"{selected.unavailable_reason}"
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _system_prompt(document_id, requested_skill),
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    initial: AgentState = {
        "run_id": run_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "document_id": document_id,
        "requested_skill": requested_skill,
        "question": question,
        "checkpoint_path": (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        ),
        "messages": messages,
        "plan": None,
        "artifacts": [],
        "trace": [],
        "pending_tool_calls": [],
        "current_tool": None,
        "evaluated_pipeline_selection": None,
        "step": 0,
        "answer": preflight_error,
        "status": "completed" if preflight_error else "running",
    }

    with _open_checkpointer(checkpoint_path) as saver:
        graph = build_agent_graph(saver)
        existing = graph.get_state(_graph_config(run_id))
        if existing.values:
            raise AgentRunStateError(f"Agent run_id 已存在: {run_id}")
        _record_persistent_run(checkpoint_path, initial, run_id, "running")
        result = graph.invoke(initial, _graph_config(run_id))
    public = _public_result(result, run_id, _interrupt_value(result))
    _record_persistent_run(checkpoint_path, result, run_id, public["status"])
    return public


def inspect_agent_run(
    run_id: str,
    *,
    checkpoint_path: str | Path,
    expected_user_id: int | None = None,
) -> dict[str, Any]:
    """读取持久化状态，不暴露内部消息和完整 checkpoint。"""
    with _open_checkpointer(checkpoint_path) as saver:
        graph = build_agent_graph(saver)
        snapshot, values = _checkpoint_state(graph, run_id, expected_user_id)
        approval = _interrupt_value(snapshot)
        recoverable = bool(snapshot.next) and approval is None
    public = _public_result(values, run_id, approval, recoverable=recoverable)
    _record_persistent_run(
        checkpoint_path,
        values,
        run_id,
        public["status"],
        only_if_missing=True,
    )
    return public


def resume_agent(
    run_id: str,
    approved: bool,
    *,
    checkpoint_path: str | Path,
    expected_user_id: int | None = None,
    comment: str | None = None,
    edited_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用同一 ``thread_id`` 恢复一个正在等待审批的图。"""
    resume_value: dict[str, Any] = {"approved": approved, "comment": comment}
    if edited_args is not None:
        resume_value["edited_args"] = edited_args

    with _open_checkpointer(checkpoint_path) as saver:
        graph = build_agent_graph(saver)
        snapshot, values = _checkpoint_state(graph, run_id, expected_user_id)
        if _interrupt_value(snapshot) is None:
            raise AgentRunStateError("Agent run 当前不在等待审批")
        _record_persistent_run(checkpoint_path, values, run_id, "running")
        result = graph.invoke(Command(resume=resume_value), _graph_config(run_id))
    public = _public_result(result, run_id, _interrupt_value(result))
    _record_persistent_run(checkpoint_path, result, run_id, public["status"])
    return public


def recover_agent(
    run_id: str,
    *,
    checkpoint_path: str | Path,
    expected_user_id: int | None = None,
) -> dict[str, Any]:
    """从非 interrupt 的未完成节点继续，工具执行回执负责防重复。"""
    with _open_checkpointer(checkpoint_path) as saver:
        graph = build_agent_graph(saver)
        snapshot, values = _checkpoint_state(graph, run_id, expected_user_id)
        if _interrupt_value(snapshot) is not None:
            raise AgentRunStateError("Agent run 正在等待人工审批，请使用 resume")
        if not snapshot.next:
            raise AgentRunStateError("Agent run 已完成，没有可恢复节点")
        _record_persistent_run(checkpoint_path, values, run_id, "running")
        result = graph.invoke(None, _graph_config(run_id))
    public = _public_result(result, run_id, _interrupt_value(result))
    _record_persistent_run(checkpoint_path, result, run_id, public["status"])
    return public
