"""Durable, explicit task plans for the outer Agent.

Planning is intentionally separate from tool execution.  The model proposes a
small, inspectable plan using only currently available Skills; the LangGraph
state then owns step status and acceptance evidence.  Invalid model output
falls back to an adaptive plan instead of blocking the task.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_PLAN_PROMPT = """你是 DocMind 的任务规划器。把用户目标拆成少量、可执行、可验收的步骤。

用户目标：
{question}

当前文档 ID：{document_id}
用户锁定的 Skill：{requested_skill}

可用 Skills：
{skills}

只输出严格 JSON，不要 Markdown：
{{
  "objective": "一句话描述最终目标",
  "steps": [
    {{
      "title": "面向用户的步骤标题",
      "skill": "必须从可用 Skills 中选择；纯总结步骤可为 null",
      "success_criteria": "可观察的完成标准"
    }}
  ]
}}

约束：
- 最多 {max_steps} 步；
- 不得编造 Skill；
- 简单问候或无需工具的问题可返回空 steps；
- 若用户锁定了 Skill，计划必须包含该 Skill；
- 生成逐句证据校验报告前，若可用，先调用 select_evaluated_rag_pipeline，
  再把其 selected.pipeline_id 交给 generate_verified_research_report；
- 计划描述结果，不要假装工具已经执行。"""

_QUALITY_ADVISOR_SKILL = "select_evaluated_rag_pipeline"
_EVIDENCE_DELIVERY_SKILLS = {"generate_verified_research_report"}


def _normalize_step(
    raw: Mapping[str, Any],
    *,
    index: int,
    allowed_skills: set[str],
) -> dict[str, Any] | None:
    title = str(raw.get("title") or "").strip()
    skill_value = raw.get("skill")
    skill = str(skill_value).strip() if skill_value is not None else None
    criteria = str(raw.get("success_criteria") or "").strip()
    if skill and skill not in allowed_skills:
        return None
    if not title:
        title = f"完成步骤 {index}"
    if not criteria:
        criteria = "该步骤返回成功结果"
    return {
        "id": f"step-{index}",
        "title": title[:160],
        "skill": skill or None,
        "success_criteria": criteria[:300],
        "status": "pending",
        "attempts": 0,
    }


def _fallback_plan(
    question: str,
    *,
    requested_skill: str | None,
    allowed_skills: set[str],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if requested_skill and requested_skill in allowed_skills:
        steps.append(
            {
                "id": "step-1",
                "title": f"执行 {requested_skill}",
                "skill": requested_skill,
                "success_criteria": "Skill 成功返回并形成可交付结果",
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "contract": "agent_plan_v1",
        "objective": question.strip()[:500] or "完成用户任务",
        "mode": "adaptive" if not steps else "explicit",
        "status": "pending",
        "steps": steps,
    }


def _inject_quality_selection_step(
    steps: list[dict[str, Any]],
    *,
    allowed_skills: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Enforce evaluation-before-delivery without trusting planner prose alone."""

    if (
        _QUALITY_ADVISOR_SKILL not in allowed_skills
        or limit < 2
        or any(step.get("skill") == _QUALITY_ADVISOR_SKILL for step in steps)
    ):
        return steps
    delivery_index = next(
        (
            index
            for index, step in enumerate(steps)
            if step.get("skill") in _EVIDENCE_DELIVERY_SKILLS
        ),
        None,
    )
    if delivery_index is None:
        return steps
    if len(steps) >= limit:
        removable = next(
            (
                index
                for index in range(len(steps) - 1, -1, -1)
                if index != delivery_index
                and steps[index].get("skill")
                not in _EVIDENCE_DELIVERY_SKILLS
            ),
            None,
        )
        if removable is None:
            return steps
        steps.pop(removable)
        if removable < delivery_index:
            delivery_index -= 1
    steps.insert(
        delivery_index,
        {
            "title": "依据公开回归评测选择知识管线",
            "skill": _QUALITY_ADVISOR_SKILL,
            "success_criteria": (
                "返回同一公开数据集内、未被错误级门禁阻断的管线决策，"
                "或明确说明没有合格管线"
            ),
            "status": "pending",
            "attempts": 0,
        },
    )
    for index, step in enumerate(steps, start=1):
        step["id"] = f"step-{index}"
    return steps


def create_task_plan(
    *,
    question: str,
    available_skills: Sequence[Mapping[str, Any]],
    llm: Callable[..., Any],
    max_steps: int,
    document_id: int | None = None,
    requested_skill: str | None = None,
) -> dict[str, Any]:
    """Ask the model for a bounded plan and normalize it for durable state."""

    allowed_skills = {
        str(item.get("name") or "").strip()
        for item in available_skills
        if str(item.get("name") or "").strip()
    }
    limit = max(1, min(int(max_steps), 12))

    # A user-selected Skill is already an explicit one-step plan.  Avoid an
    # unnecessary model round and guarantee that UI intent cannot be rewritten.
    if requested_skill:
        return _fallback_plan(
            question,
            requested_skill=requested_skill,
            allowed_skills=allowed_skills,
        )

    skill_lines = "\n".join(
        f"- {item.get('name')}: {str(item.get('description') or '')[:240]}"
        for item in available_skills
        if item.get("name")
    )
    try:
        response = llm(
            [
                {
                    "role": "user",
                    "content": _PLAN_PROMPT.format(
                        question=question,
                        document_id=document_id if document_id is not None else "无",
                        requested_skill=requested_skill or "无",
                        skills=skill_lines or "无",
                        max_steps=limit,
                    ),
                }
            ],
            temperature=0.1,
        )
        raw = _FENCE_RE.sub("", str(response.content or "").strip())
        payload = json.loads(raw)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return _fallback_plan(
            question,
            requested_skill=requested_skill,
            allowed_skills=allowed_skills,
        )

    if not isinstance(payload, dict):
        return _fallback_plan(
            question,
            requested_skill=requested_skill,
            allowed_skills=allowed_skills,
        )

    rows = payload.get("steps")
    steps: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for raw_step in rows[:limit]:
            if not isinstance(raw_step, Mapping):
                continue
            step = _normalize_step(
                raw_step,
                index=len(steps) + 1,
                allowed_skills=allowed_skills,
            )
            if step is not None:
                steps.append(step)
    steps = _inject_quality_selection_step(
        steps,
        allowed_skills=allowed_skills,
        limit=limit,
    )

    objective = str(payload.get("objective") or question).strip()[:500]
    return {
        "contract": "agent_plan_v1",
        "objective": objective or "完成用户任务",
        "mode": "explicit",
        "status": "pending",
        "steps": steps,
    }


def claim_plan_step(
    plan: Mapping[str, Any] | None,
    *,
    skill: str,
    waiting_approval: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Mark a matching step as started, appending an adaptive step if needed."""

    if not plan:
        return None, None
    updated = deepcopy(dict(plan))
    steps = list(updated.get("steps") or [])
    selected: dict[str, Any] | None = None
    for step in steps:
        if step.get("skill") == skill and step.get("status") == "pending":
            selected = step
            break
    if selected is None:
        selected = {
            "id": f"step-{len(steps) + 1}",
            "title": f"自适应调用 {skill}",
            "skill": skill,
            "success_criteria": "Skill 成功返回",
            "status": "pending",
            "attempts": 0,
            "adaptive": True,
        }
        steps.append(selected)
    selected["attempts"] = int(selected.get("attempts") or 0) + 1
    selected["status"] = "waiting_approval" if waiting_approval else "running"
    updated["steps"] = steps
    updated["status"] = "running"
    return updated, str(selected["id"])


def finish_plan_step(
    plan: Mapping[str, Any] | None,
    *,
    step_id: str | None,
    succeeded: bool,
    summary: str | None = None,
) -> dict[str, Any] | None:
    if not plan or not step_id:
        return deepcopy(dict(plan)) if plan else None
    updated = deepcopy(dict(plan))
    for step in updated.get("steps") or []:
        if step.get("id") != step_id:
            continue
        step["status"] = "completed" if succeeded else "failed"
        if summary:
            step["result"] = summary[:500]
        break
    updated["status"] = _status(updated, terminal=False)
    return updated


def finalize_plan(plan: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not plan:
        return None
    updated = deepcopy(dict(plan))
    # A plan step without a Skill represents synthesis/communication.  Reaching
    # a final assistant answer is the observable completion evidence for it.
    for step in updated.get("steps") or []:
        if step.get("status") == "pending" and not step.get("skill"):
            step["status"] = "completed"
            step["result"] = "已由最终答复完成"
    updated["status"] = _status(updated, terminal=True)
    return updated


def fail_plan(
    plan: Mapping[str, Any] | None,
    *,
    summary: str,
) -> dict[str, Any] | None:
    """Terminally fail unfinished steps without turning them into successes."""

    if not plan:
        return None
    updated = deepcopy(dict(plan))
    for step in updated.get("steps") or []:
        if step.get("status") not in {"pending", "running", "waiting_approval"}:
            continue
        step["status"] = "failed"
        step["result"] = summary[:500]
    updated["status"] = "failed"
    return updated


def _status(plan: Mapping[str, Any], *, terminal: bool) -> str:
    states = [str(step.get("status") or "pending") for step in plan.get("steps") or []]
    if not states:
        return "completed" if terminal else "running"
    if all(state in {"completed", "skipped"} for state in states):
        return "completed"
    if terminal:
        if all(state == "failed" for state in states):
            return "failed"
        return "partial"
    return "running"


def plan_context(plan: Mapping[str, Any] | None) -> str:
    """Compact progress context for the supervisor; never mutate message history."""

    if not plan:
        return ""
    compact = {
        "objective": plan.get("objective"),
        "status": plan.get("status"),
        "steps": [
            {
                "id": step.get("id"),
                "title": step.get("title"),
                "skill": step.get("skill"),
                "success_criteria": step.get("success_criteria"),
                "status": step.get("status"),
            }
            for step in plan.get("steps") or []
        ],
    }
    return json.dumps(compact, ensure_ascii=False)


__all__ = [
    "claim_plan_step",
    "create_task_plan",
    "fail_plan",
    "finalize_plan",
    "finish_plan_step",
    "plan_context",
]
