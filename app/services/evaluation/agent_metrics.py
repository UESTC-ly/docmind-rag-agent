"""Agent-task metrics derived from observable execution contracts.

Unlike retrieval metrics, these scores evaluate whether the Agent completed the
requested workflow: plan progress, tool choice, interventions, artifacts, and
their verification gates.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Iterable, Mapping


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 1.0


def _non_negative_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value >= 0 else None


def _provider_usage(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider telemetry separate from Agent outcome metrics.

    Most OpenAI-compatible Agent endpoints intentionally expose no token usage.
    In that case the harness must preserve the absence rather than derive a
    synthetic estimate from prompts, model names, or elapsed time.
    """

    raw = result.get("provider_usage") or result.get("usage")
    if not isinstance(raw, Mapping):
        return {
            "status": "unavailable",
            "reason": "Agent API did not expose provider token usage.",
        }
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    observed: dict[str, int | float] = {}
    for target, keys in aliases.items():
        for key in keys:
            value = _non_negative_number(raw.get(key))
            if value is not None:
                observed[target] = value
                break
    if not observed:
        return {
            "status": "unavailable",
            "reason": "Agent API usage payload did not contain token counts.",
        }
    if "total_tokens" not in observed and {
        "input_tokens",
        "output_tokens",
    } <= set(observed):
        observed["total_tokens"] = (
            observed["input_tokens"] + observed["output_tokens"]
        )
    return {"status": "observed", **observed}


def _provider_cost(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicitly reported provider cost, never an inferred price."""

    raw = result.get("provider_cost")
    if not isinstance(raw, Mapping):
        return {
            "status": "unavailable",
            "reason": (
                "Agent API did not expose provider cost with an explicit "
                "currency."
            ),
        }
    amount = _non_negative_number(raw.get("amount"))
    currency = str(raw.get("currency") or "").strip().upper()
    if amount is None or not currency:
        return {
            "status": "unavailable",
            "reason": (
                "Agent API did not expose provider cost with an explicit "
                "currency."
            ),
        }
    return {"status": "observed", "amount": amount, "currency": currency}


def _artifact_verified(artifact: Mapping[str, Any]) -> bool:
    verification = artifact.get("verification")
    return isinstance(verification, Mapping) and bool(verification.get("passed"))


def _ordered_subsequence(required: list[str], observed: list[str]) -> bool:
    if not required:
        return True
    cursor = 0
    for value in observed:
        if value == required[cursor]:
            cursor += 1
            if cursor == len(required):
                return True
    return False


def score_agent_case(
    expected: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    latency_seconds: float = 0.0,
) -> dict[str, Any]:
    trace = [item for item in result.get("trace") or [] if isinstance(item, Mapping)]
    artifacts = [
        item for item in result.get("artifacts") or [] if isinstance(item, Mapping)
    ]
    raw_plan = result.get("plan")
    plan: Mapping[str, Any] = raw_plan if isinstance(raw_plan, Mapping) else {}
    plan_steps = [
        item for item in plan.get("steps") or [] if isinstance(item, Mapping)
    ]

    required_skills = {
        str(item) for item in expected.get("required_skills") or [] if str(item)
    }
    successful_skills = {
        str(item.get("skill"))
        for item in trace
        if item.get("skill") and item.get("ok") is not False
    }
    skill_hits = required_skills & successful_skills
    tool_recall = _rate(len(skill_hits), len(required_skills))
    tool_precision = (
        _rate(len(skill_hits), len(successful_skills))
        if successful_skills
        else (0.0 if required_skills else 1.0)
    )

    required_artifacts = {
        str(item)
        for item in expected.get("required_artifact_types") or []
        if str(item)
    }
    artifact_types = {str(item.get("type")) for item in artifacts if item.get("type")}
    missing_artifacts = sorted(required_artifacts - artifact_types)
    verified_count = sum(1 for item in artifacts if _artifact_verified(item))
    response_failed = str(result.get("status") or "") == "failed"
    verified_rate = (
        _rate(verified_count, len(artifacts))
        if artifacts
        else (
            0.0
            if response_failed or expected.get("require_verified_artifact")
            else 1.0
        )
    )
    required_verified = True
    if expected.get("require_verified_artifact"):
        required_verified = bool(artifacts) and all(
            _artifact_verified(item)
            for item in artifacts
            if not required_artifacts or str(item.get("type")) in required_artifacts
        )
    trace_verification_passed = any(
        isinstance(item.get("verification"), Mapping)
        and bool(item["verification"].get("passed"))
        for item in trace
    )
    evidence_gate_passed = required_verified and (
        any(_artifact_verified(item) for item in artifacts)
        or trace_verification_passed
    )
    require_evidence_gate = bool(expected.get("require_evidence_gate"))

    required_skill_order = [
        str(item)
        for item in expected.get("required_skill_order") or []
        if str(item)
    ]
    observed_skill_order = [
        str(item.get("skill"))
        for item in trace
        if item.get("skill") and item.get("ok") is not False
    ]
    trajectory_order_ok = _ordered_subsequence(
        required_skill_order,
        observed_skill_order,
    )

    completed_plan_steps = sum(
        1 for step in plan_steps if step.get("status") in {"completed", "skipped"}
    )
    plan_completion = (
        _rate(completed_plan_steps, len(plan_steps))
        if plan_steps
        else (0.0 if plan.get("status") == "failed" or response_failed else 1.0)
    )

    answer = str(result.get("answer") or "")
    expected_fragments = [
        str(item) for item in expected.get("answer_contains") or [] if str(item)
    ]
    missing_fragments = [
        fragment for fragment in expected_fragments if fragment not in answer
    ]
    expected_status = str(expected.get("expected_status") or "completed")
    status_ok = str(result.get("status") or "") == expected_status

    intervention_count = sum(
        1
        for item in trace
        if isinstance(item.get("approval"), Mapping)
        and item["approval"].get("required")
    )
    if result.get("approval") and not intervention_count:
        intervention_count = 1
    recovery_count = sum(
        1
        for item in trace
        if item.get("recovery") or item.get("receipt_reused")
    )
    receipt_reuse_count = sum(1 for item in trace if item.get("receipt_reused"))

    failure_reasons: list[str] = []
    benchmark_error = result.get("benchmark_error")
    if isinstance(benchmark_error, Mapping):
        failure_reasons.append(
            "Agent API 请求失败："
            + str(benchmark_error.get("message") or "未知错误")
        )
    if any(
        item.get("failure_code") == "provider_unavailable"
        for item in trace
    ):
        failure_reasons.append("模型服务不可用，Agent 已安全终止")
    if not status_ok:
        failure_reasons.append(
            f"status={result.get('status')}，期望 {expected_status}"
        )
    missing_skills = sorted(required_skills - successful_skills)
    if missing_skills:
        failure_reasons.append("缺少成功 Skill：" + "、".join(missing_skills))
    if missing_artifacts:
        failure_reasons.append("缺少产物：" + "、".join(missing_artifacts))
    if not required_verified:
        failure_reasons.append("必需产物未通过质量门")
    if require_evidence_gate and not evidence_gate_passed:
        failure_reasons.append("任务没有通过证据闭环质量门")
    if not trajectory_order_ok:
        failure_reasons.append(
            "Skill 执行顺序不符合要求："
            + " → ".join(required_skill_order)
        )
    if missing_fragments:
        failure_reasons.append("回答缺少：" + "、".join(missing_fragments))
    max_interventions = expected.get("max_interventions")
    if max_interventions is not None and intervention_count > int(max_interventions):
        failure_reasons.append(
            f"人工介入 {intervention_count} 次，超过上限 {int(max_interventions)}"
        )
    quality_adaptation_count = 0
    for artifact in artifacts:
        workflow = artifact.get("workflow")
        if not isinstance(workflow, Mapping):
            continue
        quality_adaptation_count += int(bool(workflow.get("repair_attempted")))
        quality_adaptation_count += int(bool(workflow.get("fail_safe_applied")))
        quality_adaptation_count += sum(
            1
            for step in workflow.get("steps") or []
            if isinstance(step, Mapping)
            and (
                step.get("adapted")
                or step.get("step") == "switch_pipeline"
            )
        )

    return {
        "case_id": str(expected.get("id") or ""),
        "task_success": not failure_reasons,
        "verified_task_completion": not failure_reasons
        and (not require_evidence_gate or evidence_gate_passed),
        "failure_reasons": failure_reasons,
        "plan_completion_rate": plan_completion,
        "tool_selection_recall": tool_recall,
        "tool_selection_precision": tool_precision,
        "artifact_completion_rate": _rate(
            len(required_artifacts & artifact_types),
            len(required_artifacts),
        ),
        "verified_artifact_rate": verified_rate,
        "evidence_gate_passed": evidence_gate_passed,
        "trajectory_order_ok": trajectory_order_ok,
        "quality_adaptation_count": quality_adaptation_count,
        "human_intervention_count": intervention_count,
        "recovery_event_count": recovery_count,
        "receipt_reuse_count": receipt_reuse_count,
        "latency_seconds": round(max(0.0, float(latency_seconds)), 6),
        "provider_usage": _provider_usage(result),
        "provider_cost": _provider_cost(result),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[rank], 6)


def _aggregate_provider_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        row["provider_usage"]
        for row in rows
        if isinstance(row.get("provider_usage"), Mapping)
        and row["provider_usage"].get("status") == "observed"
    ]
    if not observed:
        return {
            "status": "unavailable",
            "observed_case_count": 0,
            "case_count": len(rows),
            "reason": "No Agent case exposed provider token usage.",
        }
    fields = ("input_tokens", "output_tokens", "total_tokens")
    totals = {
        field: sum(
            float(item.get(field) or 0.0)
            for item in observed
            if _non_negative_number(item.get(field)) is not None
        )
        for field in fields
    }
    status = "observed" if len(observed) == len(rows) else "partial"
    payload: dict[str, Any] = {
        "status": status,
        "observed_case_count": len(observed),
        "case_count": len(rows),
        **{
            field: round(value, 6)
            for field, value in totals.items()
            if value
        },
    }
    if status == "partial":
        payload["reason"] = (
            "Some Agent cases did not expose provider token usage; totals are "
            "not a complete-suite usage claim."
        )
    return payload


def _aggregate_provider_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        row["provider_cost"]
        for row in rows
        if isinstance(row.get("provider_cost"), Mapping)
        and row["provider_cost"].get("status") == "observed"
    ]
    if not observed:
        return {
            "status": "unavailable",
            "observed_case_count": 0,
            "case_count": len(rows),
            "reason": "No Agent case exposed provider cost with a currency.",
        }
    currencies = {str(item.get("currency")) for item in observed}
    if len(currencies) != 1:
        return {
            "status": "partial",
            "observed_case_count": len(observed),
            "case_count": len(rows),
            "reason": (
                "Observed provider costs use multiple currencies and cannot be "
                "summed without an external exchange-rate policy."
            ),
        }
    status = "observed" if len(observed) == len(rows) else "partial"
    payload: dict[str, Any] = {
        "status": status,
        "observed_case_count": len(observed),
        "case_count": len(rows),
        "currency": next(iter(currencies)),
        "amount": round(sum(float(item["amount"]) for item in observed), 6),
    }
    if status == "partial":
        payload["reason"] = (
            "Some Agent cases did not expose provider cost; the amount is not "
            "a complete-suite cost claim."
        )
    return payload


def aggregate_agent_metrics(
    case_scores: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in case_scores]
    if not rows:
        return {
            "contract": "agent_eval_v1",
            "case_count": 0,
            "task_success_rate": 0.0,
        }

    def average(key: str) -> float:
        return round(mean(float(row.get(key) or 0.0) for row in rows), 6)

    latencies = [float(row.get("latency_seconds") or 0.0) for row in rows]
    return {
        "contract": "agent_eval_v1",
        "case_count": len(rows),
        "task_success_rate": _rate(
            sum(1 for row in rows if row.get("task_success")),
            len(rows),
        ),
        "verified_task_completion_rate": _rate(
            sum(
                1
                for row in rows
                if row.get("verified_task_completion")
            ),
            len(rows),
        ),
        "mean_plan_completion_rate": average("plan_completion_rate"),
        "mean_tool_selection_recall": average("tool_selection_recall"),
        "mean_tool_selection_precision": average("tool_selection_precision"),
        "mean_artifact_completion_rate": average("artifact_completion_rate"),
        "mean_verified_artifact_rate": average("verified_artifact_rate"),
        "mean_quality_adaptation_count": average(
            "quality_adaptation_count"
        ),
        "mean_human_intervention_count": average("human_intervention_count"),
        "mean_recovery_event_count": average("recovery_event_count"),
        "mean_latency_seconds": round(mean(latencies), 6),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "provider_usage": _aggregate_provider_usage(rows),
        "provider_cost": _aggregate_provider_cost(rows),
        "failed_case_ids": [
            row.get("case_id") for row in rows if not row.get("task_success")
        ],
        "cases": rows,
    }


__all__ = ["aggregate_agent_metrics", "score_agent_case"]
