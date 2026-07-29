"""Agent-level evaluation must measure task execution, not retrieval alone."""

from app.services.evaluation.agent_metrics import (
    aggregate_agent_metrics,
    score_agent_case,
)


def _result(*, verified=True):
    return {
        "status": "completed",
        "answer": "报告已经生成",
        "plan": {
            "status": "completed",
            "steps": [
                {"id": "step-1", "status": "completed"},
                {"id": "step-2", "status": "completed"},
            ],
        },
        "trace": [
            {
                "skill": "search_knowledge_base",
                "ok": True,
                "approval": {"required": True, "approved": True},
            },
            {"skill": "generate_report", "ok": True},
        ],
        "artifacts": [
            {
                "type": "report",
                "verification": {"passed": verified},
                "workflow": {
                    "repair_attempted": True,
                    "fail_safe_applied": False,
                    "steps": [{"step": "retrieve", "adapted": True}],
                },
            }
        ],
    }


def test_scores_agent_task_execution_dimensions():
    score = score_agent_case(
        {
            "id": "report-task",
            "required_skills": ["search_knowledge_base", "generate_report"],
            "required_artifact_types": ["report"],
            "require_verified_artifact": True,
            "answer_contains": ["报告"],
        },
        _result(),
        latency_seconds=2.5,
    )

    assert score["task_success"] is True
    assert score["plan_completion_rate"] == 1.0
    assert score["tool_selection_recall"] == 1.0
    assert score["tool_selection_precision"] == 1.0
    assert score["human_intervention_count"] == 1
    assert score["verified_artifact_rate"] == 1.0
    assert score["quality_adaptation_count"] == 2
    assert score["latency_seconds"] == 2.5


def test_aggregate_reports_failure_instead_of_hiding_it():
    passed = score_agent_case(
        {"id": "ok", "required_artifact_types": ["report"]},
        _result(),
    )
    failed = score_agent_case(
        {
            "id": "failed",
            "required_artifact_types": ["presentation"],
            "require_verified_artifact": True,
        },
        _result(verified=False),
    )

    metrics = aggregate_agent_metrics([passed, failed])

    assert metrics["case_count"] == 2
    assert metrics["task_success_rate"] == 0.5
    assert metrics["verified_task_completion_rate"] == 0.5
    assert metrics["mean_plan_completion_rate"] == 1.0
    assert metrics["mean_verified_artifact_rate"] == 0.5


def test_verified_completion_requires_evidence_gate_and_skill_order():
    expected = {
        "id": "verified-report",
        "required_skills": ["search_knowledge_base", "generate_report"],
        "required_skill_order": [
            "search_knowledge_base",
            "generate_report",
        ],
        "required_artifact_types": ["report"],
        "require_verified_artifact": True,
        "require_evidence_gate": True,
    }

    passed = score_agent_case(expected, _result())
    failed_result = _result(verified=False)
    failed_result["trace"] = list(reversed(failed_result["trace"]))
    failed = score_agent_case(expected, failed_result)

    assert passed["verified_task_completion"] is True
    assert passed["evidence_gate_passed"] is True
    assert passed["trajectory_order_ok"] is True
    assert failed["verified_task_completion"] is False
    assert "任务没有通过证据闭环质量门" in failed["failure_reasons"]
    assert any(
        reason.startswith("Skill 执行顺序不符合要求")
        for reason in failed["failure_reasons"]
    )


def test_provider_telemetry_is_explicit_or_unavailable():
    observed_result = _result()
    observed_result["usage"] = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
    }
    observed_result["provider_cost"] = {"amount": 0.0025, "currency": "usd"}
    unavailable_result = _result()
    # An app-side estimate does not prove the provider reported a cost.
    unavailable_result["estimated_cost"] = 999
    unavailable_result["cost_currency"] = "USD"

    observed = score_agent_case({"id": "observed"}, observed_result)
    unavailable = score_agent_case({"id": "unavailable"}, unavailable_result)
    metrics = aggregate_agent_metrics([observed, unavailable])

    assert observed["provider_usage"] == {
        "status": "observed",
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert observed["provider_cost"] == {
        "status": "observed",
        "amount": 0.0025,
        "currency": "USD",
    }
    assert unavailable["provider_usage"]["status"] == "unavailable"
    assert unavailable["provider_cost"]["status"] == "unavailable"
    assert metrics["provider_usage"]["status"] == "partial"
    assert metrics["provider_usage"]["total_tokens"] == 18.0
    assert metrics["provider_cost"]["status"] == "partial"
    assert metrics["provider_cost"]["amount"] == 0.0025


def test_failed_agent_api_does_not_receive_vacuous_success_metrics():
    failed = score_agent_case(
        {
            "id": "provider-down",
            "required_skills": ["generate_verified_research_report"],
            "required_artifact_types": ["verified_research_report"],
            "require_verified_artifact": True,
            "require_evidence_gate": True,
        },
        {
            "status": "failed",
            "answer": "",
            "plan": {"status": "failed", "steps": []},
            "trace": [
                {
                    "skill": "model_provider",
                    "ok": False,
                    "failure_code": "provider_unavailable",
                }
            ],
            "artifacts": [],
            "benchmark_error": {
                "contract": "agent_benchmark_error_v1",
                "message": "Agent API returned HTTP 500.",
            },
        },
    )

    assert failed["task_success"] is False
    assert failed["plan_completion_rate"] == 0.0
    assert failed["tool_selection_recall"] == 0.0
    assert failed["tool_selection_precision"] == 0.0
    assert failed["verified_artifact_rate"] == 0.0
    assert failed["evidence_gate_passed"] is False
    assert "模型服务不可用，Agent 已安全终止" in failed["failure_reasons"]
