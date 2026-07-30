"""Explicit Agent plan contracts.

The plan is part of the durable Agent state, not display-only metadata.  Tool
execution must advance the matching step and expose the same plan through the
public response.
"""

from __future__ import annotations

import json

from app.agent import orchestrator
from app.agent.planning import create_task_plan
from app.agent.planning import finalize_plan
from app.skills import registry
from app.skills.base import BaseSkill, SkillContext


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMsg:
    def __init__(self, content: str | None = None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _PlanSkill(BaseSkill):
    name = "plan_test_skill"
    description = "执行显式计划中的测试步骤"
    parameters = {"type": "object", "properties": {}}

    def run(self, context: SkillContext, **kwargs) -> dict:
        return {"type": "echo", "ok": True}


class _PipelineAdvisorSkill(BaseSkill):
    name = "select_evaluated_rag_pipeline"
    description = "根据公开回归评测选择管线"
    parameters = {"type": "object", "properties": {}}
    last_kwargs = None

    def run(self, context: SkillContext, **kwargs) -> dict:
        type(self).last_kwargs = kwargs
        return {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "selected",
            "target": "retrieval",
            "selection_policy": "same_public_snapshot_v1",
            "dataset": {
                "id": 3,
                "name": "Public fixture",
                "source_name": "Public fixture",
                "source_version": "v1",
                "split": "test",
                "corpus_fingerprint": "a" * 64,
                "source_snapshot_fingerprint": "b" * 64,
            },
            "selected": {
                "run_id": 8,
                "dataset_id": 3,
                "pipeline_id": "hybrid",
                "pipeline_fingerprint": "c" * 64,
                "comparison_role": "candidate",
                "release_status": "approved",
                "regression_gates": [
                    {
                        "severity": "error",
                        "applicable": True,
                        "passed": True,
                    }
                ],
            },
            "candidates": [{"pipeline_id": "hybrid"}],
        }


class _VerifiedDeliverySkill(BaseSkill):
    name = "generate_verified_research_report"
    description = "生成通过证据门的报告"
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "pipeline_id": {"type": "string"},
        },
        "required": ["topic"],
    }
    last_pipeline_id = None

    def run(self, context: SkillContext, **kwargs) -> dict:
        type(self).last_pipeline_id = kwargs.get("pipeline_id")
        return {
            "type": "verified_research_report",
            "artifact_kind": "file",
            "verification": {
                "contract": "artifact_quality_v1",
                "passed": True,
            },
            "workflow": {"status": "completed"},
        }


class _NoPipelineAdvisorSkill(BaseSkill):
    name = "select_evaluated_rag_pipeline"
    description = "没有合格公开回归管线"
    parameters = {"type": "object", "properties": {}}

    def run(self, context: SkillContext, **kwargs) -> dict:
        return {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "no_eligible_pipeline",
            "selected": None,
            "candidates": [],
        }


def test_create_task_plan_filters_unknown_skills_and_normalizes_steps():
    response = _FakeMsg(
        content=json.dumps(
            {
                "objective": "整理材料并形成结论",
                "steps": [
                    {
                        "title": "检索材料",
                        "skill": "search_knowledge_base",
                        "success_criteria": "返回至少一个来源",
                    },
                    {
                        "title": "越权动作",
                        "skill": "unknown_tool",
                        "success_criteria": "不应进入计划",
                    },
                ],
            },
            ensure_ascii=False,
        )
    )

    plan = create_task_plan(
        question="整理材料",
        available_skills=[
            {
                "name": "search_knowledge_base",
                "description": "检索上传文档",
            }
        ],
        llm=lambda *args, **kwargs: response,
        max_steps=4,
    )

    assert plan["contract"] == "agent_plan_v1"
    assert plan["objective"] == "整理材料并形成结论"
    assert plan["status"] == "pending"
    assert [step["skill"] for step in plan["steps"]] == [
        "search_knowledge_base"
    ]
    assert plan["steps"][0]["id"] == "step-1"
    assert plan["steps"][0]["status"] == "pending"
    assert plan["steps"][0]["attempts"] == 0


def test_evidence_delivery_plan_injects_public_eval_pipeline_selection():
    response = _FakeMsg(
        content=json.dumps(
            {
                "objective": "生成可信研究报告",
                "steps": [
                    {
                        "title": "生成研究报告",
                        "skill": "generate_verified_research_report",
                        "success_criteria": "产物通过证据门",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    plan = create_task_plan(
        question="生成可信研究报告",
        available_skills=[
            {
                "name": "select_evaluated_rag_pipeline",
                "description": "根据公开评测选择管线",
            },
            {
                "name": "generate_verified_research_report",
                "description": "生成证据闭环报告",
            },
        ],
        llm=lambda *args, **kwargs: response,
        max_steps=4,
    )

    assert [step["skill"] for step in plan["steps"]] == [
        "select_evaluated_rag_pipeline",
        "generate_verified_research_report",
    ]
    assert [step["id"] for step in plan["steps"]] == ["step-1", "step-2"]


def test_explicit_grounded_research_report_enforces_delivery_when_planner_omits_it():
    response = _FakeMsg(
        content=json.dumps(
            {
                "objective": "先做普通检索",
                "steps": [
                    {
                        "title": "检索知识库",
                        "skill": "search_knowledge_base",
                        "success_criteria": "找到相关片段",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    plan = create_task_plan(
        question="生成逐结论有依据的研究报告，并在证据不足时明确拒答",
        available_skills=[
            {
                "name": "search_knowledge_base",
                "description": "检索知识库",
            },
            {
                "name": "select_evaluated_rag_pipeline",
                "description": "根据公开评测选择管线",
            },
            {
                "name": "generate_verified_research_report",
                "description": "生成证据闭环报告",
            },
        ],
        llm=lambda *args, **kwargs: response,
        max_steps=4,
    )

    assert [step["skill"] for step in plan["steps"]] == [
        "search_knowledge_base",
        "select_evaluated_rag_pipeline",
        "generate_verified_research_report",
    ]


def test_public_benchmark_grounded_report_wording_enforces_delivery():
    response = _FakeMsg(
        content=json.dumps(
            {
                "objective": "只检索公开语料",
                "steps": [
                    {
                        "title": "检索知识库",
                        "skill": "search_knowledge_base",
                        "success_criteria": "找到相关片段",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    plan = create_task_plan(
        question=(
            "核验公开问题并生成简短的逐结论有依据报告。"
            "自主选择已评测管线；证据不足时明确拒答。"
        ),
        available_skills=[
            {
                "name": "search_knowledge_base",
                "description": "检索知识库",
            },
            {
                "name": "select_evaluated_rag_pipeline",
                "description": "根据公开评测选择管线",
            },
            {
                "name": "generate_verified_research_report",
                "description": "生成证据闭环报告",
            },
        ],
        llm=lambda *args, **kwargs: response,
        max_steps=4,
    )

    assert [step["skill"] for step in plan["steps"]] == [
        "search_knowledge_base",
        "select_evaluated_rag_pipeline",
        "generate_verified_research_report",
    ]


def test_generic_report_is_not_promoted_to_verified_delivery():
    response = _FakeMsg(
        content=json.dumps(
            {
                "objective": "生成普通报告",
                "steps": [
                    {
                        "title": "生成报告",
                        "skill": "generate_report",
                        "success_criteria": "返回报告",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    plan = create_task_plan(
        question="整理这份文档并生成报告",
        available_skills=[
            {
                "name": "generate_report",
                "description": "生成普通报告",
            },
            {
                "name": "select_evaluated_rag_pipeline",
                "description": "根据公开评测选择管线",
            },
            {
                "name": "generate_verified_research_report",
                "description": "生成证据闭环报告",
            },
        ],
        llm=lambda *args, **kwargs: response,
        max_steps=4,
    )

    assert [step["skill"] for step in plan["steps"]] == ["generate_report"]


def test_grounded_research_report_survives_invalid_planner_output():
    plan = create_task_plan(
        question="请生成有依据的研究报告，并逐结论核验引用",
        available_skills=[
            {
                "name": "select_evaluated_rag_pipeline",
                "description": "根据公开评测选择管线",
            },
            {
                "name": "generate_verified_research_report",
                "description": "生成证据闭环报告",
            },
        ],
        llm=lambda *args, **kwargs: _FakeMsg(content="not-json"),
        max_steps=4,
    )

    assert plan["mode"] == "explicit"
    assert [step["skill"] for step in plan["steps"]] == [
        "select_evaluated_rag_pipeline",
        "generate_verified_research_report",
    ]


def test_evidence_delivery_plan_runs_complete_report_workflow_once():
    response = _FakeMsg(
        content=json.dumps(
            {
                "objective": "生成可信研究报告",
                "steps": [
                    {
                        "title": "起草研究报告",
                        "skill": "generate_verified_research_report",
                        "success_criteria": "产物通过证据门",
                    },
                    {
                        "title": "补充报告证据",
                        "skill": "generate_verified_research_report",
                        "success_criteria": "补充更多原文引用",
                    },
                    {
                        "title": "再次输出报告",
                        "skill": "generate_verified_research_report",
                        "success_criteria": "形成最终版本",
                    },
                ],
            },
            ensure_ascii=False,
        )
    )

    plan = create_task_plan(
        question="生成可信研究报告",
        available_skills=[
            {
                "name": "select_evaluated_rag_pipeline",
                "description": "根据公开评测选择管线",
            },
            {
                "name": "generate_verified_research_report",
                "description": "生成证据闭环报告",
            },
        ],
        llm=lambda *args, **kwargs: response,
        max_steps=4,
    )

    assert [step["skill"] for step in plan["steps"]] == [
        "select_evaluated_rag_pipeline",
        "generate_verified_research_report",
    ]
    assert plan["steps"][1]["title"] == "起草研究报告"


def test_langgraph_executes_and_completes_explicit_plan(monkeypatch):
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_PlanSkill)
    try:
        monkeypatch.setattr(orchestrator.settings, "agent_planning_mode", "explicit")
        monkeypatch.setattr(
            orchestrator,
            "create_task_plan",
            lambda **kwargs: {
                "contract": "agent_plan_v1",
                "objective": "执行测试技能",
                "mode": "explicit",
                "status": "pending",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "执行测试技能",
                        "skill": "plan_test_skill",
                        "success_criteria": "工具成功返回",
                        "status": "pending",
                        "attempts": 0,
                    }
                ],
            },
        )
        responses = iter(
            [
                _FakeMsg(
                    tool_calls=[
                        _FakeToolCall("planned-call", "plan_test_skill", "{}")
                    ]
                ),
                _FakeMsg(content="计划已完成"),
            ]
        )
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda messages, tools=None, tool_choice="auto": next(responses),
        )

        result = orchestrator.run_agent(user_id=1, question="执行计划")

        assert result["answer"] == "计划已完成"
        assert result["plan"]["status"] == "completed"
        assert result["plan"]["steps"][0]["status"] == "completed"
        assert result["plan"]["steps"][0]["attempts"] == 1
        assert result["trace"][0]["plan_step_id"] == "step-1"
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_provider_failure_ends_agent_with_failed_plan(monkeypatch):
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_PlanSkill)
    try:
        monkeypatch.setattr(orchestrator.settings, "agent_planning_mode", "explicit")
        monkeypatch.setattr(
            orchestrator,
            "create_task_plan",
            lambda **kwargs: {
                "contract": "agent_plan_v1",
                "objective": "生成有依据的报告",
                "status": "pending",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "执行报告技能",
                        "skill": "plan_test_skill",
                        "success_criteria": "返回经过验证的产物",
                        "status": "pending",
                        "attempts": 0,
                    }
                ],
            },
        )
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("provider unavailable")
            ),
        )

        result = orchestrator.run_agent(
            user_id=1,
            question="生成有依据的报告",
        )

        assert result["status"] == "failed"
        assert result["answer"] == "模型服务暂不可用，本次任务未完成。请稍后重试。"
        assert result["plan"]["status"] == "failed"
        assert result["plan"]["steps"][0]["status"] == "failed"
        assert result["trace"][-1] == {
            "step": 0,
            "skill": "model_provider",
            "ok": False,
            "graph_node": "supervisor",
            "failure_code": "provider_unavailable",
        }
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_agent_feeds_evaluation_selection_into_evidence_delivery(monkeypatch):
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_PipelineAdvisorSkill)
    registry.register_skill(_VerifiedDeliverySkill)
    _VerifiedDeliverySkill.last_pipeline_id = None
    _PipelineAdvisorSkill.last_kwargs = None
    try:
        monkeypatch.setattr(orchestrator.settings, "agent_planning_mode", "explicit")
        monkeypatch.setattr(
            orchestrator,
            "create_task_plan",
            lambda **kwargs: {
                "contract": "agent_plan_v1",
                "objective": "生成评测驱动的证据报告",
                "mode": "explicit",
                "status": "pending",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "选择管线",
                        "skill": "select_evaluated_rag_pipeline",
                        "success_criteria": "返回公开评测决策",
                        "status": "pending",
                        "attempts": 0,
                    },
                    {
                        "id": "step-2",
                        "title": "生成报告",
                        "skill": "generate_verified_research_report",
                        "success_criteria": "通过证据质量门",
                        "status": "pending",
                        "attempts": 0,
                    },
                ],
            },
        )
        responses = iter(
            [
                _FakeMsg(
                    tool_calls=[
                        _FakeToolCall(
                            "quality-call",
                            "select_evaluated_rag_pipeline",
                            (
                                '{"dataset_id":3,"language":"en",'
                                '"target":"grounded_generation"}'
                            ),
                        )
                    ]
                ),
                _FakeMsg(
                    tool_calls=[
                        _FakeToolCall(
                            "delivery-call",
                            "generate_verified_research_report",
                            '{"topic":"DocMind","pipeline_id":"configured"}',
                        )
                    ]
                ),
                _FakeMsg(content="报告已通过证据门"),
            ]
        )
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda messages, tools=None, tool_choice="auto": next(responses),
        )

        result = orchestrator.run_agent(
            user_id=1,
            question="生成评测驱动的证据报告",
        )

        assert [row["skill"] for row in result["trace"]] == [
            "select_evaluated_rag_pipeline",
            "generate_verified_research_report",
        ]
        assert _PipelineAdvisorSkill.last_kwargs == {"target": "retrieval"}
        assert _VerifiedDeliverySkill.last_pipeline_id == "hybrid"
        assert result["plan"]["status"] == "completed"
        assert result["artifacts"][0]["verification"]["passed"] is True
        selection = result["trace"][0]["pipeline_selection"]
        assert selection["dataset"]["source_snapshot_fingerprint"] == "b" * 64
        assert selection["selected"] == {
            "run_id": 8,
            "dataset_id": 3,
            "pipeline_id": "hybrid",
            "pipeline_fingerprint": "c" * 64,
            "comparison_role": "candidate",
            "release_status": "approved",
            "error_gate_summary": {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "unavailable": 0,
            },
        }
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_agent_blocks_evidence_delivery_without_an_evaluated_pipeline(
    monkeypatch,
):
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    registry.register_skill(_NoPipelineAdvisorSkill)
    registry.register_skill(_VerifiedDeliverySkill)
    _VerifiedDeliverySkill.last_pipeline_id = None
    try:
        monkeypatch.setattr(orchestrator.settings, "agent_planning_mode", "explicit")
        monkeypatch.setattr(
            orchestrator,
            "create_task_plan",
            lambda **kwargs: {
                "contract": "agent_plan_v1",
                "objective": "生成评测驱动的证据报告",
                "mode": "explicit",
                "status": "pending",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "选择管线",
                        "skill": "select_evaluated_rag_pipeline",
                        "success_criteria": "返回公开评测决策",
                        "status": "pending",
                        "attempts": 0,
                    },
                    {
                        "id": "step-2",
                        "title": "生成报告",
                        "skill": "generate_verified_research_report",
                        "success_criteria": "通过证据质量门",
                        "status": "pending",
                        "attempts": 0,
                    },
                ],
            },
        )
        responses = iter(
            [
                _FakeMsg(
                    tool_calls=[
                        _FakeToolCall(
                            "quality-call",
                            "select_evaluated_rag_pipeline",
                            '{"target":"retrieval"}',
                        )
                    ]
                ),
                _FakeMsg(
                    tool_calls=[
                        _FakeToolCall(
                            "delivery-call",
                            "generate_verified_research_report",
                            '{"topic":"DocMind","pipeline_id":"configured"}',
                        )
                    ]
                ),
                _FakeMsg(content="当前没有合格管线，无法生成证据报告。"),
            ]
        )
        monkeypatch.setattr(
            orchestrator,
            "chat_completion",
            lambda messages, tools=None, tool_choice="auto": next(responses),
        )

        result = orchestrator.run_agent(
            user_id=1,
            question="生成评测驱动的证据报告",
        )

        assert _VerifiedDeliverySkill.last_pipeline_id is None
        assert result["plan"]["status"] == "partial"
        assert result["plan"]["steps"][1]["status"] == "failed"
        assert result["trace"][-1]["skill"] == "generate_verified_research_report"
        assert result["trace"][-1]["ok"] is False
        assert result["trace"][-1]["workflow"] == {
            "contract": "evaluated_pipeline_delivery_gate_v1",
            "status": "failed",
        }
        assert result["artifacts"] == []
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_final_answer_completes_non_tool_synthesis_step():
    plan = {
        "contract": "agent_plan_v1",
        "objective": "检索并总结",
        "status": "running",
        "steps": [
            {"id": "step-1", "skill": "web_search", "status": "completed"},
            {"id": "step-2", "skill": None, "status": "pending"},
        ],
    }

    completed = finalize_plan(plan)

    assert completed["status"] == "completed"
    assert completed["steps"][1]["status"] == "completed"


def test_failed_artifact_quality_marks_trace_and_plan_step_failed():
    state = {
        "trace": [],
        "artifacts": [],
        "messages": [],
        "plan": {
            "contract": "agent_plan_v1",
            "objective": "生成可信报告",
            "status": "running",
            "steps": [
                {
                    "id": "step-1",
                    "skill": "generate_report",
                    "status": "running",
                }
            ],
        },
    }
    current = {
        "id": "quality-call",
        "name": "generate_report",
        "args": {},
        "step": 0,
        "plan_step_id": "step-1",
    }
    result = {
        "type": "report",
        "artifact_kind": "file",
        "verification": {
            "contract": "artifact_quality_v1",
            "passed": False,
            "claims": [{"text": "不应进入 trace 的产物正文"}],
            "checks": [
                {
                    "id": "evidence_coverage",
                    "passed": False,
                    "detail": "产物没有实质内容",
                }
            ],
        },
        "download": {
            "filename": "failed.md",
            "encoding": "text",
            "content": "不应进入工具消息的下载正文",
        },
    }

    updated = orchestrator._record_tool_result(state, current, result)

    assert updated["trace"][0]["ok"] is False
    assert updated["plan"]["steps"][0]["status"] == "failed"
    assert "claims" not in updated["trace"][0]["verification"]
    assert updated["trace"][0]["verification"]["checks"] == [
        {"id": "evidence_coverage", "passed": False}
    ]
    assert updated["artifacts"][0]["download"]["content"].startswith("不应进入")
    assert "不应进入工具消息的下载正文" not in updated["messages"][0]["content"]


def test_quality_interventions_are_retained_in_agent_trace_without_query_text():
    state = {"trace": [], "artifacts": [], "messages": [], "plan": None}
    current = {
        "id": "quality-retrieval",
        "name": "search_knowledge_base",
        "args": {"query": "PRIVATE_QUERY"},
        "step": 0,
    }
    result = {
        "type": "kb_search",
        "grounding": {
            "mode": "quality_adaptive_rag",
            "quality_interventions": [
                {
                    "contract": "quality_adaptive_retrieval_v1",
                    "action": "switch_pipeline",
                    "reason": "weak_retrieval",
                    "status": "applied",
                    "attempt": 2,
                    "max_interventions": 4,
                    "from_pipeline_id": "configured",
                    "to_pipeline_id": "hybrid",
                    "query": "PRIVATE_QUERY",
                    "quality_before": {
                        "status": "weak_retrieval",
                        "hit_count": 0,
                        "unique_chunk_count": 0,
                        "stale_hit_count": 0,
                        "best_local_rerank_score": None,
                        "query": "PRIVATE_QUERY",
                    },
                    "quality_after": {
                        "status": "sufficient",
                        "hit_count": 3,
                        "unique_chunk_count": 3,
                        "stale_hit_count": 0,
                        "best_local_rerank_score": 0.72,
                        "source_text": "PRIVATE_SOURCE_TEXT",
                    },
                }
            ],
        },
    }

    updated = orchestrator._record_tool_result(state, current, result)
    trace = updated["trace"][0]
    serialized = str(trace["grounding"]["quality_interventions"])

    assert trace["grounding"]["quality_interventions"][0]["action"] == (
        "switch_pipeline"
    )
    observation = trace["grounding"]["quality_interventions"][0]
    assert observation["quality_before"]["status"] == "weak_retrieval"
    assert observation["quality_after"]["hit_count"] == 3
    assert "PRIVATE_QUERY" not in serialized
    assert "PRIVATE_SOURCE_TEXT" not in serialized
