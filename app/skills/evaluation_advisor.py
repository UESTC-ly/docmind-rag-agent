"""Agent Skill that turns RAG regression evidence into a pipeline decision."""

from app.database import SyncSessionLocal
from app.services.evaluation.pipeline_selection import (
    recommend_evaluated_pipeline,
)
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill


@register_skill
class EvaluatedPipelineAdvisorSkill(BaseSkill):
    name = "select_evaluated_rag_pipeline"
    description = (
        "从同一公开数据集上的可复现评测、badcase 回归门禁和当前管线指纹中，"
        "选择可用于后续文档任务的 RAG pipeline；不会跨数据集直接比较分数。"
    )
    grounding_mode = "public_evaluation_evidence"
    parameters = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["retrieval", "grounded_generation"],
                "description": "选择要优化的能力层，默认 retrieval。",
            },
            "dataset_id": {
                "type": "integer",
                "description": "可选；只使用指定的公开评测数据集。",
            },
            "language": {
                "type": "string",
                "description": "可选；只使用同语言的公开评测数据集，如 zh、en。",
            },
        },
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        with SyncSessionLocal() as db:
            return recommend_evaluated_pipeline(
                db,
                user_id=context.user_id,
                target=str(kwargs.get("target") or "retrieval"),
                dataset_id=kwargs.get("dataset_id"),
                language=kwargs.get("language"),
            )
