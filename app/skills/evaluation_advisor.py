"""Agent Skill that turns RAG regression evidence into a pipeline decision."""

from typing import Any

from sqlalchemy import select

from app.database import SyncSessionLocal
from app.models.evaluation import EvalDataset
from app.services.evaluation.pipeline_selection import (
    recommend_evaluated_pipeline,
)
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill


def _resolve_dataset_scope(
    db: Any,
    *,
    context: SkillContext,
    requested_dataset_id: Any,
) -> tuple[int | None, str]:
    """Prefer an explicit eval dataset, then map the current document scope."""

    try:
        requested_id = (
            int(requested_dataset_id)
            if requested_dataset_id is not None
            else None
        )
    except (TypeError, ValueError):
        requested_id = None

    if requested_id is not None:
        requested = db.get(EvalDataset, requested_id)
        if requested is not None and requested.user_id == context.user_id:
            return requested.id, "explicit_evaluation_dataset"

    if context.document_id is not None:
        scoped_id = db.scalar(
            select(EvalDataset.id)
            .where(
                EvalDataset.user_id == context.user_id,
                EvalDataset.document_id == context.document_id,
                EvalDataset.label_source == "public_ground_truth",
                EvalDataset.release_eligible.is_(True),
            )
            .order_by(EvalDataset.id.desc())
            .limit(1)
        )
        if scoped_id is not None:
            return int(scoped_id), "current_document_public_dataset"

    if requested_id is not None:
        return requested_id, "unresolved_explicit_dataset"
    return None, "latest_public_dataset"


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
                "description": (
                    "可选；仅在用户明确指定内部评测集 ID 时填写，"
                    "不是当前 document_id。"
                ),
            },
            "language": {
                "type": "string",
                "description": "可选；只使用同语言的公开评测数据集，如 zh、en。",
            },
        },
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        with SyncSessionLocal() as db:
            dataset_id, resolution = _resolve_dataset_scope(
                db,
                context=context,
                requested_dataset_id=kwargs.get("dataset_id"),
            )
            decision = recommend_evaluated_pipeline(
                db,
                user_id=context.user_id,
                target=str(kwargs.get("target") or "retrieval"),
                dataset_id=dataset_id,
                language=kwargs.get("language"),
            )
        decision["scope_resolution"] = {
            "requested_dataset_id": kwargs.get("dataset_id"),
            "resolved_dataset_id": dataset_id,
            "strategy": resolution,
        }
        return decision
