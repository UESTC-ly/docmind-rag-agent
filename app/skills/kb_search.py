"""Quality-adaptive knowledge-base retrieval for the outer Agent."""

from app.config import settings
from app.database import SyncSessionLocal
from app.services.evaluation.pipeline_selection import (
    recommend_evaluated_pipeline,
)
from app.services.quality_adaptation import (
    QualityPolicyConfig,
    adaptive_retrieve_sync,
    evaluated_pipeline_allowlist,
)
from app.services.skill_retrieval import run_pipeline_for_skill
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill


@register_skill
class KnowledgeBaseSearchSkill(BaseSkill):
    name = "search_knowledge_base"
    description = "在用户已上传的文档知识库中做语义检索，返回相关片段。回答文档相关问题时优先用它。"
    grounding_mode = "hybrid_rag"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索问题"},
            "document_id": {
                "type": "integer",
                "description": "可选，限定只在某篇文档内检索",
            },
        },
        "required": ["query"],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        query = kwargs["query"]
        document_id = kwargs.get("document_id") or context.document_id
        decision: dict = {
            "contract": "evaluated_pipeline_selection_v1",
            "status": "unavailable",
            "selected": None,
            "candidates": [],
        }
        try:
            with SyncSessionLocal() as db:
                decision = recommend_evaluated_pipeline(
                    db,
                    user_id=context.user_id,
                    target="retrieval",
                )
        except Exception:
            # Evaluation evidence is an optimization input. Retrieval itself
            # remains available and records that no evaluated switch was used.
            pass
        selected = decision.get("selected") or {}
        pipeline_id = str(selected.get("pipeline_id") or "configured")

        def _retrieve(
            effective_query: str,
            effective_pipeline_id: str,
            top_k: int,
        ):
            return run_pipeline_for_skill(
                user_id=context.user_id,
                query=effective_query,
                top_k=top_k,
                document_id=document_id,
                pipeline=effective_pipeline_id,
            )

        retrieval = adaptive_retrieve_sync(
            retrieve=_retrieve,
            query=query,
            pipeline_id=pipeline_id,
            top_k=settings.retrieval_top_k,
            allowed_pipeline_ids=evaluated_pipeline_allowlist(decision),
            config=QualityPolicyConfig.from_settings(),
        )
        hits = retrieval.hits
        sources = [
            {
                "document_id": h["document_id"],
                "chunk_index": h.get("chunk_index"),
                "citation_id": h.get("citation_id"),
                "score": round(h.get("score", 0.0), 4),
            }
            for h in hits
        ]
        return {
            "type": "kb_search",
            "summary": (
                "已获得可用证据"
                if retrieval.deliverable
                else "没有当前、可交付的文档证据"
            ),
            "grounding": {
                "mode": "quality_adaptive_rag",
                "pipeline_id": retrieval.pipeline_id,
                "pipeline_fingerprint": getattr(
                    retrieval.execution,
                    "fingerprint",
                    None,
                ),
                "sources": sources,
                "quality_interventions": retrieval.quality_interventions,
            },
            "query": query,
            "effective_query": retrieval.query,
            "quality": retrieval.quality.compact(),
            "quality_interventions": retrieval.quality_interventions,
            "terminal_reason": retrieval.terminal_reason,
            "workflow": {
                "contract": "quality_adaptive_retrieval_v1",
                "status": (
                    "completed" if retrieval.deliverable else "failed"
                ),
                "quality_interventions": retrieval.quality_interventions,
            },
            "pipeline_selection": {
                "status": decision.get("status"),
                "selected_pipeline_id": selected.get("pipeline_id"),
                "allowlisted_pipeline_ids": list(
                    evaluated_pipeline_allowlist(decision)
                ),
            },
            "results": [
                {
                    "content": h["content"],
                    "document_id": h["document_id"],
                    "chunk_index": h.get("chunk_index"),
                    "citation_id": h.get("citation_id"),
                    "score": round(h.get("score", 0.0), 4),
                }
                for h in hits
            ],
        }
