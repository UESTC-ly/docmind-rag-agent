"""知识库检索技能：把 RAG 检索包装成 Agent 的一个工具。

有了它，Agent 才能自主决定"这个问题该查知识库"还是"该联网搜索"。
"""

from app.config import settings
from app.services.skill_retrieval import retrieve_for_skill
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

        hits = retrieve_for_skill(
            context.user_id,
            query,
            top_k=settings.retrieval_top_k,
            document_id=document_id,
        )
        sources = [
            {
                "document_id": h["document_id"],
                "chunk_index": h.get("chunk_index"),
                "score": round(h.get("score", 0.0), 4),
            }
            for h in hits
        ]
        return {
            "type": "kb_search",
            "grounding": {"mode": "hybrid_rag", "sources": sources},
            "query": query,
            "results": [
                {
                    "content": h["content"],
                    "document_id": h["document_id"],
                    "chunk_index": h.get("chunk_index"),
                    "score": round(h.get("score", 0.0), 4),
                }
                for h in hits
            ],
        }
