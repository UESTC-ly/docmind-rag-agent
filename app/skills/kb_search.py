"""知识库检索技能：把 RAG 检索包装成 Agent 的一个工具。

有了它，Agent 才能自主决定"这个问题该查知识库"还是"该联网搜索"。
"""

from app.config import settings
from app.services.embedding_service import embed_query
from app.services.vector_store import search
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill


@register_skill
class KnowledgeBaseSearchSkill(BaseSkill):
    name = "search_knowledge_base"
    description = "在用户已上传的文档知识库中做语义检索，返回相关片段。回答文档相关问题时优先用它。"
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

        query_vector = embed_query(query)
        hits = search(
            query_vector,
            context.user_id,
            top_k=settings.retrieval_top_k,
            document_id=document_id,
        )
        return {
            "type": "kb_search",
            "query": query,
            "results": [
                {
                    "content": h["content"],
                    "document_id": h["document_id"],
                    "score": round(h["score"], 4),
                }
                for h in hits
            ],
        }
