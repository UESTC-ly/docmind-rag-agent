"""报告生成技能：根据知识库内容，就某主题生成可下载结构化报告。

技术点：多步规划。检索相关内容 → 组织大纲 → 分段生成 → 汇总成文。
"""

from app.config import settings
from app.services.llm_service import chat_completion
from app.services.skill_retrieval import retrieve_for_skill
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill

_OUTLINE_PROMPT = """根据主题「{topic}」和下面的资料，列出报告大纲（3-5 个小节标题）。
只输出标题，每行一个，不要编号和额外文字。

资料：
{context}"""

_SECTION_PROMPT = """你在写一份关于「{topic}」的报告。现在写「{section}」这一节。
只用下面资料里的信息，200-400 字，用中文，不要写标题。

资料：
{context}"""


@register_skill
class ReportSkill(BaseSkill):
    name = "generate_report"
    description = "就某个主题，基于已上传的知识库内容生成一份结构化报告。当用户要求'写报告''总结成文档''整理成材料'时使用。"
    grounding_mode = "hybrid_rag"
    produces_download = True
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "报告主题"},
            "document_id": {
                "type": "integer",
                "description": "可选，限定只基于某篇文档生成报告；未传时使用当前选中文档。",
            },
        },
        "required": ["topic"],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        topic = kwargs["topic"]
        document_id = kwargs.get("document_id") or context.document_id

        # 1. 检索主题相关内容
        hits = retrieve_for_skill(
            context.user_id,
            topic,
            top_k=settings.retrieval_top_k * 2,
            document_id=document_id,
        )
        if not hits:
            return {
                "error": "知识库中未找到与该主题相关的内容",
                "topic": topic,
                "document_id": document_id,
            }
        context_text = "\n\n".join(h["content"] for h in hits)

        # 2. 生成大纲
        outline_msg = chat_completion(
            [
                {
                    "role": "user",
                    "content": _OUTLINE_PROMPT.format(
                        topic=topic, context=context_text[:6000]
                    ),
                }
            ],
            temperature=0.4,
        )
        sections = [
            s.strip()
            for s in (outline_msg.content or "").split("\n")
            if s.strip()
        ][:5]

        # 3. 逐节生成
        parts = []
        for section in sections:
            sec_msg = chat_completion(
                [
                    {
                        "role": "user",
                        "content": _SECTION_PROMPT.format(
                            topic=topic,
                            section=section,
                            context=context_text[:6000],
                        ),
                    }
                ],
                temperature=0.5,
            )
            parts.append(f"## {section}\n\n{(sec_msg.content or '').strip()}")

        report = f"# {topic}\n\n" + "\n\n".join(parts)
        source_items = [
            {
                "document_id": h["document_id"],
                "chunk_index": h.get("chunk_index"),
                "score": h.get("score"),
            }
            for h in hits
        ]
        return {
            "type": "report",
            "artifact_kind": "file",
            "topic": topic,
            "document_id": document_id,
            "outline": sections,
            "content": report,
            "grounding": {"mode": "hybrid_rag", "sources": source_items},
            "download": {
                "filename": "report.md",
                "mime_type": "text/markdown;charset=utf-8",
                "encoding": "text",
                "content": report,
            },
        }
