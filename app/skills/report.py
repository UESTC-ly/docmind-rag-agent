"""报告生成技能：根据知识库内容，就某主题生成可下载结构化报告。

技术点：多步规划。检索相关内容 → 组织大纲 → 分段生成 → 汇总成文。
"""

from app.config import settings
from app.services.artifact_verification import (
    sanitize_cited_markdown,
    verify_cited_markdown,
)
from app.services.evidence import attach_citation_ids, build_evidence_context
from app.services.llm_service import chat_completion
from app.services.skill_retrieval import retrieve_for_skill
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill

_OUTLINE_PROMPT = """根据主题「{topic}」和下面的资料，列出报告大纲（3-5 个小节标题）。
只输出标题，每行一个，不要编号和额外文字。

资料：
{context}"""

_SECTION_PROMPT = """你在写一份关于「{topic}」的报告。现在写「{section}」这一节。
只用下面证据里的信息，200-400 字，用中文，不要写标题。
每个事实性句子末尾必须复制一个或多个真实证据编号，例如 [D12:C3]。
不得创造证据编号；没有证据支持的内容不要写。

证据：
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
        hits = attach_citation_ids(hits)
        context_text = build_evidence_context(hits)

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
        verification = verify_cited_markdown(
            report,
            hits,
            artifact_type="report",
            required_headings=tuple(sections),
        )
        fail_safe_applied = not verification["passed"]
        if fail_safe_applied:
            report = sanitize_cited_markdown(report, hits)
            report = (
                f"{report}\n\n> 证据不足或引用无效的句子已由质量门自动移除。"
            ).strip()
            verification = verify_cited_markdown(
                report,
                hits,
                artifact_type="report",
                required_headings=tuple(sections),
            )
        verification["fail_safe_applied"] = fail_safe_applied
        source_items = [
            {
                "citation_id": h["citation_id"],
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
            "verification": verification,
            "grounding": {"mode": "hybrid_rag", "sources": source_items},
            "download": {
                "filename": "report.md",
                "mime_type": "text/markdown;charset=utf-8",
                "encoding": "text",
                "content": report,
            },
        }
