"""周报生成技能：根据用户上传材料生成可下载 Markdown 周报。"""

import re
from datetime import date

from app.services.llm_service import chat_completion
from app.skills._helpers import fetch_material_text
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill

_PROMPT = """你是专业项目周报撰写助手。请根据下面材料写一份中文周报。

要求：
- 输出 Markdown，不要使用 ``` 包裹。
- 结构必须包含：本周概览、本周完成、关键进展、问题与风险、下周计划。
- 内容必须基于材料，不要编造材料之外的事实；材料不足时明确写“材料未提及”。
- 语气适合给导师/主管/团队同步。

周报主题：{topic}
周期：{week}

材料：
{content}"""


def _safe_filename(name: str, suffix: str) -> str:
    base = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name.strip(), flags=re.UNICODE)
    base = base.strip("_")[:60] or "docmind"
    return f"{base}{suffix}"


@register_skill
class WeeklyReportSkill(BaseSkill):
    name = "generate_weekly_report"
    description = "根据已上传材料生成一份结构化中文周报，并提供 Markdown 文件下载。当用户要求写周报、工作汇报、进度周总结时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "周报主题或工作方向，可选；未提供时使用“项目周报”。",
            },
            "week": {
                "type": "string",
                "description": "周报周期，可选，如 2026-07-01 至 2026-07-07。",
            },
            "document_id": {
                "type": "integer",
                "description": "可选，限定使用某个文档材料；不传则汇总用户全部文档。",
            },
        },
        "required": [],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        document_id = kwargs.get("document_id") or context.document_id
        topic = (kwargs.get("topic") or "项目周报").strip()
        week = (kwargs.get("week") or date.today().isoformat()).strip()

        material, used_doc_ids = fetch_material_text(
            context.user_id, document_id=document_id, max_chars=12000
        )
        if not material:
            return {"error": "未找到可用于生成周报的材料", "document_id": document_id}

        msg = chat_completion(
            [
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        topic=topic, week=week, content=material
                    ),
                }
            ],
            temperature=0.3,
        )
        markdown = (msg.content or "").strip().removeprefix("```markdown").removeprefix(
            "```"
        ).removesuffix("```").strip()
        if not markdown:
            markdown = f"# {topic}\n\n材料未提及可写入周报的有效信息。"

        filename = _safe_filename(f"{topic}_{week}_周报", ".md")
        return {
            "type": "weekly_report",
            "artifact_kind": "file",
            "topic": topic,
            "week": week,
            "document_ids": used_doc_ids,
            "content": markdown,
            "download": {
                "filename": filename,
                "mime_type": "text/markdown;charset=utf-8",
                "encoding": "text",
                "content": markdown,
            },
        }
