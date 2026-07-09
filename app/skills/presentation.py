"""PPT 生成技能：根据材料生成可下载 .pptx 文件。"""

import base64
import json
import re

from app.services.llm_service import chat_completion
from app.skills._helpers import fetch_material_text
from app.skills.base import BaseSkill, SkillContext
from app.skills.pptx_builder import Slide, build_pptx
from app.skills.registry import register_skill

_PROMPT = """你是 PPT 内容策划助手。请根据材料生成演示文稿结构。

严格输出 JSON，不要解释，不要 ``` 包裹。格式：
{{
  "slides": [
    {{"title": "封面标题", "bullets": ["副标题或说明"]}},
    {{"title": "页面标题", "bullets": ["要点1", "要点2", "要点3"]}}
  ]
}}

要求：
- 总页数控制在 {slide_count} 页。
- 每页标题简短，每页 2-5 个要点。
- 内容必须基于材料；材料不足时用“材料未提及”说明，不要编造。
- 用中文。

主题：{topic}

材料：
{content}"""


def _safe_filename(name: str, suffix: str) -> str:
    base = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name.strip(), flags=re.UNICODE)
    base = base.strip("_")[:60] or "docmind_presentation"
    return f"{base}{suffix}"


def _strip_json_fence(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _parse_slides(raw: str, topic: str, slide_count: int) -> list[Slide]:
    try:
        data = json.loads(_strip_json_fence(raw))
        items = data.get("slides", [])
    except json.JSONDecodeError:
        items = []

    slides: list[Slide] = []
    for item in items[:slide_count]:
        title = str(item.get("title") or topic).strip()[:80]
        bullets = [
            str(b).strip()[:160]
            for b in item.get("bullets", [])
            if str(b).strip()
        ][:5]
        if title:
            slides.append(Slide(title=title, bullets=bullets or ["材料未提及"]))

    if not slides:
        slides = [Slide(title=topic, bullets=["材料不足，未生成有效页面结构"])]
    return slides


@register_skill
class PresentationSkill(BaseSkill):
    name = "generate_presentation"
    description = "根据已上传材料制作一份可下载的 PowerPoint 演示文稿（.pptx）。当用户要求制作 PPT、汇报演示、答辩材料、展示稿时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "PPT 主题"},
            "slide_count": {
                "type": "integer",
                "description": "期望页数，默认 6，范围 3-10",
                "minimum": 3,
                "maximum": 10,
            },
            "document_id": {
                "type": "integer",
                "description": "可选，限定使用某个文档材料；不传则汇总用户全部文档。",
            },
        },
        "required": ["topic"],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        topic = str(kwargs["topic"]).strip()
        document_id = kwargs.get("document_id") or context.document_id
        slide_count = max(3, min(10, int(kwargs.get("slide_count") or 6)))

        material, used_doc_ids = fetch_material_text(
            context.user_id, document_id=document_id, max_chars=14000
        )
        if not material:
            return {"error": "未找到可用于制作 PPT 的材料", "document_id": document_id}

        msg = chat_completion(
            [
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        topic=topic,
                        slide_count=slide_count,
                        content=material,
                    ),
                }
            ],
            temperature=0.35,
        )
        slides = _parse_slides(msg.content or "", topic, slide_count)
        pptx = build_pptx(slides)
        filename = _safe_filename(f"{topic}_PPT", ".pptx")

        return {
            "type": "presentation",
            "artifact_kind": "file",
            "topic": topic,
            "document_ids": used_doc_ids,
            "slides": [
                {"title": slide.title, "bullets": slide.bullets}
                for slide in slides
            ],
            "download": {
                "filename": filename,
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "encoding": "base64",
                "content": base64.b64encode(pptx).decode("ascii"),
            },
        }
