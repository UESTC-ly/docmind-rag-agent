"""思维导图技能：读文档 → LLM 抽取层级结构 → 输出 Mermaid mindmap 文本。

技术点：结构化输出。前端拿到 Mermaid 文本可直接渲染成图。
"""

from app.services.llm_service import chat_completion
from app.services.artifact_verification import verify_mermaid_artifact
from app.skills._helpers import fetch_document_text
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill

_PROMPT = """你是思维导图生成器。根据下面的文档内容，提炼核心主题和层级结构，
输出 Mermaid mindmap 语法。要求：
- 只输出 Mermaid 代码，不要额外解释，不要 ```markdown 包裹。
- 根节点是文档主题，往下 2-3 层展开关键点。
- 用中文。

格式示例：
mindmap
  root((文档主题))
    分支一
      要点A
      要点B
    分支二
      要点C

文档内容：
{content}"""


def _strip_mermaid_fence(raw: str) -> str:
    """移除模型常见的 ```mermaid / ``` 代码围栏，不误留语言标签。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    return text


@register_skill
class MindmapSkill(BaseSkill):
    name = "generate_mindmap"
    description = "把指定文档的内容生成为思维导图（Mermaid 格式）。当用户想要梳理文档结构、生成脑图/思维导图时使用。"
    grounding_mode = "document_prefix"
    produces_download = True
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "integer",
                "description": "可选，要生成思维导图的文档 ID；未传时使用当前选中文档。",
            }
        },
        "required": [],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        document_id = kwargs.get("document_id") or context.document_id
        if document_id is None:
            return {"error": "请先选择文档或传入 document_id"}
        content = fetch_document_text(context.user_id, document_id)
        if not content:
            return {"error": "文档不存在或无内容", "document_id": document_id}

        # 内容过长时截断，思维导图不需要全文细节
        content = content[:8000]
        msg = chat_completion(
            [{"role": "user", "content": _PROMPT.format(content=content)}],
            temperature=0.2,
        )
        mermaid = _strip_mermaid_fence(msg.content or "")
        verification = verify_mermaid_artifact(
            mermaid,
            artifact_type="mindmap",
        )
        return {
            "type": "mindmap",
            "artifact_kind": "file",
            "format": "mermaid",
            "document_id": document_id,
            "content": mermaid,
            "verification": verification,
            "grounding": {
                "mode": "document_prefix",
                "document_ids": [document_id],
                "max_chars": 8000,
            },
            "download": {
                "filename": f"document_{document_id}_mindmap.mmd",
                "mime_type": "text/plain;charset=utf-8",
                "encoding": "text",
                "content": mermaid,
            },
        }
