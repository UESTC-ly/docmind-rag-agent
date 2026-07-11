"""关系拓扑图技能：从文档抽取实体及其关系，构成知识图谱。

技术点：GraphRAG。弥补纯向量检索抓不住"实体间关系"的短板。
LLM 输出结构化的 nodes + edges，前端可用图库渲染，也提供 Mermaid 备用。
"""

import json

from app.services.llm_service import chat_completion
from app.skills._helpers import fetch_document_text
from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill

_PROMPT = """你是知识图谱抽取器。从下面文档中抽取关键实体及它们之间的关系。
严格输出 JSON，格式：
{{
  "nodes": [{{"id": "实体名", "type": "人物|组织|概念|事件|地点"}}],
  "edges": [{{"source": "实体A", "target": "实体B", "relation": "关系描述"}}]
}}
要求：
- 只输出 JSON，不要解释，不要 ``` 包裹。
- 实体去重，关系聚焦重要的，控制在 20 个节点内。
- 用中文。

文档内容：
{content}"""


def _to_mermaid(nodes: list[dict], edges: list[dict]) -> str:
    """把 nodes/edges 转成 Mermaid graph 语法（备用渲染）。"""
    lines = ["graph TD"]
    names = []
    for node in nodes:
        name = str(node.get("id", "")).strip()
        if name and name not in names:
            names.append(name)
    for edge in edges:
        for key in ("source", "target"):
            name = str(edge.get(key, "")).strip()
            if name and name not in names:
                names.append(name)
    node_ids = {name: f"n{idx}" for idx, name in enumerate(names)}

    def _label(value: str) -> str:
        return value.replace("\n", " ").replace('"', "&quot;").replace("|", "&#124;")

    for name in names:
        lines.append(f'  {node_ids[name]}["{_label(name)}"]')
    for e in edges:
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        rel = _label(str(e.get("relation", "")).strip())
        if src and tgt:
            lines.append(f'  {node_ids[src]} -->|"{rel}"| {node_ids[tgt]}')
    return "\n".join(lines)


@register_skill
class GraphSkill(BaseSkill):
    name = "generate_relation_graph"
    description = "从指定文档抽取实体和关系，生成知识图谱/关系拓扑图。当用户想看文档中人物、概念、事件之间的关系网络时使用。"
    grounding_mode = "document_prefix"
    produces_download = True
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "integer",
                "description": "可选，要生成关系图的文档 ID；未传时使用当前选中文档。",
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

        content = content[:8000]
        msg = chat_completion(
            [{"role": "user", "content": _PROMPT.format(content=content)}],
            temperature=0.2,
        )
        raw = (msg.content or "").strip().removeprefix("```json").removeprefix(
            "```"
        ).removesuffix("```")

        try:
            data = json.loads(raw)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
        except json.JSONDecodeError:
            return {"error": "图谱抽取结果解析失败", "raw": raw[:500]}

        graph_data = {"nodes": nodes, "edges": edges}
        return {
            "type": "relation_graph",
            "artifact_kind": "file",
            "document_id": document_id,
            "nodes": nodes,
            "edges": edges,
            "mermaid": _to_mermaid(nodes, edges),
            "grounding": {
                "mode": "document_prefix",
                "document_ids": [document_id],
                "max_chars": 8000,
            },
            "download": {
                "filename": f"document_{document_id}_relation_graph.json",
                "mime_type": "application/json;charset=utf-8",
                "encoding": "text",
                "content": json.dumps(graph_data, ensure_ascii=False, indent=2),
            },
        }
