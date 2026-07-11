"""联网搜索技能：知识库答不了时，Agent 自主判断并调用网络搜索补充。

技术点：工具自主调用——这是 agentic 的精髓。用免费的 DuckDuckGo，无需 API key。
"""

from app.skills.base import BaseSkill, SkillContext
from app.skills.registry import register_skill


@register_skill
class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "联网搜索实时/外部信息。当用户问题超出已上传文档的范围，或需要最新信息时使用。"
    grounding_mode = "web"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {
                "type": "integer",
                "description": "返回结果数，默认 5",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def run(self, context: SkillContext, **kwargs) -> dict:
        query = kwargs["query"]
        max_results = kwargs.get("max_results", 5)

        try:
            # ddgs：DuckDuckGo 搜索库，import 放函数内避免未装时影响其他技能
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"搜索失败: {exc}", "query": query}

        return {
            "type": "web_search",
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
                for r in results
            ],
        }
