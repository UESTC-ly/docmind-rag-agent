"""Agent 编排器：agentic 系统的核心主循环。

流程（ReAct 式）：
  1. 把用户消息 + 历史 + 系统提示交给 LLM，附上所有可用 Skill 的工具定义
  2. LLM 要么直接回答，要么返回 tool_calls（决定调哪个 Skill、传什么参数）
  3. 若有 tool_calls：逐个执行 Skill，把结果作为 tool 消息塞回对话
  4. 回到 1，直到 LLM 给出最终答案，或达到最大步数（防死循环）

返回：最终答案 + 产出物(artifacts，如思维导图/图谱数据) + 执行轨迹(trace)
"""

import json

import app.skills  # noqa: F401  触发所有技能注册
from app.config import settings
from app.services.llm_service import chat_completion
from app.skills.base import SkillContext
from app.skills.registry import all_tools, get_skill

SYSTEM_PROMPT = """你是 DocMind 的智能文档助手。你可以调用工具来完成任务：
- 回答文档相关问题：用 search_knowledge_base 检索知识库
- 生成思维导图：用 generate_mindmap
- 生成关系图谱：用 generate_relation_graph
- 写报告：用 generate_report
- 知识库答不了或需要外部信息：用 web_search

规则：
1. 根据用户意图自主选择合适的工具，可以多步调用。
2. 拿到工具结果后，用中文给用户清晰的最终回复。
3. 如果生成了思维导图/图谱/报告，在回复里说明已生成，正文数据在产出物里。
4. 不要编造工具没返回的信息。"""


def run_agent(
    user_id: int,
    question: str,
    history: list[dict] | None = None,
    document_id: int | None = None,
) -> dict:
    """执行 Agent 主循环，返回 {answer, artifacts, trace}。"""
    context = SkillContext(user_id=user_id, document_id=document_id)
    tools = all_tools()

    # 已选定文档时，明确告知 LLM 无需再向用户索要文档 ID，
    # 否则模型会保守地反问"请提供文档"，而不是直接调 search_knowledge_base。
    system_prompt = SYSTEM_PROMPT
    if document_id is not None:
        system_prompt += (
            f"\n\n当前用户已选定文档（document_id={document_id}），"
            "检索类工具会自动作用于该文档，不要向用户索要文档 ID，直接调用工具。"
        )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    artifacts: list[dict] = []  # 结构化产出（思维导图/图谱/报告）
    trace: list[dict] = []  # 执行轨迹，便于调试和前端展示"思考过程"

    for step in range(settings.agent_max_steps):
        llm_msg = chat_completion(messages, tools=tools)

        # LLM 没有调用工具 → 得到最终答案，结束
        if not llm_msg.tool_calls:
            return {
                "answer": llm_msg.content or "",
                "artifacts": artifacts,
                "trace": trace,
            }

        # 把 assistant 的 tool_calls 消息加回上下文（OpenAI 协议要求）
        messages.append(
            {
                "role": "assistant",
                "content": llm_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in llm_msg.tool_calls
                ],
            }
        )

        # 逐个执行工具调用
        for tc in llm_msg.tool_calls:
            skill_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            skill = get_skill(skill_name)
            if skill is None:
                result = {"error": f"未知技能: {skill_name}"}
            else:
                try:
                    result = skill.run(context, **args)
                except Exception as exc:  # noqa: BLE001
                    result = {"error": f"技能执行失败: {exc}"}

            trace.append({"step": step, "skill": skill_name, "args": args})

            # 有结构化产出的技能，收集到 artifacts
            if result.get("type") in {"mindmap", "relation_graph", "report"}:
                artifacts.append(result)

            # 把工具结果塞回对话，供 LLM 下一步参考
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)[:4000],
                }
            )

    # 达到最大步数仍没收敛，兜底返回
    return {
        "answer": "任务较复杂，未能在限定步数内完成，请尝试拆分问题。",
        "artifacts": artifacts,
        "trace": trace,
    }
