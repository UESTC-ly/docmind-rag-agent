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
- 写周报/进度周总结：用 generate_weekly_report
- 制作 PPT/演示文稿/答辩材料：用 generate_presentation
- 知识库答不了或需要外部信息：用 web_search
- 目录化通用技能：可调用 Codex-style SKILL.md package（如 codex_note），由技能内部按 Markdown 指令规划并使用受控工具执行

规则：
1. 根据用户意图自主选择合适的工具，可以多步调用。
2. 拿到工具结果后，用中文给用户清晰的最终回复。
3. 如果生成了思维导图/图谱/报告/周报/PPT/通用技能文件，在回复里说明已生成，正文或下载文件在产出物里。
4. 不要编造工具没返回的信息。"""

ARTIFACT_TYPES = {
    "mindmap",
    "relation_graph",
    "report",
    "weekly_report",
    "presentation",
}


def _tool_message_content(result: dict) -> str:
    """压缩给 LLM 看的工具结果，避免把 base64 文件内容塞回上下文。"""
    if result.get("artifact_kind") != "file":
        return json.dumps(result, ensure_ascii=False)[:4000]

    compact = {
        key: value
        for key, value in result.items()
        if key != "download"
    }
    if "download" in result:
        compact["download"] = {
            "filename": result["download"].get("filename"),
            "mime_type": result["download"].get("mime_type"),
            "encoding": result["download"].get("encoding"),
            "content_omitted": True,
        }
    return json.dumps(compact, ensure_ascii=False)[:4000]


def run_agent(
    user_id: int,
    question: str,
    history: list[dict] | None = None,
    document_id: int | None = None,
    requested_skill: str | None = None,
) -> dict:
    """执行 Agent 主循环，返回 {answer, artifacts, trace}。"""
    context = SkillContext(user_id=user_id, document_id=document_id)
    tools = all_tools()
    selected = get_skill(requested_skill) if requested_skill is not None else None
    if requested_skill is not None and selected is None:
        return {
            "answer": f"指定技能不存在或未加载：{requested_skill}",
            "artifacts": [],
            "trace": [],
        }
    if selected is not None and not selected.available:
        return {
            "answer": (
                f"技能 {requested_skill} 当前未通过 DocMind 运行时兼容性审计："
                f"{selected.unavailable_reason}"
            ),
            "artifacts": [],
            "trace": [],
        }

    # 已选定文档时，明确告知 LLM 无需再向用户索要文档 ID，
    # 否则模型会保守地反问"请提供文档"，而不是直接调 search_knowledge_base。
    system_prompt = SYSTEM_PROMPT
    if document_id is not None:
        system_prompt += (
            f"\n\n当前用户已选定文档（document_id={document_id}），"
            "检索类工具会自动作用于该文档，不要向用户索要文档 ID，直接调用工具。"
        )
    if requested_skill is not None:
        system_prompt += (
            f"\n\n用户已在技能面板明确选择 {requested_skill}。"
            "第一步必须调用该技能，不允许只在聊天正文中模拟技能产出。"
        )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    artifacts: list[dict] = []  # 结构化产出（思维导图/图谱/报告）
    trace: list[dict] = []  # 执行轨迹，便于调试和前端展示"思考过程"

    for step in range(settings.agent_max_steps):
        tool_choice: str | dict = "auto"
        forced_skill_name: str | None = None
        if step == 0 and requested_skill is not None:
            forced_skill_name = requested_skill
            tool_choice = {
                "type": "function",
                "function": {"name": requested_skill},
            }
        elif step == 0 and document_id is not None:
            # 未锁定具体技能时，选定文档的 Agent 请求先强制走知识库 RAG；拿到真实
            # 片段后，模型仍可继续调用脑图、图谱、报告等后续技能。
            forced_skill_name = "search_knowledge_base"
            tool_choice = {
                "type": "function",
                "function": {"name": "search_knowledge_base"},
            }
        llm_msg = chat_completion(messages, tools=tools, tool_choice=tool_choice)

        # LLM 没有调用工具 → 得到最终答案，结束
        if not llm_msg.tool_calls:
            if forced_skill_name is not None:
                return {
                    "answer": (
                        f"模型未按要求调用技能 {forced_skill_name}，"
                        "为避免绕过 RAG/文件产出，本次没有接受模型的直接回答。"
                    ),
                    "artifacts": artifacts,
                    "trace": trace,
                }
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
            elif not skill.available:
                result = {
                    "error": (
                        f"技能 {skill_name} 当前不可调用："
                        f"{skill.unavailable_reason}"
                    )
                }
            else:
                try:
                    result = skill.run(context, **args)
                except Exception as exc:  # noqa: BLE001
                    result = {"error": f"技能执行失败: {exc}"}

            trace_item = {
                "step": step,
                "skill": skill_name,
                "args": args,
                "ok": "error" not in result,
            }
            if skill is not None:
                trace_item["grounding_mode"] = skill.grounding_mode
            grounding = result.get("grounding")
            if grounding:
                trace_item["grounding"] = grounding
            trace.append(trace_item)

            # 有结构化 / 文件产出的技能，收集到 artifacts
            if result.get("type") in ARTIFACT_TYPES or result.get("artifact_kind") == "file":
                artifacts.append(result)

            # 把工具结果塞回对话，供 LLM 下一步参考
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _tool_message_content(result),
                }
            )

    # 达到最大步数仍没收敛，兜底返回
    return {
        "answer": "任务较复杂，未能在限定步数内完成，请尝试拆分问题。",
        "artifacts": artifacts,
        "trace": trace,
    }
