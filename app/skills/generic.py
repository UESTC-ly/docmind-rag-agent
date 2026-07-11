"""Codex-style 通用 Skill 执行器。

GenericPackageSkill 把一个只有 `SKILL.md` 的 package 暴露成 OpenAI
Function Calling 工具。被 Agent 调用后，它会启动一个“技能内部 ReAct 循环”：

1. 把 SKILL.md、资源索引、用户任务交给 LLM；
2. LLM 可调用受控动作工具（读写文件、shell、MCP/browser/app adapter 等）；
3. 工具结果回填上下文，直到 LLM 给出最终答复；
4. 若写出了文件，把工作区产物打成 zip artifact 返回给前端下载。

这让 DocMind 从 v0.4 的“Python 技能注册表”演进到 v0.5 的
“Python 技能 + Codex-style 通用技能包”并存。
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from app.config import settings
from app.services.llm_service import chat_completion
from app.skills.base import BaseSkill, SkillContext
from app.skills.package_loader import SkillPackage
from app.skills.toolkit import SkillToolExecutor


class GenericPackageSkill(BaseSkill):
    """由 SKILL.md 驱动的通用技能。

    与 Python-backed skill 不同，它没有手写 `run()` 业务逻辑，而是在 `run()` 中
    解释当前 package 的 Markdown 指令，并让 LLM 通过受控工具完成任务。
    """

    execution_mode = "generic_package"
    grounding_mode = "rag_when_document_selected"
    produces_download = True

    def __init__(self, package: SkillPackage) -> None:
        self._package = package
        self.package_slug = package.slug
        self.name = package.name
        self.description = package.description
        self.parameters = package.parameters
        self.available = package.runtime_status == "ready"
        self.unavailable_reason = package.runtime_reason

    def load_package(self) -> SkillPackage:
        return self._package

    def apply_package_metadata(self) -> None:
        self.name = self._package.name
        self.description = self._package.description
        self.parameters = self._package.parameters

    def run(self, context: SkillContext, **kwargs) -> dict:
        task = str(kwargs.get("task") or kwargs.get("prompt") or "").strip()
        if not task:
            task = json.dumps(kwargs, ensure_ascii=False)
        inputs = kwargs.get("inputs") if isinstance(kwargs.get("inputs"), dict) else {}
        document_id = kwargs.get("document_id") or context.document_id

        executor = SkillToolExecutor(
            package=self._package,
            context=context,
            shell_enabled=settings.skill_shell_enabled,
            shell_allowed_commands=settings.skill_shell_allowed_command_set,
            shell_timeout_seconds=settings.skill_shell_timeout_seconds,
        )
        tools = executor.tool_definitions()
        actions: list[dict] = []
        grounding = None
        task_mentions_documents = any(
            marker in task.lower()
            for marker in ("文档", "材料", "上传", "document", "material", "attachment")
        )
        if document_id is not None or task_mentions_documents:
            grounding = executor.execute(
                "search_uploaded_documents",
                {
                    "query": task,
                    "document_id": document_id,
                    "max_chars": 12000,
                },
            )
            actions.append(
                {
                    "step": -1,
                    "tool": "search_uploaded_documents",
                    "args": {"query": task, "document_id": document_id},
                    "ok": bool(grounding.get("ok")),
                }
            )

        if grounding is not None and not grounding.get("ok"):
            final_answer = (
                "无法基于上传文档执行该技能："
                f"{grounding.get('error', 'RAG 检索失败')}"
            )
            write_result = executor.execute(
                "write_file",
                {
                    "path": f"outputs/{self._package.slug}-result.md",
                    "content": final_answer,
                },
            )
            actions.append(
                {
                    "step": 0,
                    "tool": "write_file",
                    "args": {"path": f"outputs/{self._package.slug}-result.md"},
                    "ok": bool(write_result.get("ok")),
                    "fallback": True,
                }
            )
            return self._build_result(final_answer, actions, executor, grounding)

        messages = [
            {"role": "system", "content": self._system_prompt(executor)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "inputs": inputs,
                        "document_id": document_id,
                        "rag_grounding": grounding,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        final_answer = ""
        for step in range(settings.skill_runner_max_steps):
            llm_msg = chat_completion(messages, tools=tools, temperature=0.2)

            if not llm_msg.tool_calls:
                final_answer = llm_msg.content or ""
                break

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

            for tc in llm_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = executor.execute(tool_name, args)
                actions.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "args": args,
                        "ok": bool(result.get("ok")),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _compact_tool_result(result),
                    }
                )
        else:
            final_answer = "通用技能执行步骤过多，已达到上限，请尝试缩小任务范围。"

        if not executor.generated_files:
            fallback_content = (
                final_answer.strip()
                or "该技能本次没有生成正文或文件，请检查调用轨迹与运行时能力。"
            )
            write_result = executor.execute(
                "write_file",
                {
                    "path": f"outputs/{self._package.slug}-result.md",
                    "content": fallback_content,
                },
            )
            actions.append(
                {
                    "step": settings.skill_runner_max_steps,
                    "tool": "write_file",
                    "args": {"path": f"outputs/{self._package.slug}-result.md"},
                    "ok": bool(write_result.get("ok")),
                    "fallback": True,
                }
            )

        return self._build_result(final_answer, actions, executor, grounding)

    def _system_prompt(self, executor: SkillToolExecutor) -> str:
        references = ", ".join(self._package.reference_names) or "无"
        templates = ", ".join(self._package.template_names) or "无"
        scripts = ", ".join(self._package.script_names) or "无"
        assets = ", ".join(self._package.asset_names) or "无"
        return f"""你是 DocMind 的 Codex-style 通用 Skill Runner。

你正在执行 Skill Package:
- slug: {self._package.slug}
- name: {self._package.name}
- 工作区: {executor.workspace}

你必须遵循下面的 SKILL.md 指令。若需要产出文件，优先写入 outputs/ 目录。
只能通过提供的受控工具读写文件、运行 shell、调用 MCP/browser/app adapter。
不要声称完成了工具没有实际返回成功的动作；工具失败时要调整方案或在最终答复中说明。

可用 package 资源：
- references: {references}
- templates: {templates}
- scripts: {scripts}
- assets: {assets}

如任务需要基于用户上传文档，必须先用 search_uploaded_documents 走 DocMind 的
向量+关键词 RRF 检索；只有思维导图、全文结构分析等确实需要完整顺序时，才继续用
read_uploaded_document 补充全文。不要把工作区文件工具误认为可以读取已上传文档。

SKILL.md:
{self._package.instructions}
"""

    def _build_result(
        self,
        answer: str,
        actions: list[dict],
        executor: SkillToolExecutor,
        grounding: dict | None,
    ) -> dict:
        generated_files = list(executor.generated_files)
        result = {
            "type": "generic_skill",
            "skill": self.name,
            "package_slug": self._package.slug,
            "answer": answer,
            "actions": actions,
            "generated_files": generated_files,
            "grounding": {
                "mode": "hybrid_rag" if grounding is not None else "tool_driven",
                "document_ids": (grounding or {}).get("document_ids", []),
                "sources": (grounding or {}).get("sources", []),
            },
        }
        if generated_files:
            result["artifact_kind"] = "file"
            result["download"] = {
                "filename": f"{self._package.slug}_outputs.zip",
                "mime_type": "application/zip",
                "encoding": "base64",
                "content": _zip_generated_files(executor),
            }
        return result


def _compact_tool_result(result: dict) -> str:
    compact = dict(result)
    if "content" in compact and isinstance(compact["content"], str):
        content = compact["content"]
        compact["content"] = content[:4000]
        if len(content) > 4000:
            compact["content_truncated"] = True
    if "stdout" in compact and isinstance(compact["stdout"], str):
        compact["stdout"] = compact["stdout"][-4000:]
    if "stderr" in compact and isinstance(compact["stderr"], str):
        compact["stderr"] = compact["stderr"][-4000:]
    return json.dumps(compact, ensure_ascii=False)


def _zip_generated_files(executor: SkillToolExecutor) -> str:
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
        for rel in executor.generated_files:
            path = executor.workspace / rel
            if path.is_file():
                zf.write(path, rel)
    return base64.b64encode(buf.getvalue()).decode("ascii")
