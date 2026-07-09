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

    def __init__(self, package: SkillPackage) -> None:
        self._package = package
        self.package_slug = package.slug
        self.name = package.name
        self.description = package.description
        self.parameters = package.parameters

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
        messages = [
            {"role": "system", "content": self._system_prompt(executor)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "inputs": inputs,
                        "document_id": document_id,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        actions: list[dict] = []
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

        return self._build_result(final_answer, actions, executor)

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

SKILL.md:
{self._package.instructions}
"""

    def _build_result(
        self,
        answer: str,
        actions: list[dict],
        executor: SkillToolExecutor,
    ) -> dict:
        generated_files = list(executor.generated_files)
        result = {
            "type": "generic_skill",
            "skill": self.name,
            "package_slug": self._package.slug,
            "answer": answer,
            "actions": actions,
            "generated_files": generated_files,
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
