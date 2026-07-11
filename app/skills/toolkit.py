"""Codex-style 通用 Skill 的受控工具集。

主流 Agent Skills 的关键不是“把技能写成 Python 函数”，而是让 Agent 能按
`SKILL.md` 指令使用一组动作能力：读写文件、运行脚本、调用外部工具等。

这些能力如果直接暴露给 Web 用户会非常危险。因此 DocMind v0.5 采用：

- 文件能力限定在用户隔离工作区：`skill_workspaces/user_<id>/<skill>/`
- shell 默认关闭，开启后仍需命令 allowlist，且不使用 `shell=True`
- MCP / browser / app 工具先提供 adapter 扩展点；未配置时安全失败

后续要接入真正 MCP、Playwright 或桌面应用控制，只需要注册 adapter，而不是改
GenericPackageSkill 的主循环。
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import settings
from app.skills._helpers import (
    fetch_material_text,
    fetch_retrieved_material,
    fetch_user_documents,
)
from app.skills.base import SkillContext
from app.skills.package_loader import SkillPackage

Adapter = Callable[[dict[str, Any], SkillContext], dict[str, Any]]

_MCP_ADAPTERS: dict[tuple[str, str], Adapter] = {}
_BROWSER_ADAPTERS: dict[str, Adapter] = {}
_APP_ADAPTERS: dict[tuple[str, str], Adapter] = {}


def register_mcp_adapter(server: str, tool: str, adapter: Adapter) -> None:
    """注册 MCP tool adapter，供未来接入真实 MCP bridge 使用。"""
    _MCP_ADAPTERS[(server, tool)] = adapter


def register_browser_adapter(action: str, adapter: Adapter) -> None:
    """注册浏览器动作 adapter，例如 open/click/screenshot。"""
    _BROWSER_ADAPTERS[action] = adapter


def register_app_adapter(app: str, action: str, adapter: Adapter) -> None:
    """注册本地/外部应用动作 adapter。"""
    _APP_ADAPTERS[(app, action)] = adapter


class SkillToolExecutor:
    """执行通用 Skill 内部 tool_calls 的安全边界。"""

    def __init__(
        self,
        package: SkillPackage,
        context: SkillContext,
        workspace_root: str | Path | None = None,
        shell_enabled: bool | None = None,
        shell_allowed_commands: set[str] | None = None,
        shell_timeout_seconds: int | None = None,
    ) -> None:
        self.package = package
        self.context = context
        self.workspace_base = Path(workspace_root or settings.skill_workspace_dir)
        self.workspace = (
            self.workspace_base / f"user_{context.user_id}" / package.slug
        ).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.shell_enabled = (
            settings.skill_shell_enabled if shell_enabled is None else shell_enabled
        )
        self.shell_allowed_commands = (
            settings.skill_shell_allowed_command_set
            if shell_allowed_commands is None
            else shell_allowed_commands
        )
        self.shell_timeout_seconds = (
            settings.skill_shell_timeout_seconds
            if shell_timeout_seconds is None
            else shell_timeout_seconds
        )
        self.generated_files: list[str] = []

    # ── OpenAI Function Calling schema ──────────────────────────────
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_skill_reference",
                    "description": "读取当前 Skill 包 references/ 下的参考资料。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "reference 相对路径"}
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_skill_template",
                    "description": "读取当前 Skill 包 templates/ 下的模板。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "template 相对路径"}
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "列出当前用户 Skill 工作区内的文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "工作区相对路径，默认根目录。",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_uploaded_documents",
                    "description": "列出当前登录用户已上传到 DocMind 的文档，返回文档 ID 和文件名。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_uploaded_documents",
                    "description": "通过 DocMind 的向量+关键词 RRF 检索已上传文档。凡是根据文档回答或生成材料，应先调用此工具取得相关片段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "围绕用户任务构造的检索问题或主题。",
                            },
                            "document_id": {
                                "type": "integer",
                                "description": "可选，限定当前用户的一篇文档。",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "最多返回字符数，默认 12000。",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_uploaded_document",
                    "description": "读取当前用户已上传并解析完成的文档文本；不传 document_id 时优先使用当前选中文档，否则汇总用户文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "integer",
                                "description": "可选，指定要读取的文档 ID。",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "最多返回字符数，默认 12000。",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取当前用户 Skill 工作区内的文本文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "工作区相对路径"},
                            "max_chars": {
                                "type": "integer",
                                "description": "最多返回字符数，默认 20000。",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "写入当前用户 Skill 工作区内的文本文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "工作区相对路径"},
                            "content": {"type": "string", "description": "文件内容"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_code",
                    "description": "在受控工作区内写入/替换代码文件。默认不改 DocMind 仓库源码。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "工作区相对路径"},
                            "content": {"type": "string", "description": "新的完整文件内容"},
                            "instruction": {
                                "type": "string",
                                "description": "可选：本次修改意图说明，记录到结果中。",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "在当前用户 Skill 工作区运行 allowlist 命令；默认由配置关闭。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "命令字符串；会用 shlex 拆分，不使用 shell=True。",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "call_mcp",
                    "description": "调用已配置的 MCP 工具；未配置 adapter 时安全失败。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "server": {"type": "string"},
                            "tool": {"type": "string"},
                            "arguments": {"type": "object", "additionalProperties": True},
                        },
                        "required": ["server", "tool"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "use_browser_tool",
                    "description": "调用已配置的浏览器工具；未配置 adapter 时安全失败。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "arguments": {"type": "object", "additionalProperties": True},
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "use_app_tool",
                    "description": "调用已配置的应用工具；未配置 adapter 时安全失败。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app": {"type": "string"},
                            "action": {"type": "string"},
                            "arguments": {"type": "object", "additionalProperties": True},
                        },
                        "required": ["app", "action"],
                    },
                },
            },
        ]

    # ── Dispatcher ─────────────────────────────────────────────────
    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return {"ok": False, "error": f"未知动作工具: {name}"}
            return handler(args)
        except Exception as exc:  # noqa: BLE001 - tool 失败要回传给 LLM 而不是打断主循环
            return {"ok": False, "error": str(exc)}

    # ── File boundary ──────────────────────────────────────────────
    def _resolve_workspace_path(self, raw_path: str) -> Path:
        rel = Path(str(raw_path))
        if rel.is_absolute():
            raise PermissionError("只能访问 Skill 工作区内的相对路径")
        target = (self.workspace / rel).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise PermissionError("路径超出 Skill 工作区")
        return target

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def _record_generated_file(self, path: Path) -> None:
        rel = self._relative(path)
        if rel not in self.generated_files:
            self.generated_files.append(rel)

    def _tool_list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self._resolve_workspace_path(args.get("path") or ".")
        if not root.exists():
            return {"ok": True, "files": []}
        if root.is_file():
            return {"ok": True, "files": [self._relative(root)]}
        files = [
            self._relative(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
        return {"ok": True, "files": files}

    def _tool_read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_workspace_path(args["path"])
        if not path.is_file():
            return {"ok": False, "error": f"文件不存在: {args['path']}"}
        max_chars = int(args.get("max_chars") or 20000)
        content = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return {"ok": True, "path": self._relative(path), "content": content}

    def _tool_write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_workspace_path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args.get("content", "")), encoding="utf-8")
        self._record_generated_file(path)
        return {
            "ok": True,
            "path": self._relative(path),
            "bytes": path.stat().st_size,
        }

    def _tool_modify_code(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self._tool_write_file(args)
        result["instruction"] = args.get("instruction", "")
        result["mode"] = "replace"
        return result

    # ── Uploaded DocMind documents ────────────────────────────────
    def _tool_list_uploaded_documents(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "documents": fetch_user_documents(self.context.user_id),
        }

    def _tool_read_uploaded_document(self, args: dict[str, Any]) -> dict[str, Any]:
        document_id = args.get("document_id") or self.context.document_id
        max_chars = int(args.get("max_chars") or 12000)
        content, used_doc_ids = fetch_material_text(
            self.context.user_id,
            document_id=document_id,
            max_chars=max_chars,
        )
        if not content:
            return {
                "ok": False,
                "error": "未找到可读取的已上传文档内容；请确认文档归属当前用户且已解析完成。",
                "document_id": document_id,
            }
        return {
            "ok": True,
            "document_id": document_id,
            "document_ids": used_doc_ids,
            "content": content,
        }

    def _tool_search_uploaded_documents(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "检索 query 不能为空"}
        document_id = args.get("document_id") or self.context.document_id
        max_chars = int(args.get("max_chars") or 12000)
        content, used_doc_ids, sources = fetch_retrieved_material(
            self.context.user_id,
            query=query,
            document_id=document_id,
            max_chars=max_chars,
        )
        if not content:
            return {
                "ok": False,
                "error": "RAG 未检索到相关文档片段。",
                "document_id": document_id,
                "retrieval_mode": "hybrid",
            }
        return {
            "ok": True,
            "document_id": document_id,
            "document_ids": used_doc_ids,
            "retrieval_mode": "hybrid",
            "sources": sources,
            "content": content,
        }

    # ── Package resources ──────────────────────────────────────────
    def _tool_read_skill_reference(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        content = self.package.references.get(name)
        if content is None:
            return {"ok": False, "error": f"reference 不存在: {name}"}
        return {"ok": True, "name": name, "content": content}

    def _tool_read_skill_template(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"])
        content = self.package.templates.get(name)
        if content is None:
            return {"ok": False, "error": f"template 不存在: {name}"}
        return {"ok": True, "name": name, "content": content}

    # ── Shell / external adapters ──────────────────────────────────
    def _tool_run_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args.get("command") or "").strip()
        if not command:
            return {"ok": False, "error": "命令不能为空"}
        if not self.shell_enabled:
            return {
                "ok": False,
                "disabled": True,
                "error": "shell 工具未启用；请通过 SKILL_SHELL_ENABLED=true 显式开启。",
            }

        argv = shlex.split(command)
        if not argv:
            return {"ok": False, "error": "命令不能为空"}
        executable = Path(argv[0]).name
        if executable not in self.shell_allowed_commands:
            return {
                "ok": False,
                "error": f"命令 {executable} 不在允许列表",
                "allowed": sorted(self.shell_allowed_commands),
            }

        completed = subprocess.run(  # noqa: S603 - argv + allowlist + cwd sandbox
            argv,
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=self.shell_timeout_seconds,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    def _tool_call_mcp(self, args: dict[str, Any]) -> dict[str, Any]:
        server = str(args.get("server") or "")
        tool = str(args.get("tool") or "")
        adapter = _MCP_ADAPTERS.get((server, tool))
        if adapter is None:
            return {
                "ok": False,
                "disabled": True,
                "error": f"MCP adapter 未配置: {server}.{tool}",
            }
        return adapter(args.get("arguments") or {}, self.context)

    def _tool_use_browser_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "")
        adapter = _BROWSER_ADAPTERS.get(action)
        if adapter is None:
            return {
                "ok": False,
                "disabled": True,
                "error": f"浏览器 adapter 未配置: {action}",
            }
        return adapter(args.get("arguments") or {}, self.context)

    def _tool_use_app_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        app = str(args.get("app") or "")
        action = str(args.get("action") or "")
        adapter = _APP_ADAPTERS.get((app, action))
        if adapter is None:
            return {
                "ok": False,
                "disabled": True,
                "error": f"应用 adapter 未配置: {app}.{action}",
            }
        return adapter(args.get("arguments") or {}, self.context)
