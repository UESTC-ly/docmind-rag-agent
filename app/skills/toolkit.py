"""Codex-style 通用 Skill 的受控工具集。

主流 Agent Skills 的关键不是“把技能写成 Python 函数”，而是让 Agent 能按
`SKILL.md` 指令使用一组动作能力：读写文件、运行脚本、调用外部工具等。

这些能力如果直接暴露给 Web 用户会非常危险。因此 DocMind v2.2 采用：

- 文件能力限定在用户隔离工作区：`skill_workspaces/user_<id>/<skill>/`
- shell 默认关闭，开启后仍需命令 allowlist，且不使用 `shell=True`
- MCP / browser / app 工具使用生产 adapter，并在未配置或越权时安全失败

运行时仍保留 adapter 注册接口，便于替换具体后端而不改
GenericPackageSkill 的主循环。
"""

from __future__ import annotations

import base64
import os
import shlex
import shutil
import subprocess
import sys
import uuid
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
from app.skills.capabilities import capability_states
from app.skills.package_loader import SkillPackage
from app.skills.script_sandbox import (
    changed_workspace_files,
    confine_command,
    confined_environment,
    validate_untrusted_arguments,
    workspace_snapshot,
)
from app.skills.adapters.contracts import failure, normalize_observation

Adapter = Callable[[dict[str, Any], SkillContext], dict[str, Any]]

_MCP_ADAPTERS: dict[tuple[str, str], Adapter] = {}
_BROWSER_ADAPTERS: dict[str, Adapter] = {}
_APP_ADAPTERS: dict[tuple[str, str], Adapter] = {}

# Fail closed: every action handler must have an explicit package capability
# policy. Host availability alone never grants a copied SKILL.md new authority.
_TOOL_CAPABILITY_POLICY: dict[str, frozenset[str]] = {
    "read_skill_reference": frozenset(),
    "read_skill_template": frozenset(),
    "list_files": frozenset({"workspace"}),
    "read_file": frozenset({"workspace"}),
    "write_file": frozenset({"workspace"}),
    "modify_code": frozenset({"workspace"}),
    "list_uploaded_documents": frozenset({"documents"}),
    "search_uploaded_documents": frozenset({"documents"}),
    "read_uploaded_document": frozenset({"documents"}),
    "read_skill_asset": frozenset({"package_assets"}),
    "copy_skill_asset": frozenset({"package_assets", "workspace"}),
    "run_skill_script": frozenset({"package_scripts", "workspace"}),
    "run_shell": frozenset({"shell", "workspace"}),
    "call_mcp": frozenset({"mcp"}),
    "use_browser_tool": frozenset({"browser", "workspace"}),
    "use_app_tool": frozenset({"app", "workspace"}),
    "list_repository_files": frozenset({"repository"}),
    "read_repository_file": frozenset({"repository"}),
    "search_repository": frozenset({"repository"}),
    "git_history": frozenset({"git", "repository"}),
}


def register_mcp_adapter(server: str, tool: str, adapter: Adapter) -> None:
    """注册或覆盖指定 MCP server/tool 的运行时 adapter。"""
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
        self.session_id = uuid.uuid4().hex
        self.host_capabilities = {
            name for name, state in capability_states().items() if state.available
        }
        # No copied package receives a general command interpreter. Audited
        # package scripts and typed adapters are the production action surface.
        self.host_capabilities.discard("shell")
        self.declared_capabilities = set(package.required_capabilities)
        self.capabilities = self.host_capabilities & self.declared_capabilities

    def _tool_is_authorized(self, name: str) -> bool:
        requirements = _TOOL_CAPABILITY_POLICY.get(name)
        return requirements is not None and requirements <= self.capabilities

    # ── OpenAI Function Calling schema ──────────────────────────────
    def tool_definitions(self) -> list[dict]:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "read_skill_reference",
                    "description": "读取当前 Skill 包 references/ 下的参考资料。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "reference 相对路径",
                            }
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
                            "name": {
                                "type": "string",
                                "description": "template 相对路径",
                            }
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
                            "content": {
                                "type": "string",
                                "description": "新的完整文件内容",
                            },
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
                            "arguments": {
                                "type": "object",
                                "additionalProperties": True,
                            },
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
                            "arguments": {
                                "type": "object",
                                "additionalProperties": True,
                            },
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
                            "arguments": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["app", "action"],
                    },
                },
            },
        ]
        definitions.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_skill_asset",
                        "description": "读取当前 Skill package 的二进制 asset，返回受限 base64 内容。",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "copy_skill_asset",
                        "description": "把当前 Skill package 的 asset 复制到用户隔离工作区。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "destination": {"type": "string"},
                            },
                            "required": ["name", "destination"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_skill_script",
                        "description": "运行当前 package scripts/ 中的显式脚本；受 capability、解释器和超时限制。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_repository_files",
                        "description": "列出宿主显式授权仓库中的文件。只读且有数量上限。",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": [],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_repository_file",
                        "description": "读取宿主显式授权仓库内的文本文件。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "max_chars": {"type": "integer"},
                            },
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_repository",
                        "description": "在宿主显式授权仓库中进行固定字符串只读搜索。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "path": {"type": "string"},
                                "max_results": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "git_history",
                        "description": "读取授权仓库的有限 Git 提交历史。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                            "required": [],
                        },
                    },
                },
            ]
        )

        filtered: list[dict] = []
        for definition in definitions:
            name = definition["function"]["name"]
            if not self._tool_is_authorized(name):
                continue
            if name == "run_skill_script":
                if not self.package.script_names:
                    continue
                definition["function"]["parameters"]["properties"]["name"]["enum"] = (
                    list(self.package.script_names)
                )
            if name in {"read_skill_asset", "copy_skill_asset"}:
                if not self.package.asset_names:
                    continue
                definition["function"]["parameters"]["properties"]["name"]["enum"] = (
                    list(self.package.asset_names)
                )
            filtered.append(definition)
        return filtered

    # ── Dispatcher ─────────────────────────────────────────────────
    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        try:
            if not self._tool_is_authorized(name):
                requirements = _TOOL_CAPABILITY_POLICY.get(name)
                if requirements is None:
                    root_cause = f"未知或未登记权限策略的动作工具: {name}"
                else:
                    missing = sorted(requirements - self.capabilities)
                    root_cause = (
                        f"package 未声明或宿主未授权 capability: {', '.join(missing)}"
                    )
                return failure(
                    f"Skill action denied: {name}",
                    root_cause=root_cause,
                    retry="只调用当前 tool definitions 中公开的动作。",
                    stop_condition="不要尝试绕过 package capability manifest。",
                    data={"denied": True},
                )
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return failure(
                    f"Unknown Skill action: {name}",
                    root_cause=f"未知动作工具: {name}",
                    retry="从当前 tool definitions 中选择动作。",
                    stop_condition="不要重复调用不存在的动作。",
                )
            return normalize_observation(
                handler(args),
                operation=name,
                max_chars=settings.skill_adapter_observation_max_chars,
            )
        except Exception as exc:  # noqa: BLE001 - tool 失败要回传给 LLM 而不是打断主循环
            return failure(
                f"Skill action {name} failed",
                root_cause=str(exc),
                retry="检查参数与 capability 状态后修正一次。",
                stop_condition="相同参数持续失败时停止。",
            )

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
        if path.is_symlink():
            raise PermissionError("artifact 不允许符号链接")
        path = path.resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise PermissionError("artifact 路径超出 Skill 工作区")
        if not path.is_file():
            raise FileNotFoundError("artifact 文件不存在")
        size = path.stat().st_size
        if size > settings.skill_artifact_max_bytes:
            raise ValueError(
                f"artifact 超过 {settings.skill_artifact_max_bytes} bytes 上限"
            )
        rel = self._relative(path)
        if rel not in self.generated_files:
            self.generated_files.append(rel)

    def validated_generated_paths(self) -> list[Path]:
        paths: list[Path] = []
        for relative in self.generated_files:
            path = self.workspace / relative
            self._record_generated_file(path)
            paths.append(path.resolve())
        return paths

    def _tool_list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self._resolve_workspace_path(args.get("path") or ".")
        if not root.exists():
            return {"ok": True, "files": []}
        if root.is_file():
            return {"ok": True, "files": [self._relative(root)]}
        files = [
            self._relative(path) for path in sorted(root.rglob("*")) if path.is_file()
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
        content = str(args.get("content", ""))
        encoded = content.encode("utf-8")
        if len(encoded) > settings.skill_artifact_max_bytes:
            return {
                "ok": False,
                "error": (
                    f"artifact 超过 {settings.skill_artifact_max_bytes} bytes 上限"
                ),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
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

    def _resolve_package_resource(
        self, root_name: str, raw_name: str, allowed: list[str]
    ) -> Path:
        name = Path(str(raw_name))
        if name.is_absolute() or name.as_posix() not in allowed:
            raise PermissionError(f"未声明的 package {root_name} 资源: {raw_name}")
        root = (self.package.path / root_name).resolve()
        path = (root / name).resolve()
        if root != path and root not in path.parents:
            raise PermissionError(f"package {root_name} 路径越界")
        if not path.is_file():
            raise FileNotFoundError(f"package 资源不存在: {raw_name}")
        return path

    def _tool_read_skill_asset(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_package_resource(
            "assets", str(args["name"]), self.package.asset_names
        )
        max_bytes = 5 * 1024 * 1024
        size = path.stat().st_size
        if size > max_bytes:
            return {"ok": False, "error": f"asset 超过 {max_bytes} bytes 上限"}
        return {
            "ok": True,
            "name": str(args["name"]),
            "encoding": "base64",
            "bytes": size,
            "content": base64.b64encode(path.read_bytes()).decode("ascii"),
        }

    def _tool_copy_skill_asset(self, args: dict[str, Any]) -> dict[str, Any]:
        source = self._resolve_package_resource(
            "assets", str(args["name"]), self.package.asset_names
        )
        if source.stat().st_size > settings.skill_artifact_max_bytes:
            return {
                "ok": False,
                "error": (
                    f"artifact 超过 {settings.skill_artifact_max_bytes} bytes 上限"
                ),
            }
        destination = self._resolve_workspace_path(str(args["destination"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        self._record_generated_file(destination)
        return {
            "ok": True,
            "source": str(args["name"]),
            "path": self._relative(destination),
            "bytes": destination.stat().st_size,
        }

    def _tool_run_skill_script(self, args: dict[str, Any]) -> dict[str, Any]:
        if "package_scripts" not in self.capabilities:
            return {
                "ok": False,
                "disabled": True,
                "error": "package script capability 未启用",
            }
        script = self._resolve_package_resource(
            "scripts", str(args["name"]), self.package.script_names
        )
        suffix = script.suffix.lower()
        interpreter_by_suffix = {
            ".py": ("python", sys.executable),
            ".js": ("node", "node"),
            ".mjs": ("node", "node"),
            ".sh": ("bash", "bash"),
            ".ps1": ("powershell", "pwsh"),
            ".swift": ("swift", "swift"),
        }
        if suffix not in interpreter_by_suffix:
            return {"ok": False, "error": f"不支持的 package script 类型: {suffix}"}
        interpreter_name, executable = interpreter_by_suffix[suffix]
        if interpreter_name not in settings.skill_package_script_interpreter_set:
            return {"ok": False, "error": f"解释器未获授权: {interpreter_name}"}
        if executable != sys.executable and shutil.which(executable) is None:
            return {"ok": False, "error": f"解释器不可用: {executable}"}
        raw_arguments = args.get("arguments") or []
        if not isinstance(raw_arguments, list) or not all(
            isinstance(item, str) for item in raw_arguments
        ):
            return {"ok": False, "error": "script arguments 必须是字符串数组"}
        arguments = validate_untrusted_arguments(raw_arguments)
        environment_source = {
            key: value
            for key, value in os.environ.items()
            if key in settings.skill_package_script_env_allowlist_set
        }
        environment = confined_environment(self.workspace, environment_source)
        command = [executable, str(script), *arguments]
        if suffix == ".py" and getattr(sys, "frozen", False):
            # A PyInstaller executable is not a general Python launcher.  The
            # sidecar exposes a second, resource-root-validated package-script
            # entrypoint so bundled Skills can still use their shipped helpers.
            command = [sys.executable, "run-script", str(script), "--", *arguments]
        before = workspace_snapshot(self.workspace)
        command = confine_command(command, self.workspace)
        completed = subprocess.run(  # noqa: S603 - fixed script root/interpreter
            command,
            cwd=self.workspace,
            env=environment,
            text=True,
            capture_output=True,
            timeout=settings.skill_package_script_timeout_seconds,
            check=False,
        )
        changed = changed_workspace_files(
            self.workspace,
            before,
            max_file_bytes=settings.skill_artifact_max_bytes,
            max_total_bytes=settings.skill_artifact_max_bytes,
        )
        for path in changed:
            self._record_generated_file(path)
        return {
            "ok": completed.returncode == 0,
            "script": str(args["name"]),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
            "generated_files": [self._relative(path) for path in changed],
        }

    # ── Explicitly mounted repository (read-only) ───────────────────
    def _repository_root(self) -> Path:
        if "repository" not in self.capabilities or not settings.skill_repository_root:
            raise PermissionError("repository capability 未启用")
        root = Path(settings.skill_repository_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("授权 repository root 不存在")
        return root

    def _resolve_repository_path(self, raw_path: str) -> Path:
        root = self._repository_root()
        rel = Path(str(raw_path or "."))
        if rel.is_absolute():
            raise PermissionError("repository 工具只接受相对路径")
        path = (root / rel).resolve()
        if path != root and root not in path.parents:
            raise PermissionError("repository 路径越界")
        return path

    def _tool_list_repository_files(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self._repository_root()
        path = self._resolve_repository_path(str(args.get("path") or "."))
        if not path.exists():
            return {"ok": False, "error": "repository 路径不存在"}
        candidates = [path] if path.is_file() else path.rglob("*")
        files: list[str] = []
        for item in candidates:
            if not item.is_file() or ".git" in item.relative_to(root).parts:
                continue
            files.append(item.relative_to(root).as_posix())
            if len(files) >= 5000:
                break
        return {"ok": True, "files": sorted(files), "truncated": len(files) >= 5000}

    def _tool_read_repository_file(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self._repository_root()
        path = self._resolve_repository_path(str(args["path"]))
        if not path.is_file():
            return {"ok": False, "error": "repository 文件不存在"}
        max_chars = min(
            int(args.get("max_chars") or settings.skill_repository_max_chars),
            settings.skill_repository_max_chars,
        )
        return {
            "ok": True,
            "path": path.relative_to(root).as_posix(),
            "content": path.read_text(encoding="utf-8", errors="replace")[:max_chars],
        }

    def _tool_search_repository(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "repository query 不能为空"}
        root = self._repository_root()
        search_root = self._resolve_repository_path(str(args.get("path") or "."))
        max_results = max(1, min(int(args.get("max_results") or 100), 500))
        rg = shutil.which("rg")
        if rg:
            completed = subprocess.run(  # noqa: S603 - fixed executable/argv
                [rg, "-n", "-F", "--glob", "!.git/**", "--", query, str(search_root)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            lines = completed.stdout.splitlines()[:max_results]
        else:
            lines = []
            candidates = (
                [search_root] if search_root.is_file() else search_root.rglob("*")
            )
            for path in candidates:
                if not path.is_file() or ".git" in path.relative_to(root).parts:
                    continue
                try:
                    for number, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(),
                        1,
                    ):
                        if query in line:
                            lines.append(
                                f"{path.relative_to(root)}:{number}:{line[:500]}"
                            )
                            if len(lines) >= max_results:
                                break
                except OSError:
                    continue
                if len(lines) >= max_results:
                    break
        return {"ok": True, "query": query, "matches": lines}

    def _tool_git_history(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self._repository_root()
        limit = max(1, min(int(args.get("limit") or 50), 200))
        command = [
            "git",
            "-C",
            str(root),
            "log",
            f"-{limit}",
            "--date=iso-strict",
            "--pretty=format:%H%x09%ad%x09%an%x09%s",
        ]
        raw_path = str(args.get("path") or "").strip()
        if raw_path:
            path = self._resolve_repository_path(raw_path)
            command.extend(["--", path.relative_to(root).as_posix()])
        completed = subprocess.run(  # noqa: S603 - fixed git argv
            command,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "history": completed.stdout.splitlines(),
            "stderr": completed.stderr[-2000:],
        }

    # ── Shell / external adapters ──────────────────────────────────
    def _tool_run_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args.get("command") or "").strip()
        if not command:
            return {"ok": False, "error": "命令不能为空"}
        if not self.shell_enabled:
            return {
                "ok": False,
                "disabled": True,
                "error": (
                    "生产 Generic Skill 不开放任意 shell；"
                    "请使用受审计 package script 或专用 adapter。"
                ),
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

        validate_untrusted_arguments(argv[1:])
        environment_source = {
            key: value
            for key, value in os.environ.items()
            if key in settings.skill_package_script_env_allowlist_set
        }
        environment = confined_environment(self.workspace, environment_source)
        before = workspace_snapshot(self.workspace)
        confined = confine_command(argv, self.workspace)
        completed = subprocess.run(  # noqa: S603 - allowlist + macOS confinement
            confined,
            cwd=self.workspace,
            env=environment,
            text=True,
            capture_output=True,
            timeout=self.shell_timeout_seconds,
            check=False,
        )
        changed = changed_workspace_files(
            self.workspace,
            before,
            max_file_bytes=settings.skill_artifact_max_bytes,
            max_total_bytes=settings.skill_artifact_max_bytes,
        )
        for path in changed:
            self._record_generated_file(path)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "generated_files": [self._relative(path) for path in changed],
        }

    def _tool_call_mcp(self, args: dict[str, Any]) -> dict[str, Any]:
        server = str(args.get("server") or "")
        tool = str(args.get("tool") or "")
        adapter = _MCP_ADAPTERS.get((server, tool)) or _MCP_ADAPTERS.get((server, "*"))
        if adapter is None:
            return {
                "ok": False,
                "disabled": True,
                "error": f"MCP adapter 未配置: {server}.{tool}",
            }
        arguments = dict(args.get("arguments") or {})
        arguments["__tool_name"] = tool
        arguments["__session_id"] = self.session_id
        return adapter(arguments, self.context)

    def _tool_use_browser_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "")
        adapter = _BROWSER_ADAPTERS.get(action)
        if adapter is None:
            return {
                "ok": False,
                "disabled": True,
                "error": f"浏览器 adapter 未配置: {action}",
            }
        arguments = dict(args.get("arguments") or {})
        arguments.update(
            __action=action,
            __session_id=self.session_id,
            __workspace=str(self.workspace),
        )
        result = adapter(arguments, self.context)
        for artifact in result.get("artifacts") or []:
            raw_path = artifact.get("path") if isinstance(artifact, dict) else None
            if raw_path:
                path = self._resolve_workspace_path(str(raw_path))
                if path.is_file():
                    self._record_generated_file(path)
        return result

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
        arguments = dict(args.get("arguments") or {})
        arguments.update(__workspace=str(self.workspace), __session_id=self.session_id)
        return adapter(arguments, self.context)

    def close(self) -> None:
        """Release per-run adapter resources without affecting other executions."""
        adapters = [*_MCP_ADAPTERS.values(), *_BROWSER_ADAPTERS.values()]
        seen: set[int] = set()
        for adapter in adapters:
            identity = id(adapter)
            if identity in seen:
                continue
            seen.add(identity)
            close_session = getattr(adapter, "close_session", None)
            if close_session is not None:
                try:
                    close_session(self.session_id)
                except Exception:  # noqa: BLE001 - cleanup cannot replace Skill result
                    pass
