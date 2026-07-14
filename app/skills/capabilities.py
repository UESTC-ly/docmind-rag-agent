"""Runtime capability discovery for Codex-style Skill packages.

Package manifests describe *what* a package needs.  This module decides whether
the current DocMind process has both an implementation and explicit authority
for each capability.  Keeping those concerns separate prevents a copied
``docmind.json`` file from granting itself host privileges.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.skills.script_sandbox import sandbox_backend_available


@dataclass(frozen=True)
class CapabilityState:
    name: str
    available: bool
    reason: str


@dataclass(frozen=True)
class CapabilityReport:
    required: tuple[str, ...]
    states: tuple[CapabilityState, ...]

    @property
    def available(self) -> bool:
        return all(state.available for state in self.states)

    @property
    def missing(self) -> tuple[CapabilityState, ...]:
        return tuple(state for state in self.states if not state.available)

    @property
    def reason(self) -> str:
        if self.available:
            return "运行时能力已满足。"
        return "；".join(f"{state.name}: {state.reason}" for state in self.missing)


def _json_object(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _repository_root() -> Path | None:
    raw = (settings.skill_repository_root or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.is_dir() else None


def _valid_mcp_configuration(servers: dict) -> bool:
    for server, config in servers.items():
        if not str(server).strip() or not isinstance(config, dict):
            continue
        parsed = urlparse(str(config.get("url") or "").strip())
        tools = config.get("tools")
        headers = config.get("headers") or {}
        token_env = str(config.get("token_env") or "").strip()
        if (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and isinstance(tools, list)
            and any(str(tool).strip() for tool in tools)
            and isinstance(headers, dict)
            and (not token_env or bool(os.getenv(token_env)))
        ):
            return True
    return False


def _valid_app_configuration(actions: dict) -> bool:
    parsed = urlparse(str(settings.skill_app_bridge_url or ""))
    has_actions = any(
        isinstance(values, list)
        and bool(str(app).strip())
        and any(str(action).strip() for action in values)
        for app, values in actions.items()
    )
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and bool(settings.skill_app_bridge_token)
        and has_actions
    )


@lru_cache(maxsize=1)
def _playwright_runtime_available() -> bool:
    """Check the driver and Chromium executable once per process."""
    try:
        if importlib.util.find_spec("playwright") is None:
            return False
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            return Path(playwright.chromium.executable_path).is_file()
    except Exception:  # noqa: BLE001 - capability discovery must fail closed
        return False


def capability_states() -> dict[str, CapabilityState]:
    """Return host-authoritative capability states for this process."""

    repository = _repository_root()
    mcp_servers = _json_object(settings.skill_mcp_servers_json)
    app_actions = _json_object(settings.skill_app_allowed_actions_json)
    browser_runtime = settings.skill_browser_enabled and _playwright_runtime_available()

    states = {
        "workspace": CapabilityState("workspace", True, "用户隔离工作区可用"),
        "documents": CapabilityState("documents", True, "DocMind 文档工具可用"),
        "package_assets": CapabilityState(
            "package_assets", True, "package assets 受控读取/复制已实现"
        ),
        "package_scripts": CapabilityState(
            "package_scripts",
            settings.skill_package_scripts_enabled and sandbox_backend_available(),
            "需要启用 package scripts 且运行于带 sandbox-exec confinement 的 macOS",
        ),
        "shell": CapabilityState(
            "shell",
            False,
            "生产 Generic Skill 不开放任意 shell；请使用受审计 package script 或专用 adapter",
        ),
        "repository": CapabilityState(
            "repository",
            settings.skill_repository_enabled and repository is not None,
            "设置 SKILL_REPOSITORY_ENABLED=true 和有效 SKILL_REPOSITORY_ROOT",
        ),
        "git": CapabilityState(
            "git",
            settings.skill_repository_enabled
            and repository is not None
            and shutil.which("git") is not None,
            "需要已授权 repository root 和可执行 git",
        ),
        "mcp": CapabilityState(
            "mcp",
            settings.skill_mcp_enabled and _valid_mcp_configuration(mcp_servers),
            "设置 SKILL_MCP_ENABLED=true，并配置有效 URL、headers 和非空 tool allowlist",
        ),
        "browser": CapabilityState(
            "browser",
            settings.skill_browser_enabled
            and bool(settings.skill_browser_allowed_host_set)
            and browser_runtime,
            "需要启用 Browser、非空 host allowlist、Playwright driver 和 Chromium",
        ),
        "app": CapabilityState(
            "app",
            settings.skill_app_enabled and _valid_app_configuration(app_actions),
            "需要启用带 token、有效动作 allowlist 和 loopback URL 的 App bridge",
        ),
    }
    return states


def check_capabilities(required: tuple[str, ...] | list[str]) -> CapabilityReport:
    states = capability_states()
    normalized = tuple(
        dict.fromkeys(str(item).strip() for item in required if str(item).strip())
    )
    resolved = tuple(
        states.get(
            name,
            CapabilityState(name, False, "宿主未实现该 capability"),
        )
        for name in normalized
    )
    return CapabilityReport(required=normalized, states=resolved)


__all__ = [
    "CapabilityReport",
    "CapabilityState",
    "capability_states",
    "check_capabilities",
]
