"""Install configured production adapters into the Generic Skill registry."""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.skills.adapters.app_bridge import HttpAppBridgeAdapter
from app.skills.adapters.browser import PlaywrightBrowserAdapter
from app.skills.adapters.mcp import StreamableHttpMcpAdapter


def _object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def install_runtime_adapters() -> None:
    from app.skills.toolkit import (
        register_app_adapter,
        register_browser_adapter,
        register_mcp_adapter,
    )

    if settings.skill_mcp_enabled:
        for server, config in _object(settings.skill_mcp_servers_json).items():
            if not str(server).strip() or not isinstance(config, dict):
                continue
            try:
                adapter = StreamableHttpMcpAdapter(
                    str(server),
                    config,
                    timeout_seconds=settings.skill_mcp_timeout_seconds,
                )
            except (TypeError, ValueError):
                continue
            register_mcp_adapter(str(server), "*", adapter)

    if settings.skill_browser_enabled:
        browser = PlaywrightBrowserAdapter(
            headless=settings.skill_browser_headless,
            allowed_hosts=settings.skill_browser_allowed_host_set,
            timeout_seconds=settings.skill_browser_timeout_seconds,
        )
        for action in browser.ACTIONS:
            register_browser_adapter(action, browser)

    if (
        settings.skill_app_enabled
        and settings.skill_app_bridge_url
        and settings.skill_app_bridge_token
    ):
        for app, actions in _object(settings.skill_app_allowed_actions_json).items():
            if not str(app).strip() or not isinstance(actions, list):
                continue
            for action in actions:
                if not str(action).strip():
                    continue
                try:
                    adapter = HttpAppBridgeAdapter(
                        settings.skill_app_bridge_url,
                        settings.skill_app_bridge_token,
                        str(app),
                        str(action),
                        timeout_seconds=settings.skill_app_timeout_seconds,
                    )
                except (TypeError, ValueError):
                    continue
                register_app_adapter(str(app), str(action), adapter)


__all__ = ["install_runtime_adapters"]
