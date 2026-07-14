"""Synchronous MCP Streamable HTTP adapter for Generic Skills."""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from app.skills.adapters.contracts import failure, success
from app.skills.base import SkillContext


def _json_rpc_payload(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        payloads = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                raw = line[5:].strip()
                if raw and raw != "[DONE]":
                    payloads.append(json.loads(raw))
        if not payloads:
            raise ValueError("MCP SSE response did not contain a JSON-RPC payload")
        return payloads[-1]
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("MCP response must be a JSON object")
    return value


class StreamableHttpMcpAdapter:
    def __init__(
        self,
        server: str,
        config: dict[str, Any],
        *,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.server = server
        self.url = str(config.get("url") or "").strip()
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"MCP server {server} requires an http(s) URL")
        tools = config.get("tools") or []
        if not isinstance(tools, list):
            raise ValueError(f"MCP server {server} tools must be a list")
        self.allowed_tools = {str(item).strip() for item in tools if str(item).strip()}
        if not self.allowed_tools:
            raise ValueError(f"MCP server {server} requires a non-empty tools allowlist")

        raw_headers = config.get("headers") or {}
        if not isinstance(raw_headers, dict):
            raise ValueError(f"MCP server {server} headers must be an object")
        headers = {
            str(key): str(value)
            for key, value in raw_headers.items()
        }
        token_env = str(config.get("token_env") or "").strip()
        if token_env:
            token = os.getenv(token_env)
            if not token:
                raise ValueError(
                    f"MCP server {server} token environment variable is missing: {token_env}"
                )
            headers["Authorization"] = f"Bearer {token}"
        headers.setdefault("Accept", "application/json, text/event-stream")
        headers.setdefault("Content-Type", "application/json")
        self._headers = headers
        self._client = client or httpx.Client(timeout=timeout_seconds)
        # MCP sessions are execution-scoped.  A process-wide adapter is shared by
        # many users, so protocol state must never leak across Agent executions.
        self._local = threading.local()

    def _states(self) -> dict[str, dict[str, Any]]:
        if not hasattr(self._local, "states"):
            self._local.states = {}
        return self._local.states

    def _post(
        self,
        state: dict[str, Any],
        method: str,
        params: dict[str, Any] | None,
        *,
        notification=False,
    ):
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            payload["id"] = uuid.uuid4().hex
        if params is not None:
            payload["params"] = params
        headers = dict(self._headers)
        if state.get("session_id"):
            headers["Mcp-Session-Id"] = state["session_id"]
        response = self._client.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            state["session_id"] = session_id
        if notification and not response.content:
            return {}
        return _json_rpc_payload(response)

    def _initialize(self, state: dict[str, Any]) -> None:
        if state.get("initialized"):
            return
        response = self._post(
            state,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "docmind", "version": "2.2.0"},
            },
        )
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        self._post(state, "notifications/initialized", None, notification=True)
        state["initialized"] = True

    def close_session(self, execution: str) -> None:
        """Forget execution state and best-effort terminate the remote session."""
        state = self._states().pop(execution, None)
        session_id = state.get("session_id") if state else None
        if not session_id:
            return
        headers = dict(self._headers)
        headers["Mcp-Session-Id"] = str(session_id)
        try:
            self._client.delete(self.url, headers=headers)
        except Exception:  # noqa: BLE001 - cleanup must not hide the Skill result
            pass

    def __call__(self, arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        tool = str(arguments.pop("__tool_name", ""))
        execution = str(arguments.pop("__session_id", f"user-{context.user_id}"))
        if tool not in self.allowed_tools and "*" not in self.allowed_tools:
            return failure(
                f"MCP tool {self.server}.{tool} was denied",
                root_cause="tool is not in the configured allowlist",
                retry="选择该 server allowlist 中的工具。",
                stop_condition="不要用不同参数重复调用被拒绝的工具。",
                disabled=True,
            )
        try:
            state = self._states().setdefault(execution, {})
            self._initialize(state)
            response = self._post(
                state,
                "tools/call",
                {"name": tool, "arguments": arguments},
            )
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            result = response.get("result") or {}
            return success(
                f"MCP tool {self.server}.{tool} completed",
                data={"server": self.server, "tool": tool, "result": result},
            )
        except Exception as exc:  # noqa: BLE001 - external boundary
            return failure(
                f"MCP tool {self.server}.{tool} failed",
                root_cause=str(exc),
                retry="检查 MCP URL、凭据、协议版本和服务健康状态后重试一次。",
                stop_condition="同一配置连续失败时停止并报告 adapter 诊断。",
            )


__all__ = ["StreamableHttpMcpAdapter"]
