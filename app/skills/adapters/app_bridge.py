"""Authenticated local HTTP bridge for explicitly allowed application actions."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlparse

import httpx

from app.skills.adapters.contracts import failure, success
from app.skills.base import SkillContext


class HttpAppBridgeAdapter:
    def __init__(
        self,
        base_url: str,
        token: str,
        app: str,
        action: str,
        *,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("App bridge requires an http(s) URL")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("App bridge must use a loopback host")
        if not token:
            raise ValueError("App bridge requires an authentication token")
        self.url = (
            f"{base_url.rstrip('/')}/v1/apps/{quote(app, safe='')}/actions/"
            f"{quote(action, safe='')}"
        )
        self.app = app
        self.action = action
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._headers = {"Authorization": f"Bearer {token}"}

    def __call__(self, arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        public_arguments = {
            key: value for key, value in arguments.items() if not key.startswith("__")
        }
        try:
            response = self._client.post(
                self.url,
                headers=self._headers,
                json={"arguments": public_arguments, "user_id": context.user_id},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("App bridge response must be a JSON object")
            if payload.get("ok") is False:
                raise RuntimeError(str(payload.get("error") or "application action failed"))
            return success(
                f"Application action {self.app}.{self.action} completed",
                data={"app": self.app, "action": self.action, "result": payload},
                artifacts=list(payload.get("artifacts") or []),
            )
        except Exception as exc:  # noqa: BLE001 - external boundary
            return failure(
                f"Application action {self.app}.{self.action} failed",
                root_cause=str(exc),
                retry="确认本地 bridge 正在运行、token 匹配且应用权限已授予。",
                stop_condition="bridge 不可达或权限持续拒绝时停止。",
            )


__all__ = ["HttpAppBridgeAdapter"]
