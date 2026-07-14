"""Bounded Playwright browser adapter with per-execution sessions."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.skills.adapters.contracts import failure, success
from app.skills.base import SkillContext


class PlaywrightBrowserAdapter:
    ACTIONS = {"open", "navigate", "click", "fill", "text", "screenshot", "close"}
    INTERNAL_SCHEMES = {"about", "blob", "data"}

    def __init__(self, *, headless: bool, allowed_hosts: set[str], timeout_seconds: int):
        self.headless = headless
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self.timeout_ms = timeout_seconds * 1000
        self._local = threading.local()

    def _sessions(self) -> dict[str, tuple[Any, Any, Any, Any]]:
        if not hasattr(self._local, "sessions"):
            self._local.sessions = {}
        return self._local.sessions

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise PermissionError("browser URL must use http(s)")
        host = parsed.hostname.lower()
        if "*" not in self.allowed_hosts and host not in self.allowed_hosts:
            raise PermissionError(f"browser host is not allowed: {host}")

    def _route_request(self, route) -> None:
        """Block subresources and redirects that leave the host allowlist."""
        url = route.request.url
        parsed = urlparse(url)
        try:
            if parsed.scheme in {"http", "https"}:
                self._validate_url(url)
            elif parsed.scheme not in self.INTERNAL_SCHEMES:
                raise PermissionError(f"browser URL scheme is not allowed: {parsed.scheme}")
        except PermissionError:
            route.abort("blockedbyclient")
            return
        route.continue_()

    def _page(self, session: str):
        sessions = self._sessions()
        if session in sessions:
            return sessions[session][3]
        from playwright.sync_api import sync_playwright

        manager = sync_playwright().start()
        browser = manager.chromium.launch(headless=self.headless)
        context = browser.new_context()
        context.route("**/*", self._route_request)
        page = context.new_page()
        page.set_default_timeout(self.timeout_ms)
        sessions[session] = (manager, browser, context, page)
        return page

    def close_session(self, session: str) -> None:
        values = self._sessions().pop(session, None)
        if values is None:
            return
        manager, browser, context, _ = values
        try:
            context.close()
            browser.close()
        finally:
            manager.stop()

    def __call__(self, arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        action = str(arguments.get("__action") or "")
        session = str(arguments.get("__session_id") or f"user-{context.user_id}")
        workspace = Path(str(arguments.get("__workspace") or ".")).resolve()
        public = {key: value for key, value in arguments.items() if not key.startswith("__")}
        try:
            if action not in self.ACTIONS:
                raise ValueError(f"unsupported browser action: {action}")
            if action == "close":
                self.close_session(session)
                return success("Browser session closed")

            if action in {"open", "navigate"}:
                url = str(public.get("url") or "")
                # Validate the network boundary before launching a browser process.
                self._validate_url(url)
                page = self._page(session)
                page.goto(url, wait_until="domcontentloaded")
                self._validate_url(page.url)
                return success(
                    f"Browser navigated to {url}",
                    data={"url": page.url, "title": page.title()},
                )
            page = self._page(session)
            self._validate_url(page.url)
            if action == "click":
                selector = str(public["selector"])
                page.locator(selector).click()
                self._validate_url(page.url)
                return success(f"Clicked {selector}", data={"url": page.url})
            if action == "fill":
                selector = str(public["selector"])
                page.locator(selector).fill(str(public.get("text") or ""))
                self._validate_url(page.url)
                return success(f"Filled {selector}", data={"url": page.url})
            if action == "text":
                selector = str(public.get("selector") or "body")
                text = page.locator(selector).inner_text()[:40000]
                return success(
                    f"Read text from {selector}",
                    data={"url": page.url, "text": text},
                )
            if action == "screenshot":
                output = (workspace / "outputs" / f"browser-{int(time.time() * 1000)}.png").resolve()
                if workspace != output and workspace not in output.parents:
                    raise PermissionError("browser artifact escaped the Skill workspace")
                output.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(output), full_page=bool(public.get("full_page", True)))
                return success(
                    "Browser screenshot captured",
                    artifacts=[
                        {
                            "path": output.relative_to(workspace).as_posix(),
                            "mime_type": "image/png",
                        }
                    ],
                )
            raise AssertionError("validated browser action was not handled")
        except Exception as exc:  # noqa: BLE001 - external boundary
            return failure(
                f"Browser action {action} failed",
                root_cause=str(exc),
                retry="检查 URL/selector、允许域名和 Playwright 浏览器安装后重试一次。",
                stop_condition="同一页面状态下重复失败时关闭会话并报告诊断。",
            )


__all__ = ["PlaywrightBrowserAdapter"]
