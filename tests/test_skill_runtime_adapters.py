"""Capability-gated production adapter and package action tests."""

import json
import sys
import types
from pathlib import Path

import httpx
import pytest

from app.config import settings
from app.skills import package_loader
from app.skills.adapters.app_bridge import HttpAppBridgeAdapter
from app.skills.adapters import browser as browser_adapter
from app.skills.adapters.browser import PlaywrightBrowserAdapter
from app.skills.adapters.contracts import normalize_observation
from app.skills.adapters.mcp import StreamableHttpMcpAdapter
from app.skills.adapters import runtime as adapter_runtime
from app.skills.base import SkillContext
from app.skills.capabilities import capability_states, check_capabilities
from app.skills.generic import GenericPackageSkill
from app.skills import toolkit
from app.skills.toolkit import SkillToolExecutor


def _package(root: Path, *, requires=None):
    package = root / "runtime-package"
    for name in ("assets", "scripts", "references", "templates"):
        (package / name).mkdir(parents=True, exist_ok=True)
    (package / "SKILL.md").write_text(
        "---\nname: runtime_package\ndescription: Runtime capability test.\n---\n",
        encoding="utf-8",
    )
    (package / "assets" / "payload.bin").write_bytes(b"\x00asset")
    (package / "scripts" / "hello.py").write_text(
        "print('script-ok')\n", encoding="utf-8"
    )
    if requires is None:
        requires = [
            "workspace",
            "documents",
            "package_assets",
            "package_scripts",
            "repository",
            "git",
            "mcp",
            "browser",
            "app",
        ]
    (package / "docmind.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "reason": "tested",
                "requires": requires,
            }
        ),
        encoding="utf-8",
    )
    return package


def _load(monkeypatch, root: Path, *, requires=None):
    _package(root, requires=requires)
    monkeypatch.setattr(package_loader, "PACKAGE_ROOT", root)
    package_loader.load_skill_package.cache_clear()
    return package_loader.load_skill_package("runtime-package")


def test_observation_contract_enriches_legacy_adapter_results():
    result = normalize_observation({"ok": False, "error": "boom"}, operation="test")

    assert result["status"] == "error"
    assert result["summary"] == "test failed"
    assert result["root_cause"] == "boom"
    assert result["next_actions"] == []
    assert result["artifacts"] == []


def test_observation_contract_bounds_untrusted_nested_output():
    result = normalize_observation(
        {
            "ok": True,
            "summary": "large result",
            "result": {"content": "x" * 10000},
            "artifacts": [
                {
                    "path": "outputs/result.txt",
                    "mime_type": "text/plain",
                    "content": "secret" * 1000,
                }
            ],
        },
        operation="call_mcp",
        max_chars=800,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["artifacts"] == [
        {"path": "outputs/result.txt", "mime_type": "text/plain"}
    ]
    assert len(json.dumps(result, ensure_ascii=False, default=str)) <= 800


def test_generic_package_availability_is_capability_gated(tmp_path, monkeypatch):
    package = _load(monkeypatch, tmp_path, requires=["repository"])
    monkeypatch.setattr(settings, "skill_repository_enabled", False)
    monkeypatch.setattr(settings, "skill_repository_root", None)

    skill = GenericPackageSkill(package)

    assert package.required_capabilities == ("repository",)
    assert skill.available is False
    assert "SKILL_REPOSITORY" in skill.unavailable_reason


def test_capability_report_rejects_unknown_host_capability():
    report = check_capabilities(["capability-that-does-not-exist"])

    assert report.available is False
    assert "未实现" in report.reason


def test_host_capability_does_not_grant_an_undeclared_package(tmp_path, monkeypatch):
    package = _load(monkeypatch, tmp_path / "packages", requires=[])
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "secret.txt").write_text("host secret", encoding="utf-8")
    monkeypatch.setattr(settings, "skill_repository_enabled", True)
    monkeypatch.setattr(settings, "skill_repository_root", str(repository))
    monkeypatch.setattr(settings, "skill_package_scripts_enabled", True)
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=9),
        workspace_root=tmp_path / "workspaces",
    )

    names = {item["function"]["name"] for item in executor.tool_definitions()}
    denied_repo = executor.execute("read_repository_file", {"path": "secret.txt"})
    denied_script = executor.execute("run_skill_script", {"name": "hello.py"})
    denied_write = executor.execute(
        "write_file", {"path": "output.txt", "content": "no"}
    )

    assert names == {"read_skill_reference", "read_skill_template"}
    assert denied_repo["denied"] is True
    assert denied_script["denied"] is True
    assert denied_write["denied"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["/tmp/escape"],
        ["../escape"],
        ["--out=/tmp/escape"],
        ["--out=../../escape"],
        ["file:///tmp/escape"],
        ["~/escape"],
    ],
)
def test_package_script_rejects_host_paths_before_subprocess(
    tmp_path, monkeypatch, arguments
):
    package = _load(
        monkeypatch,
        tmp_path / "packages",
        requires=["package_scripts", "workspace"],
    )
    monkeypatch.setattr(settings, "skill_package_scripts_enabled", True)
    monkeypatch.setattr(
        "app.skills.capabilities.sandbox_backend_available", lambda: True
    )
    monkeypatch.setattr(
        toolkit.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid arguments reached subprocess"),
    )
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=2),
        workspace_root=tmp_path / "workspaces",
    )

    result = executor.execute(
        "run_skill_script",
        {"name": "hello.py", "arguments": arguments},
    )

    assert result["ok"] is False
    assert "路径" in result["error"] or "argument" in result["error"]


def test_jupyter_script_defaults_to_workspace_and_registers_artifact(
    tmp_path, monkeypatch
):
    package_loader.load_skill_package.cache_clear()
    package = package_loader.load_skill_package("jupyter-notebook")
    monkeypatch.setattr(settings, "skill_package_scripts_enabled", True)
    monkeypatch.setattr(settings, "skill_package_script_interpreters", "python")
    monkeypatch.setattr(
        "app.skills.capabilities.sandbox_backend_available", lambda: True
    )
    monkeypatch.setattr(toolkit, "confine_command", lambda command, workspace: command)
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=3),
        workspace_root=tmp_path / "workspaces",
    )

    result = executor.execute(
        "run_skill_script",
        {
            "name": "new_notebook.py",
            "arguments": ["--title", "Safe Notebook", "--kind", "tutorial"],
        },
    )

    assert result["ok"] is True, result.get("stderr")
    output = executor.workspace / "output/jupyter-notebook/safe-notebook.ipynb"
    assert output.is_file()
    assert "output/jupyter-notebook/safe-notebook.ipynb" in executor.generated_files


def test_zip_rejects_workspace_symlink(tmp_path, monkeypatch):
    from app.skills.generic import _zip_generated_files

    package = _load(
        monkeypatch,
        tmp_path / "packages",
        requires=["workspace"],
    )
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=4),
        workspace_root=tmp_path / "workspaces",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = executor.workspace / "outputs" / "link.txt"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    executor.generated_files.append("outputs/link.txt")

    with pytest.raises(PermissionError, match="符号链接"):
        _zip_generated_files(executor)


def test_capability_report_fails_closed_for_invalid_adapter_configs(monkeypatch):
    monkeypatch.setattr(settings, "skill_mcp_enabled", True)
    monkeypatch.setattr(
        settings,
        "skill_mcp_servers_json",
        '{"broken":{"url":"file:///tmp/socket","tools":"search"}}',
    )
    monkeypatch.setattr(settings, "skill_browser_enabled", True)
    monkeypatch.setattr(settings, "skill_browser_allowed_hosts", "")
    monkeypatch.setattr(settings, "skill_app_enabled", True)
    monkeypatch.setattr(settings, "skill_app_bridge_url", "https://example.com")
    monkeypatch.setattr(settings, "skill_app_bridge_token", "token")
    monkeypatch.setattr(
        settings,
        "skill_app_allowed_actions_json",
        '{"Finder":[]}',
    )
    monkeypatch.setattr(
        "app.skills.capabilities._playwright_runtime_available",
        lambda: True,
    )

    states = capability_states()
    assert states["mcp"].available is False
    assert states["browser"].available is False
    assert states["app"].available is False


def test_asset_script_and_repository_tools_use_separate_roots(tmp_path, monkeypatch):
    package_root = tmp_path / "packages"
    package = _load(monkeypatch, package_root)
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "app.py").write_text("needle = 1\n", encoding="utf-8")
    monkeypatch.setattr(settings, "skill_package_scripts_enabled", True)
    monkeypatch.setattr(
        "app.skills.capabilities.sandbox_backend_available", lambda: True
    )
    monkeypatch.setattr(toolkit, "confine_command", lambda command, workspace: command)
    monkeypatch.setattr(settings, "skill_repository_enabled", True)
    monkeypatch.setattr(settings, "skill_repository_root", str(repository))

    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=9),
        workspace_root=tmp_path / "workspaces",
    )
    copied = executor.execute(
        "copy_skill_asset",
        {"name": "payload.bin", "destination": "outputs/payload.bin"},
    )
    script = executor.execute("run_skill_script", {"name": "hello.py"})
    read = executor.execute("read_repository_file", {"path": "app.py"})
    searched = executor.execute("search_repository", {"query": "needle"})
    denied = executor.execute("read_repository_file", {"path": "../secret"})

    assert copied["ok"] is True
    assert (executor.workspace / "outputs/payload.bin").read_bytes() == b"\x00asset"
    assert script["ok"] is True and "script-ok" in script["stdout"], script["stderr"]
    assert read["content"] == "needle = 1\n"
    assert searched["matches"]
    assert denied["ok"] is False and "越界" in denied["error"]


def test_workspace_artifacts_enforce_configured_size_limit(tmp_path, monkeypatch):
    package = _load(monkeypatch, tmp_path / "packages")
    monkeypatch.setattr(settings, "skill_artifact_max_bytes", 5)
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=9),
        workspace_root=tmp_path / "workspaces",
    )

    written = executor.execute(
        "write_file",
        {"path": "outputs/too-large.txt", "content": "123456"},
    )
    copied = executor.execute(
        "copy_skill_asset",
        {"name": "payload.bin", "destination": "outputs/payload.bin"},
    )

    assert written["ok"] is False and "bytes 上限" in written["error"]
    assert copied["ok"] is False and "bytes 上限" in copied["error"]
    assert executor.generated_files == []


def test_generated_artifact_bundle_enforces_aggregate_limit(tmp_path, monkeypatch):
    from app.skills.generic import _zip_generated_files

    package = _load(monkeypatch, tmp_path / "packages")
    monkeypatch.setattr(settings, "skill_artifact_max_bytes", 5)
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=9),
        workspace_root=tmp_path / "workspaces",
    )
    assert executor.execute("write_file", {"path": "outputs/a.txt", "content": "1234"})[
        "ok"
    ]
    assert executor.execute("write_file", {"path": "outputs/b.txt", "content": "5678"})[
        "ok"
    ]

    with pytest.raises(ValueError, match="总大小"):
        _zip_generated_files(executor)


def test_frozen_runtime_routes_python_package_script_through_sidecar(
    tmp_path, monkeypatch
):
    package = _load(monkeypatch, tmp_path / "packages")
    monkeypatch.setattr(settings, "skill_package_scripts_enabled", True)
    monkeypatch.setattr(
        "app.skills.capabilities.sandbox_backend_available", lambda: True
    )
    monkeypatch.setattr(
        toolkit,
        "confine_command",
        lambda command, workspace: ["/usr/bin/sandbox-exec", *command],
    )
    monkeypatch.setattr(toolkit.sys, "frozen", True, raising=False)
    captured = {}

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _run(command, **kwargs):
        captured["command"] = command
        return _Completed()

    monkeypatch.setattr(toolkit.subprocess, "run", _run)
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=7),
        workspace_root=tmp_path / "workspaces",
    )
    result = executor.execute(
        "run_skill_script",
        {"name": "hello.py", "arguments": ["--demo"]},
    )
    assert result["ok"] is True
    assert captured["command"][0] == "/usr/bin/sandbox-exec"
    sidecar_index = captured["command"].index(toolkit.sys.executable)
    assert captured["command"][sidecar_index : sidecar_index + 2] == [
        toolkit.sys.executable,
        "run-script",
    ]
    assert captured["command"][-2:] == ["--", "--demo"]


def test_extended_toolkit_actions_and_adapter_artifacts(tmp_path, monkeypatch):
    package = _load(monkeypatch, tmp_path / "packages")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "a.py").write_text("print('a')\n", encoding="utf-8")
    monkeypatch.setattr(settings, "skill_repository_enabled", True)
    monkeypatch.setattr(settings, "skill_repository_root", str(repository))
    monkeypatch.setattr(settings, "skill_browser_enabled", True)
    monkeypatch.setattr(settings, "skill_browser_allowed_hosts", "localhost")
    monkeypatch.setattr(
        "app.skills.capabilities._playwright_runtime_available", lambda: True
    )
    monkeypatch.setattr(settings, "skill_app_enabled", True)
    monkeypatch.setattr(settings, "skill_app_bridge_url", "http://127.0.0.1:18999")
    monkeypatch.setattr(settings, "skill_app_bridge_token", "token")
    monkeypatch.setattr(
        settings,
        "skill_app_allowed_actions_json",
        '{"Finder":["reveal"]}',
    )
    monkeypatch.setattr(toolkit, "_BROWSER_ADAPTERS", {})
    monkeypatch.setattr(toolkit, "_APP_ADAPTERS", {})
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=8),
        workspace_root=tmp_path / "workspaces",
    )

    asset = executor.execute("read_skill_asset", {"name": "payload.bin"})
    listed = executor.execute("list_repository_files", {})
    modified = executor.execute(
        "modify_code",
        {"path": "outputs/code.py", "content": "x = 1\n", "instruction": "demo"},
    )
    missing = executor.execute("list_files", {"path": "missing"})
    unknown = executor.execute("does_not_exist", {})

    class _Completed:
        returncode = 0
        stdout = "abc\t2026-01-01\tDev\tCommit\n"
        stderr = ""

    monkeypatch.setattr(toolkit.subprocess, "run", lambda *args, **kwargs: _Completed())
    history = executor.execute("git_history", {"path": "a.py", "limit": 2})

    class _Browser:
        def __call__(self, arguments, context):
            output = Path(arguments["__workspace"]) / "outputs" / "shot.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"png")
            return {"ok": True, "artifacts": [{"path": "outputs/shot.png"}]}

        def close_session(self, session_id):
            self.closed = session_id

    browser = _Browser()
    monkeypatch.setitem(toolkit._BROWSER_ADAPTERS, "screenshot", browser)
    monkeypatch.setitem(toolkit._BROWSER_ADAPTERS, "close", browser)
    monkeypatch.setitem(
        toolkit._APP_ADAPTERS,
        ("Finder", "reveal"),
        lambda arguments, context: {
            "ok": True,
            "workspace": arguments["__workspace"],
            "user_id": context.user_id,
        },
    )
    screenshot = executor.execute("use_browser_tool", {"action": "screenshot"})
    app_result = executor.execute("use_app_tool", {"app": "Finder", "action": "reveal"})
    executor.close()

    assert asset["bytes"] == 6 and asset["encoding"] == "base64"
    assert listed["files"] == ["a.py"]
    assert modified["instruction"] == "demo" and modified["mode"] == "replace"
    assert missing["files"] == []
    assert unknown["status"] == "error"
    assert history["history"] == ["abc\t2026-01-01\tDev\tCommit"]
    assert screenshot["ok"] is True
    assert "outputs/shot.png" in executor.generated_files
    assert app_result["user_id"] == 8
    assert browser.closed == executor.session_id


def test_mcp_adapter_initializes_session_and_calls_allowlisted_tool(monkeypatch):
    calls = []
    monkeypatch.setenv("DOCS_MCP_TOKEN", "mcp-secret")

    def handler(request: httpx.Request):
        assert request.headers["Authorization"] == "Bearer mcp-secret"
        if request.method == "DELETE":
            calls.append(f"delete:{request.headers['Mcp-Session-Id']}")
            return httpx.Response(200)
        payload = json.loads(request.content)
        calls.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "session-1"},
                json={"jsonrpc": "2.0", "id": payload["id"], "result": {}},
            )
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"content": [{"type": "text", "text": "ok"}]},
            },
        )

    adapter = StreamableHttpMcpAdapter(
        "docs",
        {
            "url": "https://mcp.example.test",
            "tools": ["search"],
            "token_env": "DOCS_MCP_TOKEN",
        },
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = adapter(
        {"__tool_name": "search", "__session_id": "run-a", "query": "DocMind"},
        SkillContext(user_id=1),
    )
    adapter(
        {"__tool_name": "search", "__session_id": "run-a", "query": "again"},
        SkillContext(user_id=1),
    )
    adapter(
        {"__tool_name": "search", "__session_id": "run-b", "query": "isolated"},
        SkillContext(user_id=1),
    )

    assert result["ok"] is True
    assert calls == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert result["result"]["content"][0]["text"] == "ok"
    adapter.close_session("run-a")
    adapter.close_session("run-b")
    adapter.close_session("missing")
    assert calls[-2:] == ["delete:session-1", "delete:session-1"]
    assert adapter._states() == {}


def test_executor_close_releases_each_adapter_session_once(tmp_path, monkeypatch):
    package = _load(monkeypatch, tmp_path / "packages")
    closed = []

    class _Adapter:
        def close_session(self, session_id):
            closed.append(session_id)

    adapter = _Adapter()
    monkeypatch.setattr(toolkit, "_MCP_ADAPTERS", {("docs", "*"): adapter})
    monkeypatch.setattr(
        toolkit,
        "_BROWSER_ADAPTERS",
        {"open": adapter, "close": adapter},
    )
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=1),
        workspace_root=tmp_path / "workspaces",
    )

    executor.close()
    assert closed == [executor.session_id]


def test_mcp_adapter_denies_tool_before_network_call():
    adapter = StreamableHttpMcpAdapter(
        "docs",
        {"url": "https://mcp.example.test", "tools": ["search"]},
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError("network called"))
            )
        ),
    )

    result = adapter({"__tool_name": "delete"}, SkillContext(user_id=1))

    assert result["ok"] is False
    assert result["disabled"] is True
    assert "allowlist" in result["root_cause"]


def test_mcp_adapter_requires_configured_token_environment(monkeypatch):
    monkeypatch.delenv("MISSING_MCP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MISSING_MCP_TOKEN"):
        StreamableHttpMcpAdapter(
            "docs",
            {
                "url": "https://mcp.example.test",
                "tools": ["search"],
                "token_env": "MISSING_MCP_TOKEN",
            },
            timeout_seconds=1,
        )


def test_mcp_timeout_returns_recoverable_observation():
    adapter = StreamableHttpMcpAdapter(
        "docs",
        {"url": "https://mcp.example.test", "tools": ["search"]},
        timeout_seconds=0.01,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    httpx.ReadTimeout("timed out", request=request)
                )
            )
        ),
    )
    result = adapter(
        {"__tool_name": "search", "query": "DocMind"},
        SkillContext(user_id=1),
    )
    assert result["status"] == "error"
    assert "timed out" in result["root_cause"]
    assert result["next_actions"]


def test_generic_executor_discovers_and_calls_registered_mcp(tmp_path, monkeypatch):
    package = _load(monkeypatch, tmp_path / "packages")
    monkeypatch.setattr(settings, "skill_mcp_enabled", True)
    monkeypatch.setattr(
        settings,
        "skill_mcp_servers_json",
        '{"docs":{"url":"https://mcp.example.test","tools":["search"]}}',
    )

    def _adapter(arguments, context):
        return {
            "ok": True,
            "server": "docs",
            "tool": arguments["__tool_name"],
            "user_id": context.user_id,
        }

    monkeypatch.setitem(toolkit._MCP_ADAPTERS, ("docs", "*"), _adapter)
    executor = SkillToolExecutor(
        package=package,
        context=SkillContext(user_id=42),
        workspace_root=tmp_path / "workspaces",
    )
    names = {item["function"]["name"] for item in executor.tool_definitions()}
    result = executor.execute(
        "call_mcp",
        {"server": "docs", "tool": "search", "arguments": {"q": "DocMind"}},
    )

    assert "call_mcp" in names
    assert result["status"] == "success"
    assert result["user_id"] == 42


def test_runtime_installer_registers_enabled_allowlisted_adapters(monkeypatch):
    monkeypatch.setattr(toolkit, "_MCP_ADAPTERS", {})
    monkeypatch.setattr(toolkit, "_BROWSER_ADAPTERS", {})
    monkeypatch.setattr(toolkit, "_APP_ADAPTERS", {})
    monkeypatch.setattr(settings, "skill_mcp_enabled", True)
    monkeypatch.setattr(
        settings,
        "skill_mcp_servers_json",
        '{"docs":{"url":"https://mcp.test","tools":["search"]}}',
    )
    monkeypatch.setattr(settings, "skill_browser_enabled", True)
    monkeypatch.setattr(settings, "skill_app_enabled", True)
    monkeypatch.setattr(settings, "skill_app_bridge_url", "http://localhost:18999")
    monkeypatch.setattr(settings, "skill_app_bridge_token", "token")
    monkeypatch.setattr(
        settings,
        "skill_app_allowed_actions_json",
        '{"Finder":["open","reveal"]}',
    )

    class _Mcp:
        def __init__(self, server, config, **kwargs):
            self.server = server

        def __call__(self, arguments, context):
            return {"ok": True}

    class _Browser:
        ACTIONS = {"open", "close"}

        def __init__(self, **kwargs):
            pass

        def __call__(self, arguments, context):
            return {"ok": True}

    class _App:
        def __init__(self, base_url, token, app, action, **kwargs):
            self.app = app
            self.action = action

        def __call__(self, arguments, context):
            return {"ok": True}

    monkeypatch.setattr(adapter_runtime, "StreamableHttpMcpAdapter", _Mcp)
    monkeypatch.setattr(adapter_runtime, "PlaywrightBrowserAdapter", _Browser)
    monkeypatch.setattr(adapter_runtime, "HttpAppBridgeAdapter", _App)
    adapter_runtime.install_runtime_adapters()

    assert ("docs", "*") in toolkit._MCP_ADAPTERS
    assert set(toolkit._BROWSER_ADAPTERS) == {"open", "close"}
    assert set(toolkit._APP_ADAPTERS) == {("Finder", "open"), ("Finder", "reveal")}


def test_runtime_installer_ignores_invalid_json_and_adapter_config(monkeypatch):
    monkeypatch.setattr(toolkit, "_MCP_ADAPTERS", {})
    monkeypatch.setattr(settings, "skill_mcp_enabled", True)
    monkeypatch.setattr(settings, "skill_mcp_servers_json", '{"broken": {}}')
    monkeypatch.setattr(settings, "skill_browser_enabled", False)
    monkeypatch.setattr(settings, "skill_app_enabled", False)
    monkeypatch.setattr(
        adapter_runtime,
        "StreamableHttpMcpAdapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    )
    adapter_runtime.install_runtime_adapters()
    assert toolkit._MCP_ADAPTERS == {}
    assert adapter_runtime._object("not-json") == {}
    assert adapter_runtime._object("[]") == {}


def test_browser_rejects_unlisted_host_before_launch(monkeypatch):
    adapter = PlaywrightBrowserAdapter(
        headless=True,
        allowed_hosts={"localhost"},
        timeout_seconds=1,
    )
    monkeypatch.setattr(
        adapter,
        "_page",
        lambda session: (_ for _ in ()).throw(AssertionError("browser launched")),
    )
    result = adapter(
        {
            "__action": "open",
            "__session_id": "s1",
            "url": "https://example.com",
        },
        SkillContext(user_id=1),
    )
    assert result["status"] == "error"
    assert "not allowed" in result["root_cause"]


def test_browser_rejects_redirect_and_click_navigation_outside_allowlist(monkeypatch):
    class _Locator:
        def __init__(self, page):
            self.page = page

        def click(self):
            self.page.url = "https://example.com/escaped"

    class _Page:
        url = "http://localhost/start"

        def goto(self, _url, wait_until):
            assert wait_until == "domcontentloaded"
            self.url = "https://example.com/redirected"

        def locator(self, _selector):
            return _Locator(self)

    page = _Page()
    adapter = PlaywrightBrowserAdapter(
        headless=True,
        allowed_hosts={"localhost"},
        timeout_seconds=1,
    )
    monkeypatch.setattr(adapter, "_page", lambda _session: page)
    context = SkillContext(user_id=1)

    redirected = adapter(
        {
            "__action": "open",
            "__session_id": "redirect",
            "url": "http://localhost/start",
        },
        context,
    )
    assert redirected["status"] == "error"
    assert "not allowed" in redirected["root_cause"]

    page.url = "http://localhost/start"
    clicked = adapter(
        {"__action": "click", "__session_id": "redirect", "selector": "#go"},
        context,
    )
    assert clicked["status"] == "error"
    assert "not allowed" in clicked["root_cause"]


def test_browser_request_routing_blocks_disallowed_subresources():
    class _Request:
        url = "https://example.com/tracker.js"

    class _Route:
        request = _Request()

        def __init__(self):
            self.action = None

        def abort(self, reason):
            self.action = ("abort", reason)

        def continue_(self):
            self.action = ("continue", None)

    route = _Route()
    adapter = PlaywrightBrowserAdapter(
        headless=True,
        allowed_hosts={"localhost"},
        timeout_seconds=1,
    )
    adapter._route_request(route)
    assert route.action == ("abort", "blockedbyclient")


def test_browser_actions_share_session_and_write_scoped_screenshot(
    tmp_path, monkeypatch
):
    class _Locator:
        def __init__(self):
            self.clicked = False
            self.value = None

        def click(self):
            self.clicked = True

        def fill(self, value):
            self.value = value

        def inner_text(self):
            return "page text"

    class _Page:
        url = "http://localhost/current"

        def __init__(self):
            self.locators = {}
            self.visited = None

        def goto(self, url, wait_until):
            self.visited = (url, wait_until)
            self.url = url

        def title(self):
            return "DocMind"

        def locator(self, selector):
            return self.locators.setdefault(selector, _Locator())

        def screenshot(self, path, full_page):
            Path(path).write_bytes(b"png")

    page = _Page()
    adapter = PlaywrightBrowserAdapter(
        headless=True,
        allowed_hosts={"localhost"},
        timeout_seconds=1,
    )
    monkeypatch.setattr(adapter, "_page", lambda session: page)
    context = SkillContext(user_id=1)
    internal = {"__session_id": "s1", "__workspace": str(tmp_path)}

    opened = adapter(
        {**internal, "__action": "open", "url": "http://localhost/page"},
        context,
    )
    clicked = adapter({**internal, "__action": "click", "selector": "#go"}, context)
    filled = adapter(
        {**internal, "__action": "fill", "selector": "#q", "text": "hello"},
        context,
    )
    text_result = adapter({**internal, "__action": "text", "selector": "main"}, context)
    screenshot = adapter(
        {**internal, "__action": "screenshot", "full_page": False}, context
    )

    assert opened["title"] == "DocMind"
    assert clicked["ok"] is True and page.locators["#go"].clicked
    assert filled["ok"] is True and page.locators["#q"].value == "hello"
    assert text_result["text"] == "page text"
    artifact = tmp_path / screenshot["artifacts"][0]["path"]
    assert artifact.read_bytes() == b"png"


def test_browser_session_lifecycle_uses_one_playwright_context(monkeypatch):
    events = []

    class _Page:
        def set_default_timeout(self, timeout):
            events.append(("timeout", timeout))

    page = _Page()

    class _Context:
        def route(self, pattern, handler):
            events.append(("route", pattern, handler.__name__))

        def new_page(self):
            return page

        def close(self):
            events.append("context-close")

    class _Browser:
        def new_context(self):
            return _Context()

        def close(self):
            events.append("browser-close")

    class _Chromium:
        def launch(self, headless):
            events.append(("launch", headless))
            return _Browser()

    class _Manager:
        chromium = _Chromium()

        def start(self):
            events.append("start")
            return self

        def stop(self):
            events.append("stop")

    sync_module = types.ModuleType("playwright.sync_api")
    sync_module.sync_playwright = lambda: _Manager()
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_module)

    adapter = browser_adapter.PlaywrightBrowserAdapter(
        headless=True,
        allowed_hosts={"localhost"},
        timeout_seconds=2,
    )
    assert adapter._page("one") is page
    assert adapter._page("one") is page
    adapter.close_session("one")
    adapter.close_session("missing")
    assert events.count("start") == 1
    assert ("timeout", 2000) in events
    assert events[-3:] == ["context-close", "browser-close", "stop"]


def test_app_bridge_is_loopback_authenticated_and_structured():
    def handler(request: httpx.Request):
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"ok": True, "window": "main"})

    adapter = HttpAppBridgeAdapter(
        "http://127.0.0.1:18999",
        "secret",
        "Finder",
        "open",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = adapter({"path": "/tmp/a"}, SkillContext(user_id=3))

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["result"]["window"] == "main"


def test_app_bridge_rejects_non_loopback_and_missing_auth():
    with pytest.raises(ValueError, match="loopback"):
        HttpAppBridgeAdapter(
            "https://bridge.example.test",
            "secret",
            "Finder",
            "open",
            timeout_seconds=1,
        )
    with pytest.raises(ValueError, match="authentication"):
        HttpAppBridgeAdapter(
            "http://127.0.0.1:18999",
            "",
            "Finder",
            "open",
            timeout_seconds=1,
        )
