"""Codex-style 通用 Skill 包测试。

目标：让 DocMind 不再只能注册 Python BaseSkill，也能扫描只包含 SKILL.md
的通用技能包，并用受控工具集执行 Markdown 指令。
"""

import base64
from io import BytesIO
from zipfile import ZipFile

import pytest

from app.skills import package_loader, registry
from app.skills.base import SkillContext


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def _write_codex_package(root, slug="codex-note"):
    package_dir = root / slug
    (package_dir / "references").mkdir(parents=True)
    (package_dir / "templates").mkdir()
    (package_dir / "scripts").mkdir()
    (package_dir / "assets").mkdir()
    (package_dir / "SKILL.md").write_text(
        """---
name: codex_note
description: Read materials and write a concise note when the user asks for note generation.
---

# Codex Note

1. Read references/style.md when tone guidance is needed.
2. Write the final note to `outputs/note.md`.
""",
        encoding="utf-8",
    )
    (package_dir / "references" / "style.md").write_text("保持简洁。", encoding="utf-8")
    (package_dir / "templates" / "note.md").write_text("# {title}\n", encoding="utf-8")
    (package_dir / "scripts" / "helper.py").write_text("print('helper')\n", encoding="utf-8")
    (package_dir / "assets" / "logo.txt").write_text("logo", encoding="utf-8")
    return package_dir


@pytest.fixture
def isolated_package_root(tmp_path, monkeypatch):
    monkeypatch.setattr(package_loader, "PACKAGE_ROOT", tmp_path)
    package_loader.load_skill_package.cache_clear()
    yield tmp_path
    package_loader.load_skill_package.cache_clear()


class TestCodexStylePackageLoader:
    def test_loads_codex_style_package_without_skill_json(self, isolated_package_root):
        _write_codex_package(isolated_package_root)

        package = package_loader.load_skill_package("codex-note")

        assert package.slug == "codex-note"
        assert package.name == "codex_note"
        assert "note generation" in package.description
        assert package.source == "codex"
        assert package.parameters["required"] == ["task"]
        assert "style.md" in package.references
        assert "note.md" in package.templates
        assert package.script_names == ["helper.py"]
        assert package.asset_names == ["logo.txt"]

    def test_rejects_skill_without_description(self, isolated_package_root):
        package_dir = isolated_package_root / "bad-skill"
        package_dir.mkdir()
        (package_dir / "SKILL.md").write_text(
            "---\nname: bad_skill\n---\n\nMissing description.", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="description"):
            package_loader.load_skill_package("bad-skill")


class TestGenericPackageRegistration:
    def test_registers_unbacked_codex_package_as_generic_skill(self, isolated_package_root):
        _write_codex_package(isolated_package_root)
        saved = dict(registry._REGISTRY)
        registry._REGISTRY.clear()
        try:
            registry.register_generic_package_skills()
            skill = registry.get_skill("codex_note")
            assert skill is not None
            assert skill.execution_mode == "generic_package"
            assert skill.to_tool()["function"]["parameters"]["required"] == ["task"]

            # 幂等：启动或测试重复调用不应重复注册/抛错。
            registry.register_generic_package_skills()
            assert [s.name for s in registry.all_skills()] == ["codex_note"]
        finally:
            registry._REGISTRY.clear()
            registry._REGISTRY.update(saved)


class TestSkillToolExecutor:
    def test_file_tools_are_scoped_to_user_workspace(self, isolated_package_root, tmp_path):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills.toolkit import SkillToolExecutor

        executor = SkillToolExecutor(
            package=package,
            context=SkillContext(user_id=7),
            workspace_root=tmp_path / "workspaces",
        )

        write_result = executor.execute(
            "write_file", {"path": "outputs/note.md", "content": "# Note"}
        )
        assert write_result["ok"] is True

        read_result = executor.execute("read_file", {"path": "outputs/note.md"})
        assert read_result["content"] == "# Note"

        listed = executor.execute("list_files", {"path": "outputs"})
        assert listed["files"] == ["outputs/note.md"]

        denied = executor.execute("read_file", {"path": "../secret.txt"})
        assert denied["ok"] is False
        assert "工作区" in denied["error"]

    def test_shell_tool_is_disabled_by_default_and_allowlisted_when_enabled(
        self, isolated_package_root, tmp_path
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills.toolkit import SkillToolExecutor

        executor = SkillToolExecutor(
            package=package,
            context=SkillContext(user_id=7),
            workspace_root=tmp_path / "workspaces",
        )
        disabled = executor.execute("run_shell", {"command": "echo hello"})
        assert disabled["ok"] is False
        assert disabled["disabled"] is True

        enabled = SkillToolExecutor(
            package=package,
            context=SkillContext(user_id=7),
            workspace_root=tmp_path / "workspaces",
            shell_enabled=True,
            shell_allowed_commands={"echo"},
        )
        result = enabled.execute("run_shell", {"command": "echo hello"})
        assert result["ok"] is True
        assert result["stdout"].strip() == "hello"

        denied = enabled.execute("run_shell", {"command": "rm -rf ."})
        assert denied["ok"] is False
        assert "不在允许列表" in denied["error"]

    def test_mcp_browser_and_app_tools_have_safe_unconfigured_fallback(
        self, isolated_package_root, tmp_path
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills.toolkit import SkillToolExecutor

        executor = SkillToolExecutor(
            package=package,
            context=SkillContext(user_id=7),
            workspace_root=tmp_path / "workspaces",
        )

        assert executor.execute("call_mcp", {"server": "x", "tool": "y"})["ok"] is False
        assert executor.execute("use_browser_tool", {"action": "open"})["ok"] is False
        assert executor.execute("use_app_tool", {"app": "Finder", "action": "open"})["ok"] is False


class TestGenericPackageSkillRunner:
    def test_generic_skill_follows_markdown_instructions_and_returns_zip_artifact(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills.generic import GenericPackageSkill
        from app.skills import generic

        calls = {"n": 0}
        captured = {}

        def _llm(messages, tools=None, temperature=0.2):
            calls["n"] += 1
            captured["tools"] = [t["function"]["name"] for t in tools]
            if calls["n"] == 1:
                return _FakeMsg(
                    tool_calls=[
                        _FakeToolCall(
                            "c1",
                            "read_skill_reference",
                            '{"name":"style.md"}',
                        ),
                        _FakeToolCall(
                            "c2",
                            "write_file",
                            '{"path":"outputs/note.md","content":"# Note\\n完成"}',
                        ),
                    ]
                )
            return _FakeMsg(content="已按 SKILL.md 生成 note.md。")

        monkeypatch.setattr(generic, "chat_completion", _llm)
        monkeypatch.setattr(generic.settings, "skill_workspace_dir", str(tmp_path / "workspaces"))

        skill = GenericPackageSkill(package)
        result = skill.run(SkillContext(user_id=3), task="写一份项目笔记")

        assert result["type"] == "generic_skill"
        assert result["artifact_kind"] == "file"
        assert result["answer"] == "已按 SKILL.md 生成 note.md。"
        assert result["generated_files"] == ["outputs/note.md"]
        assert "read_skill_reference" in captured["tools"]
        assert "write_file" in captured["tools"]

        payload = base64.b64decode(result["download"]["content"])
        with ZipFile(BytesIO(payload)) as zf:
            assert zf.read("outputs/note.md").decode("utf-8") == "# Note\n完成"
