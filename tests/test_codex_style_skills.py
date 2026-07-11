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


def _write_codex_package(root, slug="codex-note", runtime_ready=True):
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
    if runtime_ready:
        (package_dir / "docmind.json").write_text(
            '{"status":"ready","reason":"test package"}', encoding="utf-8"
        )
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
        assert package.runtime_status == "ready"

    def test_rejects_skill_without_description(self, isolated_package_root):
        package_dir = isolated_package_root / "bad-skill"
        package_dir.mkdir()
        (package_dir / "SKILL.md").write_text(
            "---\nname: bad_skill\n---\n\nMissing description.", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="description"):
            package_loader.load_skill_package("bad-skill")

    def test_list_skips_invalid_package_without_breaking_startup(self, isolated_package_root):
        _write_codex_package(isolated_package_root)
        bad_dir = isolated_package_root / "bad-skill"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text(
            "---\nname: bad_skill\n---\n\nMissing description.", encoding="utf-8"
        )

        packages = package_loader.list_skill_packages()

        assert [package.name for package in packages] == ["codex_note"]


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

    def test_unreviewed_package_is_registered_but_quarantined_from_agent_tools(
        self, isolated_package_root
    ):
        _write_codex_package(
            isolated_package_root, slug="unreviewed", runtime_ready=False
        )
        saved = dict(registry._REGISTRY)
        registry._REGISTRY.clear()
        try:
            registry.register_generic_package_skills()
            skill = registry.get_skill("codex_note")
            assert skill is not None
            assert skill.available is False
            assert "尚未经过" in skill.unavailable_reason
            assert registry.all_tools() == []
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

    def test_uploaded_document_tools_respect_user_and_context_document(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills import toolkit
        from app.skills.toolkit import SkillToolExecutor

        monkeypatch.setattr(
            toolkit,
            "fetch_user_documents",
            lambda user_id: [{"id": 10, "filename": "a.md"}],
        )
        captured = {}

        def _fake_fetch_material_text(user_id, document_id=None, max_chars=12000):
            captured["user_id"] = user_id
            captured["document_id"] = document_id
            captured["max_chars"] = max_chars
            return "用户上传文档内容", [document_id]

        monkeypatch.setattr(toolkit, "fetch_material_text", _fake_fetch_material_text)

        executor = SkillToolExecutor(
            package=package,
            context=SkillContext(user_id=7, document_id=10),
            workspace_root=tmp_path / "workspaces",
        )

        listed = executor.execute("list_uploaded_documents", {})
        read = executor.execute("read_uploaded_document", {})

        assert listed["documents"] == [{"id": 10, "filename": "a.md"}]
        assert read["ok"] is True
        assert read["content"] == "用户上传文档内容"
        assert read["document_ids"] == [10]
        assert captured == {"user_id": 7, "document_id": 10, "max_chars": 12000}

    def test_uploaded_document_search_uses_rag_retrieval(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills import toolkit
        from app.skills.toolkit import SkillToolExecutor

        captured = {}

        def _fake_retrieve(user_id, query, document_id=None, max_chars=12000):
            captured.update(
                user_id=user_id,
                query=query,
                document_id=document_id,
                max_chars=max_chars,
            )
            return "RAG 命中片段", [10], [{"document_id": 10, "chunk_index": 2}]

        monkeypatch.setattr(toolkit, "fetch_retrieved_material", _fake_retrieve)
        executor = SkillToolExecutor(
            package=package,
            context=SkillContext(user_id=7, document_id=10),
            workspace_root=tmp_path / "workspaces",
        )

        result = executor.execute(
            "search_uploaded_documents", {"query": "项目进展", "max_chars": 6000}
        )

        assert result["ok"] is True
        assert result["content"] == "RAG 命中片段"
        assert result["retrieval_mode"] == "hybrid"
        assert captured == {
            "user_id": 7,
            "query": "项目进展",
            "document_id": 10,
            "max_chars": 6000,
        }


class TestGenericPackageSkillRunner:
    def test_generic_skill_follows_markdown_instructions_and_returns_zip_artifact(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills.generic import GenericPackageSkill
        from app.skills import generic, toolkit

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
                            "c-doc",
                            "read_uploaded_document",
                            '{}',
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
        assert "read_uploaded_document" in captured["tools"]
        assert "write_file" in captured["tools"]

        payload = base64.b64decode(result["download"]["content"])
        with ZipFile(BytesIO(payload)) as zf:
            assert zf.read("outputs/note.md").decode("utf-8") == "# Note\n完成"

    def test_direct_text_result_still_becomes_downloadable_artifact(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills import generic, toolkit
        from app.skills.generic import GenericPackageSkill

        monkeypatch.setattr(
            generic,
            "chat_completion",
            lambda messages, tools=None, temperature=0.2: _FakeMsg(content="# 完整材料\n正文"),
        )
        monkeypatch.setattr(
            toolkit,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=12000: (
                "材料片段",
                [10],
                [{"document_id": 10, "chunk_index": 0, "score": 0.9}],
            ),
        )
        monkeypatch.setattr(generic.settings, "skill_workspace_dir", str(tmp_path / "workspaces"))

        result = GenericPackageSkill(package).run(
            SkillContext(user_id=3), task="根据材料生成文档"
        )

        assert result["artifact_kind"] == "file"
        assert result["generated_files"] == ["outputs/codex-note-result.md"]
        payload = base64.b64decode(result["download"]["content"])
        with ZipFile(BytesIO(payload)) as zf:
            assert "完整材料" in zf.read("outputs/codex-note-result.md").decode("utf-8")

    def test_selected_document_is_rag_grounded_before_inner_llm(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills import generic, toolkit
        from app.skills.generic import GenericPackageSkill

        captured = {}
        monkeypatch.setattr(
            toolkit,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=12000: (
                "【文档 10 · 片段 2】\n真实材料",
                [10],
                [{"document_id": 10, "chunk_index": 2, "score": 0.9}],
            ),
        )

        def _llm(messages, tools=None, temperature=0.2):
            captured["messages"] = messages
            return _FakeMsg(content="基于真实材料的结果")

        monkeypatch.setattr(generic, "chat_completion", _llm)
        monkeypatch.setattr(generic.settings, "skill_workspace_dir", str(tmp_path / "workspaces"))

        result = GenericPackageSkill(package).run(
            SkillContext(user_id=3, document_id=10), task="根据文档生成材料"
        )

        assert result["actions"][0]["tool"] == "search_uploaded_documents"
        assert result["actions"][0]["ok"] is True
        assert result["grounding"]["mode"] == "hybrid_rag"
        assert result["grounding"]["document_ids"] == [10]
        assert "真实材料" in captured["messages"][1]["content"]

    def test_document_task_stops_instead_of_hallucinating_when_rag_fails(
        self, isolated_package_root, tmp_path, monkeypatch
    ):
        _write_codex_package(isolated_package_root)
        package = package_loader.load_skill_package("codex-note")

        from app.skills import generic, toolkit
        from app.skills.generic import GenericPackageSkill

        monkeypatch.setattr(
            toolkit,
            "fetch_retrieved_material",
            lambda *args, **kwargs: ("", [], []),
        )
        monkeypatch.setattr(
            generic,
            "chat_completion",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("RAG 失败后不应继续让 LLM 无依据生成")
            ),
        )
        monkeypatch.setattr(generic.settings, "skill_workspace_dir", str(tmp_path / "workspaces"))

        result = GenericPackageSkill(package).run(
            SkillContext(user_id=3, document_id=10), task="根据文档生成材料"
        )

        assert "无法基于上传文档" in result["answer"]
        assert result["actions"][0]["ok"] is False
        assert result["artifact_kind"] == "file"
