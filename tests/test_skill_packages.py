"""Skill Package 两层架构测试。

验证：目录化 package 可加载、metadata 可驱动 Function Calling schema、
templates/references 可供 Python 执行层读取。
"""

import app.skills  # noqa: F401 触发注册
from app.skills.package_loader import list_skill_packages, load_skill_package
from app.skills.registry import get_skill


class TestSkillPackageLoader:
    def test_loads_weekly_report_package(self):
        package = load_skill_package("weekly-report")
        assert package.name == "generate_weekly_report"
        assert "周报" in package.description
        assert "prompt.md" in package.templates
        assert "writing-guide.md" in package.references
        assert (
            "SKILL" in package.instructions or "Weekly Report" in package.instructions
        )

    def test_loads_presentation_package(self):
        package = load_skill_package("presentation")
        assert package.name == "generate_presentation"
        assert package.parameters["required"] == ["topic"]
        assert "prompt.md" in package.templates
        assert "slide-guide.md" in package.references

    def test_lists_packages(self):
        names = {package.name for package in list_skill_packages()}
        assert {
            "generate_weekly_report",
            "generate_presentation",
            "codex_note",
        } <= names

    def test_every_bundled_package_has_explicit_runtime_audit(self):
        packages = list_skill_packages()
        assert packages
        assert all(
            package.runtime_status in {"ready", "blocked"} for package in packages
        )
        assert not [p.slug for p in packages if p.runtime_status == "unreviewed"]
        assert not [p.slug for p in packages if not p.required_capabilities]

    def test_bundled_runtime_statuses_match_the_supported_action_surface(self):
        packages = list_skill_packages()
        ready = {
            package.slug for package in packages if package.runtime_status == "ready"
        }
        blocked = {
            package.slug for package in packages if package.runtime_status == "blocked"
        }

        assert ready == {
            "codex-note",
            "jupyter-notebook",
            "openai-docs",
            "playwright",
            "presentation",
            "screenshot",
            "security-best-practices",
            "security-threat-model",
            "weekly-report",
        }
        assert blocked == {
            "gh-fix-ci",
            "pdf",
            "playwright-interactive",
            "security-ownership-map",
        }

    def test_ready_script_packages_have_scripts_and_workspace_authority(self):
        packages = list_skill_packages()
        scripted = [
            package
            for package in packages
            if package.runtime_status == "ready"
            and "package_scripts" in package.required_capabilities
        ]

        assert {package.slug for package in scripted} == {
            "jupyter-notebook",
            "screenshot",
        }
        assert all(package.script_names for package in scripted)
        assert all("workspace" in package.required_capabilities for package in scripted)

    def test_known_incompatible_copied_packages_remain_quarantined(self):
        assert load_skill_package("pdf").script_names == []
        assert "js_repl" in load_skill_package("playwright-interactive").instructions
        for slug in (
            "gh-fix-ci",
            "pdf",
            "playwright-interactive",
            "security-ownership-map",
        ):
            skill = get_skill(load_skill_package(slug).name)
            assert skill is not None
            skill.refresh_availability()
            assert skill.available is False

    def test_loads_codex_style_package_without_skill_json(self):
        package = load_skill_package("codex-note")
        assert package.name == "codex_note"
        assert package.source == "codex"
        assert package.parameters["required"] == ["task"]
        assert "style.md" in package.references


class TestPackageBackedSkill:
    def test_registered_skill_uses_package_metadata(self):
        skill = get_skill("generate_weekly_report")
        assert skill is not None
        package = skill.load_package()
        assert package is not None
        assert skill.description == package.description
        assert skill.to_tool()["function"]["parameters"] == package.parameters

    def test_codex_style_package_registered_as_generic_skill(self):
        skill = get_skill("codex_note")
        assert skill is not None
        assert skill.execution_mode == "generic_package"
        package = skill.load_package()
        assert package.source == "codex"
