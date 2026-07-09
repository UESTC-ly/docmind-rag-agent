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
        assert "SKILL" in package.instructions or "Weekly Report" in package.instructions

    def test_loads_presentation_package(self):
        package = load_skill_package("presentation")
        assert package.name == "generate_presentation"
        assert package.parameters["required"] == ["topic"]
        assert "prompt.md" in package.templates
        assert "slide-guide.md" in package.references

    def test_lists_packages(self):
        names = {package.name for package in list_skill_packages()}
        assert {"generate_weekly_report", "generate_presentation"} <= names


class TestPackageBackedSkill:
    def test_registered_skill_uses_package_metadata(self):
        skill = get_skill("generate_weekly_report")
        assert skill is not None
        package = skill.load_package()
        assert package is not None
        assert skill.description == package.description
        assert skill.to_tool()["function"]["parameters"] == package.parameters
