"""Skill Package 加载器。

两层架构：
1. Python `BaseSkill` 子类仍然负责执行逻辑、I/O 边界和测试；
2. `app/skills/packages/<slug>/` 负责 Agent 可读说明、function metadata、
   templates 和 references。

这样既保留现有 Function Calling / registry 执行链路，又让复杂技能具备主流
Agent Skills 的目录化组织形式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent / "packages"


@dataclass(frozen=True)
class SkillPackage:
    slug: str
    name: str
    description: str
    parameters: dict
    instructions: str
    templates: dict[str, str]
    references: dict[str, str]
    path: Path

    @property
    def template_names(self) -> list[str]:
        return sorted(self.templates)

    @property
    def reference_names(self) -> list[str]:
        return sorted(self.references)


def _read_text_files(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            files[rel] = path.read_text(encoding="utf-8")
    return files


@lru_cache(maxsize=None)
def load_skill_package(slug: str) -> SkillPackage:
    package_dir = PACKAGE_ROOT / slug
    metadata_path = package_dir / "skill.json"
    instructions_path = package_dir / "SKILL.md"

    if not package_dir.is_dir():
        raise FileNotFoundError(f"Skill package not found: {slug}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Skill package missing skill.json: {slug}")
    if not instructions_path.is_file():
        raise FileNotFoundError(f"Skill package missing SKILL.md: {slug}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Skill package {slug} has invalid parameters")

    return SkillPackage(
        slug=slug,
        name=metadata["name"],
        description=metadata["description"],
        parameters=parameters,
        instructions=instructions_path.read_text(encoding="utf-8"),
        templates=_read_text_files(package_dir / "templates"),
        references=_read_text_files(package_dir / "references"),
        path=package_dir,
    )


def list_skill_packages() -> list[SkillPackage]:
    if not PACKAGE_ROOT.exists():
        return []
    return [
        load_skill_package(path.name)
        for path in sorted(PACKAGE_ROOT.iterdir())
        if path.is_dir()
    ]
