"""Skill Package 加载器。

v0.4 的两层架构要求每个 package 都有 `skill.json`，Python `BaseSkill`
负责执行。v0.5 在此基础上兼容 Codex-style 通用技能包：

1. **Python-backed package**：`skill.json + SKILL.md + templates/references`，
   仍由显式 Python 类执行；
2. **Codex-style package**：只有 `SKILL.md` 也能被加载，metadata 从
   YAML frontmatter 读取，并由 GenericPackageSkill 解释执行。

这样可以直接复制符合主流 Agent Skills 结构的文件夹到
`app/skills/packages/`，先作为受控通用技能运行；需要更强确定性时，再补一层
Python 执行类。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    source: str
    metadata: dict[str, Any]
    script_names: list[str]
    asset_names: list[str]
    runtime_status: str
    runtime_reason: str
    required_capabilities: tuple[str, ...]

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


def _list_files(root: Path) -> list[str]:
    """列出资源文件相对路径，不读取内容（assets/scripts 可能是二进制或很大）。"""
    if not root.exists():
        return []
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _parse_frontmatter(markdown: str) -> dict[str, str]:
    """解析 SKILL.md 顶部极简 YAML frontmatter。

    为避免新增 PyYAML 依赖，这里只支持 Codex Skills 必需的简单
    `key: value` 字段；复杂结构仍应放到 `skill.json`。
    """
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = idx
            break
    if end is None:
        return {}

    data: dict[str, str] = {}
    for raw in lines[1:end]:
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            data[key] = value
    return data


def _default_generic_parameters() -> dict:
    """Codex-style 通用技能默认 Function Calling schema。

    通用技能通常不是固定函数签名，而是“按 SKILL.md 完成一个任务”。因此默认只
    要求 task；inputs 用于传入结构化补充信息，document_id 复用 DocMind 当前文档。
    """
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "要按该 SKILL.md 完成的具体任务。",
            },
            "inputs": {
                "type": "object",
                "description": "可选：任务相关结构化输入。",
                "additionalProperties": True,
            },
            "document_id": {
                "type": "integer",
                "description": "可选：当前任务关联的 DocMind 文档 ID。",
            },
        },
        "required": ["task"],
    }


def _metadata_from_skill_json(path: Path) -> tuple[dict[str, Any], str]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "parameters" not in metadata:
        metadata["parameters"] = _default_generic_parameters()
    elif not isinstance(metadata["parameters"], dict):
        raise ValueError(f"Skill package {path.parent.name} has invalid parameters")
    return metadata, "skill_json"


def _metadata_from_frontmatter(slug: str, instructions: str) -> tuple[dict[str, Any], str]:
    frontmatter = _parse_frontmatter(instructions)
    metadata = {
        "name": frontmatter.get("name"),
        "description": frontmatter.get("description"),
        "parameters": _default_generic_parameters(),
    }
    return metadata, "codex"


def _validate_metadata(slug: str, metadata: dict[str, Any]) -> None:
    name = metadata.get("name")
    description = metadata.get("description")
    parameters = metadata.get("parameters")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Skill package {slug} missing name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Skill package {slug} missing description")
    if not isinstance(parameters, dict):
        raise ValueError(f"Skill package {slug} has invalid parameters")


@lru_cache(maxsize=None)
def load_skill_package(slug: str) -> SkillPackage:
    package_dir = PACKAGE_ROOT / slug
    metadata_path = package_dir / "skill.json"
    instructions_path = package_dir / "SKILL.md"
    runtime_path = package_dir / "docmind.json"

    if not package_dir.is_dir():
        raise FileNotFoundError(f"Skill package not found: {slug}")
    if not instructions_path.is_file():
        raise FileNotFoundError(f"Skill package missing SKILL.md: {slug}")

    instructions = instructions_path.read_text(encoding="utf-8")
    if metadata_path.is_file():
        metadata, source = _metadata_from_skill_json(metadata_path)
    else:
        metadata, source = _metadata_from_frontmatter(slug, instructions)
    _validate_metadata(slug, metadata)
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {}
    )
    runtime_status = str(runtime.get("status") or "unreviewed").strip().lower()
    if runtime_status not in {"ready", "blocked", "unreviewed"}:
        raise ValueError(f"Skill package {slug} has invalid runtime status")
    runtime_reason = str(
        runtime.get("reason")
        or (
            "尚未经过 DocMind 工具/权限兼容性审计，暂不允许 Agent 自动调用。"
            if runtime_status == "unreviewed"
            else ""
        )
    ).strip()
    raw_capabilities = runtime.get("requires") or []
    if not isinstance(raw_capabilities, list) or not all(
        isinstance(item, str) for item in raw_capabilities
    ):
        raise ValueError(f"Skill package {slug} has invalid capability requirements")
    required_capabilities = tuple(
        dict.fromkeys(item.strip() for item in raw_capabilities if item.strip())
    )

    return SkillPackage(
        slug=slug,
        name=metadata["name"].strip(),
        description=metadata["description"].strip(),
        parameters=metadata["parameters"],
        instructions=instructions,
        templates=_read_text_files(package_dir / "templates"),
        references=_read_text_files(package_dir / "references"),
        path=package_dir,
        source=source,
        metadata=metadata,
        script_names=_list_files(package_dir / "scripts"),
        asset_names=_list_files(package_dir / "assets"),
        runtime_status=runtime_status,
        runtime_reason=runtime_reason,
        required_capabilities=required_capabilities,
    )


def list_skill_packages() -> list[SkillPackage]:
    if not PACKAGE_ROOT.exists():
        return []
    packages: list[SkillPackage] = []
    for path in sorted(PACKAGE_ROOT.iterdir()):
        if not path.is_dir():
            continue
        try:
            packages.append(load_skill_package(path.name))
        except (FileNotFoundError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            # 直接 load 某个坏包时仍会抛错；自动扫描时跳过，避免一个复制失败的
            # GitHub skill 让整个 FastAPI 应用无法启动。
            continue
    return packages
