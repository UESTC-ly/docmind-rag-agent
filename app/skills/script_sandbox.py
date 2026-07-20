"""macOS confinement helpers for bundled Generic Skill scripts and shell tools."""

from __future__ import annotations

import platform
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_PROFILE = """(version 1)
(allow default)
(deny file-write*)
(allow file-write* (subpath (param \"WORKSPACE\")))
"""


@lru_cache(maxsize=1)
def sandbox_backend_available() -> bool:
    """Return whether the validated v2.2 macOS confinement backend exists."""

    if platform.system() != "Darwin" or not SANDBOX_EXEC.is_file():
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="docmind-sandbox-probe-") as root:
            # macOS resolves /var and /tmp through /private before matching a
            # sandbox profile.  Use the canonical path in both -D and argv;
            # otherwise a valid confinement backend is reported unavailable.
            probe_root = Path(root).resolve()
            workspace = probe_root / "workspace"
            workspace.mkdir()
            allowed = workspace / "allowed"
            denied = probe_root / "denied"
            subprocess.run(  # noqa: S603 - fixed system binary/profile
                [
                    str(SANDBOX_EXEC),
                    "-p",
                    _PROFILE,
                    "-D",
                    f"WORKSPACE={workspace}",
                    "/bin/sh",
                    "-c",
                    'printf allowed > "$1"; printf denied > "$2"',
                    "docmind-probe",
                    str(allowed),
                    str(denied),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            return allowed.read_text(encoding="utf-8") == "allowed" and not denied.exists()
    except (OSError, subprocess.SubprocessError):
        return False


def _argument_value(argument: str) -> str:
    if argument.startswith("-") and "=" in argument:
        return argument.split("=", 1)[1]
    return argument


def validate_untrusted_arguments(arguments: Iterable[str]) -> list[str]:
    """Reject path forms that could intentionally address host files.

    The OS sandbox is the authoritative write boundary.  This validation also
    blocks obvious read attempts such as ``/etc/passwd`` and ``../secret``.
    Trusted interpreter/script paths are added only after this check.
    """

    validated: list[str] = []
    for raw in arguments:
        if not isinstance(raw, str):
            raise ValueError("script/shell arguments 必须是字符串")
        if "\x00" in raw:
            raise ValueError("argument 包含 NUL 字符")
        value = _argument_value(raw).strip()
        lowered = value.lower()
        if value.startswith("~") or lowered.startswith("file://"):
            raise PermissionError(f"argument 不允许宿主路径: {raw}")
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise PermissionError(f"argument 不允许绝对路径: {raw}")
        path_parts = PurePosixPath(value.replace("\\", "/")).parts
        if ".." in path_parts:
            raise PermissionError(f"argument 不允许路径越界: {raw}")
        validated.append(raw)
    return validated


def confined_environment(workspace: Path, source: dict[str, str]) -> dict[str, str]:
    runtime = workspace / ".runtime"
    home = runtime / "home"
    temp = runtime / "tmp"
    cache = runtime / "cache"
    for path in (home, temp, cache):
        path.mkdir(parents=True, exist_ok=True)
    environment = dict(source)
    environment.update(
        {
            "HOME": str(home),
            "TMPDIR": str(temp),
            "TEMP": str(temp),
            "TMP": str(temp),
            "XDG_CACHE_HOME": str(cache),
            "DOCMIND_SKILL_WORKSPACE": str(workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def confine_command(command: list[str], workspace: Path) -> list[str]:
    if not sandbox_backend_available():
        raise RuntimeError("macOS sandbox-exec confinement backend 不可用")
    return [
        str(SANDBOX_EXEC),
        "-p",
        _PROFILE,
        "-D",
        f"WORKSPACE={workspace}",
        *command,
    ]


def workspace_snapshot(workspace: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not workspace.exists():
        return snapshot
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if relative.parts and relative.parts[0] == ".runtime":
            continue
        if path.is_symlink():
            raise PermissionError(f"Skill 工作区不允许符号链接: {relative}")
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[relative.as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def changed_workspace_files(
    workspace: Path,
    before: dict[str, tuple[int, int]],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[Path]:
    after = workspace_snapshot(workspace)
    changed: list[Path] = []
    total = 0
    for relative, state in after.items():
        if before.get(relative) == state:
            continue
        path = workspace / relative
        resolved = path.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise PermissionError(f"脚本产物越出工作区: {relative}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f"脚本产物超过单文件上限: {relative}")
        total += size
        if total > max_total_bytes:
            raise ValueError("脚本产物总大小超过上限")
        changed.append(path)
    return changed


__all__ = [
    "changed_workspace_files",
    "confine_command",
    "confined_environment",
    "sandbox_backend_available",
    "validate_untrusted_arguments",
    "workspace_snapshot",
]
