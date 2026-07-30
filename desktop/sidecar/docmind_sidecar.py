"""Frozen DocMind desktop backend entry point.

The installed desktop application launches this executable directly.  It owns
all local-runtime choices before importing :mod:`app`, which is important
because DocMind's settings and database engines are module-level singletons.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Protocol, Sequence

SIDECAR_VERSION = "3.2.0"


class _Closable(Protocol):
    def close(self) -> None: ...


def _resource_root() -> Path:
    """Return the immutable directory embedded by PyInstaller."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parents[2]


def _sqlite_url(database_path: Path) -> str:
    normalized = database_path.resolve().as_posix()
    return f"sqlite+aiosqlite:///{normalized}"


def _read_or_create_secret(data_dir: Path) -> str:
    secret_dir = data_dir / "secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret_path = secret_dir / "jwt-signing-key"
    try:
        return secret_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass

    secret = secrets.token_urlsafe(48)
    try:
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return secret_path.read_text(encoding="utf-8").strip()
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(secret)
        output.write("\n")
    if os.name != "nt":
        secret_path.chmod(0o600)
    return secret


def configure_runtime(data_dir: Path, config_path: Path) -> dict[str, str]:
    """Configure the backend for a self-contained, per-user local runtime."""
    data_dir = data_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    for directory in (
        data_dir,
        data_dir / "uploads",
        data_dir / "skill_workspaces",
        data_dir / "qdrant",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    resource_root = _resource_root()
    values = {
        "DOCMIND_DESKTOP": "1",
        "DOCMIND_ENV_FILE": str(config_path),
        "DOCMIND_DATA_DIR": str(data_dir),
        "DOCMIND_RESOURCE_DIR": str(resource_root),
        "DOCMIND_FRONTEND_DIR": str(resource_root / "frontend"),
        "DATABASE_URL": _sqlite_url(data_dir / "docmind.db"),
        "QDRANT_PATH": str(data_dir / "qdrant"),
        "TASK_EXECUTION_MODE": "local",
        "UPLOAD_DIR": str(data_dir / "uploads"),
        "SKILL_WORKSPACE_DIR": str(data_dir / "skill_workspaces"),
        "AGENT_CHECKPOINT_PATH": str(data_dir / "agent-checkpoints.sqlite3"),
        "AGENT_RUN_LOCK_BACKEND": "sqlite",
        "SECRET_KEY": _read_or_create_secret(data_dir),
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    alembic_config = resource_root / "alembic.ini"
    if alembic_config.is_file():
        values["ALEMBIC_CONFIG"] = str(alembic_config)

    # Runtime-owned values intentionally override both the parent environment and
    # the user's dotenv file.  Provider credentials remain user configurable.
    os.environ.update(values)
    return values


def runtime_manifest(values: dict[str, str]) -> dict[str, object]:
    """Return a secret-free runtime description for build and support checks."""
    return {
        "status": "ok",
        "version": SIDECAR_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "database": "sqlite",
        "vector_store": "qdrant-local",
        "task_executor": "local",
        "agent_checkpointer": "sqlite",
        "agent_run_lock": "sqlite-lease",
        "package_script_confinement": "macos-sandbox-exec",
        "external_runtime_required": False,
        "data_dir": values["DOCMIND_DATA_DIR"],
        "resource_dir": values["DOCMIND_RESOURCE_DIR"],
    }


def _parent_alive(parent_pid: int) -> bool:
    if parent_pid <= 0:
        return True
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _watch_parent(parent_pid: int, server: object) -> None:
    while _parent_alive(parent_pid):
        time.sleep(2)
    setattr(server, "should_exit", True)


def _verify_backend_imports() -> _Closable:
    # Import only after configure_runtime() so module-level settings cannot bind
    # to a developer's PostgreSQL/Qdrant/Redis environment.
    from app.config import settings  # noqa: PLC0415
    from app.database import engine  # noqa: F401, PLC0415
    from app.main import app  # noqa: F401, PLC0415
    from app.services import vector_store  # noqa: PLC0415
    from app.skills.capabilities import capability_states  # noqa: PLC0415

    if not settings.database_url.startswith("sqlite+aiosqlite:///"):
        raise RuntimeError("desktop backend did not select SQLite")
    if getattr(settings, "task_execution_mode", None) != "local":
        raise RuntimeError("desktop backend did not select the local task executor")
    if not getattr(settings, "qdrant_path", None):
        raise RuntimeError("desktop backend did not select Qdrant local storage")
    if Path(settings.agent_checkpoint_path).parent != Path(
        os.environ["DOCMIND_DATA_DIR"]
    ):
        raise RuntimeError("desktop backend did not isolate Agent checkpoints")
    if (
        settings.skill_package_scripts_enabled
        and not capability_states()["package_scripts"].available
    ):
        raise RuntimeError("desktop package-script confinement self-check failed")
    return vector_store._client


def _resolve_package_script(raw_script: Path) -> Path:
    """Resolve one packaged Python script without exposing a general interpreter."""
    scripts_root = (_resource_root() / "app" / "skills" / "packages").resolve()
    raw_parts = raw_script.parts
    package_marker = ("app", "skills", "packages")
    mapped_relative: Path | None = None
    for index in range(len(raw_parts) - len(package_marker) + 1):
        if tuple(raw_parts[index : index + len(package_marker)]) == package_marker:
            mapped_relative = Path(*raw_parts[index + len(package_marker) :])
            break

    # A frozen child may unpack to a different _MEI directory than its parent.
    # In that case map only the package-relative suffix into this process's own
    # signed resources; never execute the caller-provided absolute file.
    candidate = scripts_root / mapped_relative if mapped_relative else raw_script
    try:
        script = candidate.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise PermissionError("package script does not exist") from error
    try:
        relative = script.relative_to(scripts_root)
    except ValueError as error:
        raise PermissionError("script is outside packaged skills") from error
    if len(relative.parts) < 3 or relative.parts[1] != "scripts":
        raise PermissionError("script is not in a packaged skill scripts directory")
    if script.suffix.lower() != ".py" or not script.is_file():
        raise PermissionError("the frozen script runner only accepts packaged .py files")
    return script


def _run_package_script(raw_script: Path, arguments: Sequence[str]) -> int:
    script = _resolve_package_script(raw_script)
    previous_argv = sys.argv
    sys.argv = [str(script), *arguments]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
    return 0


def _serve(args: argparse.Namespace) -> int:
    values = configure_runtime(args.data_dir, args.config)
    _verify_backend_imports()

    import uvicorn  # noqa: PLC0415
    from app.main import app  # noqa: PLC0415

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    server = uvicorn.Server(config)
    if args.parent_pid:
        threading.Thread(
            target=_watch_parent,
            args=(args.parent_pid, server),
            name="docmind-parent-watchdog",
            daemon=True,
        ).start()

    print(json.dumps(runtime_manifest(values), ensure_ascii=False), flush=True)
    server.run()
    return 0


def _check(args: argparse.Namespace) -> int:
    values = configure_runtime(args.data_dir, args.config)
    vector_client = _verify_backend_imports()
    try:
        print(json.dumps(runtime_manifest(values), ensure_ascii=False), flush=True)
    finally:
        # `check` imports the application but does not enter FastAPI's lifespan,
        # so it owns cleanup of the local Qdrant file lock itself.
        vector_client.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocMind frozen desktop backend")
    parser.add_argument("--version", action="version", version=SIDECAR_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("serve", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        if command == "serve":
            subparser.add_argument("--host", default="127.0.0.1")
            subparser.add_argument("--port", type=int, default=8000)
            subparser.add_argument("--parent-pid", type=int, default=0)
    script_parser = subparsers.add_parser("run-script")
    script_parser.add_argument("script", type=Path)
    script_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Required by PyInstaller for Windows process bootstrap, harmless elsewhere.
    import multiprocessing

    multiprocessing.freeze_support()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    # Some packaged scripts launch a sibling with ``sys.executable child.py``.
    # Preserve that idiom, but route through the exact same package-root check.
    if arguments and arguments[0].lower().endswith(".py"):
        return _run_package_script(Path(arguments[0]), arguments[1:])

    args = _parser().parse_args(arguments)
    if args.command == "run-script":
        script_arguments = list(args.arguments)
        if script_arguments[:1] == ["--"]:
            script_arguments.pop(0)
        return _run_package_script(args.script, script_arguments)
    if args.command == "check":
        return _check(args)
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
