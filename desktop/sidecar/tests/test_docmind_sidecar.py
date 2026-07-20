from __future__ import annotations

import importlib.util
import os
from argparse import Namespace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "docmind_sidecar.py"
SPEC = importlib.util.spec_from_file_location("docmind_sidecar", MODULE_PATH)
assert SPEC and SPEC.loader
sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sidecar)


def test_configure_runtime_owns_all_local_service_paths(tmp_path, monkeypatch):
    config = tmp_path / "desktop.env"
    config.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://do-not-use")
    monkeypatch.setenv("QDRANT_PATH", "/do-not-use")
    monkeypatch.setenv("TASK_EXECUTION_MODE", "celery")

    values = sidecar.configure_runtime(tmp_path / "data", config)

    assert values["DATABASE_URL"].startswith("sqlite+aiosqlite:////")
    assert values["QDRANT_PATH"] == str((tmp_path / "data/qdrant").resolve())
    assert values["AGENT_CHECKPOINT_PATH"] == str(
        (tmp_path / "data/agent-checkpoints.sqlite3").resolve()
    )
    assert values["AGENT_RUN_LOCK_BACKEND"] == "sqlite"
    assert values["TASK_EXECUTION_MODE"] == "local"
    assert os.environ["DATABASE_URL"] == values["DATABASE_URL"]
    assert (tmp_path / "data/uploads").is_dir()
    assert (tmp_path / "data/skill_workspaces").is_dir()


def test_secret_is_stable_and_not_exposed_by_manifest(tmp_path):
    config = tmp_path / "desktop.env"
    config.touch()

    first = sidecar.configure_runtime(tmp_path / "data", config)
    second = sidecar.configure_runtime(tmp_path / "data", config)
    manifest = sidecar.runtime_manifest(second)

    assert first["SECRET_KEY"] == second["SECRET_KEY"]
    assert len(first["SECRET_KEY"]) >= 48
    assert "secret" not in manifest
    assert manifest["external_runtime_required"] is False
    assert manifest["database"] == "sqlite"
    assert manifest["vector_store"] == "qdrant-local"
    assert manifest["task_executor"] == "local"
    assert manifest["agent_checkpointer"] == "sqlite"
    assert manifest["agent_run_lock"] == "sqlite-lease"
    assert manifest["version"] == "3.1.0"
    assert manifest["package_script_confinement"] == "macos-sandbox-exec"


def test_parent_probe_treats_current_process_as_alive():
    assert sidecar._parent_alive(os.getpid()) is True


def test_check_closes_vector_store_without_entering_app_lifespan(
    tmp_path, monkeypatch, capsys
):
    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()
    config = tmp_path / "desktop.env"
    config.touch()
    monkeypatch.setattr(sidecar, "_verify_backend_imports", lambda: client)

    assert sidecar._check(Namespace(data_dir=tmp_path / "data", config=config)) == 0

    assert client.closed is True
    assert '"status": "ok"' in capsys.readouterr().out


def test_tauri_release_overlay_bundles_sidecar_and_host_has_no_external_launchers():
    desktop_root = Path(__file__).resolve().parents[2]
    release_overlay = (
        desktop_root / "src-tauri/tauri.sidecar.conf.json"
    ).read_text(encoding="utf-8")
    host = (desktop_root / "src-tauri/src/lib.rs").read_text(encoding="utf-8")

    assert '"binaries/docmind-sidecar"' in release_overlay
    for forbidden in (
        "ensure_docker",
        "start_compose",
        "docker compose",
        'desktop_command("uv")',
        '"-m", "celery"',
    ):
        assert forbidden not in host.lower()


def test_desktop_config_enables_only_the_embedded_python_script_runner():
    config = (Path(__file__).resolve().parents[1] / "default.env").read_text(
        encoding="utf-8"
    )

    assert "SKILL_PACKAGE_SCRIPTS_ENABLED=true" in config
    assert "SKILL_PACKAGE_SCRIPT_INTERPRETERS=python" in config
    assert "SKILL_SHELL_ENABLED=true" not in config


def test_frozen_script_runner_only_executes_packaged_python(tmp_path, monkeypatch, capsys):
    resource_root = tmp_path / "resources"
    script = resource_root / "app/skills/packages/demo/scripts/echo.py"
    script.parent.mkdir(parents=True)
    script.write_text("import sys\nprint('|'.join(sys.argv[1:]))\n", encoding="utf-8")
    monkeypatch.setattr(sidecar, "_resource_root", lambda: resource_root)

    assert sidecar._run_package_script(script, ["one", "two"]) == 0
    assert capsys.readouterr().out.strip() == "one|two"


@pytest.mark.parametrize("relative", ["outside.py", "app/skills/packages/demo/not-scripts/a.py"])
def test_frozen_script_runner_rejects_paths_outside_skill_scripts(
    tmp_path, monkeypatch, relative
):
    resource_root = tmp_path / "resources"
    candidate = resource_root / relative
    candidate.parent.mkdir(parents=True)
    candidate.write_text("raise AssertionError('must not run')\n", encoding="utf-8")
    monkeypatch.setattr(sidecar, "_resource_root", lambda: resource_root)

    with pytest.raises(PermissionError):
        sidecar._run_package_script(candidate, [])


def test_frozen_script_runner_rejects_non_python_files(tmp_path, monkeypatch):
    resource_root = tmp_path / "resources"
    script = resource_root / "app/skills/packages/demo/scripts/unsafe.sh"
    script.parent.mkdir(parents=True)
    script.write_text("exit 0\n", encoding="utf-8")
    monkeypatch.setattr(sidecar, "_resource_root", lambda: resource_root)

    with pytest.raises(PermissionError):
        sidecar._run_package_script(script, [])


def test_frozen_script_runner_remaps_parent_extraction_to_current_resources(
    tmp_path, monkeypatch, capsys
):
    resource_root = tmp_path / "current-mei"
    packaged = resource_root / "app/skills/packages/demo/scripts/child.py"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("print('current-resource')\n", encoding="utf-8")
    stale_parent_path = Path(
        "/private/tmp/old-mei/app/skills/packages/demo/scripts/child.py"
    )
    monkeypatch.setattr(sidecar, "_resource_root", lambda: resource_root)

    sidecar._run_package_script(stale_parent_path, [])

    assert capsys.readouterr().out.strip() == "current-resource"
