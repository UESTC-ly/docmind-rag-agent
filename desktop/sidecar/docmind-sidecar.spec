# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-file build for DocMind's desktop-local backend."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

sidecar_dir = Path(SPECPATH).resolve()
project_root = sidecar_dir.parents[1]

datas = []
binaries = []
hiddenimports = [
    "aiosqlite",
    "email_validator",
    "passlib.handlers.bcrypt",
    "sqlalchemy.dialects.sqlite.aiosqlite",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

def is_qdrant_runtime_module(name):
    parts = name.split(".")
    return "tests" not in parts and not any(part.startswith("test_") for part in parts)


datas += collect_data_files(
    "qdrant_client",
    excludes=["**/tests/**", "**/test_*.py", "**/__pycache__/**"],
)
binaries += collect_dynamic_libs("qdrant_client")
hiddenimports += collect_submodules("qdrant_client", filter=is_qdrant_runtime_module)

for distribution in ("fastapi", "pydantic", "qdrant-client", "sqlalchemy", "uvicorn"):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        # Package metadata is diagnostic-only for these dependencies.
        pass

for source, destination in (
    (project_root / "frontend", "frontend"),
    (project_root / "app" / "skills" / "packages", "app/skills/packages"),
    (project_root / "alembic", "alembic"),
):
    if source.exists():
        datas.append((str(source), destination))

alembic_config = project_root / "alembic.ini"
if alembic_config.is_file():
    datas.append((str(alembic_config), "."))

analysis = Analysis(
    [str(sidecar_dir / "docmind_sidecar.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["asyncpg", "psycopg2", "pytest", "_pytest"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="docmind-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
