import json
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_desktop_release_keeps_tauri_backend_and_frontend_contracts():
    config = json.loads(
        (PROJECT_ROOT / "desktop" / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["version"] == "2.0.1"
    assert config["build"]["frontendDist"] == "../dist"
    assert config["bundle"]["resources"]["../../app"] == "backend/app"
    assert "http://127.0.0.1:8000" in config["app"]["security"]["csp"]


def test_api_allows_only_declared_desktop_webview_origins():
    middleware = next(item for item in app.user_middleware if item.cls is CORSMiddleware)
    origins = middleware.kwargs["allow_origins"]

    assert app.version == "2.0.1"
    assert "tauri://localhost" in origins
    assert "https://tauri.localhost" in origins
    assert "*" not in origins
