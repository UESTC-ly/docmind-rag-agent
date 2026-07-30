import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.skills  # noqa: F401  触发所有技能注册
from app.agent.checkpoint_cleanup import checkpoint_cleanup_loop
from app.config import settings
from app.database import engine, sync_engine
from app.routers import agent, auth, chat, documents, evaluation as eval_router
from app.services import vector_store
from app.services.task_dispatcher import (
    recover_local_evaluations,
    shutdown_local_executor,
)
from app.utils.logging import logger, setup_logging
from app.utils.middleware import RequestLoggingMiddleware


def _upgrade_database() -> None:
    """Apply immutable schema migrations before accepting any request."""
    default_config = Path(__file__).resolve().parent.parent / "alembic.ini"
    config_path = Path(settings.alembic_config or default_config).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Alembic config not found: {config_path}")
    command.upgrade(Config(str(config_path)), "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("DocMind starting up")
    # Alembic's async environment owns its own event loop; isolate it from the
    # ASGI loop and fail closed before serving if a migration cannot complete.
    await asyncio.to_thread(_upgrade_database)
    await asyncio.to_thread(recover_local_evaluations)
    cleanup_stop = asyncio.Event()
    cleanup_task = (
        asyncio.create_task(
            checkpoint_cleanup_loop(cleanup_stop),
            name="agent-checkpoint-cleanup",
        )
        if settings.agent_checkpoint_cleanup_enabled
        else None
    )
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_stop.set()
            await cleanup_task
        # Embedded Qdrant holds a filesystem lock, and both SQLAlchemy engines
        # own connection pools. Close them explicitly so the desktop sidecar
        # exits cleanly instead of relying on interpreter finalizers.
        await asyncio.to_thread(shutdown_local_executor)
        await asyncio.to_thread(vector_store.close)
        await engine.dispose()
        await asyncio.to_thread(sync_engine.dispose)
        logger.info("DocMind shutting down")


app = FastAPI(title="DocMind", version="3.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:1420",
        "http://localhost:1420",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(agent.router)
app.include_router(eval_router.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# 前端静态托管（放在所有 API 路由之后；html=True 让 / 返回 index.html）。
# 已注册的 API 前缀（/auth /documents /chat /agent /eval /health）优先匹配，
# 其余路径交给前端单页。
_FRONTEND_DIR = (
    Path(
        settings.docmind_frontend_dir
        or (Path(__file__).resolve().parent.parent / "frontend")
    )
    .expanduser()
    .resolve()
)
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend"
    )
