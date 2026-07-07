from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.skills  # noqa: F401  触发所有技能注册
from app.database import Base, engine

# 导入所有模型，确保建表时被 SQLAlchemy 识别
from app.models import conversation, document, evaluation, user  # noqa: F401
from app.routers import agent, auth, chat, documents, evaluation as eval_router
from app.utils.logging import logger, setup_logging
from app.utils.middleware import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("DocMind starting up")
    # 启动时建表（开发阶段用，生产环境改用 Alembic 迁移）
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    logger.info("DocMind shutting down")


app = FastAPI(title="DocMind", version="0.3.0", lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(agent.router)
app.include_router(eval_router.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
