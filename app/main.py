from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.skills  # noqa: F401  触发所有技能注册
from app.database import Base, engine

# 导入所有模型，确保建表时被 SQLAlchemy 识别
from app.models import conversation, document, evaluation, user  # noqa: F401
from app.routers import agent, auth, chat, documents, evaluation as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时建表（开发阶段用，生产环境改用 Alembic 迁移）
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="DocMind", version="0.2.0", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(agent.router)
app.include_router(eval_router.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
