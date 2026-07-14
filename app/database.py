from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

_is_sqlite = settings.database_url.startswith("sqlite+")
_async_engine_options: dict = {"echo": False}
_sync_engine_options: dict = {"echo": False}
if _is_sqlite:
    # Local background threads share the same desktop SQLite database file.
    _async_engine_options["connect_args"] = {"check_same_thread": False}
    _sync_engine_options["connect_args"] = {"check_same_thread": False}
else:
    _async_engine_options["pool_pre_ping"] = True
    _sync_engine_options["pool_pre_ping"] = True

# ── 异步引擎（FastAPI 请求处理用）────────────────────────────
engine = create_async_engine(
    settings.database_url,
    **_async_engine_options,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── 同步引擎（Celery worker 用）─────────────────────────────
# Celery 任务是同步函数，用同步驱动(psycopg2)最简单，避免在任务里跑事件循环
sync_engine = create_engine(
    settings.sync_database_url,
    **_sync_engine_options,
)

SyncSessionLocal = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)


# 所有 ORM 模型的基类
class Base(DeclarativeBase):
    pass


# FastAPI 依赖注入：获取异步数据库 Session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
