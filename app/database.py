from datetime import UTC, datetime

from sqlalchemy import create_engine, event
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


def _install_sqlite_compatibility_functions(
    dbapi_connection: object, _connection_record: object
) -> None:
    """Support v2.2 SQLite files whose timestamp defaults call ``now()``.

    PostgreSQL provides ``now()`` natively, while SQLite does not.  The v2.2
    Alembic baseline used that expression verbatim, so existing desktop files
    need a connection-local compatibility function until their schema is
    rebuilt.  Registering it on both engines keeps request and background-task
    writes consistent without a destructive migration.
    """
    create_function = getattr(dbapi_connection, "create_function")
    create_function(
        "now",
        0,
        lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
    )


if _is_sqlite:
    event.listen(
        engine.sync_engine, "connect", _install_sqlite_compatibility_functions
    )
    event.listen(sync_engine, "connect", _install_sqlite_compatibility_functions)


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
