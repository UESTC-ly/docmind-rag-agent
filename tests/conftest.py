"""集成测试共享 fixture。

策略：
- DB：用内存 sqlite（aiosqlite）替换真实 PG，每个测试独立建表/销毁，互不污染。
- 外部边界：Celery / embedding / vector_store / LLM 全部 mock，
  测试只验证「我们的业务逻辑与 HTTP 契约」，不依赖任何在线服务。
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.utils.security import create_access_token

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
SYNC_TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture
def sync_db():
    """同步内存库 session，供 runner / dataset_gen 等同步服务测试用。"""
    engine = create_engine(SYNC_TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    """每个测试一个内存库 + 独立 session。建表 → yield → 拆表。"""
    engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    """注入测试 DB 的 HTTP 客户端。覆盖 get_db 依赖，让路由用内存库。"""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def registered_user(client):
    """注册一个用户并返回 (email, 认证头)。用于需要登录态的测试。"""
    email, password = "tester@example.com", "test123"
    await client.post("/auth/register", json={"email": email, "password": password})
    token = create_access_token(email)
    return {"email": email, "headers": {"Authorization": f"Bearer {token}"}}
