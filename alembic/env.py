"""Alembic runtime configuration for DocMind."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import settings
from app.database import Base

# Import every model module so autogenerate sees the complete metadata graph.
from app.models import conversation, document, evaluation, user  # noqa: F401,E402


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _configured_url() -> str:
    """Prefer an explicit Alembic URL, otherwise use application settings."""

    configured = config.get_main_option("sqlalchemy.url").strip()
    return configured or settings.database_url


def _configure_context(*, connection: Connection | None = None, url: str | None = None) -> None:
    options: dict[str, object] = {
        "target_metadata": target_metadata,
        "compare_type": True,
        # Each revision owns its commit boundary.  In particular, the
        # transactional v2.2 column revision must be stamped before the next
        # revision enters PostgreSQL AUTOCOMMIT for CREATE INDEX CONCURRENTLY.
        "transaction_per_migration": True,
    }
    if connection is not None:
        options["connection"] = connection
        options["render_as_batch"] = connection.dialect.name == "sqlite"
    else:
        options.update(
            {
                "url": url,
                "literal_binds": True,
                "dialect_opts": {"paramstyle": "named"},
                "render_as_batch": bool(url and url.startswith("sqlite")),
            }
        )
    context.configure(**options)


def run_migrations_offline() -> None:
    """Render migration SQL without opening a database connection."""

    _configure_context(url=_configured_url())
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection: Connection) -> None:
    _configure_context(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations(url: str) -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = url
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(_run_sync_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Apply migrations with either the configured sync or async driver."""

    url = _configured_url()
    driver = make_url(url).get_driver_name()
    if driver in {"asyncpg", "aiosqlite"}:
        asyncio.run(_run_async_migrations(url))
        return

    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = url
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _run_sync_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
