"""Configuration contracts shared by Web, Celery, and frozen desktop mode."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings(**overrides):
    values = {
        "database_url": "postgresql+asyncpg://user:pass@localhost/docmind",
        "secret_key": "secret",
        "openai_api_key": "sk-test",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_sync_database_url_supports_postgresql_and_desktop_sqlite():
    assert _settings().sync_database_url.startswith("postgresql://")
    desktop = _settings(database_url="sqlite+aiosqlite:////tmp/docmind.db")
    assert desktop.sync_database_url == "sqlite:////tmp/docmind.db"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrieval_mode", "unknown"),
        ("reranker_mode", "magic"),
        ("task_execution_mode", "redis-or-local-ish"),
    ],
)
def test_execution_modes_fail_closed(field, value):
    with pytest.raises(ValidationError):
        _settings(**{field: value})
