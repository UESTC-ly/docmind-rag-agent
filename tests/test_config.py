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


def test_agent_run_lock_backend_resolves_by_runtime_profile():
    assert _settings(agent_run_lock_backend="auto").resolved_agent_run_lock_backend == "redis"
    desktop = _settings(
        agent_run_lock_backend="auto",
        docmind_desktop=True,
        task_execution_mode="local",
    )
    assert desktop.resolved_agent_run_lock_backend == "sqlite"


def test_agent_run_lock_configuration_fails_closed():
    with pytest.raises(ValidationError):
        _settings(agent_run_lock_backend="filesystem-ish")
    with pytest.raises(ValidationError):
        _settings(
            agent_run_lock_ttl_seconds=30,
            agent_run_lock_heartbeat_seconds=30,
        )


def test_checkpoint_retention_requires_incomplete_runs_to_live_longer():
    with pytest.raises(ValidationError):
        _settings(
            agent_checkpoint_completed_retention_days=30,
            agent_checkpoint_incomplete_retention_days=7,
        )
