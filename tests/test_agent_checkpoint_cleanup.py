"""Retention and cleanup contracts for LangGraph checkpoint data."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.agent import orchestrator
from app.agent.checkpoint_cleanup import (
    CheckpointCleanupResult,
    CheckpointCleanupPolicy,
    checkpoint_cleanup_loop,
    cleanup_agent_checkpoints,
    cleanup_configured_agent_checkpoints,
    configured_cleanup_policy,
    configured_run_lock,
)
from app.agent.checkpoint_store import record_agent_run
from app.agent.run_lock import RunLockConfig, create_agent_run_lease


def _finish_immediately(monkeypatch):
    class Message:
        content = "done"
        tool_calls = []

    monkeypatch.setattr(
        orchestrator,
        "chat_completion",
        lambda messages, tools=None, tool_choice="auto": Message(),
    )


def _lock_config(path) -> RunLockConfig:
    return RunLockConfig(
        backend="sqlite",
        checkpoint_path=path,
        redis_url="redis://unused",
        ttl_seconds=30,
        heartbeat_seconds=10,
        namespace="test:agent-run",
    )


def _age_run(path, run_id: str, updated_at: float) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE docmind_agent_runs SET updated_at = ? WHERE run_id = ?",
            (updated_at, run_id),
        )
        connection.commit()


def _policy() -> CheckpointCleanupPolicy:
    return CheckpointCleanupPolicy(
        completed_retention_seconds=100,
        incomplete_retention_seconds=1_000,
        interval_seconds=0,
        batch_size=20,
        maintenance_lease_seconds=30,
    )


def test_cleanup_deletes_expired_completed_checkpoint_and_receipts(
    monkeypatch, tmp_path
):
    _finish_immediately(monkeypatch)
    path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=1,
        question="done",
        thread_id="expired-completed",
        conversation_id=2,
        checkpoint_path=path,
    )
    _age_run(path, "expired-completed", updated_at=1.0)

    result = cleanup_agent_checkpoints(
        path,
        lock_config=_lock_config(path),
        policy=_policy(),
        now=1_000.0,
        force=True,
    )

    assert result.ran is True
    assert result.deleted == 1
    with pytest.raises(orchestrator.AgentRunNotFoundError):
        orchestrator.inspect_agent_run(
            "expired-completed", checkpoint_path=path, expected_user_id=1
        )
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT 1 FROM docmind_agent_runs WHERE run_id = ?",
            ("expired-completed",),
        ).fetchone() is None


def test_cleanup_preserves_recent_completed_run(monkeypatch, tmp_path):
    _finish_immediately(monkeypatch)
    path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=3,
        question="done",
        thread_id="recent-completed",
        checkpoint_path=path,
    )
    _age_run(path, "recent-completed", updated_at=950.0)

    result = cleanup_agent_checkpoints(
        path,
        lock_config=_lock_config(path),
        policy=_policy(),
        now=1_000.0,
        force=True,
    )

    assert result.deleted == 0
    assert orchestrator.inspect_agent_run(
        "recent-completed", checkpoint_path=path, expected_user_id=3
    )["status"] == "completed"


def test_cleanup_applies_longer_retention_to_incomplete_run(tmp_path):
    path = tmp_path / "agent-checkpoints.sqlite3"
    record_agent_run(
        path,
        run_id="waiting-approval",
        user_id=3,
        conversation_id=5,
        status="waiting_approval",
        now=500.0,
    )

    result = cleanup_agent_checkpoints(
        path,
        lock_config=_lock_config(path),
        policy=_policy(),
        now=1_000.0,
        force=True,
    )

    assert result.candidates == 0
    assert result.deleted == 0
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status FROM docmind_agent_runs WHERE run_id = ?",
            ("waiting-approval",),
        ).fetchone()
    assert row == ("waiting_approval",)


def test_cleanup_skips_run_held_by_another_worker(monkeypatch, tmp_path):
    _finish_immediately(monkeypatch)
    path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=4,
        question="done",
        thread_id="locked-old-run",
        checkpoint_path=path,
    )
    _age_run(path, "locked-old-run", updated_at=1.0)
    lease = create_agent_run_lease("locked-old-run", _lock_config(path))
    lease.acquire()
    try:
        result = cleanup_agent_checkpoints(
            path,
            lock_config=_lock_config(path),
            policy=_policy(),
            now=1_000.0,
            force=True,
        )
    finally:
        lease.release()

    assert result.deleted == 0
    assert result.skipped_locked == 1
    assert orchestrator.inspect_agent_run(
        "locked-old-run", checkpoint_path=path, expected_user_id=4
    )["status"] == "completed"


def test_cleanup_interval_prevents_every_worker_from_repeating_work(
    monkeypatch, tmp_path
):
    _finish_immediately(monkeypatch)
    path = tmp_path / "agent-checkpoints.sqlite3"
    policy = CheckpointCleanupPolicy(
        completed_retention_seconds=100,
        incomplete_retention_seconds=1_000,
        interval_seconds=300,
        batch_size=20,
        maintenance_lease_seconds=30,
    )

    first = cleanup_agent_checkpoints(
        path,
        lock_config=_lock_config(path),
        policy=policy,
        now=1_000.0,
    )
    second = cleanup_agent_checkpoints(
        path,
        lock_config=_lock_config(path),
        policy=policy,
        now=1_001.0,
    )

    assert first.ran is True
    assert second.ran is False


def test_cleanup_backfills_v3_checkpoint_before_starting_retention(
    monkeypatch, tmp_path
):
    _finish_immediately(monkeypatch)
    path = tmp_path / "agent-checkpoints.sqlite3"
    orchestrator.run_agent(
        user_id=8,
        question="old v3 checkpoint",
        thread_id="pre-v31-run",
        conversation_id=9,
        checkpoint_path=path,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM docmind_agent_runs WHERE run_id = ?", ("pre-v31-run",)
        )
        connection.commit()

    result = cleanup_agent_checkpoints(
        path,
        lock_config=_lock_config(path),
        policy=_policy(),
        now=1_000.0,
        force=True,
    )

    assert result.backfilled == 1
    assert result.deleted == 0
    assert orchestrator.inspect_agent_run(
        "pre-v31-run", checkpoint_path=path, expected_user_id=8
    )["status"] == "completed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed_retention_seconds": 0},
        {"completed_retention_seconds": 100, "incomplete_retention_seconds": 99},
        {"interval_seconds": -1},
        {"batch_size": 0},
    ],
)
def test_cleanup_policy_rejects_unsafe_values(kwargs):
    values = {
        "completed_retention_seconds": 100,
        "incomplete_retention_seconds": 1_000,
        "interval_seconds": 300,
        "batch_size": 20,
        "maintenance_lease_seconds": 30,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        CheckpointCleanupPolicy(**values)


def test_configured_cleanup_values_follow_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.agent.checkpoint_cleanup.settings.agent_checkpoint_path",
        str(tmp_path / "checkpoints.sqlite3"),
    )
    lock_config = configured_run_lock()
    policy = configured_cleanup_policy()

    assert lock_config.backend == "off"
    assert lock_config.checkpoint_path == str(tmp_path / "checkpoints.sqlite3")
    assert policy.incomplete_retention_seconds > policy.completed_retention_seconds


def test_configured_cleanup_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        "app.agent.checkpoint_cleanup.settings.agent_checkpoint_cleanup_enabled", False
    )
    assert cleanup_configured_agent_checkpoints() == CheckpointCleanupResult(ran=False)


@pytest.mark.asyncio
async def test_cleanup_loop_reports_success_and_stops(monkeypatch):
    stop = asyncio.Event()

    def _once():
        stop.set()
        return CheckpointCleanupResult(ran=True, candidates=1, deleted=1)

    monkeypatch.setattr(
        "app.agent.checkpoint_cleanup.cleanup_configured_agent_checkpoints", _once
    )
    await checkpoint_cleanup_loop(stop)


@pytest.mark.asyncio
async def test_cleanup_loop_survives_error(monkeypatch):
    stop = asyncio.Event()

    def _failure():
        stop.set()
        raise RuntimeError("maintenance failed")

    monkeypatch.setattr(
        "app.agent.checkpoint_cleanup.cleanup_configured_agent_checkpoints", _failure
    )
    await checkpoint_cleanup_loop(stop)
