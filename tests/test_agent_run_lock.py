"""Agent run lease contracts for multi-worker and multi-process execution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from redis.exceptions import LockError, RedisError

from app.agent.run_lock import (
    AgentRunLease,
    AgentRunLeaseBackendError,
    AgentRunLeaseBusyError,
    AgentRunLeaseError,
    AgentRunLeaseLostError,
    RedisLeaseBackend,
    RunLockConfig,
    SQLiteLeaseBackend,
    create_agent_run_lease,
)


@dataclass
class _Clock:
    value: float = 1_000.0

    def __call__(self) -> float:
        return self.value


def _sqlite_config(tmp_path) -> RunLockConfig:
    return RunLockConfig(
        backend="sqlite",
        checkpoint_path=tmp_path / "agent-checkpoints.sqlite3",
        redis_url="redis://unused",
        ttl_seconds=30,
        heartbeat_seconds=10,
        namespace="test:agent-run",
    )


def test_sqlite_lease_excludes_another_worker_until_release(tmp_path):
    config = _sqlite_config(tmp_path)
    first = create_agent_run_lease("same-run", config)
    second = create_agent_run_lease("same-run", config)

    first.acquire()
    with pytest.raises(AgentRunLeaseBusyError):
        second.acquire()

    first.release()
    second.acquire()
    second.release()


def test_expired_sqlite_lease_can_be_taken_over(tmp_path):
    clock = _Clock()
    path = tmp_path / "agent-checkpoints.sqlite3"
    first = SQLiteLeaseBackend(path, "run", "owner-a", 30, clock=clock)
    second = SQLiteLeaseBackend(path, "run", "owner-b", 30, clock=clock)

    assert first.acquire() is True
    clock.value += 31
    assert second.acquire() is True


def test_stale_owner_cannot_release_new_sqlite_owner(tmp_path):
    clock = _Clock()
    path = tmp_path / "agent-checkpoints.sqlite3"
    first = SQLiteLeaseBackend(path, "run", "owner-a", 30, clock=clock)
    second = SQLiteLeaseBackend(path, "run", "owner-b", 30, clock=clock)
    third = SQLiteLeaseBackend(path, "run", "owner-c", 30, clock=clock)

    assert first.acquire() is True
    clock.value += 31
    assert second.acquire() is True
    assert first.release() is False
    assert third.acquire() is False
    assert second.release() is True
    assert third.acquire() is True


def test_sqlite_renewal_requires_the_current_owner(tmp_path):
    clock = _Clock()
    path = tmp_path / "agent-checkpoints.sqlite3"
    owner = SQLiteLeaseBackend(path, "run", "owner", 30, clock=clock)
    stranger = SQLiteLeaseBackend(path, "run", "stranger", 30, clock=clock)

    assert owner.acquire() is True
    clock.value += 10
    assert owner.renew() is True
    assert stranger.renew() is False
    clock.value += 25
    assert SQLiteLeaseBackend(path, "run", "next", 30, clock=clock).acquire() is False


class _FakeRedisLock:
    def __init__(self):
        self.acquired_with = None
        self.reacquired = 0
        self.released = 0

    def acquire(self, *, blocking, token):
        self.acquired_with = (blocking, token)
        return True

    def reacquire(self):
        self.reacquired += 1
        return True

    def release(self):
        self.released += 1


class _FakeRedis:
    def __init__(self):
        self.created_with = None
        self.lock_instance = _FakeRedisLock()

    def lock(self, name, *, timeout, blocking, thread_local):
        self.created_with = (name, timeout, blocking, thread_local)
        return self.lock_instance


def test_redis_backend_uses_nonblocking_token_owned_lock():
    client = _FakeRedis()
    backend = RedisLeaseBackend(
        client,
        "docmind:agent-run:abc",
        "owner-token",
        ttl_seconds=60,
    )

    assert backend.acquire() is True
    assert client.created_with == ("docmind:agent-run:abc", 60, False, False)
    assert client.lock_instance.acquired_with == (False, "owner-token")
    assert backend.renew() is True
    assert backend.release() is True
    assert client.lock_instance.reacquired == 1
    assert client.lock_instance.released == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"ttl_seconds": 0},
        {"heartbeat_seconds": 0},
        {"ttl_seconds": 10, "heartbeat_seconds": 10},
        {"namespace": "  "},
    ],
)
def test_run_lock_config_rejects_unsafe_values(tmp_path, overrides):
    values = {
        "backend": "sqlite",
        "checkpoint_path": tmp_path / "checkpoints.sqlite3",
        "redis_url": "redis://unused",
        "ttl_seconds": 30,
        "heartbeat_seconds": 10,
        "namespace": "test",
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        RunLockConfig(**values)


class _ErrorRedisLock(_FakeRedisLock):
    def __init__(self, *, acquire_error=None, renew_error=None, release_error=None):
        super().__init__()
        self.acquire_error = acquire_error
        self.renew_error = renew_error
        self.release_error = release_error

    def acquire(self, *, blocking, token):
        if self.acquire_error:
            raise self.acquire_error
        return super().acquire(blocking=blocking, token=token)

    def reacquire(self):
        if self.renew_error:
            raise self.renew_error
        return super().reacquire()

    def release(self):
        if self.release_error:
            raise self.release_error
        return super().release()


def test_redis_backend_fails_closed_on_backend_and_ownership_errors():
    client = _FakeRedis()
    client.lock_instance = _ErrorRedisLock(acquire_error=RedisError("down"))
    backend = RedisLeaseBackend(client, "key", "owner", ttl_seconds=30)
    with pytest.raises(AgentRunLeaseBackendError):
        backend.acquire()

    client.lock_instance = _ErrorRedisLock(renew_error=LockError("lost"))
    backend = RedisLeaseBackend(client, "key", "owner", ttl_seconds=30)
    assert backend.renew() is False

    client.lock_instance = _ErrorRedisLock(release_error=LockError("lost"))
    backend = RedisLeaseBackend(client, "key", "owner", ttl_seconds=30)
    assert backend.release() is False

    client.lock_instance = _ErrorRedisLock(release_error=RedisError("down"))
    backend = RedisLeaseBackend(client, "key", "owner", ttl_seconds=30)
    with pytest.raises(AgentRunLeaseBackendError):
        backend.release()


class _Backend:
    def __init__(
        self,
        *,
        acquire=True,
        renew=True,
        release=True,
        acquire_error=None,
        release_error=None,
    ):
        self.acquire_result = acquire
        self.renew_result = renew
        self.release_result = release
        self.acquire_error = acquire_error
        self.release_error = release_error

    def acquire(self):
        if self.acquire_error:
            raise self.acquire_error
        return self.acquire_result

    def renew(self):
        return self.renew_result

    def release(self):
        if self.release_error:
            raise self.release_error
        return self.release_result


def test_agent_run_lease_lifecycle_and_failure_mapping():
    lease = AgentRunLease("run", _Backend(), 1, heartbeat_enabled=False)
    lease.release()  # releasing an unheld lease is idempotent
    with lease:
        with pytest.raises(AgentRunLeaseError):
            lease.acquire()

    with pytest.raises(AgentRunLeaseBusyError):
        AgentRunLease(
            "busy", _Backend(acquire=False), 1, heartbeat_enabled=False
        ).acquire()
    with pytest.raises(AgentRunLeaseBackendError):
        AgentRunLease(
            "io", _Backend(acquire_error=OSError("disk")), 1, heartbeat_enabled=False
        ).acquire()
    lost = AgentRunLease(
        "lost", _Backend(release=False), 1, heartbeat_enabled=False
    )
    lost.acquire()
    with pytest.raises(AgentRunLeaseLostError):
        lost.release()

    backend_error = AgentRunLease(
        "release-io",
        _Backend(release_error=OSError("disk")),
        1,
        heartbeat_enabled=False,
    )
    backend_error.acquire()
    with pytest.raises(AgentRunLeaseBackendError):
        backend_error.release()


def test_off_backend_is_explicit_noop(tmp_path):
    config = RunLockConfig(
        backend="off",
        checkpoint_path=tmp_path / "unused.sqlite3",
        redis_url="redis://unused",
        ttl_seconds=30,
        heartbeat_seconds=10,
        namespace="test",
    )
    lease = create_agent_run_lease("noop", config)
    lease.acquire()
    assert lease.backend.renew() is True
    lease.release()


def test_redis_factory_builds_client(monkeypatch, tmp_path):
    fake = _FakeRedis()
    captured = {}

    def _from_url(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("app.agent.run_lock.Redis.from_url", _from_url)
    config = RunLockConfig(
        backend="redis",
        checkpoint_path=tmp_path / "unused.sqlite3",
        redis_url="redis://example/3",
        ttl_seconds=30,
        heartbeat_seconds=10,
        namespace="deployment-a",
    )
    lease = create_agent_run_lease("run", config, owner_token="fixed")
    lease.acquire()
    lease.release()

    assert captured["url"] == "redis://example/3"
    assert fake.created_with[0] == "deployment-a:run"
