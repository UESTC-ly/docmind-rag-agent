"""Cross-worker Agent run leases.

Web deployments use a shared Redis lock so separate processes or hosts cannot
mutate the same LangGraph ``thread_id`` concurrently.  The self-contained
desktop runtime uses the checkpoint SQLite file as a process-shared lease
store.  Both backends use an owner token, expiry, renewal, and conditional
release; an old owner can never delete a newer owner's lease.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from redis import Redis
from redis.exceptions import LockError, RedisError

from app.agent.checkpoint_store import prepare_checkpoint_connection


class AgentRunLeaseError(RuntimeError):
    """Base error for Agent run lease operations."""


class AgentRunLeaseBusyError(AgentRunLeaseError):
    """Another worker currently owns the run lease."""


class AgentRunLeaseBackendError(AgentRunLeaseError):
    """The configured lease backend is unavailable or invalid."""


class AgentRunLeaseLostError(AgentRunLeaseError):
    """The worker lost ownership while the Agent was still running."""


@dataclass(frozen=True)
class RunLockConfig:
    backend: Literal["redis", "sqlite", "off"]
    checkpoint_path: str | Path
    redis_url: str
    ttl_seconds: int
    heartbeat_seconds: int
    namespace: str

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("Agent run lease TTL must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("Agent run lease heartbeat must be positive")
        if self.heartbeat_seconds >= self.ttl_seconds:
            raise ValueError("Agent run lease heartbeat must be shorter than its TTL")
        if not self.namespace.strip():
            raise ValueError("Agent run lease namespace cannot be empty")


class LeaseBackend(Protocol):
    def acquire(self) -> bool: ...

    def renew(self) -> bool: ...

    def release(self) -> bool: ...


class SQLiteLeaseBackend:
    """Process-safe lease for workers sharing one checkpoint SQLite file."""

    def __init__(
        self,
        path: str | Path,
        lock_key: str,
        owner_token: str,
        ttl_seconds: int,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_key = lock_key
        self.owner_token = owner_token
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(self.path), timeout=30, check_same_thread=False
        )
        prepare_checkpoint_connection(connection)
        return connection

    def acquire(self) -> bool:
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT owner_token, expires_at
                FROM docmind_agent_run_leases
                WHERE lock_key = ?
                """,
                (self.lock_key,),
            ).fetchone()
            if row is not None and float(row[1]) > now:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO docmind_agent_run_leases
                    (lock_key, owner_token, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lock_key) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (self.lock_key, self.owner_token, now + self.ttl_seconds, now),
            )
            connection.commit()
            return True

    def renew(self) -> bool:
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE docmind_agent_run_leases
                SET expires_at = ?, updated_at = ?
                WHERE lock_key = ? AND owner_token = ? AND expires_at > ?
                """,
                (
                    now + self.ttl_seconds,
                    now,
                    self.lock_key,
                    self.owner_token,
                    now,
                ),
            )
            connection.commit()
            return cursor.rowcount == 1

    def release(self) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM docmind_agent_run_leases
                WHERE lock_key = ? AND owner_token = ?
                """,
                (self.lock_key, self.owner_token),
            )
            connection.commit()
            return cursor.rowcount == 1


class RedisLeaseBackend:
    """Redis-backed lease shared across processes and hosts."""

    def __init__(
        self,
        client: Any,
        lock_key: str,
        owner_token: str,
        ttl_seconds: int,
    ) -> None:
        self.owner_token = owner_token
        self._lock = client.lock(
            lock_key,
            timeout=ttl_seconds,
            blocking=False,
            # Heartbeats run in a dedicated thread, so ownership cannot live
            # in redis-py's thread-local token storage.
            thread_local=False,
        )

    def acquire(self) -> bool:
        try:
            return bool(
                self._lock.acquire(blocking=False, token=self.owner_token)
            )
        except RedisError as exc:
            raise AgentRunLeaseBackendError(
                f"Redis Agent run lock unavailable: {exc}"
            ) from exc

    def renew(self) -> bool:
        try:
            return bool(self._lock.reacquire())
        except (LockError, RedisError):
            return False

    def release(self) -> bool:
        try:
            self._lock.release()
            return True
        except LockError:
            return False
        except RedisError as exc:
            raise AgentRunLeaseBackendError(
                f"Redis Agent run lock release failed: {exc}"
            ) from exc


class _NoopLeaseBackend:
    def acquire(self) -> bool:
        return True

    def renew(self) -> bool:
        return True

    def release(self) -> bool:
        return True


class AgentRunLease:
    """Own one renewable lease and maintain it with a daemon heartbeat."""

    def __init__(
        self,
        run_id: str,
        backend: LeaseBackend,
        heartbeat_seconds: int,
        *,
        heartbeat_enabled: bool = True,
    ) -> None:
        self.run_id = run_id
        self.backend = backend
        self.heartbeat_seconds = heartbeat_seconds
        self.heartbeat_enabled = heartbeat_enabled
        self._held = False
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> None:
        if self._held:
            raise AgentRunLeaseError(f"Agent run lease already held: {self.run_id}")
        try:
            acquired = self.backend.acquire()
        except AgentRunLeaseError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise AgentRunLeaseBackendError(
                f"Agent run lock backend unavailable: {exc}"
            ) from exc
        if not acquired:
            raise AgentRunLeaseBusyError(
                f"Agent run 正由另一个 worker 执行: {self.run_id}"
            )
        self._stop.clear()
        self._lost.clear()
        self._thread = None
        self._held = True
        if self.heartbeat_enabled:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"agent-run-lease-{self.run_id[:16]}",
                daemon=True,
            )
            self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                renewed = self.backend.renew()
            except Exception:  # noqa: BLE001 - background ownership boundary
                renewed = False
            if not renewed:
                self._lost.set()
                return

    def release(self) -> None:
        if not self._held:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1, self.heartbeat_seconds))
        try:
            released = self.backend.release()
        except AgentRunLeaseError:
            self._held = False
            raise
        except (OSError, sqlite3.Error) as exc:
            self._held = False
            raise AgentRunLeaseBackendError(
                f"Agent run lock backend unavailable during release: {exc}"
            ) from exc
        self._held = False
        if self._lost.is_set() or not released:
            raise AgentRunLeaseLostError(
                f"Agent run lease ownership lost: {self.run_id}"
            )

    def __enter__(self) -> AgentRunLease:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.release()
        except (AgentRunLeaseBackendError, AgentRunLeaseLostError):
            if exc_type is None:
                raise


def create_agent_run_lease(
    run_id: str,
    config: RunLockConfig,
    *,
    owner_token: str | None = None,
    redis_client: Any | None = None,
    clock: Callable[[], float] = time.time,
) -> AgentRunLease:
    token = owner_token or uuid.uuid4().hex
    lock_key = f"{config.namespace.rstrip(':')}:{run_id}"
    if config.backend == "redis":
        client = redis_client or Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        backend: LeaseBackend = RedisLeaseBackend(
            client, lock_key, token, config.ttl_seconds
        )
    elif config.backend == "sqlite":
        backend = SQLiteLeaseBackend(
            config.checkpoint_path,
            lock_key,
            token,
            config.ttl_seconds,
            clock=clock,
        )
    elif config.backend == "off":
        backend = _NoopLeaseBackend()
    else:  # pragma: no cover - RunLockConfig is constructed from validated settings.
        raise ValueError(f"Unsupported Agent run lock backend: {config.backend}")
    return AgentRunLease(
        run_id,
        backend,
        config.heartbeat_seconds,
        heartbeat_enabled=config.backend != "off",
    )
