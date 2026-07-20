"""Retention-based cleanup for Agent checkpoints, receipts, and run metadata."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from app.agent.checkpoint_store import (
    claim_cleanup_cycle,
    cleanup_candidate_ids,
    delete_agent_run_data,
    finish_cleanup_cycle,
    prune_expired_sqlite_leases,
)
from app.agent.run_lock import (
    AgentRunLeaseBusyError,
    RunLockConfig,
    create_agent_run_lease,
)
from app.config import settings
from app.utils.logging import logger


@dataclass(frozen=True)
class CheckpointCleanupPolicy:
    completed_retention_seconds: int
    incomplete_retention_seconds: int
    interval_seconds: int
    batch_size: int
    maintenance_lease_seconds: int

    def __post_init__(self) -> None:
        if self.completed_retention_seconds <= 0:
            raise ValueError("completed checkpoint retention must be positive")
        if self.incomplete_retention_seconds < self.completed_retention_seconds:
            raise ValueError("incomplete checkpoints must outlive completed checkpoints")
        if self.interval_seconds < 0:
            raise ValueError("checkpoint cleanup interval cannot be negative")
        if self.batch_size <= 0 or self.maintenance_lease_seconds <= 0:
            raise ValueError("checkpoint cleanup batch and lease must be positive")


@dataclass(frozen=True)
class CheckpointCleanupResult:
    ran: bool
    backfilled: int = 0
    candidates: int = 0
    deleted: int = 0
    skipped_locked: int = 0
    expired_leases_pruned: int = 0


def configured_run_lock() -> RunLockConfig:
    return RunLockConfig(
        backend=settings.resolved_agent_run_lock_backend,
        checkpoint_path=settings.agent_checkpoint_path,
        redis_url=settings.redis_url,
        ttl_seconds=settings.agent_run_lock_ttl_seconds,
        heartbeat_seconds=settings.agent_run_lock_heartbeat_seconds,
        namespace=settings.agent_run_lock_namespace,
    )


def configured_cleanup_policy() -> CheckpointCleanupPolicy:
    seconds_per_day = 24 * 60 * 60
    return CheckpointCleanupPolicy(
        completed_retention_seconds=(
            settings.agent_checkpoint_completed_retention_days * seconds_per_day
        ),
        incomplete_retention_seconds=(
            settings.agent_checkpoint_incomplete_retention_days * seconds_per_day
        ),
        interval_seconds=settings.agent_checkpoint_cleanup_interval_seconds,
        batch_size=settings.agent_checkpoint_cleanup_batch_size,
        maintenance_lease_seconds=max(
            60, settings.agent_checkpoint_cleanup_interval_seconds
        ),
    )


def cleanup_agent_checkpoints(
    checkpoint_path: str | Path,
    *,
    lock_config: RunLockConfig,
    policy: CheckpointCleanupPolicy,
    now: float | None = None,
    force: bool = False,
) -> CheckpointCleanupResult:
    timestamp = time.time() if now is None else now
    owner_token = uuid.uuid4().hex
    if not claim_cleanup_cycle(
        checkpoint_path,
        owner_token=owner_token,
        now=timestamp,
        interval_seconds=policy.interval_seconds,
        lease_seconds=policy.maintenance_lease_seconds,
        force=force,
    ):
        return CheckpointCleanupResult(ran=False)

    successful = False
    try:
        # v3.0 checkpoint files have no DocMind retention index. Backfill a
        # bounded batch and start their retention clock at upgrade time instead
        # of deleting old user state immediately.
        from app.agent.orchestrator import backfill_agent_run_index

        backfilled = backfill_agent_run_index(
            checkpoint_path,
            batch_size=policy.batch_size,
            now=timestamp,
        )
        candidates = cleanup_candidate_ids(
            checkpoint_path,
            completed_before=timestamp - policy.completed_retention_seconds,
            incomplete_before=timestamp - policy.incomplete_retention_seconds,
            batch_size=policy.batch_size,
        )
        deleted = 0
        skipped_locked = 0
        effective_lock_config = replace(
            lock_config, checkpoint_path=checkpoint_path
        )
        for run_id in candidates:
            lease = create_agent_run_lease(run_id, effective_lock_config)
            try:
                lease.acquire()
            except AgentRunLeaseBusyError:
                skipped_locked += 1
                continue
            try:
                delete_agent_run_data(checkpoint_path, run_id)
                deleted += 1
            finally:
                lease.release()
        pruned = prune_expired_sqlite_leases(checkpoint_path, now=timestamp)
        successful = True
        return CheckpointCleanupResult(
            ran=True,
            backfilled=backfilled,
            candidates=len(candidates),
            deleted=deleted,
            skipped_locked=skipped_locked,
            expired_leases_pruned=pruned,
        )
    finally:
        finish_cleanup_cycle(
            checkpoint_path,
            owner_token=owner_token,
            now=timestamp,
            successful=successful,
        )


def cleanup_configured_agent_checkpoints() -> CheckpointCleanupResult:
    if not settings.agent_checkpoint_cleanup_enabled:
        return CheckpointCleanupResult(ran=False)
    return cleanup_agent_checkpoints(
        settings.agent_checkpoint_path,
        lock_config=configured_run_lock(),
        policy=configured_cleanup_policy(),
    )


async def checkpoint_cleanup_loop(stop_event: asyncio.Event) -> None:
    """Run one bounded cleanup cycle per configured interval until shutdown."""
    interval = settings.agent_checkpoint_cleanup_interval_seconds
    while not stop_event.is_set():
        try:
            result = await asyncio.to_thread(cleanup_configured_agent_checkpoints)
            if result.ran:
                logger.info(
                    "Agent checkpoint cleanup completed: candidates={}, "
                    "backfilled={}, deleted={}, skipped_locked={}, "
                    "expired_leases_pruned={}",
                    result.candidates,
                    result.backfilled,
                    result.deleted,
                    result.skipped_locked,
                    result.expired_leases_pruned,
                )
        except Exception as exc:  # noqa: BLE001 - maintenance must not stop the API
            logger.error("Agent checkpoint cleanup failed: {}", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue
