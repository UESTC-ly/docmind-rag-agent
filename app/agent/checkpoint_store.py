"""DocMind metadata beside LangGraph's SQLite checkpoint tables."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def prepare_checkpoint_connection(connection: sqlite3.Connection) -> None:
    """Configure SQLite and create DocMind-owned support tables idempotently."""
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS docmind_tool_receipts (
            run_id TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_id, tool_call_id)
        );

        CREATE TABLE IF NOT EXISTS docmind_agent_runs (
            run_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL
        );
        CREATE INDEX IF NOT EXISTS ix_docmind_agent_runs_retention
            ON docmind_agent_runs(status, updated_at);

        CREATE TABLE IF NOT EXISTS docmind_agent_run_leases (
            lock_key TEXT PRIMARY KEY,
            owner_token TEXT NOT NULL,
            expires_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS docmind_checkpoint_maintenance (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            owner_token TEXT,
            lease_expires_at REAL,
            last_completed_at REAL
        );
        INSERT OR IGNORE INTO docmind_checkpoint_maintenance(id) VALUES (1);
        """
    )
    connection.commit()


def open_checkpoint_connection(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(resolved), timeout=30, check_same_thread=False)
    prepare_checkpoint_connection(connection)
    return connection


def record_agent_run(
    path: str | Path,
    *,
    run_id: str,
    user_id: int,
    conversation_id: int | None,
    status: str,
    now: float | None = None,
    only_if_missing: bool = False,
) -> None:
    timestamp = time.time() if now is None else now
    completed_at = timestamp if status == "completed" else None
    with open_checkpoint_connection(path) as connection:
        if only_if_missing:
            connection.execute(
                """
                INSERT OR IGNORE INTO docmind_agent_runs
                    (run_id, user_id, conversation_id, status,
                     created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    user_id,
                    conversation_id,
                    status,
                    timestamp,
                    timestamp,
                    completed_at,
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO docmind_agent_runs
                    (run_id, user_id, conversation_id, status,
                     created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    conversation_id = excluded.conversation_id,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    completed_at = CASE
                        WHEN excluded.status = 'completed'
                        THEN COALESCE(docmind_agent_runs.completed_at, excluded.completed_at)
                        ELSE NULL
                    END
                """,
                (
                    run_id,
                    user_id,
                    conversation_id,
                    status,
                    timestamp,
                    timestamp,
                    completed_at,
                ),
            )
        connection.commit()


def cleanup_candidate_ids(
    path: str | Path,
    *,
    completed_before: float,
    incomplete_before: float,
    batch_size: int,
) -> list[str]:
    with open_checkpoint_connection(path) as connection:
        return [
            str(row[0])
            for row in connection.execute(
                """
                SELECT run_id
                FROM docmind_agent_runs
                WHERE (status = 'completed' AND updated_at <= ?)
                   OR (status != 'completed' AND updated_at <= ?)
                ORDER BY updated_at ASC, run_id ASC
                LIMIT ?
                """,
                (completed_before, incomplete_before, batch_size),
            ).fetchall()
        ]


def unindexed_checkpoint_ids(path: str | Path, *, batch_size: int) -> list[str]:
    """Return v3.0 thread IDs that predate the v3.1 retention index."""
    with open_checkpoint_connection(path) as connection:
        has_checkpoints = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'checkpoints'
            """
        ).fetchone()
        if has_checkpoints is None:
            return []
        return [
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT checkpoints.thread_id
                FROM checkpoints
                LEFT JOIN docmind_agent_runs
                  ON docmind_agent_runs.run_id = checkpoints.thread_id
                WHERE docmind_agent_runs.run_id IS NULL
                ORDER BY checkpoints.thread_id
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()
        ]


def delete_agent_run_data(path: str | Path, run_id: str) -> None:
    """Delete graph state and DocMind metadata while a run lease is held."""
    with open_checkpoint_connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "writes" in tables:
            connection.execute("DELETE FROM writes WHERE thread_id = ?", (run_id,))
        if "checkpoints" in tables:
            connection.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (run_id,)
            )
        connection.execute(
            "DELETE FROM docmind_tool_receipts WHERE run_id = ?", (run_id,)
        )
        connection.execute("DELETE FROM docmind_agent_runs WHERE run_id = ?", (run_id,))
        connection.commit()


def claim_cleanup_cycle(
    path: str | Path,
    *,
    owner_token: str,
    now: float,
    interval_seconds: int,
    lease_seconds: int,
    force: bool,
) -> bool:
    with open_checkpoint_connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT owner_token, lease_expires_at, last_completed_at
            FROM docmind_checkpoint_maintenance
            WHERE id = 1
            """
        ).fetchone()
        assert row is not None
        _, lease_expires_at, last_completed_at = row
        if lease_expires_at is not None and float(lease_expires_at) > now:
            connection.rollback()
            return False
        if (
            not force
            and last_completed_at is not None
            and float(last_completed_at) + interval_seconds > now
        ):
            connection.rollback()
            return False
        connection.execute(
            """
            UPDATE docmind_checkpoint_maintenance
            SET owner_token = ?, lease_expires_at = ?
            WHERE id = 1
            """,
            (owner_token, now + lease_seconds),
        )
        connection.commit()
        return True


def finish_cleanup_cycle(
    path: str | Path,
    *,
    owner_token: str,
    now: float,
    successful: bool,
) -> None:
    with open_checkpoint_connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if successful:
            connection.execute(
                """
                UPDATE docmind_checkpoint_maintenance
                SET owner_token = NULL,
                    lease_expires_at = NULL,
                    last_completed_at = ?
                WHERE id = 1 AND owner_token = ?
                """,
                (now, owner_token),
            )
        else:
            connection.execute(
                """
                UPDATE docmind_checkpoint_maintenance
                SET owner_token = NULL, lease_expires_at = NULL
                WHERE id = 1 AND owner_token = ?
                """,
                (owner_token,),
            )
        connection.commit()


def prune_expired_sqlite_leases(path: str | Path, *, now: float) -> int:
    with open_checkpoint_connection(path) as connection:
        cursor = connection.execute(
            "DELETE FROM docmind_agent_run_leases WHERE expires_at <= ?", (now,)
        )
        connection.commit()
        return cursor.rowcount
