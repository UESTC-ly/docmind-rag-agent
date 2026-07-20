"""Database-engine compatibility regressions."""

from datetime import datetime
import sqlite3

from app.database import _install_sqlite_compatibility_functions


def test_legacy_sqlite_now_default_remains_writable() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE events (created_at TEXT NOT NULL DEFAULT (now()))"
        )
        _install_sqlite_compatibility_functions(connection, object())

        connection.execute("INSERT INTO events DEFAULT VALUES")
        value = connection.execute("SELECT created_at FROM events").fetchone()[0]

        assert datetime.fromisoformat(value)
    finally:
        connection.close()
