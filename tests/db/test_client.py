"""
Tests for the db client connection helpers.

The one-writer rule (CONTRIBUTING section 7) is only meaningful if the
read-only path actually rejects writes. get_connection is the writer used by
the experiment; get_readonly_connection is the reader used by analysis. These
pin both halves of that contract.
"""

from __future__ import annotations

import sqlite3

import pytest

from maestro.db.client import SCHEMA, get_connection, get_readonly_connection


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def test_readonly_connection_rejects_writes(tmp_path):
    """The analysis read path must not be able to mutate experiment data."""
    db_path = tmp_path / "ro.db"
    _make_db(db_path)

    with get_readonly_connection(db_path) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO run_environments (environment_id, captured_at) "
                "VALUES ('x', 'y')"
            )


def test_readonly_connection_missing_db_raises(tmp_path):
    """mode=ro raises on a missing file rather than creating an empty one."""
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        with get_readonly_connection(missing):
            pass
    assert not missing.exists()


def test_writer_connection_allows_writes(tmp_path):
    """The writer path commits an insert (contrast with the reader above)."""
    db_path = tmp_path / "rw.db"
    _make_db(db_path)

    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO run_environments (environment_id, captured_at) "
            "VALUES ('env-1', '2026-01-01T00:00:00Z')"
        )

    with get_readonly_connection(db_path) as conn:
        row = conn.execute("SELECT environment_id FROM run_environments").fetchone()
    assert row[0] == "env-1"
