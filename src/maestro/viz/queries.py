"""
MAESTRO viz — read-only queries specific to the dashboard.

This module *extends* ``maestro.db.queries`` rather than rewriting it: the
experiment-side query layer there is the source of truth for the joins the
analysis pipeline relies on. Here we add only the lightweight reads the
dashboard needs — primarily "is there any data to show?" checks that drive
empty-state handling. Per-view queries live alongside their respective view
modules.

Every function takes an already-opened read-only ``sqlite3.Connection`` (see
``viz.db.get_connection``) and only reads.
"""

from __future__ import annotations

import sqlite3


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """
    Whether ``table`` exists in the database. Used to degrade gracefully when
    pointed at a brand-new or unmigrated DB that lacks the expected schema,
    rather than letting a query raise ``OperationalError``.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """
    Row count for ``table``, or 0 if the table is absent. The absent-table
    case returns 0 (not an error) so callers can treat "no table" and "empty
    table" identically for empty-state purposes.
    """
    if not table_exists(conn, table):
        return 0
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def has_any_runs(conn: sqlite3.Connection) -> bool:
    """True if at least one experiment run has been recorded."""
    return count_rows(conn, "run_configs") > 0


def has_any_metrics(conn: sqlite3.Connection) -> bool:
    """True if at least one run has been scored (metric_results populated)."""
    return count_rows(conn, "metric_results") > 0
