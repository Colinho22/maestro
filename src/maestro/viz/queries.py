"""
MAESTRO viz — read-only queries specific to the dashboard.

This module *extends* ``maestro.db.queries`` rather than rewriting it: the
experiment-side query layer there is the source of truth for the joins the
analysis pipeline relies on. Here we add only the lightweight reads the
dashboard needs — primarily "is there any data to show?" checks that drive
empty-state handling. Per-view queries live alongside their respective view
modules.

Every function takes an already-opened read-only ``sqlite3.Connection`` (see
``viz.db.connect``) and only reads.
"""

from __future__ import annotations

import re
import sqlite3

# A valid SQLite identifier for our purposes: a bare table name. Used to
# reject anything that isn't a plain identifier before it reaches an
# interpolated query (table names can't be bound as parameters).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    A table name cannot be passed as a bound parameter, so it is interpolated
    into the query. ``table`` is validated against a strict identifier
    pattern first (and rejected with ``ValueError`` otherwise) so this can
    never become an injection vector even if a caller ever passes untrusted
    input.
    """
    if not _IDENTIFIER_RE.match(table):
        raise ValueError(f"invalid table identifier: {table!r}")
    if not table_exists(conn, table):
        return 0
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def has_any_runs(conn: sqlite3.Connection) -> bool:
    """True if at least one experiment run has been recorded."""
    return count_rows(conn, "run_configs") > 0


def has_any_metrics(conn: sqlite3.Connection) -> bool:
    """True if at least one run has been scored (metric_results populated)."""
    return count_rows(conn, "metric_results") > 0
