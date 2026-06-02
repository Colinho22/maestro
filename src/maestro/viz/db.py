"""
MAESTRO viz — read-only SQLite access for the dashboard.

The visualizer must never mutate the experiment database. This module opens
connections in SQLite *read-only* mode (``file:...?mode=ro`` URI) so any
accidental write raises rather than corrupting data a long experiment
produced.

Connections are short-lived: ``connect`` is a context manager that opens a
fresh connection per operation and closes it on exit. A single sqlite3
connection is not safe to share across threads, and Streamlit may run reruns
on different threads — so rather than caching one shared handle, each query
gets its own. Opening a local SQLite file is cheap enough that this is a
non-issue for an interactive dashboard, and it sidesteps cross-thread
cursor-state hazards entirely.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """
    Yield a read-only connection to ``db_path``, closed on exit.

    Opened with ``mode=ro``: writes raise, and a missing file raises rather
    than being created (callers surface that as an empty-state). The
    connection is created and used on the same thread (the caller's), so no
    ``check_same_thread`` override is needed.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def database_exists(db_path: Path) -> bool:
    """Whether the configured database file is present and a regular file."""
    return db_path.is_file()
