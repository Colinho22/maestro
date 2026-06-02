"""
MAESTRO viz — read-only SQLite access for the dashboard.

The visualizer must never mutate the experiment database. This module opens
the connection in SQLite *read-only* mode (``file:...?mode=ro`` URI) so any
accidental write raises rather than corrupting data a long experiment
produced. The connection is cached per Streamlit session via
``st.cache_resource`` so a single handle is reused across reruns instead of
reopening the file on every widget interaction.

Path resolution is delegated to ``settings.resolve_db_path`` so the sidebar
settings panel and this module agree on which database is in view.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """
    Open ``db_path`` read-only. Raises FileNotFoundError-equivalent via the
    URI layer if the file is absent (``mode=ro`` refuses to create it), which
    the caller surfaces as an empty-state rather than a crash.

    ``check_same_thread=False`` because Streamlit may touch the connection
    from a different thread than the one that created it; the read-only mode
    makes concurrent reads safe.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource(show_spinner=False)
def get_connection(db_path_str: str) -> sqlite3.Connection:
    """
    Return a cached read-only connection for ``db_path_str``.

    Keyed on the path string: pointing the sidebar at a different database
    yields a distinct cache entry and a fresh connection, so switching DBs
    in the UI works without a manual restart. ``st.cache_resource`` keeps one
    connection per distinct path for the session's lifetime.
    """
    return _connect_ro(Path(db_path_str))


def database_exists(db_path: Path) -> bool:
    """Whether the configured database file is present and a regular file."""
    return db_path.is_file()
