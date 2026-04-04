"""
MAESTRO — DB client
Handles SQLite connection and schema initialization.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema — creates tables if they don't exist
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_configs (
    run_id       TEXT PRIMARY KEY,
    strategy     TEXT NOT NULL,
    model        TEXT NOT NULL,
    example_id   TEXT NOT NULL,
    tier         INTEGER NOT NULL,
    run_number   INTEGER NOT NULL,
    timestamp    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_results (
    run_id               TEXT PRIMARY KEY,
    output_diagram_code  TEXT,
    prompt_tokens        INTEGER NOT NULL,
    completion_tokens    INTEGER NOT NULL,
    duration_ms          INTEGER NOT NULL,
    cost_usd             REAL NOT NULL,
    error                TEXT,
    FOREIGN KEY (run_id) REFERENCES run_configs(run_id)
);

CREATE TABLE IF NOT EXISTS sub_results (
    sub_id            TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    step_number       INTEGER NOT NULL,
    step_name         TEXT NOT NULL,
    output_text       TEXT,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    duration_ms       INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    error             TEXT,
    retry_count       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES run_configs(run_id)
);
"""


def init_db(db_path: Path) -> None:
    """
    Create the SQLite file and tables if they don't exist.
    Safe to call on every run — no data is overwritten.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection(db_path: Path):
    """
    Context manager for a SQLite connection.
    Commits on success, rolls back on exception.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()