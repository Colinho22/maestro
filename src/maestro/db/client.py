"""
MAESTRO DB client
Handles SQLite connection and schema initialization.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema: creates tables if they don't exist
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_environments (
    environment_id      TEXT PRIMARY KEY,
    os                  TEXT,
    arch                TEXT,
    python              TEXT,
    hostname            TEXT,
    git_commit          TEXT,
    git_dirty           INTEGER,
    lib_versions        TEXT,
    docker_image_digest TEXT,
    captured_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_configs (
    run_id         TEXT PRIMARY KEY,
    strategy       TEXT NOT NULL,
    model          TEXT NOT NULL,
    example_id     TEXT NOT NULL,
    tier           INTEGER NOT NULL,
    run_number     INTEGER NOT NULL,
    timestamp      TEXT NOT NULL,
    environment_id TEXT,
    FOREIGN KEY (environment_id) REFERENCES run_environments(environment_id)
);

CREATE TABLE IF NOT EXISTS run_results (
    run_id               TEXT PRIMARY KEY,
    output_diagram_code  TEXT,
    prompt_tokens        INTEGER NOT NULL,
    completion_tokens    INTEGER NOT NULL,
    duration_ms          INTEGER NOT NULL,
    cost_usd             REAL NOT NULL,
    error                TEXT,
    retry_count          INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS metric_results (
    metric_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL,
    parses_valid            INTEGER,
    parse_error             TEXT,
    entity_id_precision     REAL NOT NULL,
    entity_id_recall        REAL NOT NULL,
    entity_id_f1            REAL NOT NULL,
    entity_name_precision   REAL NOT NULL,
    entity_name_recall      REAL NOT NULL,
    entity_name_f1          REAL NOT NULL,
    entity_lemma_precision  REAL NOT NULL,
    entity_lemma_recall     REAL NOT NULL,
    entity_lemma_f1         REAL NOT NULL,
    relationship_relaxed_precision  REAL NOT NULL,
    relationship_relaxed_recall     REAL NOT NULL,
    relationship_relaxed_f1         REAL NOT NULL,
    relationship_strict_precision   REAL NOT NULL,
    relationship_strict_recall      REAL NOT NULL,
    relationship_strict_f1          REAL NOT NULL,
    entities_in_output      INTEGER NOT NULL,
    entities_in_truth       INTEGER NOT NULL,
    relationships_in_output INTEGER NOT NULL,
    relationships_in_truth  INTEGER NOT NULL,
    missing_entities        INTEGER NOT NULL,
    extra_entities          INTEGER NOT NULL,
    false_entities          INTEGER NOT NULL,
    duplicate_entities      INTEGER NOT NULL,
    missing_relationships   INTEGER NOT NULL,
    extra_relationships     INTEGER NOT NULL,
    false_relationships     INTEGER NOT NULL,
    duplicate_relationships INTEGER NOT NULL,
    -- Container dimension (subgraphs). P/R/F1 nullable: NULL = no containers
    -- in the ground truth (metric not applicable for that diagram).
    container_id_precision   REAL,
    container_id_recall      REAL,
    container_id_f1          REAL,
    container_name_precision REAL,
    container_name_recall    REAL,
    container_name_f1        REAL,
    containers_in_output     INTEGER NOT NULL DEFAULT 0,
    containers_in_truth      INTEGER NOT NULL DEFAULT 0,
    -- Attachment dimension (o--o edges). P/R/F1 nullable: NULL = no
    -- attachments in the ground truth.
    attachment_precision     REAL,
    attachment_recall        REAL,
    attachment_f1            REAL,
    attachments_in_output    INTEGER NOT NULL DEFAULT 0,
    attachments_in_truth     INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES run_configs(run_id)
);
"""


def init_db(db_path: Path) -> None:
    """
    Create the SQLite file and tables if they don't exist.
    Safe to call on every run: no data is overwritten.

    Also runs the small set of additive migrations needed to bring a
    pre-existing database up to the current schema. Old rows keep their
    NULL ``run_configs.environment_id`` and default ``run_results.retry_count
    = 0``; no backfill is attempted.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        _migrate_add_environment_id_column(conn)
        _migrate_add_retry_count_column(conn)
        _migrate_add_container_attachment_columns(conn)
        conn.commit()


def _migrate_add_environment_id_column(conn: sqlite3.Connection) -> None:
    """
    Add ``run_configs.environment_id`` to databases that predate the column.

    SQLite has no portable ``ADD COLUMN IF NOT EXISTS``, so we inspect
    ``PRAGMA table_info`` first and only issue the ALTER when the column
    is missing. The FK is declarative-only on the new column (SQLite cannot
    add a constrained FK via ALTER); fresh databases created by ``SCHEMA``
    above carry the FK clause directly.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(run_configs)")}
    if "environment_id" not in cols:
        conn.execute("ALTER TABLE run_configs ADD COLUMN environment_id TEXT")


def _migrate_add_retry_count_column(conn: sqlite3.Connection) -> None:
    """
    Add ``run_results.retry_count`` to databases that predate the column.

    SQLite has no portable ``ADD COLUMN IF NOT EXISTS``, so we inspect
    ``PRAGMA table_info`` first and only ALTER when missing. Fresh
    databases created by ``SCHEMA`` above carry the column directly.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(run_results)")}
    if "retry_count" not in cols:
        conn.execute(
            "ALTER TABLE run_results ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
        )


def _migrate_add_container_attachment_columns(conn: sqlite3.Connection) -> None:
    """
    Add the container + attachment metric columns to databases that predate
    them (Phase 3b). Each is added only if missing. Nullable REAL columns get
    no default (NULL = metric not applicable); count columns default to 0.
    Old rows keep NULL P/R/F1 and 0 counts; no backfill is attempted.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(metric_results)")}
    additions = [
        ("container_id_precision", "REAL"),
        ("container_id_recall", "REAL"),
        ("container_id_f1", "REAL"),
        ("container_name_precision", "REAL"),
        ("container_name_recall", "REAL"),
        ("container_name_f1", "REAL"),
        ("containers_in_output", "INTEGER NOT NULL DEFAULT 0"),
        ("containers_in_truth", "INTEGER NOT NULL DEFAULT 0"),
        ("attachment_precision", "REAL"),
        ("attachment_recall", "REAL"),
        ("attachment_f1", "REAL"),
        ("attachments_in_output", "INTEGER NOT NULL DEFAULT 0"),
        ("attachments_in_truth", "INTEGER NOT NULL DEFAULT 0"),
    ]
    for name, decl in additions:
        if name not in cols:
            conn.execute(f"ALTER TABLE metric_results ADD COLUMN {name} {decl}")


@contextmanager
def get_connection(db_path: Path):
    """
    Context manager for a SQLite connection.
    Commits on success, rolls back on exception.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_readonly_connection(db_path: Path):
    """
    Read-only connection for the analysis read path, so the one-writer rule
    holds at the boundary: opened with ``mode=ro``, any write raises and a
    missing file raises rather than being created. There is no commit, since
    a reader has nothing to commit.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
