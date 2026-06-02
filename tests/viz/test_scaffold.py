"""
Tests for the viz scaffold: read-only DB access, the queries layer, the
view registry, and settings resolution.

What's covered here is everything verifiable WITHOUT a running Streamlit
server — module structure, the read-only connection guarantee, and the
empty-state-driving queries against a real in-memory schema. The live UI
(nav rendering, the settings panel widgets) is verified by launching the app
(`streamlit run`), not pytest, since Streamlit widgets need a script context.

streamlit must be importable for the viz package to load; importorskip turns
a missing-streamlit environment into a clean skip.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("streamlit")

from maestro.db.client import SCHEMA  # noqa: E402
from maestro.viz import db as viz_db  # noqa: E402
from maestro.viz import queries as viz_queries  # noqa: E402
from maestro.viz import settings as viz_settings  # noqa: E402
from maestro.viz.views import VIEWS  # noqa: E402

# ---------------------------------------------------------------------------
# View registry
# ---------------------------------------------------------------------------


def test_views_registry_shape():
    """VIEWS is a non-empty list of (label, callable) pairs with unique labels."""
    assert VIEWS, "view registry is empty"
    labels = [label for label, _ in VIEWS]
    assert len(labels) == len(set(labels)), "duplicate view labels"
    for label, render in VIEWS:
        assert isinstance(label, str) and label
        assert callable(render)


def test_planned_views_present():
    """The five planned data views are registered (as placeholders for now)."""
    labels = {label for label, _ in VIEWS}
    for expected in (
        "Overview",
        "Strategy Comparison",
        "Pareto",
        "Run Detail",
        "Hallucination Taxonomy",
    ):
        assert expected in labels, f"missing planned view: {expected}"


# ---------------------------------------------------------------------------
# Read-only DB access
# ---------------------------------------------------------------------------


def test_connection_is_read_only(tmp_path):
    """
    The viz connection must reject writes — a dashboard must never mutate the
    experiment data. Build a real on-disk DB, open it via the viz layer, and
    assert an INSERT raises.
    """
    db_path = tmp_path / "ro.db"
    # Create + populate via a normal (writable) connection first.
    w = sqlite3.connect(db_path)
    w.executescript(SCHEMA)
    w.commit()
    w.close()

    # viz_db.connect yields a read-only connection; a write must raise.
    with viz_db.connect(db_path) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO run_environments (environment_id, captured_at) "
                "VALUES ('x', 'y')"
            )


def test_database_exists(tmp_path):
    missing = tmp_path / "nope.db"
    assert viz_db.database_exists(missing) is False
    present = tmp_path / "yes.db"
    present.write_bytes(b"")  # a file is enough for the existence check
    assert viz_db.database_exists(present) is True


# ---------------------------------------------------------------------------
# Queries — empty-state drivers, graceful on absent tables
# ---------------------------------------------------------------------------


def _mem_db(with_schema: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    if with_schema:
        conn.executescript(SCHEMA)
    return conn


def test_queries_on_empty_schema():
    """A schema with no rows reports no runs / no metrics, no errors."""
    conn = _mem_db(with_schema=True)
    assert viz_queries.table_exists(conn, "run_configs") is True
    assert viz_queries.count_rows(conn, "run_configs") == 0
    assert viz_queries.has_any_runs(conn) is False
    assert viz_queries.has_any_metrics(conn) is False


def test_queries_on_missing_tables_do_not_raise():
    """
    Pointed at a DB without the expected schema, the count/has_* helpers
    return 0/False rather than raising OperationalError — so the dashboard
    degrades to an empty-state instead of crashing.
    """
    conn = _mem_db(with_schema=False)
    assert viz_queries.table_exists(conn, "run_configs") is False
    assert viz_queries.count_rows(conn, "run_configs") == 0
    assert viz_queries.has_any_runs(conn) is False
    assert viz_queries.has_any_metrics(conn) is False


def test_count_rows_rejects_invalid_identifier():
    """A non-identifier table name raises rather than reaching the SQL."""
    conn = _mem_db(with_schema=True)
    with pytest.raises(ValueError):
        viz_queries.count_rows(conn, "run_configs; DROP TABLE run_configs")


def test_has_any_runs_true_after_insert():
    conn = _mem_db(with_schema=True)
    conn.execute(
        "INSERT INTO run_configs "
        "(run_id, strategy, model, example_id, tier, run_number, timestamp) "
        "VALUES ('r1', 'single_agent', 'm', 'ex', 2, 1, '2026-01-01T00:00:00Z')"
    )
    assert viz_queries.has_any_runs(conn) is True


# ---------------------------------------------------------------------------
# Settings — env → default resolution (the precedence that runs without a
# Streamlit script context; the session_state/UI layer is verified live)
# ---------------------------------------------------------------------------


def test_db_path_default_falls_back_to_project_default(monkeypatch):
    monkeypatch.delenv(viz_settings.DB_PATH_ENV_VAR, raising=False)
    # The project default ends with the standard DB filename.
    assert viz_settings._default_db_path().endswith("maestro.db")


def test_db_path_default_uses_env_when_set(monkeypatch):
    monkeypatch.setenv(viz_settings.DB_PATH_ENV_VAR, "/tmp/custom.db")
    assert viz_settings._default_db_path() == "/tmp/custom.db"


def test_display_tz_default_blank_means_system_local(monkeypatch):
    monkeypatch.delenv(viz_settings.TZ_ENV_VAR, raising=False)
    # Empty string sentinel (normalized to None in current_settings).
    assert viz_settings._default_display_tz() == ""


def test_display_tz_default_uses_env_when_set(monkeypatch):
    monkeypatch.setenv(viz_settings.TZ_ENV_VAR, "Europe/Zurich")
    assert viz_settings._default_display_tz() == "Europe/Zurich"
