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


def test_raw_response_survives_a_failed_cell(tmp_path):
    """A failed cell keeps the raw model output for diagnosis even though the
    cleaned output (output_diagram_code / output_text) is None. This is what
    lets a "invalid JSON" failure be inspected after the run without re-calling
    the model."""
    from maestro.db.client import init_db
    from maestro.db.queries import (
        insert_run_config,
        insert_run_result,
        insert_sub_result,
    )
    from maestro.schemas import RunConfig, RunResult, Strategy, SubResult

    db_path = tmp_path / "raw.db"
    init_db(db_path)
    cfg = RunConfig(
        strategy=Strategy.SOP_BASED,
        model="m",
        example_id="e",
        tier=1,
        run_number=1,
    )
    failed = RunResult(
        run_id=cfg.run_id,
        output_diagram_code=None,
        raw_response="{ malformed json the model emitted",
        prompt_tokens=1,
        completion_tokens=1,
        duration_ms=1,
        cost_usd=0.0,
        error="invalid JSON",
    )
    sub = SubResult(
        run_id=cfg.run_id,
        step_number=1,
        step_name="extract_entities",
        output_text=None,
        raw_response="{ bad json from step 1",
        prompt_tokens=1,
        completion_tokens=1,
        duration_ms=1,
        cost_usd=0.0,
        error="rejected",
    )
    with get_connection(db_path) as conn:
        insert_run_config(conn, cfg)
        insert_run_result(conn, failed)
        insert_sub_result(conn, sub)

    with get_readonly_connection(db_path) as conn:
        rr = conn.execute(
            "SELECT output_diagram_code, raw_response FROM run_results"
        ).fetchone()
        sr = conn.execute(
            "SELECT output_text, raw_response FROM sub_results"
        ).fetchone()
    assert rr[0] is None and rr[1] == "{ malformed json the model emitted"
    assert sr[0] is None and sr[1] == "{ bad json from step 1"
