"""
Reported-numbers dump: aggregate arithmetic and empty-DB behaviour.

The whole point of this module is that the values docs quote are computed
from a fresh DB read, not maintained by hand. So the tests pin both
paths: an empty DB (fresh checkout) yields a valid empty-status payload,
and a populated DB reports the same totals the runner prints on exit.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from maestro.analysis.reported_numbers import (
    SCHEMA_VERSION,
    compute_reported_numbers,
)
from maestro.db.client import get_connection, init_db
from maestro.schemas import RunConfig, RunResult, Strategy, Tier


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Fresh SQLite database with the MAESTRO schema, for one test."""
    path = tmp_path / "reported.db"
    init_db(path)
    return path


def _insert_result(
    conn,
    *,
    strategy: Strategy,
    example_id: str,
    output: str | None,
    cost_usd: float,
    error: str | None,
) -> None:
    """Persist a RunConfig / RunResult pair so the aggregate query can see it."""
    from maestro.db.queries import insert_run_config, insert_run_result

    config = RunConfig(
        run_id=uuid4(),
        strategy=strategy,
        model="claude-opus-4-8",
        example_id=example_id,
        tier=Tier.SIMPLE,
        run_number=1,
    )
    insert_run_config(conn, config)
    insert_run_result(
        conn,
        RunResult(
            run_id=config.run_id,
            output_diagram_code=output,
            prompt_tokens=1,
            completion_tokens=1,
            duration_ms=1,
            cost_usd=cost_usd,
            error=error,
        ),
    )


def test_empty_db_returns_zeroed_payload(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        payload = compute_reported_numbers(conn)

    assert payload.schema_version == SCHEMA_VERSION
    assert payload.status == "empty"
    assert payload.total_runs == 0
    assert payload.successes == 0
    assert payload.failures == 0
    assert payload.total_cost_usd == 0.0


def test_populated_db_reports_split_and_cost(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        _insert_result(
            conn,
            strategy=Strategy.SINGLE_AGENT,
            example_id="bpmn_1_01",
            output="graph TD; a-->b",
            cost_usd=0.01,
            error=None,
        )
        _insert_result(
            conn,
            strategy=Strategy.SOP_BASED,
            example_id="bpmn_1_02",
            output="graph TD; c-->d",
            cost_usd=0.02,
            error=None,
        )
        _insert_result(
            conn,
            strategy=Strategy.CREW_AI,
            example_id="bpmn_1_03",
            output=None,
            cost_usd=0.0,
            error="RateLimitError: 429",
        )

    with get_connection(db_path) as conn:
        payload = compute_reported_numbers(conn)

    assert payload.status == "ok"
    assert payload.total_runs == 3
    assert payload.successes == 2
    assert payload.failures == 1
    # Cost is rounded to two decimals; matches how docs quote it.
    assert payload.total_cost_usd == 0.03


def test_empty_output_counts_as_failure(db_path: Path) -> None:
    """A run with an empty-string diagram is not a valid success, so the
    aggregate must not double-count it as one."""
    with get_connection(db_path) as conn:
        _insert_result(
            conn,
            strategy=Strategy.SINGLE_AGENT,
            example_id="bpmn_1_01",
            output="   ",
            cost_usd=0.0,
            error=None,
        )

    with get_connection(db_path) as conn:
        payload = compute_reported_numbers(conn)

    assert payload.total_runs == 1
    assert payload.successes == 0
    assert payload.failures == 1
