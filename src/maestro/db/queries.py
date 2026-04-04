"""
MAESTRO — DB queries
Insert and fetch operations for RunConfig and RunResult.
"""

import sqlite3

from maestro.schemas import RunConfig, RunResult, SubResult


def insert_run_config(conn: sqlite3.Connection, config: RunConfig) -> None:
    """Persist a RunConfig row — raises if run_id already exists."""
    conn.execute(
        """
        INSERT INTO run_configs
            (run_id, strategy, model, example_id, tier, run_number, timestamp)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(config.run_id),
            config.strategy.value,
            config.model,
            config.example_id,
            config.tier.value,
            config.run_number,
            config.timestamp.isoformat(),
        ),
    )


def insert_run_result(conn: sqlite3.Connection, result: RunResult) -> None:
    """Persist a RunResult row — raises if run_id already exists."""
    conn.execute(
        """
        INSERT INTO run_results
            (run_id, output_diagram_code, prompt_tokens, completion_tokens,
             duration_ms, cost_usd, error)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(result.run_id),
            result.output_diagram_code,
            result.prompt_tokens,
            result.completion_tokens,
            result.duration_ms,
            result.cost_usd,
            result.error,
        ),
    )

def insert_sub_result(conn: sqlite3.Connection, sub: SubResult) -> None:
    """Persist one sub-call result from a multi-step strategy."""
    conn.execute(
        """
        INSERT INTO sub_results
            (sub_id, run_id, step_number, step_name, output_text,
             prompt_tokens, completion_tokens, duration_ms, cost_usd,
             error, retry_count)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(sub.sub_id),
            str(sub.run_id),
            sub.step_number,
            sub.step_name,
            sub.output_text,
            sub.prompt_tokens,
            sub.completion_tokens,
            sub.duration_ms,
            sub.cost_usd,
            sub.error,
            sub.retry_count,
        ),
    )


def fetch_sub_results_by_run(
    conn: sqlite3.Connection, run_id: str
) -> list[sqlite3.Row]:
    """Fetch all sub-call results for a given parent run."""
    return conn.execute(
        """
        SELECT * FROM sub_results
        WHERE run_id = ?
        ORDER BY step_number
        """,
        (run_id,),
    ).fetchall()

def fetch_results_by_strategy(
    conn: sqlite3.Connection, strategy: str
) -> list[sqlite3.Row]:
    """Fetch all joined run_config + run_result rows for a given strategy."""
    return conn.execute(
        """
        SELECT c.*, r.*
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        WHERE c.strategy = ?
        ORDER BY c.timestamp
        """,
        (strategy,),
    ).fetchall()


def fetch_all_results(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Fetch all joined rows — used by the analysis script."""
    return conn.execute(
        """
        SELECT c.*, r.*
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        ORDER BY c.timestamp
        """,
    ).fetchall()