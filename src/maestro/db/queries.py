"""
MAESTRO — DB queries
Insert and fetch operations for RunConfig and RunResult.
"""

import sqlite3

from maestro.schemas import RunConfig, RunResult


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