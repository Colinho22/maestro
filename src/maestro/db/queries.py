"""
MAESTRO — DB queries
Insert and fetch operations for RunConfig and RunResult.
"""

import sqlite3

from maestro.schemas import RunConfig, RunResult, SubResult, MetricResult


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

def insert_metric_result(conn: sqlite3.Connection, metric: MetricResult) -> None:
    """Persist evaluation metrics for one run."""
    conn.execute(
        """
        INSERT INTO metric_results
            (metric_id, run_id, parses_valid, parse_error,
             entity_id_precision, entity_id_recall, entity_id_f1,
             entity_name_precision, entity_name_recall, entity_name_f1,
             entity_lemma_precision, entity_lemma_recall, entity_lemma_f1,
             relationship_relaxed_precision, relationship_relaxed_recall, relationship_relaxed_f1,
             relationship_strict_precision, relationship_strict_recall, relationship_strict_f1,
             entities_in_output, entities_in_truth,
             relationships_in_output, relationships_in_truth,
             missing_entities, extra_entities, false_entities, duplicate_entities,
             missing_relationships, extra_relationships, false_relationships, duplicate_relationships)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(metric.metric_id),
            str(metric.run_id),
            int(metric.parses_valid),
            metric.parse_error,
            metric.entity_id_precision,
            metric.entity_id_recall,
            metric.entity_id_f1,
            metric.entity_name_precision,
            metric.entity_name_recall,
            metric.entity_name_f1,
            metric.entity_lemma_precision,
            metric.entity_lemma_recall,
            metric.entity_lemma_f1,
            metric.relationship_relaxed_precision,
            metric.relationship_relaxed_recall,
            metric.relationship_relaxed_f1,
            metric.relationship_strict_precision,
            metric.relationship_strict_recall,
            metric.relationship_strict_f1,
            metric.entities_in_output,
            metric.entities_in_truth,
            metric.relationships_in_output,
            metric.relationships_in_truth,
            metric.missing_entities,
            metric.extra_entities,
            metric.false_entities,
            metric.duplicate_entities,
            metric.missing_relationships,
            metric.extra_relationships,
            metric.false_relationships,
            metric.duplicate_relationships,
        ),
    )