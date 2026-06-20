"""
MAESTRO DB queries
Insert and fetch operations for RunConfig and RunResult.
"""

from __future__ import annotations

import sqlite3

from maestro.schemas import (
    MetricResult,
    RunConfig,
    RunEnvironment,
    RunResult,
    SubResult,
)


def insert_run_environment(conn: sqlite3.Connection, env: RunEnvironment) -> None:
    """Persist a RunEnvironment row; raises if environment_id already exists."""
    conn.execute(
        """
        INSERT INTO run_environments
            (environment_id, os, arch, python, hostname,
             git_commit, git_dirty, lib_versions, docker_image_digest,
             captured_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(env.environment_id),
            env.os,
            env.arch,
            env.python,
            env.hostname,
            env.git_commit,
            # SQLite has no bool; coerce explicitly so None stays None.
            None if env.git_dirty is None else int(env.git_dirty),
            env.lib_versions,
            env.docker_image_digest,
            env.captured_at.isoformat(),
        ),
    )


def insert_run_config(conn: sqlite3.Connection, config: RunConfig) -> None:
    """Persist a RunConfig row; raises if run_id already exists."""
    conn.execute(
        """
        INSERT INTO run_configs
            (run_id, strategy, model, example_id, tier, run_number,
             timestamp, environment_id)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(config.run_id),
            config.strategy.value,
            config.model,
            config.example_id,
            config.tier.value,
            config.run_number,
            config.timestamp.isoformat(),
            str(config.environment_id) if config.environment_id is not None else None,
        ),
    )


def insert_run_result(conn: sqlite3.Connection, result: RunResult) -> None:
    """Persist a RunResult row; raises if run_id already exists."""
    conn.execute(
        """
        INSERT INTO run_results
            (run_id, output_diagram_code, raw_response, prompt_tokens,
             completion_tokens, duration_ms, cost_usd, error, retry_count)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(result.run_id),
            result.output_diagram_code,
            result.raw_response,
            result.prompt_tokens,
            result.completion_tokens,
            result.duration_ms,
            result.cost_usd,
            result.error,
            result.retry_count,
        ),
    )


def insert_sub_result(conn: sqlite3.Connection, sub: SubResult) -> None:
    """Persist one sub-call result from a multi-step strategy."""
    conn.execute(
        """
        INSERT INTO sub_results
            (sub_id, run_id, step_number, step_name, output_text, raw_response,
             prompt_tokens, completion_tokens, duration_ms, cost_usd,
             error, retry_count)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(sub.sub_id),
            str(sub.run_id),
            sub.step_number,
            sub.step_name,
            sub.output_text,
            sub.raw_response,
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


def fetch_completed_cells(conn: sqlite3.Connection) -> set[tuple[str, str, str, int]]:
    """
    Return the set of (example_id, strategy, model, run_number) tuples
    that already have a *successful* RunResult persisted.

    "Successful" mirrors ``RunResult.success`` (schemas.py): no error,
    non-empty output_diagram_code. A row with ``error IS NOT NULL`` or
    an empty/NULL diagram is *not* in this set, so resume logic will
    re-execute those cells, giving transient failures another attempt.

    Used by ``build_matrix`` to skip already-done work on resume.
    """
    rows = conn.execute(
        """
        SELECT c.example_id, c.strategy, c.model, c.run_number
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        WHERE r.error IS NULL
          AND r.output_diagram_code IS NOT NULL
          AND TRIM(r.output_diagram_code) != ''
        """
    ).fetchall()
    return {(row[0], row[1], row[2], row[3]) for row in rows}


def fetch_failed_cells(conn: sqlite3.Connection) -> set[tuple[str, str, str, int]]:
    """
    Return the set of (example_id, strategy, model, run_number) tuples
    whose persisted RunResults are *all* failures, i.e. there is at
    least one failed attempt AND no successful attempt for that cell.

    Why this matters: ``run_configs`` has no unique constraint on the
    cell tuple, so the same cell can have multiple rows (e.g. an
    initial failure followed by a successful ``--rerun-failed``
    retry). If we returned every cell with any failure, the next
    ``--rerun-failed`` invocation would re-execute cells that have
    *already* been recovered, wasting API spend and overwriting
    presumably-good metric rows.

    Used by ``--rerun-failed`` to narrow the matrix to only cells
    that still need fixing.
    """
    rows = conn.execute(
        """
        SELECT c.example_id, c.strategy, c.model, c.run_number
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        WHERE (
            r.error IS NOT NULL
            OR r.output_diagram_code IS NULL
            OR TRIM(r.output_diagram_code) = ''
        )
        AND NOT EXISTS (
            -- Exclude cells that also have *any* successful attempt;
            -- they have effectively been recovered and shouldn't be
            -- re-run by --rerun-failed.
            SELECT 1
            FROM run_configs c2
            JOIN run_results r2 ON c2.run_id = r2.run_id
            WHERE c2.example_id = c.example_id
              AND c2.strategy   = c.strategy
              AND c2.model      = c.model
              AND c2.run_number = c.run_number
              AND r2.error IS NULL
              AND r2.output_diagram_code IS NOT NULL
              AND TRIM(r2.output_diagram_code) != ''
        )
        """
    ).fetchall()
    return {(row[0], row[1], row[2], row[3]) for row in rows}


def fetch_all_results(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Fetch all joined rows; used by the analysis script."""
    return conn.execute(
        """
        SELECT c.*, r.*
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        ORDER BY c.timestamp
        """,
    ).fetchall()


def fetch_analysis_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Three-way join (run_configs ⋈ run_results ⋈ metric_results) yielding
    one row per metriced run, for the statistical analysis pipeline.

    Read-only. Columns are listed *explicitly* rather than via ``c.*, r.*,
    m.*`` on purpose: all three tables carry a ``run_id`` column, and a
    star-join would emit it three times: sqlite3.Row keeps only the last
    value under that key, silently shadowing the others. The other ``fetch_*``
    helpers use star-joins and tolerate the collision because they don't read
    the duplicated keys; this one selects every column it needs by name so the
    resulting Row maps cleanly to a DataFrame with no ambiguous columns.

    Only the columns the analysis consumes are selected: the experiment
    dimensions (strategy, model, tier, example_id, run_number), the
    efficiency DVs (cost_usd, duration_ms, retry_count), the correctness
    F1 family, and the error-taxonomy counts. ``run_id`` is included once
    for traceability. ``parses_valid`` rides along for structural-validity
    breakdowns.

    An INNER join means runs without a metric row (e.g. a failed run that
    never got scored) are excluded: analysis operates on scored runs only.
    """
    return conn.execute(
        """
        SELECT
            c.run_id        AS run_id,
            c.strategy      AS strategy,
            c.model         AS model,
            c.example_id    AS example_id,
            c.tier          AS tier,
            c.run_number    AS run_number,
            r.cost_usd      AS cost_usd,
            r.duration_ms   AS duration_ms,
            r.retry_count   AS retry_count,
            m.parses_valid  AS parses_valid,
            m.entity_id_f1            AS entity_id_f1,
            m.entity_name_f1          AS entity_name_f1,
            m.entity_lemma_f1         AS entity_lemma_f1,
            m.relationship_relaxed_f1 AS relationship_relaxed_f1,
            m.relationship_strict_f1  AS relationship_strict_f1,
            m.missing_entities        AS missing_entities,
            m.extra_entities          AS extra_entities,
            m.false_entities          AS false_entities,
            m.duplicate_entities      AS duplicate_entities,
            m.missing_relationships   AS missing_relationships,
            m.extra_relationships     AS extra_relationships,
            m.false_relationships     AS false_relationships,
            m.duplicate_relationships AS duplicate_relationships
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        JOIN metric_results m ON c.run_id = m.run_id
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
             relationship_relaxed_precision,
             relationship_relaxed_recall,
             relationship_relaxed_f1,
             relationship_strict_precision,
             relationship_strict_recall,
             relationship_strict_f1,
             entities_in_output, entities_in_truth,
             relationships_in_output, relationships_in_truth,
             missing_entities, extra_entities,
             false_entities, duplicate_entities,
             missing_relationships, extra_relationships,
             false_relationships, duplicate_relationships,
             container_id_precision, container_id_recall, container_id_f1,
             container_name_precision, container_name_recall, container_name_f1,
             containers_in_output, containers_in_truth,
             attachment_precision, attachment_recall, attachment_f1,
             attachments_in_output, attachments_in_truth)
        -- Placeholder groupings mirror the column groupings above so a
        -- visual scan can spot any bind-parameter misalignment:
        --   4 (ids+parse) · 3 (id) · 3 (name) · 3 (lemma) ·
        --   3 (rel relaxed) · 3 (rel strict) ·
        --   2 (ent counts) · 2 (rel counts) ·
        --   2 (missing/extra ent) · 2 (false/dup ent) ·
        --   2 (missing/extra rel) · 2 (false/dup rel) ·
        --   3 (container id) · 3 (container name) · 2 (container counts) ·
        --   3 (attachment) · 2 (attachment counts)  = 44
        VALUES
            (?, ?, ?, ?,
             ?, ?, ?,
             ?, ?, ?,
             ?, ?, ?,
             ?, ?, ?,
             ?, ?, ?,
             ?, ?,
             ?, ?,
             ?, ?,
             ?, ?,
             ?, ?,
             ?, ?,
             ?, ?, ?,
             ?, ?, ?,
             ?, ?,
             ?, ?, ?,
             ?, ?)
        """,
        (
            str(metric.metric_id),
            str(metric.run_id),
            int(metric.parses_valid) if metric.parses_valid is not None else None,
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
            metric.container_id_precision,
            metric.container_id_recall,
            metric.container_id_f1,
            metric.container_name_precision,
            metric.container_name_recall,
            metric.container_name_f1,
            metric.containers_in_output,
            metric.containers_in_truth,
            metric.attachment_precision,
            metric.attachment_recall,
            metric.attachment_f1,
            metric.attachments_in_output,
            metric.attachments_in_truth,
        ),
    )
