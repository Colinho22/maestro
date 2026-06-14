"""
MAESTRO viz — read-only queries specific to the dashboard.

This module *extends* ``maestro.db.queries`` rather than rewriting it: the
experiment-side query layer there is the source of truth for the joins the
analysis pipeline relies on. Here we add only the lightweight reads the
dashboard needs — primarily "is there any data to show?" checks that drive
empty-state handling. Per-view queries live alongside their respective view
modules.

Every function takes an already-opened read-only ``sqlite3.Connection`` (see
``viz.db.connect``) and only reads.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from maestro.experiment_config import CONTROL_STRATEGIES

# SQL fragment matching a "successful" run, mirroring RunResult.success
# (schemas.py): no error and a non-empty output diagram. Used wherever a
# success/failure split is needed.
_SUCCESS_SQL = (
    "(r.error IS NULL AND r.output_diagram_code IS NOT NULL "
    "AND TRIM(r.output_diagram_code) != '')"
)

# A valid SQLite identifier for our purposes: a bare table name. Used to
# reject anything that isn't a plain identifier before it reaches an
# interpolated query (table names can't be bound as parameters).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# String values of the control strategies, for excluding them from
# strategy-comparison queries (controls are reference floors/ceiling, not
# orchestration strategies under test).
_CONTROL_VALUES: tuple[str, ...] = tuple(s.value for s in CONTROL_STRATEGIES)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """
    Whether ``table`` exists in the database. Used to degrade gracefully when
    pointed at a brand-new or unmigrated DB that lacks the expected schema,
    rather than letting a query raise ``OperationalError``.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """
    Row count for ``table``, or 0 if the table is absent. The absent-table
    case returns 0 (not an error) so callers can treat "no table" and "empty
    table" identically for empty-state purposes.

    A table name cannot be passed as a bound parameter, so it is interpolated
    into the query. ``table`` is validated against a strict identifier
    pattern first (and rejected with ``ValueError`` otherwise) so this can
    never become an injection vector even if a caller ever passes untrusted
    input.
    """
    if not _IDENTIFIER_RE.match(table):
        raise ValueError(f"invalid table identifier: {table!r}")
    if not table_exists(conn, table):
        return 0
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def has_any_runs(conn: sqlite3.Connection) -> bool:
    """True if at least one experiment run has been recorded."""
    return count_rows(conn, "run_configs") > 0


def has_any_metrics(conn: sqlite3.Connection) -> bool:
    """True if at least one run has been scored (metric_results populated)."""
    return count_rows(conn, "metric_results") > 0


def mean_entity_id_f1_by_strategy(
    conn: sqlite3.Connection,
) -> list[tuple[str, float]]:
    """
    Mean ``entity_id_f1`` per orchestration strategy across all scored runs,
    as ``(strategy, mean_f1)`` pairs ordered by strategy name.

    Control strategies are excluded — they are reference floors/ceiling, not
    orchestration strategies under test, and would otherwise show as gray bars
    alongside the real strategies. Returns an empty list when there are no
    metric rows (callers show an empty-state). Joins run_configs to
    metric_results on run_id.
    """
    if not (table_exists(conn, "metric_results") and table_exists(conn, "run_configs")):
        return []
    placeholders = ",".join("?" * len(_CONTROL_VALUES))
    rows = conn.execute(
        f"""
        SELECT c.strategy AS strategy, AVG(m.entity_id_f1) AS mean_f1
        FROM run_configs c
        JOIN metric_results m ON c.run_id = m.run_id
        WHERE c.strategy NOT IN ({placeholders})
        GROUP BY c.strategy
        ORDER BY c.strategy
        """,
        _CONTROL_VALUES,
    ).fetchall()
    return [(r["strategy"], float(r["mean_f1"])) for r in rows]


# ---------------------------------------------------------------------------
# Overview view
# ---------------------------------------------------------------------------


def overview_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Headline operational counts for the Overview cards: total runs, successful
    runs, success rate, total cost (USD), and distinct environments. Returns
    zeros on an empty database.
    """
    if not (table_exists(conn, "run_configs") and table_exists(conn, "run_results")):
        return {
            "total_runs": 0,
            "successful_runs": 0,
            "success_rate": 0.0,
            "total_cost_usd": 0.0,
            "distinct_environments": 0,
        }
    row = conn.execute(
        f"""
        SELECT
            COUNT(*)                           AS total_runs,
            SUM(CASE WHEN {_SUCCESS_SQL} THEN 1 ELSE 0 END) AS successful_runs,
            COALESCE(SUM(r.cost_usd), 0.0)     AS total_cost_usd,
            COUNT(DISTINCT c.environment_id)   AS distinct_environments
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        """
    ).fetchone()
    total = row["total_runs"] or 0
    successful = row["successful_runs"] or 0
    return {
        "total_runs": total,
        "successful_runs": successful,
        "success_rate": (successful / total) if total else 0.0,
        "total_cost_usd": float(row["total_cost_usd"] or 0.0),
        "distinct_environments": row["distinct_environments"] or 0,
    }


def runs_by_strategy_success(
    conn: sqlite3.Connection,
) -> list[tuple[str, int, int]]:
    """
    Per strategy: ``(strategy, n_success, n_failure)`` across all runs.
    Includes every strategy present (controls included — this is an
    operational summary, not a strategy comparison). Empty list if no runs.
    """
    if not (table_exists(conn, "run_configs") and table_exists(conn, "run_results")):
        return []
    rows = conn.execute(
        f"""
        SELECT
            c.strategy AS strategy,
            SUM(CASE WHEN {_SUCCESS_SQL} THEN 1 ELSE 0 END)     AS n_success,
            SUM(CASE WHEN {_SUCCESS_SQL} THEN 0 ELSE 1 END)     AS n_failure
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        GROUP BY c.strategy
        ORDER BY c.strategy
        """
    ).fetchall()
    return [(r["strategy"], r["n_success"], r["n_failure"]) for r in rows]


def total_cost_by_strategy(conn: sqlite3.Connection) -> list[tuple[str, float]]:
    """Per strategy total cost (USD), all strategies. Empty list if no runs."""
    if not (table_exists(conn, "run_configs") and table_exists(conn, "run_results")):
        return []
    rows = conn.execute(
        """
        SELECT c.strategy AS strategy, COALESCE(SUM(r.cost_usd), 0.0) AS cost
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        GROUP BY c.strategy
        ORDER BY c.strategy
        """
    ).fetchall()
    return [(r["strategy"], float(r["cost"])) for r in rows]


# ---------------------------------------------------------------------------
# Strategy Comparison view
# ---------------------------------------------------------------------------

# Metric families the Strategy Comparison view can show, mapped to their
# column prefixes. The view picks precision / recall / f1 as the suffix.
ENTITY_METRICS = ("entity_id", "entity_name", "entity_lemma")
RELATIONSHIP_METRICS = ("relationship_relaxed", "relationship_strict")


def distinct_tiers(conn: sqlite3.Connection) -> list[int]:
    """Distinct tier values present in run_configs, ascending."""
    if not table_exists(conn, "run_configs"):
        return []
    rows = conn.execute(
        "SELECT DISTINCT tier FROM run_configs ORDER BY tier"
    ).fetchall()
    return [int(r["tier"]) for r in rows]


def distinct_models(conn: sqlite3.Connection) -> list[str]:
    """Distinct model ids present in run_configs (controls' 'control' excluded)."""
    if not table_exists(conn, "run_configs"):
        return []
    rows = conn.execute(
        "SELECT DISTINCT model FROM run_configs WHERE model != 'control' ORDER BY model"
    ).fetchall()
    return [r["model"] for r in rows]


def metric_means_by_strategy(
    conn: sqlite3.Connection,
    metric_columns: list[str],
    *,
    tier: int | None = None,
    models: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Mean of each column in ``metric_columns`` per strategy, optionally filtered
    to a tier and/or a set of models. Controls excluded.

    Returns ``{strategy: {metric_column: mean}}``. ``metric_columns`` are
    validated as identifiers (they are interpolated, not bindable). Empty dict
    if no rows match.
    """
    for col in metric_columns:
        if not _IDENTIFIER_RE.match(col):
            raise ValueError(f"invalid metric column: {col!r}")
    if not (table_exists(conn, "metric_results") and table_exists(conn, "run_configs")):
        return {}

    where = [f"c.strategy NOT IN ({','.join('?' * len(_CONTROL_VALUES))})"]
    params: list[Any] = list(_CONTROL_VALUES)
    if tier is not None:
        where.append("c.tier = ?")
        params.append(tier)
    if models:
        where.append(f"c.model IN ({','.join('?' * len(models))})")
        params.extend(models)

    avg_cols = ", ".join(f'AVG(m."{c}") AS "{c}"' for c in metric_columns)
    rows = conn.execute(
        f"""
        SELECT c.strategy AS strategy, {avg_cols}
        FROM run_configs c
        JOIN metric_results m ON c.run_id = m.run_id
        WHERE {" AND ".join(where)}
        GROUP BY c.strategy
        ORDER BY c.strategy
        """,
        params,
    ).fetchall()
    return {
        r["strategy"]: {c: float(r[c]) for c in metric_columns if r[c] is not None}
        for r in rows
    }


# ---------------------------------------------------------------------------
# Pareto view
# ---------------------------------------------------------------------------


def pareto_points(
    conn: sqlite3.Connection,
    *,
    strategies: list[str] | None = None,
    tiers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    One row per scored run with the fields the Pareto view plots and tabulates:
    run_id, strategy, model, tier, cost_usd, duration_ms, entity_id_f1.
    Controls excluded. Optional strategy/tier filters. Empty list if none.
    """
    needed = ("run_configs", "run_results", "metric_results")
    if not all(table_exists(conn, t) for t in needed):
        return []

    where = [f"c.strategy NOT IN ({','.join('?' * len(_CONTROL_VALUES))})"]
    params: list[Any] = list(_CONTROL_VALUES)
    if strategies:
        where.append(f"c.strategy IN ({','.join('?' * len(strategies))})")
        params.extend(strategies)
    if tiers:
        where.append(f"c.tier IN ({','.join('?' * len(tiers))})")
        params.extend(tiers)

    rows = conn.execute(
        f"""
        SELECT
            c.run_id AS run_id, c.strategy AS strategy, c.model AS model,
            c.tier AS tier, r.cost_usd AS cost_usd, r.duration_ms AS duration_ms,
            m.entity_id_f1 AS entity_id_f1
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        JOIN metric_results m ON c.run_id = m.run_id
        WHERE {" AND ".join(where)}
        ORDER BY c.strategy, c.model
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def distinct_strategies(
    conn: sqlite3.Connection, *, exclude_controls: bool = True
) -> list[str]:
    """Distinct strategy values present in run_configs, ascending."""
    if not table_exists(conn, "run_configs"):
        return []
    if exclude_controls:
        rows = conn.execute(
            f"SELECT DISTINCT strategy FROM run_configs "
            f"WHERE strategy NOT IN ({','.join('?' * len(_CONTROL_VALUES))}) "
            f"ORDER BY strategy",
            _CONTROL_VALUES,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT strategy FROM run_configs ORDER BY strategy"
        ).fetchall()
    return [r["strategy"] for r in rows]


# ---------------------------------------------------------------------------
# Run Detail view
# ---------------------------------------------------------------------------


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Every run as a selectable entry: run_id, strategy, model, tier, example_id,
    run_number, timestamp. Most recent first. Empty list if no runs.

    Returns all runs, controls included; each view filters to the subset it
    needs (the run selectors drop controls via the faceted filter, since a
    control produces no model-generated diagram to inspect). run_number is
    included so selectors can distinguish repeats of the same cell, and so the
    faceted filter can offer a run-number facet (see viz.run_filter).
    """
    if not table_exists(conn, "run_configs"):
        return []
    rows = conn.execute(
        """
        SELECT run_id, strategy, model, tier, example_id, run_number, timestamp
        FROM run_configs
        ORDER BY timestamp DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def run_detail(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """
    Full detail for one run: the config, result (output diagram, tokens, cost,
    error), and metric scores, as a flat dict. None if the run_id is unknown.
    """
    if not (table_exists(conn, "run_configs") and table_exists(conn, "run_results")):
        return None
    # Columns listed explicitly (not c.*, r.*, m.*): all three tables carry a
    # run_id, and a star-join would emit it three times, with sqlite3.Row
    # silently keeping only the last. Naming the consumed columns avoids the
    # collision and documents exactly what the Run Detail / Diagram Visualizer
    # views read.
    row = conn.execute(
        """
        SELECT
            c.run_id AS run_id, c.strategy AS strategy, c.model AS model,
            c.tier AS tier, c.example_id AS example_id, c.timestamp AS timestamp,
            r.output_diagram_code AS output_diagram_code,
            r.prompt_tokens AS prompt_tokens,
            r.completion_tokens AS completion_tokens,
            r.duration_ms AS duration_ms, r.cost_usd AS cost_usd,
            r.error AS error, r.retry_count AS retry_count,
            m.parses_valid AS parses_valid,
            m.entity_id_f1 AS entity_id_f1,
            m.entity_name_f1 AS entity_name_f1,
            m.entity_lemma_f1 AS entity_lemma_f1,
            m.relationship_relaxed_f1 AS relationship_relaxed_f1,
            m.relationship_strict_f1 AS relationship_strict_f1
        FROM run_configs c
        JOIN run_results r ON c.run_id = r.run_id
        LEFT JOIN metric_results m ON c.run_id = m.run_id
        WHERE c.run_id = ?
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def sub_results_for_run(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Sub-call trace for a run, ordered by step. Empty if none / no table."""
    if not table_exists(conn, "sub_results"):
        return []
    rows = conn.execute(
        """
        SELECT step_number, step_name, output_text, prompt_tokens,
               completion_tokens, duration_ms, cost_usd, error, retry_count
        FROM sub_results
        WHERE run_id = ?
        ORDER BY step_number
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Hallucination Taxonomy view
# ---------------------------------------------------------------------------

ENTITY_TAXONOMY = (
    "missing_entities",
    "extra_entities",
    "false_entities",
    "duplicate_entities",
)
RELATIONSHIP_TAXONOMY = (
    "missing_relationships",
    "extra_relationships",
    "false_relationships",
    "duplicate_relationships",
)


def has_any_taxonomy_data(conn: sqlite3.Connection) -> bool:
    """
    True if any error-taxonomy count is non-zero across all scored runs. Drives
    the hallucination view's gating empty-state (all-zero ⇒ nothing to show).
    """
    if not table_exists(conn, "metric_results"):
        return False
    cols = ENTITY_TAXONOMY + RELATIONSHIP_TAXONOMY
    total = " + ".join(f"COALESCE(SUM({c}), 0)" for c in cols)
    row = conn.execute(
        f"SELECT ({total}) AS grand_total FROM metric_results"
    ).fetchone()
    return bool(row["grand_total"])


def taxonomy_counts_by_strategy(
    conn: sqlite3.Connection,
    columns: tuple[str, ...],
    *,
    tier: int | None = None,
) -> dict[str, dict[str, int]]:
    """
    Summed taxonomy counts per strategy for the given taxonomy ``columns``
    (entity or relationship set), optionally filtered to a tier. Controls
    included — their error profile is itself informative. Returns
    ``{strategy: {column: total}}``.
    """
    for col in columns:
        if not _IDENTIFIER_RE.match(col):
            raise ValueError(f"invalid taxonomy column: {col!r}")
    if not (table_exists(conn, "metric_results") and table_exists(conn, "run_configs")):
        return {}

    where = []
    params: list[Any] = []
    if tier is not None:
        where.append("c.tier = ?")
        params.append(tier)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    sum_cols = ", ".join(f'SUM(m."{c}") AS "{c}"' for c in columns)
    rows = conn.execute(
        f"""
        SELECT c.strategy AS strategy, {sum_cols}
        FROM run_configs c
        JOIN metric_results m ON c.run_id = m.run_id
        {where_sql}
        GROUP BY c.strategy
        ORDER BY c.strategy
        """,
        params,
    ).fetchall()
    return {r["strategy"]: {c: int(r[c] or 0) for c in columns} for r in rows}
