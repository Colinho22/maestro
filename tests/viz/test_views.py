"""
Tests for the five data-view query functions, against a synthetic in-memory
database with multiple strategies, models, tiers, and control rows.

The view modules themselves are thin Streamlit wrappers (selectboxes, columns,
st.pyplot) that need a script context to exercise; their *logic* lives in the
queries here and in the design-system chart/theme code already covered by
test_theme.py. These tests pin the query behavior every view depends on,
especially control exclusion, filtering, and graceful degradation on missing
tables, using the real schema and the production insert helpers.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")

from maestro.db.client import SCHEMA  # noqa: E402
from maestro.db.queries import insert_metric_result  # noqa: E402
from maestro.schemas import MetricResult  # noqa: E402
from maestro.viz import queries as q  # noqa: E402

# View registry import check lives here too.
from maestro.viz.views import VIEWS  # noqa: E402


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _insert_run(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    model: str,
    tier: int,
    f1: float,
    cost: float = 0.001,
    duration_ms: int = 100,
    success: bool = True,
    missing_entities: int = 0,
    extra_entities: int = 0,
) -> str:
    """Insert a full config+result+metric triple; return the run_id string."""
    run_id = uuid.uuid4()
    rid = str(run_id)
    conn.execute(
        "INSERT INTO run_configs "
        "(run_id, strategy, model, example_id, tier, run_number, timestamp) "
        "VALUES (?, ?, ?, 'ex_01', ?, 1, '2026-01-01T00:00:00Z')",
        (rid, strategy, model, tier),
    )
    conn.execute(
        "INSERT INTO run_results "
        "(run_id, output_diagram_code, prompt_tokens, completion_tokens, "
        " duration_ms, cost_usd, error, retry_count) "
        "VALUES (?, ?, 10, 10, ?, ?, ?, 0)",
        (
            rid,
            "graph TD; a-->b" if success else None,
            duration_ms,
            cost,
            None if success else "boom",
        ),
    )
    insert_metric_result(
        conn,
        MetricResult(
            run_id=run_id,
            parses_valid=True,
            entity_id_precision=f1,
            entity_id_recall=f1,
            entity_id_f1=f1,
            entity_name_precision=0.0,
            entity_name_recall=0.0,
            entity_name_f1=0.0,
            entity_lemma_precision=0.0,
            entity_lemma_recall=0.0,
            entity_lemma_f1=0.0,
            relationship_relaxed_precision=0.0,
            relationship_relaxed_recall=0.0,
            relationship_relaxed_f1=0.0,
            relationship_strict_precision=0.0,
            relationship_strict_recall=0.0,
            relationship_strict_f1=0.0,
            entities_in_output=0,
            entities_in_truth=0,
            relationships_in_output=0,
            relationships_in_truth=0,
            missing_entities=missing_entities,
            extra_entities=extra_entities,
            false_entities=0,
            duplicate_entities=0,
            missing_relationships=0,
            extra_relationships=0,
            false_relationships=0,
            duplicate_relationships=0,
        ),
    )
    return rid


def _populate(conn: sqlite3.Connection) -> None:
    """Two LLM strategies across two tiers, plus a control row."""
    _insert_run(
        conn, strategy="single_agent", model="gpt-4o-mini-2024-07-18", tier=1, f1=0.6
    )
    _insert_run(
        conn, strategy="single_agent", model="gpt-4o-mini-2024-07-18", tier=2, f1=0.7
    )
    _insert_run(
        conn,
        strategy="crew_ai",
        model="mistral-small-2603",
        tier=1,
        f1=0.8,
        missing_entities=2,
        extra_entities=1,
    )
    _insert_run(conn, strategy="crew_ai", model="mistral-small-2603", tier=2, f1=0.9)
    _insert_run(conn, strategy="null_control", model="control", tier=1, f1=0.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_views_registry_has_all_data_views():
    labels = {label for label, _ in VIEWS}
    for expected in (
        "Home",
        "Overview",
        "Strategy Comparison",
        "Pareto",
        "Run Detail",
        "Diagram Visualizer",
        "Hallucination Taxonomy",
    ):
        assert expected in labels
    assert all(callable(fn) for _, fn in VIEWS)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def test_overview_summary_counts():
    conn = _conn()
    _populate(conn)
    s = q.overview_summary(conn)
    assert s["total_runs"] == 5
    assert s["successful_runs"] == 5
    assert s["success_rate"] == 1.0
    assert s["total_cost_usd"] > 0


def test_overview_summary_empty_db():
    s = q.overview_summary(_conn())
    assert s["total_runs"] == 0
    assert s["success_rate"] == 0.0


def test_runs_by_strategy_includes_controls():
    """Overview is operational: controls ARE included here."""
    conn = _conn()
    _populate(conn)
    rows = dict((r[0], (r[1], r[2])) for r in q.runs_by_strategy_success(conn))
    assert "null_control" in rows


def test_run_outcomes_partition_every_run():
    """
    The three funnel outcomes are mutually exclusive and cover every run.

    An errored run has no metric row at all, so it must still be counted (via
    the LEFT JOIN) rather than vanishing from its strategy's total.
    """
    conn = _conn()
    # One valid, one unrenderable, one outright failure: all three outcomes.
    _insert_run(conn, strategy="single_agent", model="m", tier=1, f1=0.6)
    rid = _insert_run(conn, strategy="single_agent", model="m", tier=1, f1=0.0)
    conn.execute("UPDATE metric_results SET parses_valid = 0 WHERE run_id = ?", (rid,))
    failed = _insert_run(
        conn, strategy="single_agent", model="m", tier=1, f1=0.0, success=False
    )
    conn.execute("DELETE FROM metric_results WHERE run_id = ?", (failed,))

    outcomes = {r[0]: r[1:] for r in q.run_outcomes_by_strategy(conn)}
    n_valid, n_invalid, n_error = outcomes["single_agent"]
    assert (n_valid, n_invalid, n_error) == (1, 1, 1)
    # The partition is what makes the funnel's percentages trustworthy.
    assert n_valid + n_invalid + n_error == 3


def test_run_outcomes_empty_db():
    assert q.run_outcomes_by_strategy(_conn()) == []


def test_mean_f1_by_convention_differ_on_failures():
    """
    A failed run drags intent-to-treat below valid-only for its strategy but
    leaves valid-only untouched: the two conventions must diverge exactly by
    how a zero is folded in.
    """
    conn = _conn()
    # single_agent: two valid runs at 0.8, plus one failure (no metric row).
    _insert_run(conn, strategy="single_agent", model="m", tier=1, f1=0.8)
    _insert_run(conn, strategy="single_agent", model="m", tier=1, f1=0.8)
    failed = _insert_run(
        conn, strategy="single_agent", model="m", tier=1, f1=0.0, success=False
    )
    conn.execute("DELETE FROM metric_results WHERE run_id = ?", (failed,))
    # crew_ai: one valid run, no failures, so the two conventions agree.
    _insert_run(conn, strategy="crew_ai", model="m", tier=1, f1=0.6)

    rows = q.mean_entity_id_f1_by_strategy_by_convention(conn)
    by = {r[0]: (r[1], r[2]) for r in rows}
    sa_vo, sa_itt = by["single_agent"]
    assert sa_vo == pytest.approx(0.8)  # only the two parsed runs
    assert sa_itt == pytest.approx(0.8 * 2 / 3)  # the failure counts as 0
    ca_vo, ca_itt = by["crew_ai"]
    assert ca_vo == pytest.approx(ca_itt)  # no failures: conventions agree
    assert "null_control" not in by  # controls excluded


def test_mean_f1_by_convention_all_invalid_strategy():
    """
    A strategy whose every run failed to parse has a NULL valid_only average
    (AVG over an empty set). It must coalesce to 0.0, not raise on float(None).
    """
    conn = _conn()
    rid = _insert_run(conn, strategy="crew_ai", model="m", tier=1, f1=0.0)
    conn.execute("UPDATE metric_results SET parses_valid = 0 WHERE run_id = ?", (rid,))

    rows = q.mean_entity_id_f1_by_strategy_by_convention(conn)
    by = {r[0]: (r[1], r[2]) for r in rows}
    valid_only, intent_to_treat = by["crew_ai"]
    assert valid_only == 0.0  # no parsed run to average; coalesced
    assert intent_to_treat == 0.0  # the one invalid run scores 0


def test_mean_f1_by_convention_empty_db():
    assert q.mean_entity_id_f1_by_strategy_by_convention(_conn()) == []


def test_run_rates_by_tier_are_pooled_fractions():
    """
    valid_rate and fail_rate are run-count fractions of the tier total, over
    experimental runs only. This query is deliberately not an F1 average:
    tier-level correctness uses per-cell means, a different grain.
    """
    conn = _conn()
    # Tier 1: two valid runs, one failure (no metric row).
    _insert_run(conn, strategy="single_agent", model="m", tier=1, f1=0.8)
    _insert_run(conn, strategy="crew_ai", model="m", tier=1, f1=0.8)
    failed = _insert_run(
        conn, strategy="crew_ai", model="m", tier=1, f1=0.0, success=False
    )
    conn.execute("DELETE FROM metric_results WHERE run_id = ?", (failed,))
    # A control at tier 1 must not enter the counts.
    _insert_run(conn, strategy="null_control", model="control", tier=1, f1=0.0)

    by = {r[0]: r for r in q.run_rates_by_tier(conn)}
    tier, n, valid_rate, fail_rate = by[1]
    assert n == 3  # control excluded
    assert valid_rate == pytest.approx(2 / 3)
    assert fail_rate == pytest.approx(1 / 3)


def test_run_rates_by_tier_empty_db():
    assert q.run_rates_by_tier(_conn()) == []


def test_efficiency_averages_include_failed_runs():
    """
    Cost and latency are averaged over every run, failures included: a failed
    run still spent tokens. A metric JOIN would silently drop it and inflate
    the per-run figures. Controls excluded.
    """
    conn = _conn()
    _insert_run(conn, strategy="crew_ai", model="m", tier=1, f1=0.8, cost=0.02)
    failed = _insert_run(
        conn, strategy="crew_ai", model="m", tier=1, f1=0.0, cost=0.04, success=False
    )
    conn.execute("DELETE FROM metric_results WHERE run_id = ?", (failed,))
    _insert_run(conn, strategy="null_control", model="c", tier=1, f1=0.0, cost=0.0)

    by = {r[0]: r for r in q.efficiency_by_strategy(conn)}
    strategy, mean_tokens, mean_cost, mean_latency_s, total_cost = by["crew_ai"]
    # Both runs counted: mean cost = (0.02 + 0.04) / 2, total = 0.06.
    assert mean_cost == pytest.approx(0.03)
    assert total_cost == pytest.approx(0.06)
    assert "null_control" not in by


def test_efficiency_by_strategy_empty_db():
    assert q.efficiency_by_strategy(_conn()) == []


def test_valid_rate_by_model_is_pooled_fraction():
    """Parsed fraction of a model's experimental runs; controls excluded."""
    conn = _conn()
    _insert_run(conn, strategy="crew_ai", model="m_x", tier=1, f1=0.8)
    inv = _insert_run(conn, strategy="crew_ai", model="m_x", tier=1, f1=0.0)
    conn.execute("UPDATE metric_results SET parses_valid = 0 WHERE run_id = ?", (inv,))
    _insert_run(conn, strategy="null_control", model="m_x", tier=1, f1=0.0)

    by = {r[0]: r for r in q.valid_rate_by_model(conn)}
    model, n, valid_rate = by["m_x"]
    assert n == 2  # control excluded
    assert valid_rate == pytest.approx(0.5)  # one of two parsed


def test_valid_rate_by_model_empty_db():
    assert q.valid_rate_by_model(_conn()) == []


def test_taxonomy_rates_average_over_valid_diagrams_only():
    """
    The rate is a mean over parsed diagrams, not all runs, and controls do
    not appear. An unparseable diagram must not drag the average.
    """
    conn = _conn()
    _insert_run(conn, strategy="crew_ai", model="m", tier=1, f1=0.9, missing_entities=2)
    _insert_run(conn, strategy="crew_ai", model="m", tier=1, f1=0.9, missing_entities=4)
    # An unrenderable run with a huge miss count must be excluded, not averaged.
    inv = _insert_run(
        conn, strategy="crew_ai", model="m", tier=1, f1=0.0, missing_entities=99
    )
    conn.execute("UPDATE metric_results SET parses_valid = 0 WHERE run_id = ?", (inv,))
    _insert_run(conn, strategy="null_control", model="c", tier=1, f1=0.0)

    rates = q.taxonomy_rates_per_valid_diagram(conn, ("missing_entities",))
    assert rates["crew_ai"]["missing_entities"] == pytest.approx(3.0)  # (2+4)/2
    assert "null_control" not in rates


def test_taxonomy_rates_rejects_bad_column():
    with pytest.raises(ValueError):
        q.taxonomy_rates_per_valid_diagram(_conn(), ("missing_entities; DROP",))


# ---------------------------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------------------------


def test_metric_means_excludes_controls():
    conn = _conn()
    _populate(conn)
    data = q.metric_means_by_strategy(conn, ["entity_id_f1"])
    assert "null_control" not in data
    assert set(data) == {"single_agent", "crew_ai"}


def test_metric_means_tier_filter():
    conn = _conn()
    _populate(conn)
    tier1 = q.metric_means_by_strategy(conn, ["entity_id_f1"], tier=1)
    # single_agent tier-1 f1 = 0.6, crew_ai tier-1 f1 = 0.8
    assert tier1["single_agent"]["entity_id_f1"] == pytest.approx(0.6)
    assert tier1["crew_ai"]["entity_id_f1"] == pytest.approx(0.8)


def test_metric_means_rejects_bad_column():
    conn = _conn()
    _populate(conn)
    with pytest.raises(ValueError):
        q.metric_means_by_strategy(conn, ["entity_id_f1; DROP TABLE x"])


def test_distinct_tiers_and_models():
    conn = _conn()
    _populate(conn)
    assert q.distinct_tiers(conn) == [1, 2]
    models = q.distinct_models(conn)
    assert "control" not in models
    assert "gpt-4o-mini-2024-07-18" in models


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


def test_pareto_points_exclude_controls():
    conn = _conn()
    _populate(conn)
    pts = q.pareto_points(conn)
    strategies = {p["strategy"] for p in pts}
    assert "null_control" not in strategies
    # Each point carries the plotted fields.
    p = pts[0]
    assert {"cost_usd", "duration_ms", "entity_id_f1", "tier", "model"} <= set(p)


def test_pareto_strategy_filter():
    conn = _conn()
    _populate(conn)
    pts = q.pareto_points(conn, strategies=["crew_ai"])
    assert {p["strategy"] for p in pts} == {"crew_ai"}


def test_distinct_strategies_excludes_controls_by_default():
    conn = _conn()
    _populate(conn)
    assert "null_control" not in q.distinct_strategies(conn)
    assert "null_control" in q.distinct_strategies(conn, exclude_controls=False)


# ---------------------------------------------------------------------------
# Run Detail
# ---------------------------------------------------------------------------


def test_list_and_detail_roundtrip():
    conn = _conn()
    rid = _insert_run(
        conn, strategy="single_agent", model="gpt-4o-mini-2024-07-18", tier=1, f1=0.6
    )
    runs = q.list_runs(conn)
    row = next(r for r in runs if r["run_id"] == rid)
    # run_number is exposed so selectors can distinguish repeats and the
    # faceted filter can offer a run-number facet.
    assert row["run_number"] == 1
    detail = q.run_detail(conn, rid)
    assert detail is not None
    assert detail["strategy"] == "single_agent"
    assert detail["entity_id_f1"] == pytest.approx(0.6)


def test_run_detail_unknown_id():
    conn = _conn()
    _populate(conn)
    assert q.run_detail(conn, str(uuid.uuid4())) is None


def test_sub_results_empty_when_none():
    conn = _conn()
    rid = _insert_run(
        conn, strategy="sop_based", model="mistral-small-2603", tier=1, f1=0.5
    )
    assert q.sub_results_for_run(conn, rid) == []


# ---------------------------------------------------------------------------
# Hallucination Taxonomy
# ---------------------------------------------------------------------------


def test_has_any_taxonomy_data():
    conn = _conn()
    # All-zero taxonomy -> False.
    _insert_run(
        conn, strategy="single_agent", model="gpt-4o-mini-2024-07-18", tier=1, f1=0.6
    )
    assert q.has_any_taxonomy_data(conn) is False
    # A row with a non-zero count -> True.
    _insert_run(
        conn,
        strategy="crew_ai",
        model="mistral-small-2603",
        tier=1,
        f1=0.8,
        missing_entities=3,
    )
    assert q.has_any_taxonomy_data(conn) is True


def test_taxonomy_counts_by_strategy():
    conn = _conn()
    _populate(conn)
    data = q.taxonomy_counts_by_strategy(conn, q.ENTITY_TAXONOMY)
    # crew_ai tier-1 row had missing_entities=2, extra_entities=1.
    assert data["crew_ai"]["missing_entities"] == 2
    assert data["crew_ai"]["extra_entities"] == 1


def test_taxonomy_rejects_bad_column():
    conn = _conn()
    _populate(conn)
    with pytest.raises(ValueError):
        q.taxonomy_counts_by_strategy(conn, ("missing_entities; DROP TABLE x",))


# ---------------------------------------------------------------------------
# Mermaid rendering (Diagram Visualizer)
# ---------------------------------------------------------------------------


def test_mermaid_render_blank_source_returns_none():
    """Empty/blank source short-circuits to None (nothing to render)."""
    from maestro.viz.mermaid_render import render_mermaid_svg

    assert render_mermaid_svg("") is None
    assert render_mermaid_svg("   \n  ") is None


def test_mermaid_render_handles_missing_mmdc(monkeypatch):
    """With mmdc absent, the renderer returns None (caller shows code)."""
    import maestro.viz.mermaid_render as mr

    monkeypatch.setattr(mr.shutil, "which", lambda _: None)
    assert mr.mmdc_available() is False
    assert mr.render_mermaid_svg("graph TD; a-->b") is None


# ---------------------------------------------------------------------------
# Graceful degradation: every query no-ops on an empty (schemaless) DB
# ---------------------------------------------------------------------------


def test_queries_safe_on_schemaless_db():
    bare = sqlite3.connect(":memory:")
    bare.row_factory = sqlite3.Row
    assert q.overview_summary(bare)["total_runs"] == 0
    assert q.runs_by_strategy_success(bare) == []
    assert q.metric_means_by_strategy(bare, ["entity_id_f1"]) == {}
    assert q.pareto_points(bare) == []
    assert q.list_runs(bare) == []
    assert q.has_any_taxonomy_data(bare) is False
    assert q.taxonomy_counts_by_strategy(bare, q.ENTITY_TAXONOMY) == {}
