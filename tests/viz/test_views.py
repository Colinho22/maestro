"""
Tests for the five data-view query functions, against a synthetic in-memory
database with multiple strategies, models, tiers, and control rows.

The view modules themselves are thin Streamlit wrappers (selectboxes, columns,
st.pyplot) that need a script context to exercise; their *logic* lives in the
queries here and in the design-system chart/theme code already covered by
test_theme.py. These tests pin the query behavior every view depends on —
especially control exclusion, filtering, and graceful degradation on missing
tables — using the real schema and the production insert helpers.
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
    """Overview is operational — controls ARE included here."""
    conn = _conn()
    _populate(conn)
    rows = dict((r[0], (r[1], r[2])) for r in q.runs_by_strategy_success(conn))
    assert "null_control" in rows


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
    assert any(r["run_id"] == rid for r in runs)
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
    # All-zero taxonomy → False.
    _insert_run(
        conn, strategy="single_agent", model="gpt-4o-mini-2024-07-18", tier=1, f1=0.6
    )
    assert q.has_any_taxonomy_data(conn) is False
    # A row with a non-zero count → True.
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
# Graceful degradation — every query no-ops on an empty (schemaless) DB
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
