"""
Tests for the statistical analysis pipeline (src/maestro/analysis/statistics.py).

Strategy: build a synthetic SQLite DB through the *real* schema (db.client
SCHEMA + db.queries inserts), so the three-way join the analysis depends on
is exercised exactly as in production — no hand-mocked DataFrames. Each test
then asserts on the analysis output shape and the two behaviors that matter
most and are easy to regress:

  1. Controls are excluded from inferential tests (ANOVA / Cohen's d) but
     retained in descriptive outputs.
  2. A factor with < 2 observed levels yields a skip-stub, not a crash.

statsmodels/pandas are required to run these; the module-level importorskip
turns a missing-dependency environment into a clean skip rather than an error.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

pytest.importorskip("pandas")
pytest.importorskip("statsmodels")

from maestro.analysis import statistics as stats  # noqa: E402
from maestro.db.client import SCHEMA  # noqa: E402
from maestro.db.queries import (  # noqa: E402
    insert_metric_result,
    insert_run_config,
    insert_run_result,
)
from maestro.schemas import (  # noqa: E402
    MetricResult,
    RunConfig,
    RunResult,
    Strategy,
    Tier,
)

# Strategies used across tests: three experimental + one control.
_EXPERIMENTAL = [
    Strategy.SINGLE_AGENT,
    Strategy.CREW_AI,
    Strategy.LANG_GRAPH,
]
_CONTROL = Strategy.NULL_CONTROL


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _zeroed_metric_fields() -> dict:
    """All MetricResult numeric fields defaulted to 0 except entity_id_f1."""
    return dict(
        parses_valid=True,
        entity_id_precision=0.0,
        entity_id_recall=0.0,
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
        missing_entities=0,
        extra_entities=0,
        false_entities=0,
        duplicate_entities=0,
        missing_relationships=0,
        extra_relationships=0,
        false_relationships=0,
        duplicate_relationships=0,
    )


def _insert_cell(
    conn: sqlite3.Connection,
    *,
    strategy: Strategy,
    model: str,
    tier: Tier,
    run_number: int,
    f1: float,
    cost: float = 0.001,
    duration_ms: int = 100,
) -> None:
    """Insert one config+result+metric triple for a single run."""
    run_id = uuid.uuid4()
    insert_run_config(
        conn,
        RunConfig(
            run_id=run_id,
            strategy=strategy,
            model=model,
            example_id="ex_01",
            tier=tier,
            run_number=run_number,
        ),
    )
    insert_run_result(
        conn,
        RunResult(
            run_id=run_id,
            output_diagram_code="graph TD; a-->b",
            prompt_tokens=10,
            completion_tokens=10,
            duration_ms=duration_ms,
            cost_usd=cost,
            error=None,
        ),
    )
    fields = _zeroed_metric_fields()
    fields["entity_id_f1"] = f1
    insert_metric_result(
        conn,
        MetricResult(run_id=run_id, **fields),
    )


def _populate_two_levels(conn: sqlite3.Connection) -> None:
    """
    Two models, three experimental strategies, several repeats, single tier
    — mirrors the real dev DB (model varies, tier does not). F1 values are
    spread so the ANOVA has variance to find.
    """
    models = ["model_a", "model_b"]
    f1_by_strategy = {
        Strategy.SINGLE_AGENT: 0.60,
        Strategy.CREW_AI: 0.75,
        Strategy.LANG_GRAPH: 0.90,
    }
    for model in models:
        for strategy in _EXPERIMENTAL:
            base = f1_by_strategy[strategy]
            for run_number, delta in enumerate([-0.05, 0.0, 0.05], start=1):
                _insert_cell(
                    conn,
                    strategy=strategy,
                    model=model,
                    tier=Tier.COMPLEX,
                    run_number=run_number,
                    f1=base + delta,
                    cost=0.001 if strategy is Strategy.SINGLE_AGENT else 0.005,
                )
    # A control cell — floor F1 = 0. Present for descriptives, must be
    # excluded from ANOVA / effect sizes.
    _insert_cell(
        conn,
        strategy=_CONTROL,
        model="control",
        tier=Tier.COMPLEX,
        run_number=1,
        f1=0.0,
        cost=0.0,
    )


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------


def test_load_dataframe_shape():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    # 2 models * 3 strategies * 3 repeats + 1 control = 19 rows.
    assert len(df) == 19
    assert stats.PRIMARY_DV in df.columns
    assert {"strategy", "model", "tier", "cost_usd", "duration_ms"} <= set(df.columns)


def test_load_dataframe_empty_db():
    conn = _conn()
    df = stats.load_dataframe(conn)
    assert df.empty


# ---------------------------------------------------------------------------
# Descriptives — controls INCLUDED
# ---------------------------------------------------------------------------


def test_describe_includes_controls():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.describe(df)

    assert out["schema_version"] == stats.SCHEMA_VERSION
    assert out["includes_controls"] is True
    # The control strategy must appear as a descriptive cell.
    control_cells = [c for c in out["cells"] if c["is_control"]]
    assert len(control_cells) == 1
    assert control_cells[0]["strategy"] == _CONTROL.value
    assert control_cells[0][stats.PRIMARY_DV]["mean"] == 0.0


def test_describe_empty():
    conn = _conn()
    df = stats.load_dataframe(conn)
    out = stats.describe(df)
    assert out["status"] == "empty"
    assert out["cells"] == []


# ---------------------------------------------------------------------------
# ANOVA — controls EXCLUDED, real stats when factors vary
# ---------------------------------------------------------------------------


def test_anova_strategy_ok_and_excludes_controls():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy(df)

    assert out["status"] == "ok"
    assert out["excludes_controls"] is True
    assert out["reference_level"] == {"strategy": "single_agent"}
    # n must be the experimental rows only: 18, not 19 (control dropped).
    assert out["n"] == 18
    # A strategy term with a finite F should be present.
    assert out["terms"]
    term = next(iter(out["terms"].values()))
    assert term["F"] is not None
    assert term["partial_eta_sq"] is not None


def test_anova_strategy_by_model_interaction_present():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy_by_model(df)
    assert out["status"] == "ok"
    # The interaction term of interest should be among the model terms.
    assert any(":" in term for term in out["terms"])


# ---------------------------------------------------------------------------
# Graceful degradation — single-level factor yields a skip-stub
# ---------------------------------------------------------------------------


def test_anova_strategy_by_tier_skips_on_single_tier():
    conn = _conn()
    _populate_two_levels(conn)  # all COMPLEX → tier has one level
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy_by_tier(df)

    assert out["status"] == "skipped"
    assert out["factor"] == "tier"
    assert "tier" in out["reason"]
    # Metadata still present so the file is self-describing even when skipped.
    assert out["dependent_variable"] == stats.PRIMARY_DV
    assert out["factors"] == ["strategy", "tier"]


def test_anova_skips_on_empty_db():
    conn = _conn()
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy(df)
    assert out["status"] == "skipped"
    assert "no experimental rows" in out["reason"]


# ---------------------------------------------------------------------------
# Post-hoc & effect sizes
# ---------------------------------------------------------------------------


def test_posthoc_strategy_pairs():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.posthoc_strategy(df)
    assert out["status"] == "ok"
    # 3 experimental strategies → C(3,2) = 3 pairwise comparisons.
    assert len(out["comparisons"]) == 3


def test_effect_sizes_excludes_controls():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.effect_sizes(df)
    assert out["status"] == "ok"
    # Pairs are over the 3 experimental strategies only — the control must
    # never appear in a pair.
    seen = {p["group_a"] for p in out["pairs"]} | {p["group_b"] for p in out["pairs"]}
    assert _CONTROL.value not in seen
    assert len(out["pairs"]) == 3


# ---------------------------------------------------------------------------
# Error taxonomy — descriptive, controls included
# ---------------------------------------------------------------------------


def test_error_taxonomy_includes_controls():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.error_taxonomy_by_strategy(df)
    assert out["analysis"] == "error_taxonomy_descriptive"
    strategies = {e["strategy"] for e in out["by_strategy"]}
    assert _CONTROL.value in strategies
    # Every taxonomy column must be summarized for each strategy.
    for entry in out["by_strategy"]:
        assert set(entry["counts"]) == set(stats.TAXONOMY_COLUMNS)


# ---------------------------------------------------------------------------
# Trade-off — controls excluded, ratio computed
# ---------------------------------------------------------------------------


def test_tradeoff_excludes_controls_and_computes_ratio():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.tradeoff_correctness_efficiency(df)
    assert out["status"] == "ok"
    strategies = {e["strategy"] for e in out["by_strategy"]}
    assert _CONTROL.value not in strategies
    for entry in out["by_strategy"]:
        # Non-zero cost in the fixture → ratio is a finite number.
        assert entry["correctness_per_usd"] is not None


# ---------------------------------------------------------------------------
# JSON-serializability of every output (no numpy types leak through)
# ---------------------------------------------------------------------------


def test_all_outputs_json_serializable():
    import json

    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    for fn in (
        stats.describe,
        stats.anova_strategy,
        stats.anova_strategy_by_tier,
        stats.anova_strategy_by_model,
        stats.posthoc_strategy,
        stats.effect_sizes,
        stats.error_taxonomy_by_strategy,
        stats.tradeoff_correctness_efficiency,
    ):
        payload = fn(df)
        # Must not raise — proves no numpy float64 / NaN leaked through.
        json.dumps(payload)
