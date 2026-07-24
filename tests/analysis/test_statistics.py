"""
Tests for the statistical analysis pipeline (src/maestro/analysis/statistics.py).

Strategy: build a synthetic SQLite DB through the *real* schema (db.client
SCHEMA + db.queries inserts), so the three-way join the analysis depends on
is exercised exactly as in production: no hand-mocked DataFrames. Each test
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
    example_id: str = "ex_01",
    cost: float = 0.001,
    duration_ms: int = 100,
    parses_valid: bool = True,
    metric: bool = True,
) -> None:
    """
    Insert one config+result(+metric) triple for a single run.

    ``metric=False`` inserts the run WITHOUT a metric_results row, mirroring an
    outright failure: the LEFT join in fetch_analysis_rows surfaces it with
    NULL metrics, which the intent_to_treat convention scores as 0.0.
    """
    run_id = uuid.uuid4()
    insert_run_config(
        conn,
        RunConfig(
            run_id=run_id,
            strategy=strategy,
            model=model,
            example_id=example_id,
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
    if not metric:
        return
    fields = _zeroed_metric_fields()
    fields["parses_valid"] = parses_valid
    fields["entity_id_f1"] = f1
    insert_metric_result(
        conn,
        MetricResult(run_id=run_id, **fields),
    )


# Two inputs and two models per strategy, so per-cell aggregation still leaves
# enough cells (2 models x 2 inputs x 3 strategies = 12) for a saturated
# strategy x model ANOVA to have residual degrees of freedom. A single input
# would collapse to 6 cells and zero residual df after averaging.
_EXAMPLES = ("ex_01", "ex_02")


def _populate_two_levels(conn: sqlite3.Connection) -> None:
    """
    Two models, two inputs, three experimental strategies, several repeats,
    single tier: mirrors the real dev DB (model and input vary, tier does not).
    F1 values are spread so the ANOVA has variance to find.
    """
    models = ["model_a", "model_b"]
    f1_by_strategy = {
        Strategy.SINGLE_AGENT: 0.60,
        Strategy.CREW_AI: 0.75,
        Strategy.LANG_GRAPH: 0.90,
    }
    for model in models:
        for example_id in _EXAMPLES:
            for strategy in _EXPERIMENTAL:
                base = f1_by_strategy[strategy]
                for run_number, delta in enumerate([-0.05, 0.0, 0.05], start=1):
                    _insert_cell(
                        conn,
                        strategy=strategy,
                        model=model,
                        tier=Tier.COMPLEX,
                        run_number=run_number,
                        example_id=example_id,
                        f1=base + delta,
                        cost=0.001 if strategy is Strategy.SINGLE_AGENT else 0.005,
                    )
    # A control cell: floor F1 = 0. Present for descriptives, must be
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


def _populate_two_tiers(conn: sqlite3.Connection) -> None:
    """
    Extend the two-level fixture with a second tier so the mixed model's
    strategy*tier fixed effect is estimable and the four-factor guard
    (strategy, tier, model, example_id) is cleared. Used by the mixed-effects
    tests that need to reach the fitter.
    """
    _populate_two_levels(conn)  # tier COMPLEX
    for model in ("model_a", "model_b"):
        for strategy in _EXPERIMENTAL:
            for run_number, delta in enumerate([-0.05, 0.0, 0.05], start=1):
                _insert_cell(
                    conn,
                    strategy=strategy,
                    model=model,
                    tier=Tier.SIMPLE,
                    run_number=run_number,
                    example_id="ex_simple",
                    f1=0.5 + delta,
                )


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------


def test_load_dataframe_shape():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    # 2 models * 2 inputs * 3 strategies * 3 repeats + 1 control = 37 rows.
    assert len(df) == 37
    assert stats.PRIMARY_DV in df.columns
    assert {"strategy", "model", "tier", "cost_usd", "duration_ms"} <= set(df.columns)


def test_load_dataframe_empty_db():
    conn = _conn()
    df = stats.load_dataframe(conn)
    assert df.empty


# ---------------------------------------------------------------------------
# Descriptives: controls INCLUDED
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
# ANOVA: controls EXCLUDED, real stats when factors vary
# ---------------------------------------------------------------------------


def test_anova_strategy_ok_and_excludes_controls():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy(df)

    assert out["status"] == "ok"
    assert out["excludes_controls"] is True
    assert out["reference_level"] == {"strategy": "single_agent"}
    # n is the aggregated CELL count, not the raw run count: the inferential
    # path averages the repeats first. 2 models x 2 inputs x 3 strategies = 12
    # cells (the 36 experimental runs and the 1 control run collapse away).
    assert out["n"] == 12
    assert out["unit_of_analysis"] == "mean over repeats per (strategy, model, input)"
    assert out["scoring_convention"] == stats.INTENT_TO_TREAT
    # The strategy term (looked up by name, not by dict order) must carry a
    # finite F and partial η².
    assert out["terms"]
    term = out["terms"][out["term_of_interest"]]
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
# Graceful degradation: single-level factor yields a skip-stub
# ---------------------------------------------------------------------------


def test_anova_strategy_by_tier_skips_on_single_tier():
    conn = _conn()
    _populate_two_levels(conn)  # all COMPLEX -> tier has one level
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
# Aggregation grain & scoring conventions
# ---------------------------------------------------------------------------


def test_aggregate_grain_one_row_per_cell():
    """
    Aggregation yields exactly one row per (strategy, model, input): the repeats
    collapse. The fixture has 2 models x 2 inputs x 3 strategies = 12 cells.
    """
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    agg = stats.aggregate_experimental(df, stats.INTENT_TO_TREAT)
    assert len(agg) == 12
    # One row per cell: no (strategy, model, example_id) key repeats.
    assert not agg.duplicated(["strategy", "model", "example_id"]).any()
    # Controls never enter the aggregated (inferential) frame.
    assert _CONTROL.value not in set(agg["strategy"])


def test_intent_to_treat_scores_failures_and_invalid_as_zero():
    """
    Under intent_to_treat a run with no metric row (outright failure) and a run
    whose diagram did not parse both contribute 0.0; under valid_only both are
    excluded. Build one cell (single strategy/model/input) whose three repeats
    are: one good (0.9, valid), one parse-invalid (0.8 but parses_valid=0), one
    outright failure (no metric row).
    """
    conn = _conn()
    _insert_cell(
        conn,
        strategy=Strategy.CREW_AI,
        model="m",
        tier=Tier.COMPLEX,
        run_number=1,
        f1=0.9,
        parses_valid=True,
    )
    _insert_cell(
        conn,
        strategy=Strategy.CREW_AI,
        model="m",
        tier=Tier.COMPLEX,
        run_number=2,
        f1=0.8,  # scorer produced a partial F1 even though it did not render
        parses_valid=False,
    )
    _insert_cell(
        conn,
        strategy=Strategy.CREW_AI,
        model="m",
        tier=Tier.COMPLEX,
        run_number=3,
        f1=0.0,
        metric=False,  # outright failure: no metric_results row at all
    )
    df = stats.load_dataframe(conn)

    itt = stats.aggregate_experimental(df, stats.INTENT_TO_TREAT)
    # One surviving cell; mean over [0.9, 0.0, 0.0] = 0.3.
    assert len(itt) == 1
    assert itt.iloc[0][stats.PRIMARY_DV] == pytest.approx(0.3)

    valid = stats.aggregate_experimental(df, stats.VALID_ONLY)
    # Only the single valid run survives; mean = 0.9.
    assert len(valid) == 1
    assert valid.iloc[0][stats.PRIMARY_DV] == pytest.approx(0.9)


def test_valid_only_drops_cell_with_no_valid_run():
    """
    A cell whose every run is invalid/failed has nothing to average under
    valid_only and disappears; under intent_to_treat it survives at 0.0.
    """
    conn = _conn()
    _insert_cell(
        conn,
        strategy=Strategy.CREW_AI,
        model="m",
        tier=Tier.COMPLEX,
        run_number=1,
        f1=0.7,
        parses_valid=False,
    )
    _insert_cell(
        conn,
        strategy=Strategy.CREW_AI,
        model="m",
        tier=Tier.COMPLEX,
        run_number=2,
        f1=0.0,
        metric=False,
    )
    df = stats.load_dataframe(conn)

    assert len(stats.aggregate_experimental(df, stats.VALID_ONLY)) == 0
    itt = stats.aggregate_experimental(df, stats.INTENT_TO_TREAT)
    assert len(itt) == 1
    assert itt.iloc[0][stats.PRIMARY_DV] == 0.0


def test_anova_n_matches_aggregated_not_raw():
    """The fitted ANOVA reports the aggregated cell count, never the raw runs."""
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    agg = stats.aggregate_experimental(df, stats.INTENT_TO_TREAT)
    out = stats.anova_strategy(df, stats.INTENT_TO_TREAT)
    assert out["n"] == len(agg)
    # And that is far below the 36 experimental raw rows.
    assert out["n"] == 12
    assert len(stats._experimental(df)) == 36


def test_valid_only_convention_recorded_in_metadata():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy(df, stats.VALID_ONLY)
    assert out["scoring_convention"] == "valid_only"


def test_unknown_convention_rejected():
    """An unrecognized convention is a programmer error, raised not swallowed."""
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    with pytest.raises(ValueError, match="unknown scoring convention"):
        stats.aggregate_experimental(df, "not_a_convention")  # type: ignore[arg-type]


def test_anova_skips_on_saturated_model():
    """
    A cell count that barely exceeds the term count leaves zero residual df:
    the ANOVA is uncomputable and must skip, not emit a status ok with null
    F/p. Build a single input per model so strategy x model saturates: with one
    input, each (strategy, model) cell is a single observation and the crossed
    model has as many parameters as observations.
    """
    conn = _conn()
    models = ["model_a", "model_b"]
    for model in models:
        for strategy in _EXPERIMENTAL:
            _insert_cell(
                conn,
                strategy=strategy,
                model=model,
                tier=Tier.COMPLEX,
                run_number=1,
                example_id="ex_only",
                f1=0.6,
            )
    df = stats.load_dataframe(conn)
    out = stats.anova_strategy_by_model(df, stats.INTENT_TO_TREAT)
    assert out["status"] == "skipped"
    assert "saturated" in out["reason"]


# ---------------------------------------------------------------------------
# Mixed-effects robustness model
# ---------------------------------------------------------------------------


def test_mixed_effects_returns_result_or_skip_stub():
    """
    The mixed model either fits (status ok, with fixed-effect estimates) or
    degrades to a skip-stub (status skipped, with a reason). Either is a valid,
    self-describing output: it must never raise. Needs >= 2 tiers for the
    strategy*tier term, so build a two-tier corpus.
    """
    conn = _conn()
    _populate_two_tiers(conn)
    df = stats.load_dataframe(conn)
    out = stats.mixed_effects_robustness(df, stats.INTENT_TO_TREAT)

    assert out["analysis"] == "mixed_effects_robustness"
    assert out["role"] == "robustness_check"
    # Either branch is valid on this small synthetic fixture (crossed random
    # effects can genuinely fail to converge here); the deterministic ok/skip
    # branches are asserted exactly in the monkeypatched tests below. What must
    # always hold: a self-describing status and strict-JSON serializability.
    assert out["status"] in {"ok", "skipped"}
    if out["status"] == "ok":
        assert out["fixed_effects_estimates"]
        assert out["scoring_convention"] == "intent_to_treat"
    else:
        assert "reason" in out
    import json

    json.dumps(out, allow_nan=False)


def test_mixed_effects_skips_on_single_tier():
    """Single tier -> strategy*tier not estimable -> graceful skip-stub."""
    conn = _conn()
    _populate_two_levels(conn)  # all COMPLEX
    df = stats.load_dataframe(conn)
    out = stats.mixed_effects_robustness(df, stats.INTENT_TO_TREAT)
    assert out["status"] == "skipped"
    # The skip must name the offending factor, not just report a generic
    # failure: tier is the one with a single level here.
    assert out.get("factor") == "tier"
    assert "tier" in out["reason"]


def test_mixed_effects_skips_on_fit_exception(monkeypatch):
    """
    A raised exception inside the fitter (not just non-convergence) degrades to
    a skip-stub whose reason names the exception, never propagating. Force it by
    monkeypatching mixedlm to raise, on a corpus that clears the factor guard.
    """
    conn = _conn()
    _populate_two_tiers(conn)
    df = stats.load_dataframe(conn)

    import statsmodels.formula.api as smf

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(smf, "mixedlm", _boom)
    out = stats.mixed_effects_robustness(df, stats.INTENT_TO_TREAT)
    assert out["status"] == "skipped"
    assert "RuntimeError" in out["reason"]
    assert "synthetic fit failure" in out["reason"]


def test_mixed_effects_skips_on_non_convergence(monkeypatch):
    """
    A model that fits but does not converge is reported as skipped, not ok:
    a non-converged estimate is not a result we stand behind. Stub the fit to
    return an object flagged not-converged.
    """
    conn = _conn()
    _populate_two_tiers(conn)
    df = stats.load_dataframe(conn)

    import statsmodels.formula.api as smf

    class _Fit:
        converged = False

    class _Model:
        def fit(self, *args, **kwargs):
            return _Fit()

    monkeypatch.setattr(smf, "mixedlm", lambda *a, **k: _Model())
    out = stats.mixed_effects_robustness(df, stats.INTENT_TO_TREAT)
    assert out["status"] == "skipped"
    assert "converge" in out["reason"]


# ---------------------------------------------------------------------------
# Post-hoc & effect sizes
# ---------------------------------------------------------------------------


def test_posthoc_strategy_pairs():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.posthoc_strategy(df)
    assert out["status"] == "ok"
    # 3 experimental strategies -> C(3,2) = 3 pairwise comparisons.
    assert len(out["comparisons"]) == 3


def test_effect_sizes_excludes_controls():
    conn = _conn()
    _populate_two_levels(conn)
    df = stats.load_dataframe(conn)
    out = stats.effect_sizes(df)
    assert out["status"] == "ok"
    # Pairs are over the 3 experimental strategies only: the control must
    # never appear in a pair.
    seen = {p["group_a"] for p in out["pairs"]} | {p["group_b"] for p in out["pairs"]}
    assert _CONTROL.value not in seen
    assert len(out["pairs"]) == 3


# ---------------------------------------------------------------------------
# Effect-size summary: the reportable |d| range
# ---------------------------------------------------------------------------


def _varying_analysis_frame() -> "object":
    """
    A frame whose per-cell means differ within each strategy, so pooled
    variance is non-zero and every pairwise d is finite.

    ``_analysis_frame`` deliberately holds each strategy at one constant score
    to exercise the zero-variance sentinel; the summary range needs the
    opposite, so the per-cell scores are spread here instead.

    The levels are chosen so the widest contrast is ``crew_ai vs lang_graph``,
    a *negative* d (crew_ai sorts first alphabetically but scores lowest).
    That makes the summary's use of absolute values load-bearing: a range over
    signed values would miss the largest magnitude entirely.
    """
    import pandas as pd

    spreads = {
        Strategy.CREW_AI: (0.10, 0.20, 0.30),
        Strategy.LANG_GRAPH: (0.80, 0.90, 1.00),
        Strategy.SINGLE_AGENT: (0.45, 0.55, 0.65),
    }
    cells = (("m_a", "ex_01"), ("m_b", "ex_02"), ("m_c", "ex_03"))
    rows = []
    for strat, scores in spreads.items():
        for (model, example_id), score in zip(cells, scores):
            rows.append(
                {
                    "strategy": strat.value,
                    "model": model,
                    "example_id": example_id,
                    "tier": 2,
                    "parses_valid": 1,
                    stats.PRIMARY_DV: score,
                    "cost_usd": 0.001,
                    "duration_ms": 100,
                    "retry_count": 0,
                }
            )
    return pd.DataFrame(rows)


def test_effect_sizes_summary_uses_absolute_values():
    """
    The summary range is over |d|, and names the widest contrast.

    The sign of a pairwise d only records which strategy sorted first, so a
    negative d must still be able to be the largest magnitude.
    """
    out = stats.effect_sizes(_varying_analysis_frame())

    summary = out["summary"]
    finite = [
        abs(p["cohens_d"])
        for p in out["pairs"]
        if isinstance(p["cohens_d"], (int, float))
    ]
    assert summary["abs_d_min"] == pytest.approx(min(finite))
    assert summary["abs_d_max"] == pytest.approx(max(finite))
    assert summary["abs_d_min"] >= 0
    assert summary["n_pairs"] == len(out["pairs"])
    assert summary["n_sentinel_pairs"] == 0
    assert summary["n_undefined_pairs"] == 0

    # The fixture puts the widest contrast on a negative d (crew_ai sorts
    # first but scores lowest). Ranging over signed values would miss it, so
    # this pins the absolute-value behaviour rather than merely asserting a
    # non-negative minimum.
    assert summary["largest_contrast"]["cohens_d"] < 0
    assert {
        summary["largest_contrast"]["group_a"],
        summary["largest_contrast"]["group_b"],
    } == {Strategy.CREW_AI.value, Strategy.LANG_GRAPH.value}

    # The named contrast is the one carrying the maximum magnitude.
    widest = summary["largest_contrast"]
    assert widest["abs_cohens_d"] == pytest.approx(summary["abs_d_max"])
    assert abs(widest["cohens_d"]) == pytest.approx(summary["abs_d_max"])
    assert {widest["group_a"], widest["group_b"]} <= {
        p["group_a"] for p in out["pairs"]
    } | {p["group_b"] for p in out["pairs"]}


def test_effect_sizes_summary_counts_sentinels_outside_the_range():
    """
    Infinite d (zero pooled variance, unequal means) is counted, not ranged.

    An infinity cannot sit in a numeric range, and dropping it silently would
    present a partial range as if it covered every contrast.
    """
    df = _analysis_frame(
        {
            Strategy.SINGLE_AGENT: 0.5,
            Strategy.CREW_AI: 0.9,
            Strategy.LANG_GRAPH: 0.1,
        }
    )
    out = stats.effect_sizes(df)
    summary = out["summary"]

    # Every pair here is deterministic-vs-deterministic: all infinite.
    assert summary["n_pairs"] == 3
    assert summary["n_sentinel_pairs"] == 3
    assert summary["n_undefined_pairs"] == 0
    assert summary["abs_d_min"] is None
    assert summary["abs_d_max"] is None
    assert summary["largest_contrast"] is None


def test_effect_sizes_summary_counts_undefined_pairs():
    """A strategy with a single cell yields d=None, counted as undefined."""
    import pandas as pd

    df = _analysis_frame({Strategy.SINGLE_AGENT: 0.5, Strategy.CREW_AI: 0.9})
    # Reduce crew_ai to one cell: with n < 2 the pooled SD is undefined, so
    # _cohens_d returns None for every pair involving it.
    single_cell = df[(df["strategy"] == Strategy.CREW_AI.value)].iloc[:1]
    df = pd.concat([df[df["strategy"] != Strategy.CREW_AI.value], single_cell])

    out = stats.effect_sizes(df)
    summary = out["summary"]
    assert summary["n_pairs"] == 1
    assert summary["n_undefined_pairs"] == 1
    assert summary["abs_d_max"] is None
    assert summary["largest_contrast"] is None


def test_effect_sizes_skip_stub_carries_no_summary():
    """
    A skipped analysis emits no summary at all, rather than an empty range: a
    null min/max would be indistinguishable from a computed-but-degenerate
    one. The skip reason is the honest record instead.
    """
    conn = _conn()
    df = stats.load_dataframe(conn)  # empty DB: no experimental rows
    out = stats.effect_sizes(df)
    assert out["status"] == "skipped"
    assert "summary" not in out
    # Metadata still present so the file is self-describing even when skipped.
    assert out["dependent_variable"] == stats.PRIMARY_DV


# ---------------------------------------------------------------------------
# Error taxonomy: descriptive, controls included
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
# Trade-off: controls excluded, ratio computed
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
        # Non-zero cost in the fixture -> ratio is a finite number.
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
        # allow_nan=False mirrors the writer in __main__._write_json: it
        # must not raise, proving no numpy float64 leaked through (TypeError)
        # AND no NaN/inf leaked through (ValueError). Infinite Cohen's d is
        # emitted as a string sentinel, not float inf, so it is unaffected.
        json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# Cohen's d: zero-variance edge case
# ---------------------------------------------------------------------------


def _analysis_frame(score_by_strategy: dict[Strategy, float]) -> "object":
    """
    Build a load_dataframe-shaped frame where each strategy scores an
    identical value across three cells (distinct model x input), all valid.

    After per-cell aggregation each strategy still has three cells at the same
    mean, so pooled within-strategy variance stays zero: the Cohen's d
    zero-variance edge case survives the aggregation step. The columns match
    what aggregate_experimental reads (strategy, model, example_id, tier,
    parses_valid, the primary DV, and the efficiency DVs).
    """
    import pandas as pd

    rows = []
    for strat, score in score_by_strategy.items():
        for model, example_id in (
            ("m_a", "ex_01"),
            ("m_b", "ex_02"),
            ("m_c", "ex_03"),
        ):
            rows.append(
                {
                    "strategy": strat.value,
                    "model": model,
                    "example_id": example_id,
                    "tier": 2,
                    "parses_valid": 1,
                    stats.PRIMARY_DV: score,
                    "cost_usd": 0.001,
                    "duration_ms": 100,
                    "retry_count": 0,
                }
            )
    return pd.DataFrame(rows)


def test_cohens_d_zero_variance_unequal_means_is_infinite():
    """
    Two deterministic groups (zero within-group variance) at different score
    levels are maximally separated: Cohen's d is infinite, not 0.0. The
    value is emitted as a signed string sentinel so it survives strict JSON.
    """
    import json

    df = _analysis_frame(
        {
            Strategy.SINGLE_AGENT: 0.5,
            Strategy.CREW_AI: 0.9,
            Strategy.LANG_GRAPH: 0.1,
        }
    )

    out = stats.effect_sizes(df)
    assert out["status"] == "ok"
    ds = {(p["group_a"], p["group_b"]): p["cohens_d"] for p in out["pairs"]}
    # crew_ai (0.9) > single_agent (0.5): groups sorted alphabetically, so
    # group_a=crew_ai, group_b=single_agent -> positive infinity.
    assert ds[("crew_ai", "single_agent")] == "inf"
    # lang_graph (0.1) < single_agent (0.5) -> negative infinity.
    assert ds[("lang_graph", "single_agent")] == "-inf"
    # Still strict-JSON serializable (string, not float inf).
    json.dumps(out, allow_nan=False)


def test_cohens_d_zero_variance_equal_means_is_zero():
    df = _analysis_frame(
        {
            Strategy.SINGLE_AGENT: 0.7,
            Strategy.CREW_AI: 0.7,
        }
    )
    out = stats.effect_sizes(df)
    assert out["pairs"][0]["cohens_d"] == 0.0


# ---------------------------------------------------------------------------
# CLI smoke test: public output contract
# ---------------------------------------------------------------------------


def test_cli_writes_output_contract(tmp_path):
    """
    End-to-end: invoke the module entry point against an on-disk DB and a
    temp output dir, and assert the public output contract: a timestamped
    run directory containing report.md, the JSON files, and the deferred
    figures/README.md, with a zero return code.
    """
    import json

    from maestro.analysis.__main__ import main

    # Materialize the synthetic dataset to a file DB (the CLI opens a path).
    db_path = tmp_path / "smoke.db"
    file_conn = sqlite3.connect(db_path)
    file_conn.row_factory = sqlite3.Row
    file_conn.executescript(SCHEMA)
    _populate_two_levels(file_conn)
    file_conn.commit()
    file_conn.close()

    out_root = tmp_path / "analysis_out"
    rc = main(["--db", str(db_path), "--out", str(out_root)])
    assert rc == 0

    # Exactly one timestamped run directory was created.
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Required artifacts exist. Convention-dependent analyses are emitted once
    # per scoring convention as ``<stem>__<convention>.json``; the descriptive
    # / taxonomy / trade-off outputs are convention-independent.
    assert (run_dir / "report.md").exists()
    assert (run_dir / "figures" / "README.md").exists()
    convention_stems = (
        "anova_strategy",
        "anova_strategy_by_tier",
        "anova_strategy_by_model",
        "posthoc_strategy",
        "effect_sizes",
        "mixed_effects_robustness",
    )
    per_convention = [
        f"{stem}__{conv}.json"
        for stem in convention_stems
        for conv in ("intent_to_treat", "valid_only")
    ]
    for filename in (
        "descriptive.json",
        "error_taxonomy_by_strategy.json",
        "tradeoff_correctness_efficiency.json",
        *per_convention,
    ):
        path = run_dir / filename
        assert path.exists(), f"missing output: {filename}"

        # Every emitted JSON must be strict-valid. json.loads has no
        # allow_nan kwarg (that's a dumps-only option); parse_constant is the
        # loads-side hook: it fires on the non-standard NaN/Infinity/-Infinity
        # tokens, so raising there fails the test if a non-finite value leaked.
        def _reject(token):
            raise ValueError(f"non-finite token in {filename}: {token}")

        json.loads(path.read_text(), parse_constant=_reject)


def test_cli_missing_db_returns_error(tmp_path):
    from maestro.analysis.__main__ import main

    rc = main(["--db", str(tmp_path / "does_not_exist.db"), "--out", str(tmp_path)])
    assert rc == 1
