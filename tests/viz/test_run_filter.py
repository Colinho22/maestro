"""
Tests for the faceted run-filter logic (viz.run_filter).

The filtering core is pure Python (no Streamlit), so facet extraction and the
AND-across / OR-within semantics are unit-tested directly here. The Streamlit
panel that wraps these functions is a thin shell and is not unit-tested.
"""

from __future__ import annotations

from maestro.viz.run_filter import (
    FACETS,
    RunFilter,
    apply_filter,
    exclude_controls,
    facet_options,
)


def _run(
    *, example_id: str, tier: int, strategy: str, model: str, run_number: int
) -> dict:
    """Build a run dict shaped like queries.list_runs rows."""
    return {
        "run_id": f"{example_id}-{strategy}-{model}-{run_number}",
        "example_id": example_id,
        "tier": tier,
        "strategy": strategy,
        "model": model,
        "run_number": run_number,
        "timestamp": "2026-01-01T00:00:00Z",
    }


_RUNS = [
    _run(
        example_id="bpmn_1_03",
        tier=1,
        strategy="single_agent",
        model="gpt",
        run_number=1,
    ),
    _run(
        example_id="bpmn_1_03",
        tier=1,
        strategy="single_agent",
        model="gpt",
        run_number=2,
    ),
    _run(
        example_id="it_2_19",
        tier=2,
        strategy="lang_graph",
        model="deepseek",
        run_number=1,
    ),
    _run(
        example_id="it_2_19", tier=2, strategy="single_agent", model="gpt", run_number=1
    ),
]


# ---------------------------------------------------------------------------
# facet_options
# ---------------------------------------------------------------------------


def _orchestration_strategies() -> list[str]:
    from maestro.experiment_config import CONTROL_STRATEGIES
    from maestro.schemas import Strategy

    return [s.value for s in Strategy if s not in CONTROL_STRATEGIES]


def test_fixed_facets_always_offer_full_domain():
    # type, tier, strategy are fixed domains: every value is offered regardless
    # of the data present (here the runs only cover tiers 1-2 and two
    # strategies, but the full domains stay selectable). strategy is sourced
    # from the Strategy enum minus controls, so the four orchestration
    # strategies under study are offered and controls are excluded.
    opts = facet_options(_RUNS)
    assert opts["type"] == ["bpmn", "it"]
    assert opts["tier"] == [1, 2, 3]
    assert opts["strategy"] == _orchestration_strategies()
    assert "null_control" not in opts["strategy"]


def test_derived_facets_distinct_and_sorted():
    # model / run_number derive their options from the runs present.
    opts = facet_options(_RUNS)
    assert opts["model"] == ["deepseek", "gpt"]
    assert opts["run_number"] == [1, 2]


def test_facet_options_empty_runs():
    opts = facet_options([])
    assert set(opts.keys()) == set(FACETS)
    # Fixed facets still offer their full domain; derived facets are empty.
    assert opts["type"] == ["bpmn", "it"]
    assert opts["tier"] == [1, 2, 3]
    assert opts["strategy"] == _orchestration_strategies()
    assert opts["model"] == []
    assert opts["run_number"] == []


def test_derived_facet_drops_none_and_blank():
    runs = [
        _run(
            example_id="bpmn_1_03",
            tier=1,
            strategy="single_agent",
            model="",
            run_number=1,
        )
    ]
    opts = facet_options(runs)
    assert opts["model"] == []  # blank model yields no derived option


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------


def test_empty_filter_returns_all():
    assert apply_filter(_RUNS, RunFilter()) == _RUNS


def test_single_facet_or_within():
    # type in {bpmn} -> only the two bpmn runs
    out = apply_filter(_RUNS, RunFilter({"type": {"bpmn"}}))
    assert len(out) == 2
    assert all(r["example_id"].startswith("bpmn") for r in out)


def test_or_within_multiple_values():
    # model in {gpt, deepseek} -> all four runs
    out = apply_filter(_RUNS, RunFilter({"model": {"gpt", "deepseek"}}))
    assert len(out) == 4


def test_and_across_facets():
    # type=it AND strategy=single_agent -> only the it_2_19 single_agent run
    out = apply_filter(_RUNS, RunFilter({"type": {"it"}, "strategy": {"single_agent"}}))
    assert len(out) == 1
    assert out[0]["example_id"] == "it_2_19"
    assert out[0]["strategy"] == "single_agent"


def test_run_number_facet_distinguishes_repeats():
    # The repeats bug: both bpmn runs share strategy/model/tier; run_number splits them.
    out = apply_filter(_RUNS, RunFilter({"run_number": {2}}))
    assert len(out) == 1
    assert out[0]["run_number"] == 2


def test_no_match_returns_empty():
    out = apply_filter(_RUNS, RunFilter({"tier": {3}}))
    assert out == []


def test_order_preserved():
    out = apply_filter(_RUNS, RunFilter({"model": {"gpt"}}))
    assert [r["run_id"] for r in out] == [
        r["run_id"] for r in _RUNS if r["model"] == "gpt"
    ]


# ---------------------------------------------------------------------------
# exclude_controls
# ---------------------------------------------------------------------------


def test_exclude_controls_drops_only_controls():
    runs = _RUNS + [
        _run(
            example_id="bpmn_1_03",
            tier=1,
            strategy="null_control",
            model="control",
            run_number=1,
        ),
        _run(
            example_id="it_2_19",
            tier=2,
            strategy="ground_truth_control",
            model="control",
            run_number=1,
        ),
    ]
    out = exclude_controls(runs)
    assert out == _RUNS  # the two control runs are dropped, order preserved
    assert all("control" not in r["strategy"] for r in out)
