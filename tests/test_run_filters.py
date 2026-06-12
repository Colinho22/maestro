"""
Filter validation in run.build_matrix.

The --strategy / --model / --example flags accept comma-separated lists. A typo
in a list would silently shrink the experiment matrix (run fewer cells than the
user intended) — costly to notice halfway through a multi-hour run. These tests
pin the fail-fast behavior:

  * unknown --strategy / --example  → exit 2 (strict)
  * unknown --model                 → exit 2 *only* when a real LLM strategy is
                                      selected; a control-only run ignores
                                      --model (controls use no model), so a
                                      bad model there is a deliberate no-op.
"""

from __future__ import annotations

import argparse

import pytest

from maestro.run import _split_csv, build_matrix


def _args(**overrides) -> argparse.Namespace:
    """Namespace with the build_matrix-relevant fields, overridable per test."""
    base = dict(strategy=None, tier=None, model=None, example=None, repeats=1)
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# _split_csv
# ---------------------------------------------------------------------------


def test_split_csv_none_passes_through():
    assert _split_csv(None) is None


def test_split_csv_trims_and_drops_empties():
    assert _split_csv("a, b ,,c,") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Valid filters build a matrix
# ---------------------------------------------------------------------------


def test_valid_list_filters_build_matrix():
    args = _args(
        example="bpmn_1_03,it_1_07",
        model="gpt-4o-mini-2024-07-18,deepseek-v4-flash",
        strategy="single_agent,lang_graph",
        repeats=2,
    )
    matrix = build_matrix(args)
    # 2 inputs × 2 strategies × 2 models × 2 repeats
    assert len(matrix) == 16
    assert {c["input_file"].example_id for c in matrix} == {"bpmn_1_03", "it_1_07"}
    assert {c["model_pricing"].model for c in matrix} == {
        "gpt-4o-mini-2024-07-18",
        "deepseek-v4-flash",
    }


# ---------------------------------------------------------------------------
# Unknown values fail fast
# ---------------------------------------------------------------------------


def test_unknown_strategy_exits_2():
    with pytest.raises(SystemExit) as exc:
        build_matrix(_args(strategy="single_agent,typo_strat"))
    assert exc.value.code == 2


def test_unknown_example_exits_2():
    with pytest.raises(SystemExit) as exc:
        build_matrix(_args(example="bpmn_1_03,not_a_real_id", strategy="single_agent"))
    assert exc.value.code == 2


def test_unknown_model_exits_2_with_real_strategy():
    with pytest.raises(SystemExit) as exc:
        build_matrix(
            _args(
                model="gpt-4o-mini-2024-07-18,typo_model",
                strategy="single_agent",
                example="bpmn_1_03",
            )
        )
    assert exc.value.code == 2


def test_unknown_model_is_noop_for_control_only_run():
    """Controls use no model, so a bad --model must not abort a control run."""
    matrix = build_matrix(
        _args(model="typo_model", strategy="null_control", example="bpmn_1_03")
    )
    # One control row for the single input — bad model ignored, no exit.
    assert len(matrix) == 1
    assert matrix[0]["strategy"].value == "null_control"
