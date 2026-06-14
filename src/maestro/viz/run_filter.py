"""
MAESTRO viz — faceted filtering of runs (pure logic, no Streamlit).

The dashboard's run selectors used to be flat dropdowns over every run, which
breaks down at scale (a ~1000-row matrix is unusable, and repeats of one cell
share a label). This module is the testable core of the fix: given the run list
(from ``queries.list_runs``) and a set of selected facet values, it derives the
available facet options and returns the filtered subset. The Streamlit panel in
``components.run_filter_panel`` is a thin shell around these functions.

Facet semantics: multi-select per facet; facets combine with AND across and OR
within. An empty selection for a facet means "no constraint" (all values pass).

The ``type`` facet (bpmn / it) is derived from the ``example_id`` prefix; every
registered id is ``<type>_<tier>_<nn>`` (e.g. ``bpmn_1_03``, ``it_2_19``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from maestro.experiment_config import CONTROL_STRATEGIES
from maestro.schemas import Strategy

# Facet keys, in display order. Each maps a run dict to a comparable value via
# _FACET_ACCESSORS below.
FACETS: tuple[str, ...] = ("type", "tier", "strategy", "model", "run_number")


def _run_type(run: dict[str, Any]) -> str:
    """Diagram type (bpmn / it) from the example_id prefix."""
    example_id = run.get("example_id") or ""
    return example_id.split("_", 1)[0] if example_id else ""


_FACET_ACCESSORS = {
    "type": _run_type,
    "tier": lambda r: r.get("tier"),
    "strategy": lambda r: r.get("strategy"),
    "model": lambda r: r.get("model"),
    "run_number": lambda r: r.get("run_number"),
}

# Facets whose option set is a fixed, known domain rather than derived from the
# data. These always offer every value (so the filter reads as intentional even
# before the full matrix has run); facets not listed here (model, run_number)
# derive their options from the runs present. The values match what the
# accessors above return.
#
# strategy is sourced from the Strategy enum (minus controls) rather than a
# literal, so adding an orchestration strategy in code surfaces it in the filter
# automatically with no edit here. Controls are deterministic pipeline checks,
# not strategies under study, and are excluded from the selector. model stays
# derived: its ids are version-pinned and change per matrix, so a hardcoded list
# would drift and could hide an older model still in the data.
_FIXED_FACET_OPTIONS: dict[str, list[Any]] = {
    "type": ["bpmn", "it"],
    "tier": [1, 2, 3],
    "strategy": [s.value for s in Strategy if s not in CONTROL_STRATEGIES],
}

# DB strategy values that are controls, for exclude_controls below.
_CONTROL_STRATEGY_VALUES: frozenset[str] = frozenset(
    s.value for s in CONTROL_STRATEGIES
)


def exclude_controls(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop control runs (null/copy/ground-truth) from a run list.

    Controls are kept in the DB (they prove the scoring floor/ceiling and feed
    reference lines in aggregate views), so list_runs returns them. Views that
    have no use for them — the per-run selectors, where a control produces no
    model-generated diagram to inspect — call this to drop them. Keeping the
    exclusion here (not in the query) lets each view/chart decide for itself.
    """
    return [r for r in runs if r.get("strategy") not in _CONTROL_STRATEGY_VALUES]


@dataclass
class RunFilter:
    """
    Selected facet values. Each entry maps a facet key to the set of values the
    user picked for it; an absent or empty key means that facet is unconstrained.
    """

    selected: dict[str, set[Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """True when no facet constrains the result (everything passes)."""
        return not any(self.selected.get(f) for f in FACETS)


def facet_options(runs: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    The option list for each facet.

    Facets with a fixed domain (type, tier) always offer every value, so the
    filter reads as intentional even before the full matrix has run. The rest
    (strategy, model, run_number) derive their options from ``runs`` so only
    values actually present are offered, and model ids are never hardcoded.
    Returns one list per facet; derived facets with no values map to [].
    """
    options: dict[str, list[Any]] = {}
    for facet in FACETS:
        if facet in _FIXED_FACET_OPTIONS:
            options[facet] = list(_FIXED_FACET_OPTIONS[facet])
            continue
        accessor = _FACET_ACCESSORS[facet]
        values = {accessor(r) for r in runs}
        values.discard(None)
        values.discard("")
        options[facet] = sorted(values)
    return options


def apply_filter(
    runs: list[dict[str, Any]], run_filter: RunFilter
) -> list[dict[str, Any]]:
    """
    Return the runs matching ``run_filter``: AND across facets, OR within a
    facet. An unconstrained facet (no values selected) imposes no restriction.
    Order is preserved from the input list.
    """
    if run_filter.is_empty():
        return list(runs)

    def matches(run: dict[str, Any]) -> bool:
        for facet in FACETS:
            chosen = run_filter.selected.get(facet)
            if not chosen:
                continue  # unconstrained facet
            if _FACET_ACCESSORS[facet](run) not in chosen:
                return False
        return True

    return [r for r in runs if matches(r)]
