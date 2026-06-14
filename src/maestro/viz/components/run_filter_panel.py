"""
MAESTRO viz — faceted run-filter panel (Streamlit shell).

Renders one multi-select per facet (type / tier / strategy / model / run
number), populated from the runs actually present, and returns the filtered
subset. This replaces the flat run dropdowns in the Diagram Visualizer and Run
Detail views: the user narrows by facet, then picks from the (now small)
result. All the filtering logic lives in ``viz.run_filter`` (pure, tested); this
module only wires it to Streamlit widgets.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from maestro.viz.run_filter import FACETS, RunFilter, apply_filter, facet_options
from maestro.viz.theme import strategy_display_name

# Human-facing labels for each facet, and per-facet option formatting.
_FACET_LABELS = {
    "type": "Diagram Type",
    "tier": "Tier",
    "strategy": "Strategy",
    "model": "Model",
    "run_number": "Run number",
}

# Display text for the fixed diagram-type values.
_TYPE_DISPLAY = {"bpmn": "BPMN", "it": "IT"}


def _format_option(facet: str, value: Any) -> str:
    """Display text for one facet option (e.g. strategy value -> display name)."""
    if facet == "strategy":
        return strategy_display_name(str(value))
    if facet == "type":
        return _TYPE_DISPLAY.get(str(value), str(value).upper())
    if facet == "tier":
        return f"Tier {value}"
    if facet == "run_number":
        return f"Run {value}"
    return str(value)


def render_run_filter(
    runs: list[dict[str, Any]], *, key_prefix: str
) -> list[dict[str, Any]]:
    """
    Draw the facet multi-selects and return the filtered runs.

    ``key_prefix`` namespaces the widget keys so two views (or two panels) can
    coexist without Streamlit key collisions. Facets with only one option are
    still shown for consistency. The count of matching runs is surfaced so the
    user can see the filter narrowing the set.
    """
    options = facet_options(runs)

    selected: dict[str, set[Any]] = {}
    columns = st.columns(len(FACETS))
    for facet, col in zip(FACETS, columns):
        with col:
            picked = st.multiselect(
                _FACET_LABELS[facet],
                options=options[facet],
                format_func=lambda v, f=facet: _format_option(f, v),
                key=f"{key_prefix}.facet.{facet}",
            )
            if picked:
                selected[facet] = set(picked)

    filtered = apply_filter(runs, RunFilter(selected))
    st.caption(f"{len(filtered)} of {len(runs)} runs match")
    return filtered


def run_label(run: dict[str, Any], *, fmt_ts) -> str:
    """
    A unique, human-readable label for one run, used by the selectbox after
    filtering. Includes run_number so repeats of the same cell are
    distinguishable (the bug this feature fixes), plus the timestamp.

    ``fmt_ts`` is the caller's timestamp formatter (views differ in tz handling).
    """
    return (
        f"{strategy_display_name(run['strategy'])} | {run['model']} | "
        f"tier {run['tier']} | {run['example_id']} | "
        f"run {run.get('run_number', '?')} | {fmt_ts(run['timestamp'])}"
    )
