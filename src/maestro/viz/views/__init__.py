"""
MAESTRO viz: view registry.

A *view* is a module exposing a no-argument ``render()`` that draws one
dashboard page (reading the DB via ``viz.db`` / ``viz.queries``). The sidebar
navigation in ``app.py`` is driven by the ``VIEWS`` registry below: each entry
is a (label, render-callable) pair, rendered when selected.

The registry holds a "Home" landing view plus the data views (Overview,
Strategy Comparison, Pareto, Run Detail, Diagram Visualizer, Hallucination
Taxonomy). Each data view lives in its own ``views/<name>.py`` module exposing
``render()``.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from maestro.viz import settings as viz_settings
from maestro.viz.charts_reference import render_reference_chart
from maestro.viz.views import (
    diagram_visualizer,
    hallucination,
    overview,
    pareto,
    run_detail,
    strategy_comparison,
)


def _render_home() -> None:
    """
    The Home landing view: orients the user and shows the design-system
    reference chart against the configured database.
    """
    st.title("MAESTRO: Results Dashboard")
    st.write(
        "Read-only dashboard over the experiment database. Use the sidebar to "
        "open a view; configure the database path and display timezone under "
        "⚙️ Settings."
    )
    st.divider()
    # Bound the chart to a left-hand portion of the wide page so the figure
    # sits in a defined region instead of dictating the full-width layout.
    chart_col, _ = st.columns([2, 1])
    with chart_col:
        render_reference_chart(viz_settings.current_settings().db_path)


# (label, render) pairs in sidebar order: Home landing, then the data views.
VIEWS: list[tuple[str, Callable[[], None]]] = [
    ("Home", _render_home),
    ("Overview", overview.render),
    ("Strategy Comparison", strategy_comparison.render),
    ("Pareto", pareto.render),
    ("Run Detail", run_detail.render),
    ("Diagram Visualizer", diagram_visualizer.render),
    ("Hallucination Taxonomy", hallucination.render),
]
