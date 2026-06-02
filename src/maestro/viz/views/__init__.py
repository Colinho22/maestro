"""
MAESTRO viz — view registry.

A *view* is a module exposing a no-argument ``render()`` that draws one
dashboard page (reading the DB via ``viz.db`` / ``viz.queries``). The sidebar
navigation in ``app.py`` is driven by the ``VIEWS`` registry below: each entry
is a (label, render-callable) pair, rendered when selected.

The registry currently holds a live "Home" view that confirms the
navigation + settings + empty-state wiring, plus placeholder entries for the
planned data views. Each planned view becomes a ``views/<name>.py`` module
exposing ``render()``, appended to ``VIEWS``; until implemented it shows a
placeholder card.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from maestro.viz import settings as viz_settings
from maestro.viz.charts_reference import render_reference_chart

# Names of the planned data views, shown in the nav as placeholders so the
# eventual structure is visible before each is implemented.
_PLANNED_VIEWS: tuple[str, ...] = (
    "Overview",
    "Strategy Comparison",
    "Pareto",
    "Run Detail",
    "Hallucination Taxonomy",
)


def _render_placeholder() -> None:
    """
    The Home view: confirms the app is wired up and shows the design-system
    reference chart against the configured database.
    """
    st.title("MAESTRO — Results Dashboard")
    st.write(
        "Navigation, read-only database access, settings, and empty-state "
        "handling are in place. Select a planned view from the sidebar to see "
        "its placeholder."
    )
    st.divider()
    # Bound the chart to a left-hand portion of the wide page so the figure
    # sits in a defined region instead of dictating the full-width layout.
    chart_col, _ = st.columns([2, 1])
    with chart_col:
        render_reference_chart(viz_settings.current_settings().db_path)


def _make_planned_placeholder(name: str) -> Callable[[], None]:
    """Build a render() that shows a 'not yet implemented' card for ``name``."""

    def _render() -> None:
        st.title(name)
        st.info(
            f"The **{name}** view is not implemented yet.",
            icon="🚧",
        )

    return _render


# (label, render) pairs in nav order. The live placeholder first, then the
# planned views as placeholders.
VIEWS: list[tuple[str, Callable[[], None]]] = [
    ("Home", _render_placeholder),
    *[(name, _make_planned_placeholder(name)) for name in _PLANNED_VIEWS],
]
