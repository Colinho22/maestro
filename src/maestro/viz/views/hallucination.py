"""
MAESTRO viz — Hallucination Taxonomy view (RQ3).

Exploratory characterization of error types per strategy: stacked bars of the
four taxonomy categories (missing / extra / false / duplicate) at the entity
and relationship levels. Controls are included — their error profile is
itself informative.

The four-category error palette is defined here rather than in the shared
theme: the design guide's section 1 covers strategy / provider / tier but not
errors yet. If this palette proves reusable it should graduate into the guide
and theme.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from maestro.viz import db as viz_db
from maestro.viz import queries as viz_queries
from maestro.viz import settings as viz_settings
from maestro.viz.chart import new_figure, render_chart
from maestro.viz.components import empty_state
from maestro.viz.theme import strategy_display_name

# Four-category error palette (view-local; candidate for the design guide).
# Distinct from the strategy/provider/tier palettes since errors co-occur with
# strategy on the same chart.
_ERROR_COLORS = {
    "missing": "#E67E22",  # dropped from truth
    "extra": "#3498DB",  # invented
    "false": "#E74C3C",  # present but wrong
    "duplicate": "#95A5A6",  # repeated
}
# Short category label (the taxonomy column suffix) → display label.
_CATEGORY_LABEL = {
    "missing": "Missing",
    "extra": "Extra",
    "false": "False",
    "duplicate": "Duplicate",
}


def render() -> None:
    """Draw the Hallucination Taxonomy page."""
    st.title("Hallucination Taxonomy")

    db_path: Path = viz_settings.current_settings().db_path
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            "Run an experiment first, or update the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        if not viz_queries.has_any_taxonomy_data(conn):
            empty_state("No hallucination / error data recorded yet.")
            return
        tiers = viz_queries.distinct_tiers(conn)
        tier = st.selectbox(
            "Tier",
            options=[None, *tiers],
            format_func=lambda t: "All tiers" if t is None else f"Tier {t}",
        )

        entity_data = viz_queries.taxonomy_counts_by_strategy(
            conn, viz_queries.ENTITY_TAXONOMY, tier=tier
        )
        rel_data = viz_queries.taxonomy_counts_by_strategy(
            conn, viz_queries.RELATIONSHIP_TAXONOMY, tier=tier
        )

    _stacked_bar(
        entity_data,
        columns=viz_queries.ENTITY_TAXONOMY,
        title="Entity errors per strategy",
        filename="entity_errors_by_strategy",
        key="halluc-entity",
    )
    _stacked_bar(
        rel_data,
        columns=viz_queries.RELATIONSHIP_TAXONOMY,
        title="Relationship errors per strategy",
        filename="relationship_errors_by_strategy",
        key="halluc-rel",
    )


def _category_of(column: str) -> str:
    """'missing_entities' / 'extra_relationships' → 'missing' / 'extra'."""
    return column.split("_", 1)[0]


def _stacked_bar(
    data: dict[str, dict[str, int]],
    *,
    columns: tuple[str, ...],
    title: str,
    filename: str,
    key: str,
) -> None:
    """Stacked bars: one bar per strategy, one segment per error category."""
    if not data:
        empty_state("No data for the current filter.", hint=title)
        return

    strategies = list(data)
    names = [strategy_display_name(s) for s in strategies]
    x = np.arange(len(strategies))
    bottom = np.zeros(len(strategies))

    fig, ax = new_figure(figsize=(9.0, 4.5))
    for col in columns:
        cat = _category_of(col)
        heights = np.array([data[s].get(col, 0) for s in strategies], dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottom,
            color=_ERROR_COLORS[cat],
            label=_CATEGORY_LABEL[cat],
        )
        # Value labels for non-zero segments (no hover on static figures).
        for xi, (h, b) in enumerate(zip(heights, bottom)):
            if h > 0:
                ax.text(
                    xi,
                    b + h / 2,
                    str(int(h)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Error count")
    ax.grid(axis="y")  # vertical bars → horizontal grid only
    ax.legend(title="Error type")
    fig.tight_layout()
    render_chart(fig, filename=filename, key=key, caption=title)
