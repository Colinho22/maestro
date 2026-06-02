"""
MAESTRO viz — reference chart.

A single, complete chart that exercises the whole rendering path
(query → themed figure → display + PNG/SVG export). It is the copy-paste
starting point for the data views: a new view follows this exact shape —
open a read-only connection, run a query, build a figure with ``new_figure``
and the named palette, hand it to ``render_chart``.

The chart itself — mean entity-ID F1 per strategy — is deliberately simple;
its job is to prove the design system works end to end against real data.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from maestro.viz import db as viz_db
from maestro.viz import queries as viz_queries
from maestro.viz.chart import new_figure, render_chart
from maestro.viz.components import empty_state
from maestro.viz.theme import strategy_color


def render_reference_chart(db_path: Path) -> None:
    """
    Draw the reference chart for the database at ``db_path``.

    Opens its own short-lived read-only connection (the standard per-view
    pattern), queries mean entity-ID F1 per strategy, and renders a bar chart
    colored by the strategy palette. Shows an empty-state if no metrics exist.
    """
    with viz_db.connect(db_path) as conn:
        data = viz_queries.mean_entity_id_f1_by_strategy(conn)

    if not data:
        empty_state(
            "No scored runs yet.",
            "Run an experiment so metric results exist, then this chart populates.",
        )
        return

    strategies = [row[0] for row in data]
    scores = [row[1] for row in data]
    colors = [strategy_color(s) for s in strategies]

    fig, ax = new_figure()
    # Fixed bar width (in category units, max 1.0) keeps a single bar from
    # stretching edge-to-edge, and leaves even gaps once several strategies
    # are present.
    bars = ax.bar(strategies, scores, color=colors, width=0.6)
    ax.set_ylabel("Entity-ID F1 (mean)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Strategy")
    # Pad the x-axis so a lone bar isn't flush against the spines.
    ax.margins(x=0.3 if len(strategies) == 1 else 0.1)

    # Place the strategy name *inside* the bar (white, vertically centered)
    # when it's tall enough to hold the text, and drop the redundant x-axis
    # tick label in that case. For short bars the name can't fit inside, so
    # keep it on the axis below instead.
    keep_xtick = []
    for bar, name in zip(bars, strategies):
        x = bar.get_x() + bar.get_width() / 2
        if bar.get_height() >= 0.25:
            ax.text(
                x,
                bar.get_height() / 2,
                name,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
            keep_xtick.append("")
        else:
            keep_xtick.append(name)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(keep_xtick)

    fig.tight_layout()

    render_chart(
        fig,
        filename="entity_id_f1_by_strategy",
        key="reference-chart",
        caption="Reference chart — mean entity-ID F1 per strategy.",
    )

    st.caption(
        "This is the design-system reference chart: the template the data views follow."
    )
