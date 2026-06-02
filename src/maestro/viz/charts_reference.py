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

from maestro.viz import db as viz_db
from maestro.viz import queries as viz_queries
from maestro.viz.chart import new_figure, render_chart
from maestro.viz.components import empty_state
from maestro.viz.theme import strategy_color, strategy_display_name


def render_reference_chart(db_path: Path) -> None:
    """
    Draw the reference chart for the database at ``db_path``.

    Opens its own short-lived read-only connection (the standard per-view
    pattern), queries mean entity-ID F1 per strategy, and renders a bar chart
    colored by the strategy palette. Shows an empty-state if the database file
    is missing or has no scored runs.
    """
    # Guard the missing-file case first: a read-only (mode=ro) connect raises
    # OperationalError on a nonexistent file, which would otherwise escape the
    # empty-data check below.
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            f"No database at `{db_path}`. Run an experiment first, or update "
            "the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        data = viz_queries.mean_entity_id_f1_by_strategy(conn)

    if not data:
        empty_state(
            "No scored runs yet.",
            "Run an experiment so metric results exist, then this chart populates.",
        )
        return

    # DB stores enum values (e.g. "single_agent"); the chart shows the design
    # guide's display names (e.g. "Single Agent"). Colors look up by the raw
    # value (the mapping lives in theme.strategy_color).
    values = [row[0] for row in data]
    scores = [row[1] for row in data]
    names = [strategy_display_name(v) for v in values]
    colors = [strategy_color(v) for v in values]

    fig, ax = new_figure()
    # Fixed bar width (in category units, max 1.0) keeps a single bar from
    # stretching edge-to-edge, and leaves even gaps once several strategies
    # are present.
    bars = ax.bar(names, scores, color=colors, width=0.6)
    ax.set_ylabel("Entity-ID F1 (mean)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Strategy")
    # Pad the x-axis so a lone bar isn't flush against the spines.
    ax.margins(x=0.3 if len(names) == 1 else 0.1)

    # Place the strategy name *inside* the bar (white, vertically centered)
    # when it's tall enough to hold the text, and drop the redundant x-axis
    # tick label in that case. For short bars the name can't fit inside, so
    # keep it on the axis below instead.
    keep_xtick = []
    for bar, name in zip(bars, names):
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
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(keep_xtick)

    fig.tight_layout()

    render_chart(
        fig,
        filename="entity_id_f1_by_strategy",
        key="reference-chart",
        caption=(
            "Reference chart — mean entity-ID F1 per strategy. This is the "
            "design-system template the data views follow."
        ),
    )
