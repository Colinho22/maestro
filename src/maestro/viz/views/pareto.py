"""
MAESTRO viz: Pareto view (RQ4).

Correctness vs. efficiency scatter: entity-ID F1 against cost and against
latency, colored by strategy and shaped by tier. Since matplotlib figures
have no hover, an accompanying data table carries the per-point detail.
Controls excluded.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from maestro.viz import db as viz_db
from maestro.viz import queries as viz_queries
from maestro.viz import settings as viz_settings
from maestro.viz.chart import new_figure, render_chart
from maestro.viz.components import empty_state
from maestro.viz.theme import strategy_color, strategy_display_name

# Tier -> matplotlib marker shape. Defined here (not the theme) as it is a
# Pareto-specific encoding.
_TIER_MARKER = {1: "o", 2: "s", 3: "D"}
_DEFAULT_MARKER = "o"


def render() -> None:
    """Draw the Pareto page."""
    st.title("Pareto: correctness vs. efficiency")

    db_path: Path = viz_settings.current_settings().db_path
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            "Run an experiment first, or update the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        all_strategies = viz_queries.distinct_strategies(conn)
        all_tiers = viz_queries.distinct_tiers(conn)

        fcol1, fcol2 = st.columns(2)
        sel_strategies = fcol1.multiselect(
            "Strategies",
            options=all_strategies,
            default=all_strategies,
            format_func=strategy_display_name,
        )
        sel_tiers = fcol2.multiselect(
            "Tiers",
            options=all_tiers,
            default=all_tiers,
            format_func=lambda t: f"Tier {t}",
        )

        points = viz_queries.pareto_points(
            conn,
            strategies=sel_strategies or None,
            tiers=sel_tiers or None,
        )

    distinct_present = {p["strategy"] for p in points}
    if len(distinct_present) < 2:
        empty_state(
            "Add results from at least two strategies to see the Pareto comparison.",
            f"Currently {len(distinct_present)} strategy with data.",
        )
        return

    _scatter(
        points,
        x_field="cost_usd",
        xlabel="Cost (USD)",
        filename="pareto_cost",
        key="pareto-cost",
    )
    _scatter(
        points,
        x_field="duration_ms",
        xlabel="Latency (ms)",
        filename="pareto_latency",
        key="pareto-latency",
    )

    # No hover on static figures: a table carries the per-run detail.
    st.caption("Per-run detail")
    st.dataframe(
        [
            {
                "run": str(p["run_id"])[:8],
                "strategy": strategy_display_name(p["strategy"]),
                "model": p["model"],
                "tier": p["tier"],
                "cost_usd": round(p["cost_usd"], 6),
                "latency_ms": p["duration_ms"],
                "entity_id_f1": round(p["entity_id_f1"], 4),
            }
            for p in points
        ],
        use_container_width=True,
        hide_index=True,
    )


def _scatter(
    points: list[dict],
    *,
    x_field: str,
    xlabel: str,
    filename: str,
    key: str,
) -> None:
    """Scatter of entity_id_f1 (y) vs ``x_field`` (x); color=strategy, marker=tier."""
    fig, ax = new_figure(figsize=(8.0, 5.0))

    # One scatter call per (strategy, tier) combo so color + marker are set
    # consistently and the legend stays meaningful.
    seen_strategies: dict[str, str] = {}
    seen_tiers: set[int] = set()
    for p in points:
        ax.scatter(
            p[x_field],
            p["entity_id_f1"],
            color=strategy_color(p["strategy"]),
            marker=_TIER_MARKER.get(p["tier"], _DEFAULT_MARKER),
            s=60,
            edgecolors="white",
            linewidths=0.5,
        )
        seen_strategies[p["strategy"]] = strategy_color(p["strategy"])
        seen_tiers.add(p["tier"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entity-ID F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="both")  # scatter -> both-axis grid

    # Two legends: color = strategy, marker = tier.
    from matplotlib.lines import Line2D

    color_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color,
            label=strategy_display_name(s),
        )
        for s, color in sorted(seen_strategies.items())
    ]
    tier_handles = [
        Line2D(
            [],
            [],
            marker=_TIER_MARKER.get(t, _DEFAULT_MARKER),
            linestyle="",
            color="#333333",
            label=f"Tier {t}",
        )
        for t in sorted(seen_tiers)
    ]
    first = ax.legend(handles=color_handles, title="Strategy", loc="lower right")
    ax.add_artist(first)
    ax.legend(handles=tier_handles, title="Tier", loc="upper left")

    fig.tight_layout()
    render_chart(
        fig,
        filename=filename,
        key=key,
        caption=f"Entity-ID F1 vs. {xlabel.lower()} (color = strategy, marker = tier).",
    )
