"""
MAESTRO viz — Overview view.

Operational summary (no RQ mapping): headline metric cards plus per-strategy
run-count and cost bars. Reads run_configs / run_results (and run_environments
indirectly via the environment count).
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


def render() -> None:
    """Draw the Overview page against the configured database."""
    st.title("Overview")

    db_path: Path = viz_settings.current_settings().db_path
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            "Run an experiment first, or update the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        summary = viz_queries.overview_summary(conn)
        runs_split = viz_queries.runs_by_strategy_success(conn)
        cost_split = viz_queries.total_cost_by_strategy(conn)

    if summary["total_runs"] == 0:
        empty_state("No runs recorded yet.", "Run an experiment first.")
        return

    # --- Metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", f"{summary['total_runs']:,}")
    c2.metric("Success rate", f"{summary['success_rate'] * 100:.0f}%")
    c3.metric("Total cost", f"${summary['total_cost_usd']:,.2f}")
    # Environment count is optional context — omit silently if none recorded.
    if summary["distinct_environments"]:
        c4.metric("Environments", f"{summary['distinct_environments']:,}")

    st.divider()

    # --- Runs per strategy, split by success ---
    _render_runs_chart(runs_split)
    # --- Total cost per strategy ---
    _render_cost_chart(cost_split)


def _render_runs_chart(runs_split: list[tuple[str, int, int]]) -> None:
    if not runs_split:
        empty_state("No run results to chart yet.")
        return
    names = [strategy_display_name(s) for s, _, _ in runs_split]
    successes = [s for _, s, _ in runs_split]
    failures = [f for _, _, f in runs_split]

    fig, ax = new_figure()
    # Stacked: success (strategy color) + failure (muted) per strategy.
    ax.bar(names, successes, color="#1ABC9C", label="Success")
    ax.bar(names, failures, bottom=successes, color="#E74C3C", label="Failure")
    ax.set_ylabel("Runs")
    ax.set_xlabel("Strategy")
    ax.grid(axis="y")  # vertical bars → horizontal grid only
    ax.legend()
    fig.tight_layout()
    render_chart(
        fig,
        filename="runs_by_strategy",
        key="overview-runs",
        caption="Run count per strategy, split by success / failure.",
    )


def _render_cost_chart(cost_split: list[tuple[str, float]]) -> None:
    if not cost_split:
        return
    names = [strategy_display_name(s) for s, _ in cost_split]
    costs = [c for _, c in cost_split]
    colors = [strategy_color(s) for s, _ in cost_split]

    fig, ax = new_figure()
    ax.bar(names, costs, color=colors)
    ax.set_ylabel("Total cost (USD)")
    ax.set_xlabel("Strategy")
    ax.grid(axis="y")
    fig.tight_layout()
    render_chart(
        fig,
        filename="cost_by_strategy",
        key="overview-cost",
        caption="Total cost (USD) per strategy.",
    )
