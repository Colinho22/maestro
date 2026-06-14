"""
MAESTRO viz: Strategy Comparison view (RQ1, RQ2).

Grouped bars of entity- and relationship-level correctness per orchestration
strategy, filterable by tier and model, with a precision / recall / F1 toggle.
Controls are excluded (this is a comparison of strategies under test).
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
from maestro.viz.theme import strategy_color, strategy_display_name

# Metric suffix per toggle choice.
_SUFFIX = {"F1": "f1", "Precision": "precision", "Recall": "recall"}


def render() -> None:
    """Draw the Strategy Comparison page."""
    st.title("Strategy Comparison")

    db_path: Path = viz_settings.current_settings().db_path
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            "Run an experiment first, or update the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        if not viz_queries.has_any_metrics(conn):
            empty_state("No metric results yet.")
            return
        tiers = viz_queries.distinct_tiers(conn)
        models = viz_queries.distinct_models(conn)

        # --- Filters (shared across both charts via widget state) ---
        fcol1, fcol2, fcol3 = st.columns([1, 2, 1])
        tier = fcol1.selectbox("Tier", options=tiers, format_func=lambda t: f"Tier {t}")
        selected_models = fcol2.multiselect("Models", options=models, default=models)
        measure = fcol3.radio("Measure", list(_SUFFIX), horizontal=True)
        suffix = _SUFFIX[measure]

        entity_cols = [f"{m}_{suffix}" for m in viz_queries.ENTITY_METRICS]
        rel_cols = [f"{m}_{suffix}" for m in viz_queries.RELATIONSHIP_METRICS]

        entity_data = viz_queries.metric_means_by_strategy(
            conn, entity_cols, tier=tier, models=selected_models or None
        )
        rel_data = viz_queries.metric_means_by_strategy(
            conn, rel_cols, tier=tier, models=selected_models or None
        )

    _grouped_bar(
        entity_data,
        entity_cols,
        labels=["ID", "Name", "Lemma"],
        title=f"Entity {measure} per strategy",
        ylabel=f"Entity {measure}",
        filename=f"entity_{suffix}_by_strategy",
        key="strat-entity",
    )
    _grouped_bar(
        rel_data,
        rel_cols,
        labels=["Relaxed", "Strict"],
        title=f"Relationship {measure} per strategy",
        ylabel=f"Relationship {measure}",
        filename=f"relationship_{suffix}_by_strategy",
        key="strat-rel",
    )


def _grouped_bar(
    data: dict[str, dict[str, float]],
    columns: list[str],
    *,
    labels: list[str],
    title: str,
    ylabel: str,
    filename: str,
    key: str,
) -> None:
    """
    Grouped bars: one group per strategy, one bar per metric column. Strategy
    identity is the bar color; the metric variant is distinguished by position
    + legend. Empty-state per chart when the filter yields nothing.
    """
    if not data:
        empty_state("No data for the current filter.", hint=title)
        return

    strategies = list(data)
    n_groups = len(strategies)
    n_bars = len(columns)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = new_figure(figsize=(9.0, 4.5))
    for i, (col, label) in enumerate(zip(columns, labels)):
        # Each metric variant offset within the strategy's group. Use the
        # strategy color but vary alpha across variants so the strategy stays
        # identifiable and the variant is distinguishable.
        heights = [data[s].get(col, 0.0) for s in strategies]
        offset = (i - (n_bars - 1) / 2) * width
        colors = [strategy_color(s) for s in strategies]
        ax.bar(
            x + offset,
            heights,
            width,
            label=label,
            color=colors,
            alpha=1.0 - 0.22 * i,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([strategy_display_name(s) for s in strategies])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.legend(title="Variant")
    fig.tight_layout()
    render_chart(fig, filename=filename, key=key, caption=title)
