"""
MAESTRO viz: Run Detail view (diagnostic, no RQ mapping).

Pick one run and inspect it: the input spec + ground truth (read from the
file system via experiment_config.INPUTS), the generated diagram with a
parse-validity badge, a per-run metric breakdown, and the sub-call trace for
multi-step strategies.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from maestro.experiment_config import INPUTS
from maestro.viz import db as viz_db
from maestro.viz import queries as viz_queries
from maestro.viz import settings as viz_settings
from maestro.viz.chart import new_figure, render_chart
from maestro.viz.components import empty_state, render_run_filter, run_label
from maestro.viz.run_filter import exclude_controls

# example_id -> InputFile, for resolving input + ground-truth file paths. The
# DB stores only example_id; the actual files live on disk per the config.
_INPUTS_BY_ID = {inp.example_id: inp for inp in INPUTS}

# Per-run metric breakdown: the score columns shown as a horizontal bar.
_METRIC_COLUMNS = [
    "entity_id_f1",
    "entity_name_f1",
    "entity_lemma_f1",
    "relationship_relaxed_f1",
    "relationship_strict_f1",
]


def render() -> None:
    """Draw the Run Detail page."""
    st.title("Run Detail")

    db_path: Path = viz_settings.current_settings().db_path
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            "Run an experiment first, or update the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        # Controls (null/copy/ground-truth) echo input/ground-truth rather than
        # a generated diagram, so drop them from this per-run selector.
        runs = exclude_controls(viz_queries.list_runs(conn))
        if not runs:
            empty_state("No runs available.")
            return

        # Faceted filter, then a selectbox over the narrowed set. run_label
        # includes run_number so repeats of the same cell are distinguishable.
        filtered = render_run_filter(runs, key_prefix="run_detail")
        if not filtered:
            empty_state("No runs match the active filter.")
            return

        labels = {
            r["run_id"]: run_label(r, fmt_ts=viz_settings.format_timestamp)
            for r in filtered
        }
        run_id = st.selectbox(
            "Run",
            options=[r["run_id"] for r in filtered],
            format_func=lambda rid: labels[rid],
        )

        detail = viz_queries.run_detail(conn, run_id)
        subs = viz_queries.sub_results_for_run(conn, run_id)

    if detail is None:
        empty_state("Selected run not found.")
        return

    _render_io(detail)
    st.divider()
    _render_metric_breakdown(detail)
    if subs:
        st.divider()
        _render_sub_trace(subs)


def _render_io(detail: dict) -> None:
    """Left: input spec + ground truth (from disk). Right: generated diagram."""
    left, right = st.columns(2)

    inp = _INPUTS_BY_ID.get(detail["example_id"])
    with left:
        st.subheader("Input & ground truth")
        if inp is None:
            st.info(
                f"No input registered for example_id "
                f"`{detail['example_id']}` in experiment_config.INPUTS."
            )
        else:
            st.markdown("**Input spec**")
            st.code(_read_or_note(inp.file_path), language="json")
            st.markdown("**Ground truth**")
            st.code(_read_or_note(inp.ground_truth_path), language="mermaid")

    with right:
        st.subheader("Generated diagram")
        parses = detail.get("parses_valid")
        if parses is True:
            st.success("Parses: valid", icon="✅")
        elif parses is False:
            st.error("Parses: invalid", icon="❌")
        else:
            st.warning("Parse validity: not checked", icon="❔")
        output = detail.get("output_diagram_code")
        if output:
            st.code(output, language="mermaid")
        else:
            st.info("No diagram produced for this run.")
        if detail.get("error"):
            st.error(f"Run error: {detail['error']}")


def _read_or_note(path: Path) -> str:
    """Read a text file, or return a short note if it's missing/unreadable."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"(file not readable: {path})"


def _render_metric_breakdown(detail: dict) -> None:
    """Horizontal bar of the run's F1 scores. Neutral color: one run only."""
    st.subheader("Metric breakdown")
    values = [(col, detail.get(col)) for col in _METRIC_COLUMNS]
    values = [(c, v) for c, v in values if v is not None]
    if not values:
        st.info("No metric scores recorded for this run.")
        return

    labels = [c.replace("_f1", "").replace("_", " ") for c, _ in values]
    scores = [float(v) for _, v in values]

    fig, ax = new_figure(figsize=(7.0, 4.0))
    ax.barh(labels, scores, color="#7F8C8D")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.grid(axis="x")  # horizontal bars -> vertical grid only
    ax.invert_yaxis()  # first metric on top
    fig.tight_layout()
    render_chart(
        fig,
        filename=f"run_{str(detail['run_id'])[:8]}_metrics",
        key="run-detail-metrics",
        caption="F1 scores for the selected run.",
    )


def _render_sub_trace(subs: list[dict]) -> None:
    """Expandable trace of each sub-call for a multi-step strategy."""
    st.subheader("Sub-call trace")
    for s in subs:
        header = f"Step {s['step_number']}: {s['step_name']}"
        with st.expander(header):
            cols = st.columns(3)
            cols[0].metric("Prompt tokens", f"{s['prompt_tokens']:,}")
            cols[1].metric("Completion tokens", f"{s['completion_tokens']:,}")
            cols[2].metric("Cost", f"${s['cost_usd']:.6f}")
            if s.get("output_text"):
                st.markdown("**Output**")
                st.code(s["output_text"])
            if s.get("error"):
                st.error(f"Sub-call error: {s['error']}")
