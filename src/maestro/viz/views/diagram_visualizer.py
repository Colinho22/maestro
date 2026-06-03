"""
MAESTRO viz — Diagram Visualizer view (diagnostic, no RQ mapping).

Side-by-side comparison of a run's ground-truth diagram and its generated
diagram. Ground truth is read from the file system via
experiment_config.INPUTS (the DB stores only example_id); the generated
diagram comes from run_results.

A Code / Visualization toggle switches both panes together. Visualization
renders with the mmdc CLI — the same engine the metric pipeline uses for
``parses_valid`` — so the rendered picture is consistent with the recorded
validity, deterministic, and reproducible. When mmdc is unavailable (or a
source fails to render), the pane falls back to showing the source code.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from maestro.experiment_config import INPUTS
from maestro.viz import db as viz_db
from maestro.viz import queries as viz_queries
from maestro.viz import settings as viz_settings
from maestro.viz.components import empty_state
from maestro.viz.mermaid_render import mmdc_available, render_mermaid_svg
from maestro.viz.theme import strategy_display_name

# example_id → InputFile, for resolving the ground-truth file path.
_INPUTS_BY_ID = {inp.example_id: inp for inp in INPUTS}


def render() -> None:
    """Draw the Diagram Visualizer page."""
    st.title("Diagram Visualizer")
    st.caption("Ground-truth vs. generated diagram source, side by side.")

    db_path: Path = viz_settings.current_settings().db_path
    if not viz_db.database_exists(db_path):
        empty_state(
            "Database not found.",
            "Run an experiment first, or update the path in ⚙️ Settings.",
        )
        return

    with viz_db.connect(db_path) as conn:
        runs = viz_queries.list_runs(conn)
        if not runs:
            empty_state("No runs available.")
            return

        labels = {
            r["run_id"]: (
                f"{strategy_display_name(r['strategy'])} | {r['model']} | "
                f"tier {r['tier']} | {r['example_id']}"
            )
            for r in runs
        }
        run_id = st.selectbox(
            "Run",
            options=[r["run_id"] for r in runs],
            format_func=lambda rid: labels[rid],
        )
        detail = viz_queries.run_detail(conn, run_id)

    if detail is None:
        empty_state("Selected run not found.")
        return

    # Code / Visualization toggle, applied to both panes together so the
    # comparison is always like-with-like. Visualization is only offered when
    # mmdc is installed; otherwise force Code mode with a note.
    if mmdc_available():
        mode = st.radio("Display", ["Code", "Visualization"], horizontal=True)
    else:
        mode = "Code"
        st.caption(
            "Install the Mermaid CLI (`npm install -g @mermaid-js/mermaid-cli`) "
            "to enable the Visualization mode."
        )

    _render_side_by_side(detail, render_visual=(mode == "Visualization"))


def _render_side_by_side(detail: dict, *, render_visual: bool) -> None:
    """Ground truth (left) next to generated (right), as code or rendered SVG."""
    left, right = st.columns(2)

    with left:
        st.subheader("Ground truth")
        inp = _INPUTS_BY_ID.get(detail["example_id"])
        if inp is None:
            st.info(
                f"No input registered for example_id "
                f"`{detail['example_id']}` in experiment_config.INPUTS."
            )
        else:
            _render_diagram(
                _read_or_note(inp.ground_truth_path), render_visual=render_visual
            )

    with right:
        st.subheader("Generated")
        parses = detail.get("parses_valid")
        if parses is True:
            st.success("Parses: valid", icon="✅")
        elif parses is False:
            st.error("Parses: invalid", icon="❌")
        else:
            st.warning("Parse validity: not checked", icon="❔")

        output = detail.get("output_diagram_code")
        if output:
            _render_diagram(output, render_visual=render_visual)
        else:
            st.info("No diagram produced for this run.")
        if detail.get("error"):
            st.error(f"Run error: {detail['error']}")


def _render_diagram(source: str, *, render_visual: bool) -> None:
    """
    Show ``source`` as a rendered SVG (Visualization mode) or as code. Falls
    back to code if rendering fails — e.g. invalid Mermaid the metric pipeline
    also rejected — so a broken diagram still shows its source for inspection.
    """
    if render_visual:
        svg = render_mermaid_svg(source)
        if svg is not None:
            # st.image treats a string as a path, so an SVG string can't go
            # there. Embed the SVG markup directly; constrain width so it
            # scales to the column. The SVG comes from our own mmdc render of
            # data already in the DB — not arbitrary user input.
            st.markdown(
                f'<div style="max-width:100%">{svg}</div>',
                unsafe_allow_html=True,
            )
            return
        st.caption("Could not render this source — showing code instead.")
    st.code(source, language="mermaid")


def _read_or_note(path: Path) -> str:
    """Read a text file, or return a short note if it's missing/unreadable."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"(file not readable: {path})"
