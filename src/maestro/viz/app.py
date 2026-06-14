"""
MAESTRO viz: Streamlit entry point.

Run with:

    streamlit run src/maestro/viz/app.py

Responsibilities:
- Page config + sidebar navigation driven by the ``views.VIEWS`` registry.
- A "gear" settings expander below the nav (DB path, display timezone),
  see ``viz.settings``, followed by a footer.
- Resolve the configured DB and, if it is missing, show an empty-state for
  the whole app rather than letting every view raise.
"""

from __future__ import annotations

import streamlit as st

from maestro.viz import db as viz_db
from maestro.viz import settings as viz_settings
from maestro.viz.components import empty_state
from maestro.viz.views import VIEWS


def _render_sidebar() -> str:
    """
    Draw the sidebar: title + view navigation, a settings expander, then a
    footer separated by a fixed vertical gap. Returns the selected view's
    label.

    The footer sits a fixed gap below settings rather than pinned to the
    viewport bottom: true bottom-anchoring needs flexbox CSS against
    Streamlit's internal sidebar DOM, which proved version-fragile, so that
    polish is left to the design-system styling pass. The fixed gap reads
    cleanly on any window height in the meantime.
    """
    st.sidebar.title("MAESTRO")
    labels = [label for label, _ in VIEWS]
    selected = st.sidebar.radio("View", labels, label_visibility="collapsed")
    st.sidebar.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        viz_settings.render_settings_panel()

    # Fixed gap, then single-line footer credits (middot-separated).
    st.sidebar.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size:0.78rem; color:#888;'>"
        "MAESTRO Viz · 2026 · "
        "👾 <a href='https://github.com/Colinho22/maestro' target='_blank' "
        "style='color:#888;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )

    return selected


def _render_view(label: str) -> None:
    """Dispatch to the selected view's render() by label."""
    for view_label, render in VIEWS:
        if view_label == label:
            render()
            return
    # Defensive: a label with no matching view should never happen (the nav is
    # built from the same registry), but fail visibly rather than silently.
    st.error(f"Unknown view: {label!r}")


def main() -> None:
    """Compose the page: config, sidebar, DB guard, then the selected view."""
    st.set_page_config(
        page_title="MAESTRO",
        page_icon="🎼",
        layout="wide",
    )

    viz_settings.init_settings()
    selected = _render_sidebar()
    cfg = viz_settings.current_settings()

    # App-wide DB guard: if the configured database is absent, no view can
    # show anything useful: surface one clear empty-state instead of letting
    # each view fail its own way.
    if not viz_db.database_exists(cfg.db_path):
        empty_state(
            f"Database not found at `{cfg.db_path}`.",
            "Run an experiment, or set the database path in ⚙️ Settings "
            "(sidebar) / the MAESTRO_DB_PATH environment variable.",
        )
        return

    _render_view(selected)


# Streamlit executes this module top-to-bottom on every rerun, so call main()
# at import time (the conventional Streamlit entry-point pattern) rather than
# guarding behind __main__: `streamlit run` imports the module, it doesn't
# exec it as __main__.
main()
