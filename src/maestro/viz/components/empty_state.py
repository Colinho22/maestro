"""
MAESTRO viz: reusable empty-state banner.

Every view renders this when its query returns no rows (no runs yet, a tier
with no data, missing extended-tier columns, etc.) instead of showing a blank
page or a stack trace. Implemented once here and imported everywhere so the
empty experience is consistent across views.
"""

from __future__ import annotations

import streamlit as st


def empty_state(message: str, hint: str | None = None) -> None:
    """
    Render a centered empty-state banner.

    ``message`` is the headline ("No runs recorded yet."). ``hint`` is an
    optional secondary line with a next step ("Run an experiment first.").
    Uses ``st.info`` so it reads as neutral guidance rather than an error:
    an empty dataset is an expected state in this dashboard, not a failure.
    """
    body = f"**{message}**"
    if hint:
        body += f"\n\n{hint}"
    st.info(body, icon="📭")
