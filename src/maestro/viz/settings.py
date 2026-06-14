"""
MAESTRO viz — view settings (the "gear" panel at the bottom of the sidebar).

This is the foundation for *view-configuration* settings, kept distinct from
*navigation* (the view list). Today it holds the two knobs whose backends
already exist:

- **Database path** — which experiment DB to read (env ``MAESTRO_DB_PATH``,
  falling back to the project-default ``maestro.db``), overridable in the UI.
- **Display timezone** — reuses ``maestro.analysis.timestamps`` (storage stays
  UTC; only display converts), env ``MAESTRO_DISPLAY_TZ``.

Future knobs (primary correctness metric, include/exclude controls, table
precision) slot in as additional fields on ``ViewSettings`` plus a widget in
``render_settings_panel`` — no structural change needed.

Resolution precedence for each setting: explicit UI value (held in
``st.session_state``) → environment variable → built-in default. The UI value
is seeded from the env/default on first load, so a user who never opens the
gear panel still gets sensible behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from maestro.analysis.timestamps import ENV_VAR as TZ_ENV_VAR
from maestro.analysis.timestamps import resolve_display_tz
from maestro.experiment_config import DB_PATH as DEFAULT_DB_PATH

# Environment variable that overrides the database path (mirrors the analysis
# CLI's --db default of experiment_config.DB_PATH).
DB_PATH_ENV_VAR = "MAESTRO_DB_PATH"

# st.session_state keys — namespaced so they can't collide with widget keys
# a view might register.
_SS_DB_PATH = "settings.db_path"
_SS_DISPLAY_TZ = "settings.display_tz"


@dataclass(frozen=True)
class ViewSettings:
    """
    Resolved, read-only snapshot of the current view settings.

    Frozen so a view can pass it around without worrying that a downstream
    widget mutates it; the canonical mutable state lives in
    ``st.session_state`` and a fresh snapshot is built each rerun by
    ``current_settings``.
    """

    db_path: Path
    # display_tz is the *raw* string the user/env chose (or None for system
    # local); call format helpers in timestamps.py to apply it. Validation
    # happens at render time via resolve_display_tz.
    display_tz: str | None


def _default_db_path() -> str:
    """Env-var DB path if set, else the project default. Returned as a string
    (session_state holds strings; Path is built at snapshot time)."""
    return os.environ.get(DB_PATH_ENV_VAR) or str(DEFAULT_DB_PATH)


def _default_display_tz() -> str:
    """
    Env-var display timezone if set, else empty string meaning "system local".
    Empty string (not None) because Streamlit text inputs return "" when blank;
    current_settings normalizes "" back to None.
    """
    return os.environ.get(TZ_ENV_VAR) or ""


def init_settings() -> None:
    """
    Seed session_state defaults once per session. Idempotent: only sets keys
    that are absent, so a user's later edits survive reruns.
    """
    st.session_state.setdefault(_SS_DB_PATH, _default_db_path())
    st.session_state.setdefault(_SS_DISPLAY_TZ, _default_display_tz())


def current_settings() -> ViewSettings:
    """Build a resolved ViewSettings snapshot from current session_state."""
    init_settings()
    tz_raw = st.session_state.get(_SS_DISPLAY_TZ, "").strip()
    return ViewSettings(
        db_path=Path(st.session_state[_SS_DB_PATH]).expanduser(),
        display_tz=tz_raw or None,
    )


def format_timestamp(ts: str | None) -> str:
    """
    Format a stored UTC timestamp string for display in the configured tz.

    Shared by the run selectors so timestamps render consistently. Returns ""
    for an empty value and passes a non-ISO string through unchanged.
    """
    from datetime import datetime

    from maestro.analysis.timestamps import format_for_display

    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return ts
    return format_for_display(dt, current_settings().display_tz)


def render_settings_panel() -> None:
    """
    Render the gear/settings controls. Call inside the sidebar (the caller
    decides placement — e.g. inside an ``st.expander`` pinned at the bottom).

    Widgets write straight into the namespaced session_state keys via their
    ``key=`` argument, so the next ``current_settings`` call reflects edits
    with no extra plumbing.
    """
    init_settings()

    st.text_input(
        "Database path",
        key=_SS_DB_PATH,
        help=(
            "Path to the experiment SQLite database (read-only). Defaults to "
            f"${DB_PATH_ENV_VAR} or {DEFAULT_DB_PATH.name}. Point this at a "
            "dev DB to inspect it without restarting."
        ),
    )

    st.text_input(
        "Display timezone",
        key=_SS_DISPLAY_TZ,
        placeholder="system local (e.g. Europe/Zurich)",
        help=(
            "IANA timezone for displayed timestamps. Storage stays UTC; this "
            f"only affects display. Blank = system local. Overrides ${TZ_ENV_VAR}."
        ),
    )

    # Validate the TZ immediately so the user sees a problem here, not as a
    # silent UTC fallback later. resolve_display_tz emits a stderr warning and
    # falls back to UTC on an unknown zone; we surface that in-UI too.
    tz_raw = st.session_state.get(_SS_DISPLAY_TZ, "").strip()
    if tz_raw:
        resolved = resolve_display_tz(tz_raw)
        if str(resolved) == "UTC" and tz_raw.upper() != "UTC":
            st.warning(
                f"Unknown timezone {tz_raw!r} — falling back to UTC.",
                icon="⚠️",
            )
