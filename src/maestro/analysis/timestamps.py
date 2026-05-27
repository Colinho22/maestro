"""
MAESTRO — Timezone-aware timestamp display helpers.

The DB stores every timestamp in UTC ISO-8601 (see schemas.py:
``RunConfig.timestamp`` and ``RunEnvironment.captured_at`` both default to
``datetime.now(timezone.utc)``). That's the right canonical form for
storage: unambiguous, sortable, replication-friendly.

Display is a different concern. A researcher reading "did run #42 hit the
US/China API peak?" wants the local wall-clock time, not UTC. This module
converts the stored UTC datetimes to a configurable display timezone and
formats them with an explicit abbreviation so the result is unambiguous
when shared across timezones.

Storage stays UTC; only display surfaces (analysis tables, plot axes, log
output, CSV columns — once #19/#24 exist) call into here.

## Configuration precedence

1. ``display_tz`` kwarg passed to ``format_for_display`` / ``resolve_display_tz``
   — for analysis CLIs that want to expose a ``--display-tz`` flag.
2. ``MAESTRO_DISPLAY_TZ`` environment variable — for persistent dev-machine
   config via ``.env`` (e.g. ``MAESTRO_DISPLAY_TZ=Europe/Zurich``).
3. System local timezone (``datetime.now().astimezone().tzinfo``) — the
   sensible default when neither override is set.

If an explicit value is provided but malformed (typo, unknown zone), we
fall back to UTC and emit a one-line stderr warning rather than crashing.
A display-formatting bug shouldn't abort an analysis run.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Name of the environment variable that overrides the display timezone.
# Documented here (not just inline) so future contributors can find it
# by grepping for the constant rather than the literal string.
ENV_VAR = "MAESTRO_DISPLAY_TZ"


def resolve_display_tz(display_tz: str | None = None) -> ZoneInfo | timezone:
    """
    Resolve the timezone to use for display, applying the precedence
    chain documented at module level.

    Returns a ``ZoneInfo`` for named zones (IANA database) or the
    ``timezone`` instance for the system local zone. Both are
    ``tzinfo`` subclasses, so callers can use the return value with
    ``datetime.astimezone(tz)`` uniformly.
    """
    # 1. Explicit kwarg from a caller's CLI flag wins.
    name = display_tz
    # 2. Fall back to the env var.
    if name is None:
        name = os.environ.get(ENV_VAR)

    if name:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            source = "argument" if display_tz else f"${ENV_VAR}"
            print(
                f"WARN: unknown timezone {name!r} from {source}; falling back to UTC.",
                file=sys.stderr,
            )
            return timezone.utc

    # 3. System local. ``datetime.now().astimezone()`` is the standard
    #    idiom for "use the current process's local timezone" — it
    #    yields a tzinfo whose ``tzname()`` resolves the local
    #    abbreviation (CEST, PST, etc.) which we want in the output.
    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is None:
        # Vanishingly rare — only on systems where Python can't
        # determine local TZ at all. Be explicit instead of returning
        # None and letting astimezone() fail later.
        return timezone.utc
    return local_tz


def format_for_display(
    utc_dt: datetime,
    display_tz: str | None = None,
    *,
    fmt: str = "%Y-%m-%d %H:%M %Z",
) -> str:
    """
    Convert a UTC datetime (as stored in the DB) into a human-readable
    string in the resolved display timezone.

    ``utc_dt`` may be timezone-aware (UTC) or naive — naive inputs are
    treated as UTC, matching the storage assumption. This is permissive
    on purpose: ISO-8601 strings from the DB round-trip through pydantic
    as aware datetimes, but downstream code that parses them with
    ``datetime.fromisoformat`` on Python 3.10 can produce naive objects
    on edge cases.

    Default ``fmt`` includes ``%Z`` so the output carries the zone
    abbreviation (e.g. ``2026-05-08 23:43 CEST``) — explicit timezones
    in shared results matter more than character count.
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    local = utc_dt.astimezone(resolve_display_tz(display_tz))
    return local.strftime(fmt)
