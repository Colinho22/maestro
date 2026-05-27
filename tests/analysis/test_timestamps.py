"""
Tests for the display-timezone helpers.

The helpers convert UTC datetimes (the DB's storage form) into
human-readable strings in a configurable timezone, used at analysis
display time. Storage stays UTC; this module only governs presentation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from maestro.analysis.timestamps import (
    ENV_VAR,
    format_for_display,
    resolve_display_tz,
)

# A fixed UTC moment used as the input across most cases. Using a
# concrete instant (mid-summer in the northern hemisphere, where Europe
# is on CEST = UTC+2) makes the expected output predictable.
SUMMER_UTC = datetime(2026, 6, 15, 21, 43, 0, tzinfo=timezone.utc)


def test_explicit_tz_arg_wins(monkeypatch):
    """The display_tz kwarg overrides the env var."""
    monkeypatch.setenv(ENV_VAR, "Asia/Tokyo")
    out = format_for_display(SUMMER_UTC, display_tz="Europe/Zurich")
    # 21:43 UTC + 2h DST = 23:43 CEST
    assert "23:43" in out
    assert "CEST" in out
    # %Z produces zone abbreviations, not full names — assert the
    # abbreviation Asia/Tokyo *would* have produced is absent, which
    # is the meaningful "env var was ignored" check.
    assert "JST" not in out


def test_env_var_used_when_arg_absent(monkeypatch):
    """MAESTRO_DISPLAY_TZ is honoured when no explicit arg is given."""
    monkeypatch.setenv(ENV_VAR, "Europe/Zurich")
    out = format_for_display(SUMMER_UTC)
    assert "23:43" in out
    assert "CEST" in out


def test_system_local_default(monkeypatch):
    """No arg, no env var → falls back to system local without crashing.

    We can't pin the exact output because the test runner's local zone
    varies, but the call must produce *some* well-formed string with
    a zone abbreviation in it.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)
    out = format_for_display(SUMMER_UTC)
    assert "2026-06-15" in out
    # Should contain *some* zone abbreviation token — not literally
    # checking which one because that depends on the runner.
    assert len(out.split()) >= 3  # date + time + zone abbr


def test_malformed_tz_falls_back_to_utc(monkeypatch, capsys):
    """An unknown zone name (via kwarg) produces UTC + a stderr warning."""
    out = format_for_display(SUMMER_UTC, display_tz="Mars/Olympus_Mons")
    assert "UTC" in out
    assert "21:43" in out  # unchanged because UTC
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "Mars/Olympus_Mons" in captured.err
    # The warning identifies the source as "argument" (the kwarg path).
    assert "argument" in captured.err


def test_malformed_env_var_falls_back_to_utc(monkeypatch, capsys):
    """A malformed MAESTRO_DISPLAY_TZ takes the env-var branch of the
    warning code path — different stderr text than the kwarg branch."""
    monkeypatch.setenv(ENV_VAR, "Mars/Olympus_Mons")
    out = format_for_display(SUMMER_UTC)
    assert "UTC" in out
    assert "21:43" in out
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "Mars/Olympus_Mons" in captured.err
    # The warning identifies the source as the env-var name, so the
    # user knows where to look. Distinguishes from the kwarg branch
    # above which says "argument".
    assert f"${ENV_VAR}" in captured.err


def test_naive_input_treated_as_utc():
    """
    Datetimes that lost their tzinfo (e.g. via fromisoformat edge
    cases) are interpreted as UTC, matching the DB storage assumption.
    """
    naive = SUMMER_UTC.replace(tzinfo=None)
    out_aware = format_for_display(SUMMER_UTC, display_tz="Europe/Zurich")
    out_naive = format_for_display(naive, display_tz="Europe/Zurich")
    assert out_aware == out_naive


def test_resolve_display_tz_returns_zoneinfo_for_named_zone(monkeypatch):
    """resolve_display_tz should produce a usable tzinfo object."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    tz = resolve_display_tz("Europe/Zurich")
    assert isinstance(tz, ZoneInfo)
    # Round-trip a datetime through it to prove it's wired correctly.
    converted = SUMMER_UTC.astimezone(tz)
    assert converted.hour == 23


@pytest.mark.parametrize(
    "tz_name,expected_hour,expected_abbr",
    [
        ("UTC", 21, "UTC"),
        ("Europe/Zurich", 23, "CEST"),  # UTC+2 in summer
        ("America/Los_Angeles", 14, "PDT"),  # UTC-7 in summer
        ("Asia/Tokyo", 6, "JST"),  # UTC+9, next-day clock (Jun 16 06:43)
    ],
)
def test_known_zones_convert_correctly(
    tz_name: str, expected_hour: int, expected_abbr: str
):
    """Spot-check a handful of zones against known DST offsets."""
    out = format_for_display(SUMMER_UTC, display_tz=tz_name)
    assert f"{expected_hour:02d}:43" in out, (
        f"expected {expected_hour:02d}:43 in output for {tz_name}, got: {out}"
    )
    assert expected_abbr in out, (
        f"expected '{expected_abbr}' in output for {tz_name}, got: {out}"
    )
