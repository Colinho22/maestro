"""
Tests for the versioned pricing package.

Pricing is the join key between token counts and reported cost, so drift
here shows up as wrong dollar figures on published charts. Three shapes
are pinned:

- **Shape of the public API.** ``load_pricing`` / ``get_pricing`` accept
  a ``YYYY-MM`` version string, return a stable structure, and reject
  unknown ids with a message that names the offender.
- **Snapshot integrity.** Each dated snapshot's ``VERSION`` matches its
  ``_VERSIONS`` key, its ``PRICING`` list has no duplicate model rows,
  and ``DEFAULT_VERSION`` points at a registered snapshot.
- **Config wiring.** ``experiment_config.MODELS`` is the default
  snapshot's pricing list (not a hardcoded copy), and the environment
  probe records the same version string on every captured run.
"""

from __future__ import annotations

import pytest

from maestro import pricing
from maestro.db.environment import capture_environment
from maestro.experiment_config import DEFAULT_PRICING_VERSION, MODELS, PRICING_VERSION
from maestro.pricing import (
    CONTROL_MODEL,
    DEFAULT_VERSION,
    available_versions,
    get_pricing,
    load_pricing,
    snapshot_2026_04,
)
from maestro.schemas import ModelPricing

# ---------------------------------------------------------------------------
# Version string format
# ---------------------------------------------------------------------------


def test_default_version_is_iso_year_month():
    """The version string is a date stamp, not a semver: enforce the shape."""
    assert DEFAULT_VERSION == "2026-04"
    parts = DEFAULT_VERSION.split("-")
    assert len(parts) == 2
    year, month = parts
    assert year.isdigit() and len(year) == 4
    assert month.isdigit() and len(month) == 2
    assert 1 <= int(month) <= 12


def test_snapshot_module_version_matches_default():
    """The snapshot file declares the same string DEFAULT_VERSION points at."""
    assert snapshot_2026_04.VERSION == DEFAULT_VERSION


def test_available_versions_lists_the_default():
    versions = available_versions()
    assert DEFAULT_VERSION in versions
    # available_versions is sorted so a future addition slots in
    # deterministically and this test does not need updating on that path.
    assert list(versions) == sorted(versions)


# ---------------------------------------------------------------------------
# load_pricing
# ---------------------------------------------------------------------------


def test_load_pricing_default_returns_version_and_rows():
    version, rows = load_pricing()
    assert version == DEFAULT_VERSION
    assert rows, "default snapshot has no pricing rows"
    assert all(isinstance(row, ModelPricing) for row in rows)


def test_load_pricing_explicit_version_matches_default():
    v_default, rows_default = load_pricing()
    v_explicit, rows_explicit = load_pricing(DEFAULT_VERSION)
    assert v_default == v_explicit
    assert rows_default == rows_explicit


def test_load_pricing_unknown_version_raises_with_message():
    with pytest.raises(KeyError) as excinfo:
        load_pricing("1999-13")
    assert "1999-13" in str(excinfo.value)
    # The message names the known versions so the user does not have to grep.
    assert DEFAULT_VERSION in str(excinfo.value)


def test_snapshot_rows_have_no_duplicate_model_ids():
    """A duplicate row would silently shadow one rate; catch it up front."""
    _, rows = load_pricing()
    ids = [row.model for row in rows]
    assert len(ids) == len(set(ids)), (
        f"duplicate model rows: {sorted({m for m in ids if ids.count(m) > 1})}"
    )


# ---------------------------------------------------------------------------
# get_pricing
# ---------------------------------------------------------------------------


def test_get_pricing_returns_row_for_known_model():
    row = get_pricing("claude-opus-4-8")
    assert isinstance(row, ModelPricing)
    assert row.model == "claude-opus-4-8"
    # Sanity: the rate matches the snapshot value; a silent typo in the
    # snapshot would be caught by the model-registry consistency suite,
    # but this test also fails if get_pricing accidentally normalises the
    # rate (rounding, currency conversion).
    assert row.input_price_per_1m == 5.00
    assert row.output_price_per_1m == 25.00


def test_get_pricing_control_model_bypasses_snapshot():
    """
    The ``"control"`` id has no vendor pricing; it resolves to CONTROL_MODEL
    regardless of the requested version so control rows keep zero cost even
    if a snapshot forgot to include it (which it should never do).
    """
    row = get_pricing(CONTROL_MODEL.model)
    assert row is CONTROL_MODEL
    assert row.input_price_per_1m == 0.0
    assert row.output_price_per_1m == 0.0


def test_get_pricing_unknown_model_raises_loudly():
    """The old silent-zero-cost behaviour is what this feature exists to remove."""
    with pytest.raises(KeyError) as excinfo:
        get_pricing("no-such-model-xyz")
    assert "no-such-model-xyz" in str(excinfo.value)
    # Version and known ids are named so the error is actionable.
    assert DEFAULT_VERSION in str(excinfo.value)


def test_get_pricing_unknown_version_raises():
    with pytest.raises(KeyError):
        get_pricing("claude-opus-4-8", version="1999-13")


# ---------------------------------------------------------------------------
# Wiring into experiment_config and environment capture
# ---------------------------------------------------------------------------


def test_experiment_config_models_is_default_snapshot_pricing():
    """
    ``experiment_config.MODELS`` is derived, not hardcoded: it must equal the
    default snapshot's pricing list row-for-row so a snapshot bump propagates
    without a stray literal being left behind.
    """
    _, expected = load_pricing()
    assert MODELS == expected


def test_experiment_config_pricing_version_matches_default():
    assert PRICING_VERSION == DEFAULT_VERSION
    assert DEFAULT_PRICING_VERSION == DEFAULT_VERSION


def test_capture_environment_records_pricing_version():
    """
    ``run_environments.pricing_version`` is the join key from a cost figure
    back to the rates that produced it, so the probe must record the active
    snapshot id (not None) on a healthy install.
    """
    env = capture_environment()
    assert env.pricing_version == DEFAULT_VERSION


# ---------------------------------------------------------------------------
# _validate: import-time invariants
# ---------------------------------------------------------------------------


def test_validate_detects_key_version_mismatch(monkeypatch):
    """A snapshot registered under the wrong key would silently return the
    wrong rates on a lookup; _validate must catch that at import time."""
    bad = {
        "1999-01": ("1999-02", []),  # key and VERSION disagree
    }
    monkeypatch.setattr(pricing, "_VERSIONS", bad)
    with pytest.raises(RuntimeError) as excinfo:
        pricing._validate()
    assert "1999-01" in str(excinfo.value) or "1999-02" in str(excinfo.value)


def test_validate_detects_duplicate_rows(monkeypatch):
    row = ModelPricing(model="dup", input_price_per_1m=1.0, output_price_per_1m=2.0)
    bad = {
        DEFAULT_VERSION: (DEFAULT_VERSION, [row, row]),
    }
    monkeypatch.setattr(pricing, "_VERSIONS", bad)
    with pytest.raises(RuntimeError) as excinfo:
        pricing._validate()
    assert "dup" in str(excinfo.value)


def test_validate_detects_orphan_default(monkeypatch):
    monkeypatch.setattr(pricing, "DEFAULT_VERSION", "9999-99")
    with pytest.raises(RuntimeError) as excinfo:
        pricing._validate()
    assert "9999-99" in str(excinfo.value)


@pytest.mark.parametrize("bad_key", ["04-2026", "2026-13", "2026-1", "2026-00"])
def test_validate_rejects_malformed_key(monkeypatch, bad_key):
    """
    A snapshot key that is not a strict ``YYYY-MM`` identifier would sort
    oddly against real ones and read confusingly wherever the id is joined,
    so ``_validate`` must reject it at import time.
    """
    monkeypatch.setattr(pricing, "_VERSIONS", {bad_key: (bad_key, [])})
    with pytest.raises(RuntimeError) as excinfo:
        pricing._validate()
    assert bad_key in str(excinfo.value)
    assert "YYYY-MM" in str(excinfo.value)


def test_validate_accepts_current_snapshot_key():
    """Positive case: the shipped ``2026-04`` key satisfies the regex."""
    # No monkeypatching: we exercise the real _VERSIONS to confirm the
    # pattern does not over-reject the snapshot the package ships with.
    pricing._validate()
