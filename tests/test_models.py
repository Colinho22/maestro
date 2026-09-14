"""
Registry-only unit tests. Cross-layer integrity (pricing, dispatch, prose)
is enforced in ``tests/test_model_registry_consistency.py``.
"""

from __future__ import annotations

import pytest

from maestro.models import (
    MODEL_REGISTRY,
    TIER_EFFICIENCY,
    TIER_FRONTIER,
    all_internal_ids,
    get_model,
    models_by_provider,
    registered_models,
)


def test_every_entry_matches_its_key():
    """The dict's key IS the ``internal_id`` field; a mismatch would
    let ``get_model(k)`` return a spec whose ``internal_id`` was `k'`,
    silently unjoining every downstream reference."""
    for internal_id, spec in MODEL_REGISTRY.items():
        assert spec.internal_id == internal_id


def test_get_model_round_trip():
    for internal_id in all_internal_ids():
        assert get_model(internal_id).internal_id == internal_id


def test_get_model_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        get_model("not-a-real-model")


def test_get_model_error_names_known_ids():
    """The KeyError message includes the known ids so the fix path is
    a one-line diff (add the id) rather than a treasure hunt."""
    with pytest.raises(KeyError) as excinfo:
        get_model("nonsense-id")
    msg = str(excinfo.value)
    for internal_id in all_internal_ids():
        assert internal_id in msg


def test_registered_models_preserves_registry_order():
    """Downstream consumers rely on the registry order for
    display; a set-driven variant would randomise it."""
    ordered = registered_models()
    assert [spec.internal_id for spec in ordered] == list(MODEL_REGISTRY.keys())


def test_every_spec_has_a_valid_tier():
    for spec in registered_models():
        assert spec.tier in {TIER_FRONTIER, TIER_EFFICIENCY}


def test_every_provider_has_both_tiers():
    """The paired matrix design (frontier + efficiency per provider) is
    what the viz slot mapping and the results-chapter narrative depend
    on, so this shape is a registry invariant, not a coincidence."""
    by_provider: dict[str, set[str]] = {}
    for spec in registered_models():
        by_provider.setdefault(spec.provider_id, set()).add(spec.tier)

    for provider_id, tiers in by_provider.items():
        assert tiers == {TIER_FRONTIER, TIER_EFFICIENCY}, (
            f"provider '{provider_id}' has tiers {tiers}, expected both"
        )


def test_models_by_provider_filters_correctly():
    for spec in registered_models():
        results = models_by_provider(spec.provider_id)
        assert spec in results
        for other in results:
            assert other.provider_id == spec.provider_id


def test_models_by_provider_unknown_returns_empty_list():
    assert models_by_provider("no-such-provider") == []


def test_snapshot_date_extracted_when_present():
    haiku = get_model("claude-haiku-4-5-20251001")
    assert haiku.snapshot_date == "20251001"


def test_snapshot_date_is_none_when_absent():
    opus = get_model("claude-opus-4-8")
    assert opus.snapshot_date is None
