"""
Canonical model registry: the single source of truth for every model name
MAESTRO uses.

Every place that names a model (pricing table, provider dispatch, viz
palettes, docs, tests) resolves through ``get_model`` or one of the
membership helpers. That eliminates the class of drift where the same
model appears under two spellings in two places, and it gives future
issues (pricing config, model-specific behaviour) a stable join key.

A dated snapshot is a distinct registry entry, not a mutable attribute
(the ``ModelSpec`` docstring in ``schemas.py`` records why): swapping
``claude-haiku-4-5-20251001`` for a newer snapshot means adding a new
row here, not editing this one.
"""

from __future__ import annotations

import re

from maestro.schemas import ModelSpec

# Frontier == flagship; efficiency == smaller/cheaper. Two tiers per provider
# is the shape MAESTRO's paired matrix depends on; if that changes, this
# constant and viz theme's frontier/efficiency slot mapping move together.
TIER_FRONTIER = "frontier"
TIER_EFFICIENCY = "efficiency"

# Regex over an internal id's terminal segment. Two vendor shapes are
# accepted: 8 consecutive digits (``20251001``) and the hyphenated form
# ``YYYY-MM-DD`` (``2026-04-23``). The captured group is normalised to the
# compact 8-digit form on return so downstream comparators do not need to
# know which shape the vendor picked. 4 consecutive digits do not count
# (that is a version number, e.g. ``mistral-small-2603``). Loose match
# is deliberate: a vendor's convention is not something the registry can
# normalise beyond this compact-form step, only expose.
_SNAPSHOT_DATE_PATTERN = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


def _snapshot_of(internal_id: str) -> str | None:
    """Extract a snapshot date from the tail of an internal id, compact form."""
    match = _SNAPSHOT_DATE_PATTERN.search(internal_id)
    if match is None:
        return None
    return match.group(1).replace("-", "")


def _spec(
    internal_id: str,
    *,
    provider_id: str,
    display_name: str,
    provider_display_name: str,
    tier: str,
) -> ModelSpec:
    """Build a ``ModelSpec`` and auto-derive its snapshot date if present."""
    return ModelSpec(
        internal_id=internal_id,
        provider_id=provider_id,
        display_name=display_name,
        provider_display_name=provider_display_name,
        tier=tier,
        snapshot_date=_snapshot_of(internal_id),
    )


# ---------------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------------
#
# Order in this list is the order every downstream consumer displays models
# in (viz legends, printed tables, docs generation). Alphabetical by
# provider, frontier-then-efficiency within each provider: the same shape
# ``experiment_config.MODELS`` uses, kept in step so the two layers can be
# diffed at a glance.

_REGISTRY_ENTRIES: tuple[ModelSpec, ...] = (
    _spec(
        "claude-opus-4-8",
        provider_id="anthropic",
        display_name="Claude Opus 4.8",
        provider_display_name="Claude",
        tier=TIER_FRONTIER,
    ),
    _spec(
        "claude-haiku-4-5-20251001",
        provider_id="anthropic",
        display_name="Claude Haiku 4.5",
        provider_display_name="Claude",
        tier=TIER_EFFICIENCY,
    ),
    _spec(
        "gpt-5.5-2026-04-23",
        provider_id="openai",
        display_name="GPT-5.5",
        provider_display_name="ChatGPT",
        tier=TIER_FRONTIER,
    ),
    _spec(
        "gpt-5.4-mini-2026-03-17",
        provider_id="openai",
        display_name="GPT-5.4 mini",
        provider_display_name="ChatGPT",
        tier=TIER_EFFICIENCY,
    ),
    _spec(
        "mistral-medium-3-5",
        provider_id="mistral",
        display_name="Mistral Medium 3.5",
        provider_display_name="Mistral",
        tier=TIER_FRONTIER,
    ),
    _spec(
        "mistral-small-2603",
        provider_id="mistral",
        display_name="Mistral Small",
        provider_display_name="Mistral",
        tier=TIER_EFFICIENCY,
    ),
    _spec(
        "gemini-3.5-flash",
        provider_id="gemini",
        display_name="Gemini 3.5 Flash",
        provider_display_name="Gemini",
        tier=TIER_FRONTIER,
    ),
    _spec(
        "gemini-3.1-flash-lite",
        provider_id="gemini",
        display_name="Gemini 3.1 Flash Lite",
        provider_display_name="Gemini",
        tier=TIER_EFFICIENCY,
    ),
    _spec(
        "deepseek-v4-pro",
        provider_id="deepseek",
        display_name="DeepSeek V4 Pro",
        provider_display_name="DeepSeek",
        tier=TIER_FRONTIER,
    ),
    _spec(
        "deepseek-v4-flash",
        provider_id="deepseek",
        display_name="DeepSeek V4 Flash",
        provider_display_name="DeepSeek",
        tier=TIER_EFFICIENCY,
    ),
)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.internal_id: spec for spec in _REGISTRY_ENTRIES
}


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_model(internal_id: str) -> ModelSpec:
    """
    Resolve an internal id to its ``ModelSpec``. Raises ``KeyError`` with a
    clear message on an unknown id: silently returning ``None`` would let a
    typo propagate into a chart or a pricing lookup, which is exactly the
    drift this registry exists to prevent.
    """
    spec = MODEL_REGISTRY.get(internal_id)
    if spec is None:
        raise KeyError(
            f"Unknown model internal_id: {internal_id!r}. "
            f"Known: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return spec


def all_internal_ids() -> set[str]:
    """Set of every registered internal id (used by consistency scans)."""
    return set(MODEL_REGISTRY)


def registered_models() -> list[ModelSpec]:
    """Ordered list of every registered ``ModelSpec`` (registry order)."""
    return list(_REGISTRY_ENTRIES)


def models_by_provider(provider_id: str) -> list[ModelSpec]:
    """
    Every ``ModelSpec`` whose ``provider_id`` matches, in registry order.
    Empty list on an unknown provider id: the caller decides whether that
    is an error (a viz palette missing a provider) or a natural no-op (a
    key-filtered run that excluded the whole vendor).
    """
    return [spec for spec in _REGISTRY_ENTRIES if spec.provider_id == provider_id]
