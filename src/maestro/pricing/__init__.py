"""
Versioned, date-stamped pricing snapshots.

Prices decay the moment a vendor updates a page, and cost is one of the
selection axes MAESTRO reports on, so every recorded run must be pinned to
the exact rates that produced its ``cost_usd``. This package holds the
snapshots and exposes two things:

- ``load_pricing(version)`` returns the ``ModelPricing`` list frozen into that
  snapshot, plus its version string. ``experiment_config.MODELS`` is derived
  from the default snapshot's pricing so a cost table change is a config bump,
  never a stray literal in application code.
- ``get_pricing(model_id, version)`` looks up a single model's rates. It
  raises loudly on an unknown model so a typo becomes a run-time failure with
  a message, not a silent zero-cost row.

**Adding a new snapshot.** Copy the newest ``snapshot_YYYY_MM.py`` into a new
dated file (for example ``snapshot_2026_10.py``), update its top-level
``VERSION`` string to the new ``YYYY-MM`` identifier and the ``PRICING`` list to
the current rates, add it to ``_VERSIONS`` below (keyed by the same string),
and bump ``DEFAULT_VERSION`` to point at it. Historical runs stay pinned to
the snapshot that was default when they ran; cross-snapshot cost comparisons
are a research decision the analyst opts into, never a silent side effect.
"""

from __future__ import annotations

import re

from maestro.pricing import snapshot_2026_04
from maestro.schemas import ModelPricing

# The version string every new run records under until a fresh snapshot is
# introduced. Deliberately the raw string rather than a re-export of the
# snapshot module's ``VERSION`` so a typo in either would fail loudly at
# import via ``_validate``, and so it is grep-able as a literal identifier
# in commit diffs and archived docs.
DEFAULT_VERSION = "2026-04"


# Strict shape a snapshot key must satisfy: four-digit year, hyphen, two-digit
# month 01 to 12. A malformed key sorts oddly in ``available_versions`` and
# reads confusingly in the DB, so ``_validate`` rejects it at import time
# before any run is written under it.
_KEY_PATTERN = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


# All snapshots the package knows about, keyed by their ISO-year-month
# version string. Adding a new snapshot means adding a row here in addition
# to bumping ``DEFAULT_VERSION``; the import-time ``_validate`` below refuses
# to load an inconsistent state.
_VERSIONS: dict[str, tuple[str, list[ModelPricing]]] = {
    snapshot_2026_04.VERSION: (snapshot_2026_04.VERSION, snapshot_2026_04.PRICING),
}


# Synthetic pricing row for control strategies (null / copy / ground-truth):
# they never invoke an LLM, so no real model applies. Kept out of the dated
# snapshots because it is not vendor pricing: rebasing every snapshot on the
# same zero-cost stub would just duplicate the row. ``run.py`` reaches for it
# directly via ``CONTROL_MODEL``.
CONTROL_MODEL = ModelPricing(
    model="control",
    input_price_per_1m=0.0,
    output_price_per_1m=0.0,
)


def available_versions() -> tuple[str, ...]:
    """Sorted tuple of every registered snapshot version, oldest first."""
    return tuple(sorted(_VERSIONS))


def load_pricing(version: str | None = None) -> tuple[str, list[ModelPricing]]:
    """
    Return ``(version, pricing)`` for the requested snapshot.

    ``version`` is a date-stamped identifier of the form ``YYYY-MM`` (for
    example ``"2026-04"``); each snapshot file publishes its own such string.
    ``None`` resolves to ``DEFAULT_VERSION`` so callers that just want "the
    current rates" (the experiment runner, most tests) do not need to name a
    version. Unknown versions raise ``KeyError`` with the list of known ones,
    so a typo fails immediately rather than silently loading the default.

    Returns the version string alongside the list so callers that persist
    provenance (``run_environments.pricing_version``, the CLI banner) never
    have to synthesise it and cannot disagree about what "current" means.
    The returned list is a shared reference, but each ``ModelPricing`` row
    is a frozen Pydantic model, so callers cannot mutate a snapshot's rates
    in place and poison another caller's read.
    """
    key = version if version is not None else DEFAULT_VERSION
    try:
        return _VERSIONS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown pricing version {key!r}. Known: {', '.join(available_versions())}"
        ) from exc


def get_pricing(model_id: str, version: str | None = None) -> ModelPricing:
    """
    Return the ``ModelPricing`` row for one model under one snapshot.

    ``version`` follows the same ``YYYY-MM`` date-stamped convention as
    ``load_pricing``; ``None`` uses ``DEFAULT_VERSION``. ``model_id`` matches
    ``ModelPricing.model`` (the same string the canonical registry keys on).
    The synthetic ``"control"`` id resolves to ``CONTROL_MODEL`` regardless of
    snapshot: controls do not use an LLM, and pretending otherwise would put
    zero-cost rows under a vendor's namespace.

    Raises ``KeyError`` with the list of known ids on a miss. Silent zero was
    the old behaviour and the reason this module exists: a typo in a run
    config would produce a plausible ``$0.000000`` row with no complaint,
    which is exactly the drift versioned pricing has to prevent.
    """
    if model_id == CONTROL_MODEL.model:
        return CONTROL_MODEL
    _, pricing = load_pricing(version)
    for row in pricing:
        if row.model == model_id:
            return row
    known = ", ".join(sorted(row.model for row in pricing))
    resolved = version if version is not None else DEFAULT_VERSION
    raise KeyError(
        f"Unknown model {model_id!r} in pricing snapshot {resolved!r}. Known: {known}"
    )


def _validate() -> None:
    """
    Import-time invariants for the whole package.

    Enforcing them at import time (not lazily) means a broken snapshot fails
    the process before any run starts, not halfway through the matrix when
    the missing row is finally reached. Four shapes of drift are caught:

    - a snapshot registered under a key that is not a valid ``YYYY-MM``
      identifier (year plus month 01 to 12), which would sort oddly and
      read confusingly wherever the id is joined against;
    - a snapshot module whose ``VERSION`` disagrees with its ``_VERSIONS``
      key (rename typo);
    - a snapshot with duplicate ``ModelPricing.model`` rows (would silently
      shadow a rate);
    - a ``DEFAULT_VERSION`` that names a snapshot no longer registered
      (typo in the bump).
    """
    for key in _VERSIONS:
        if not _KEY_PATTERN.fullmatch(key):
            raise RuntimeError(
                f"pricing snapshot key {key!r} is not a valid YYYY-MM identifier "
                f"(four-digit year, hyphen, two-digit month 01 to 12)"
            )
    if DEFAULT_VERSION not in _VERSIONS:
        raise RuntimeError(
            f"pricing DEFAULT_VERSION {DEFAULT_VERSION!r} not in _VERSIONS: "
            f"{sorted(_VERSIONS)}"
        )
    for key, (declared_version, rows) in _VERSIONS.items():
        if key != declared_version:
            raise RuntimeError(
                f"pricing snapshot key {key!r} disagrees with its VERSION "
                f"{declared_version!r}; rename one so they match"
            )
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in rows:
            if row.model in seen:
                duplicates.add(row.model)
            seen.add(row.model)
        if duplicates:
            raise RuntimeError(
                f"pricing snapshot {key!r} has duplicate model rows: "
                f"{sorted(duplicates)}"
            )


_validate()


__all__ = [
    "CONTROL_MODEL",
    "DEFAULT_VERSION",
    "available_versions",
    "get_pricing",
    "load_pricing",
]
