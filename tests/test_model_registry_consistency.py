"""
Consistency checks for the canonical model registry.

Two shapes of drift are blocked in CI here so the registry stays the single
source of truth its docstring claims:

1. **Name drift.** A model referenced by a name not in ``MODEL_REGISTRY``
   (a typo in pricing, a lingering literal in the viz or docs) fails the
   scan. The scan is deliberately strict: it walks tracked files under
   ``src/`` (excluding the registry module), ``docs/``, ``README.md``, and
   ``CHANGELOG.md``, and asserts every model-shaped literal it finds is a
   registered internal id.

2. **Number drift.** When ``output/analysis/reported_numbers.json`` exists
   (produced by ``python -m maestro.analysis.reported_numbers``), any
   docs-referenced total that appears verbatim in tracked prose must match
   the value from the file. If the file does not exist yet (a fresh
   checkout with an empty DB), the numeric check self-skips: the goal is
   to prevent silent divergence, not to force a run before the DB has
   data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from maestro.experiment_config import CONTROL_MODEL, MODELS
from maestro.models import MODEL_REGISTRY, all_internal_ids, get_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "maestro"
_DOCS_ROOT = _REPO_ROOT / "docs"
_REPORTED_NUMBERS_PATH = _REPO_ROOT / "output" / "analysis" / "reported_numbers.json"

# Files exempt from the model-literal scan. The registry module and its
# tests are the authoritative source; scanning them would just re-flag
# every entry. ``analysis_tables.ipynb`` and ``core-visualization.ipynb``
# hold viz-layer display maps that should migrate to the registry once the
# notebook-to-figures extraction lands, but they are notebooks (JSON, not
# code review targets) and blocking on them right now would delay the
# blocking check itself. Tracked here explicitly so the exemption is
# reviewable.
_SCAN_EXEMPT_RELATIVE: frozenset[Path] = frozenset(
    {
        Path("src/maestro/models.py"),
        Path("src/maestro/viz/analysis_tables.ipynb"),
        Path("src/maestro/viz/core-visualization.ipynb"),
        Path("tests/test_model_registry_consistency.py"),
    }
)

# Prose-scan file globs relative to the repo root. Deliberately narrow to
# code and human-authored docs; the DB, generated notebooks output, and
# vendor lock files are outside the review surface.
_SCAN_GLOBS: tuple[tuple[Path, str], ...] = (
    (_SRC_ROOT, "**/*.py"),
    (_DOCS_ROOT, "**/*.md"),
    (_REPO_ROOT, "README.md"),
    (_REPO_ROOT, "CHANGELOG.md"),
    (_REPO_ROOT / "tests", "**/*.py"),
)

# Regex that matches a model-shaped internal id: a known provider needle
# followed by a hyphen and a version / snapshot tail. Same needles
# ``run.py`` dispatches on, so a legitimate identifier here always
# resolves. Requiring an explicit hyphen after the needle rules out python
# module paths (``mistralai``), CamelCase class names (``DeepSeekProvider``,
# excluded also by case-sensitive matching), snake_case identifiers
# (``deepseek_p``), and vendor domains (``mistral.ai``, ``deepseek.com``,
# ``deepseek.py``). Case sensitive: real model ids are lowercase.
_PROVIDER_NEEDLES = ("claude", "gpt", "mistral", "gemini", "deepseek")
_MODEL_LITERAL_PATTERN = re.compile(
    r"\b(" + "|".join(_PROVIDER_NEEDLES) + r")-[a-z0-9][a-z0-9.\-]*"
)

# Substrings whose match is deliberately not a target model id. Kept
# narrow on purpose: broadening it here would defeat the point of the
# scan. Each entry is a literal (lowercase, matches the pattern above).
_LITERAL_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Generic version families referenced in provider docstrings.
        # Not registered because they are examples of a lineage, not the
        # pinned snapshot the matrix runs.
        "gpt-4o",
        "gpt-5",
        "gpt-5-family",
        "gpt-5.4-mini",
        "gpt-5.5",
        "mistral-large",
        "mistral-small",
        # Historical model literals mentioned in provider comments to
        # document past behaviour; not part of the active matrix.
        "claude-haiku-4-5",
        # Test fixture used by legacy viz tests: a synthetic id that
        # exercises the display path without needing a registry entry.
        # Migration to a registered id is tracked separately from the
        # registry rollout.
        "gpt-4o-mini-2024-07-18",
        # Design-guide palette label (docs/visualization_design_guide.md).
        "claude-coral",
        # URL fragment for Google's Gemini API documentation, not a model id.
        "gemini-api",
    }
)


def _tracked_files() -> list[Path]:
    """Every reviewable file matched by ``_SCAN_GLOBS``, excluding exemptions."""
    seen: set[Path] = set()
    files: list[Path] = []
    for root, pattern in _SCAN_GLOBS:
        if not root.exists():
            continue
        if pattern.startswith("**"):
            candidates = root.glob(pattern)
        else:
            # Single-file glob, e.g. README.md.
            candidate = root / pattern
            candidates = [candidate] if candidate.exists() else []
        for path in candidates:
            if not path.is_file():
                continue
            rel = path.relative_to(_REPO_ROOT)
            if rel in _SCAN_EXEMPT_RELATIVE:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            files.append(path)
    return files


def _find_unregistered_literals(text: str) -> set[str]:
    """
    Model-shaped literals in ``text`` that are neither in the registry
    nor on the allowlist. Match is case-sensitive lowercase (the pattern
    itself enforces it), which lets ``MistralProvider`` and other
    CamelCase references pass without an explicit exemption.
    """
    unregistered: set[str] = set()
    for match in _MODEL_LITERAL_PATTERN.finditer(text):
        raw = match.group(0)
        # Strip trailing punctuation that regex does not eat (a comma,
        # backtick, closing paren) so the compare is against the actual
        # identifier the author wrote.
        candidate = raw.rstrip(".,:;)\"'`")
        if candidate in MODEL_REGISTRY:
            continue
        if candidate in _LITERAL_ALLOWLIST:
            continue
        unregistered.add(candidate)
    return unregistered


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_registry_is_non_empty():
    """A registry with no entries would silently pass every downstream check."""
    assert MODEL_REGISTRY, "MODEL_REGISTRY is empty"


def test_get_model_returns_matching_spec():
    for internal_id in all_internal_ids():
        spec = get_model(internal_id)
        assert spec.internal_id == internal_id


def test_get_model_raises_keyerror_with_message_on_unknown():
    with pytest.raises(KeyError) as excinfo:
        get_model("no-such-model-xyz")
    # The error message names the offending id AND lists the registered
    # ones; a bare KeyError would leave the user to grep.
    assert "no-such-model-xyz" in str(excinfo.value)
    assert "claude" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Pricing table and provider dispatch resolve through the registry
# ---------------------------------------------------------------------------


def test_every_pricing_entry_is_registered():
    """
    Every real ``ModelPricing`` row (all except the synthetic ``control``
    model) has a matching registry entry, so cost calculations and
    scoring cannot resolve a row that no other layer recognises.
    """
    for mp in MODELS:
        if mp.model == CONTROL_MODEL.model:
            continue
        assert mp.model in MODEL_REGISTRY, (
            f"pricing entry '{mp.model}' has no matching registry row"
        )


def test_every_provider_dispatch_target_resolves_through_registry():
    """
    Each provider needle in ``run.py:_PROVIDER_DISPATCH`` claims at least
    one registered model (so a needle without a live user gets caught).
    """
    from maestro.run import _PROVIDER_DISPATCH, _dispatch_for_model

    needles = {needle for needle, _, _ in _PROVIDER_DISPATCH}
    covered: set[str] = set()
    for spec in MODEL_REGISTRY.values():
        dispatch = _dispatch_for_model(spec.internal_id)
        assert dispatch is not None, (
            f"registered model '{spec.internal_id}' does not dispatch to a provider"
        )
        needle, _, _ = dispatch
        covered.add(needle)
    assert needles == covered, (
        f"provider dispatch needles have no registered users: {needles - covered}"
    )


# ---------------------------------------------------------------------------
# Prose scan: no unregistered model literals in tracked files
# ---------------------------------------------------------------------------


def test_no_unregistered_model_literals_in_tracked_files():
    """
    Every model-shaped literal in tracked code and docs resolves through
    the registry. Deliberately strict: the whole point of the registry is
    that a rename touches one place, not five.
    """
    offenders: dict[str, set[str]] = {}
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        bad = _find_unregistered_literals(text)
        if bad:
            offenders[str(path.relative_to(_REPO_ROOT))] = bad

    assert not offenders, (
        "Unregistered model-shaped literals found. Add them to "
        "maestro.models.MODEL_REGISTRY or, if they are not model ids, "
        "extend _LITERAL_ALLOWLIST with a one-line justification. "
        f"Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# Numeric drift: docs match the machine-readable results dump
# ---------------------------------------------------------------------------


# Numbers we want the docs to keep in step with the generated file. Each
# entry is (json_field, list_of_literals_that_may_appear_in_docs).
# ``total_cost_usd`` is compared as a float; the count fields are compared
# with and without thousands separators (docs use both).
_TRANSCRIBED_NUMBERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("total_runs", ("total_runs",)),
    ("successes", ("successes",)),
    ("failures", ("failures",)),
    ("total_cost_usd", ("total_cost_usd",)),
)


def _load_reported_numbers() -> dict | None:
    """Return the JSON payload if present, else ``None`` (skip signal)."""
    if not _REPORTED_NUMBERS_PATH.exists():
        return None
    return json.loads(_REPORTED_NUMBERS_PATH.read_text(encoding="utf-8"))


def _prose_files() -> list[Path]:
    """Human-authored prose subset of ``_tracked_files``: docs + top-level."""
    return [
        path
        for path in _tracked_files()
        if path.suffix.lower() in {".md"} or path.name in {"README.md", "CHANGELOG.md"}
    ]


def test_docs_totals_match_reported_numbers_file():
    """
    If a reported-numbers dump exists, docs must not contradict it. On a
    fresh checkout (no DB, no dump) this self-skips: the check exists to
    catch drift, not to require the DB be populated before merge.
    """
    payload = _load_reported_numbers()
    if payload is None:
        pytest.skip(
            "output/analysis/reported_numbers.json not present; "
            "run `python -m maestro.analysis.reported_numbers` to enable."
        )
    if payload.get("status") == "empty":
        pytest.skip("reported_numbers.json reports an empty database")

    # Build the set of literal strings each field is allowed to appear as
    # in prose. Integers accept a bare form and a thousands-separator form
    # (docs use both), floats accept a two-decimal string.
    allowed: dict[str, set[str]] = {}
    for field, _labels in _TRANSCRIBED_NUMBERS:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, int):
            allowed[field] = {str(value), f"{value:,}"}
        elif isinstance(value, float):
            allowed[field] = {f"{value:.2f}", f"{value:,.2f}"}
        else:
            allowed[field] = {str(value)}

    if not allowed:
        pytest.skip("reported_numbers.json has no comparable fields")

    # Sanity: whenever a prose file mentions "USD ..." right before a
    # numeric literal, that literal must match total_cost_usd. Similarly
    # for the cell count. Kept pattern-driven rather than free-form so
    # this can catch reworded prose that keeps the numbers.
    mismatches: list[str] = []
    cost_pattern = re.compile(r"USD\s+(\d+(?:,\d{3})*(?:\.\d+)?)")
    cell_pattern = re.compile(r"(\d[\d,]*)\s*(?:evaluated\s+)?cells?", re.IGNORECASE)

    for path in _prose_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        cost_allowed = allowed.get("total_cost_usd")
        if cost_allowed is not None:
            for match in cost_pattern.finditer(text):
                if match.group(1) not in cost_allowed:
                    mismatches.append(
                        f"{path.relative_to(_REPO_ROOT)}: cost {match.group(1)} "
                        f"disagrees with total_cost_usd={payload['total_cost_usd']}"
                    )

        runs_allowed = allowed.get("total_runs")
        if runs_allowed is not None:
            for match in cell_pattern.finditer(text):
                # "5 repeats" and similar short numeric contexts are noise;
                # only flag when the number itself is large enough to be
                # the cell count (four+ digits).
                literal = match.group(1)
                cleaned = literal.replace(",", "")
                if cleaned.isdigit() and len(cleaned) >= 4:
                    if literal not in runs_allowed:
                        mismatches.append(
                            f"{path.relative_to(_REPO_ROOT)}: cell count "
                            f"{literal} disagrees with total_runs="
                            f"{payload['total_runs']}"
                        )

    assert not mismatches, (
        "Transcribed number(s) in prose disagree with "
        f"{_REPORTED_NUMBERS_PATH.relative_to(_REPO_ROOT)}. "
        f"Regenerate the file or update the prose. Mismatches: {mismatches}"
    )
