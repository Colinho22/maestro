"""
Failure-mode classification for runs that produced no valid output.

The harness separates reliability (did a run produce anything usable) from
accuracy (how good was it). The accuracy side is analysed in detail by
``statistics.py``; without this module the whole reliability side collapses
to a single failure count, which is not enough to say *why* a strategy is
less reliable.

## What counts as a failure

Two disjoint shapes, both of which ``RunResult.success`` rejects:

- an **errored** run: ``error`` is set, and the string is the classification
  evidence.
- a **silent-empty** run: ``error`` is None but the diagram is missing or
  blank. No error string exists, so these classify as ``EMPTY_OUTPUT`` from
  the row shape alone. They are a real category, not a data defect: a
  provider can return whitespace without raising.

## Where the evidence lives

For a failed *run*, ``run_results.raw_response`` is NULL: the top-level error
path builds its result before any text exists. The failing text is retained
one level down, on the ``sub_results`` row that failed. Classification
therefore reads the run's ``error`` string and, when a caller supplies it,
the failed sub-result's ``raw_response``. That is why ``classify_failure``
takes the raw text as a separate optional argument rather than digging it
out of the run row: the run row does not have it.

## Precedence

Categories are not mutually exclusive in practice: a truncated response is
usually *also* a JSON parse error, because the truncation is what broke the
parse. Rather than multi-label (which makes rates hard to sum and compare),
each failure gets one primary cause under a fixed precedence, most specific
first. ``_RULES`` is ordered, and the first match wins:

1. Infrastructure causes that preempt any output judgement (rate limit,
   timeout, API error, safety block). If the API never returned, nothing
   downstream is meaningful.
2. Empty output. A missing response cannot be a parse error.
3. Truncation, checked *before* the parse rules so a cut-off response is
   attributed to the token limit that caused it rather than to the parse
   error that is merely its symptom.
4. Schema and parse violations, the ordinary "model wrote the wrong thing"
   causes.
5. Orchestration errors: the framework itself misbehaved.
6. ``UNKNOWN`` as the explicit fallback, so an unrecognised string is
   visible in the breakdown instead of silently joining a real category.

Adding a provider means checking whether its error prefixes are covered
here; an unmatched prefix shows up as ``UNKNOWN`` rather than being
mis-filed, which is the failure mode this ordering is built to avoid.
"""

from __future__ import annotations

import re
from enum import StrEnum

# Heuristic ceiling for calling a response truncated. Applied only to text
# that also failed to parse: a long *valid* response is not truncated, and a
# short unparseable one is malformed rather than cut off. The value is a
# judgement call, so it is named rather than inlined at the comparison.
_TRUNCATION_MIN_CHARS = 200


class FailureCause(StrEnum):
    """
    Primary cause of one invalid generation.

    A ``StrEnum`` so the value serialises to a plain string in the JSON
    breakdown and compares cleanly against DataFrame columns, matching how
    ``Strategy`` and ``Tier`` are already handled in ``schemas.py``.
    """

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    SAFETY_BLOCK = "safety_block"
    EMPTY_OUTPUT = "empty_output"
    TRUNCATION = "truncation"
    PARSE_ERROR = "parse_error"
    SCHEMA_VIOLATION = "schema_violation"
    ORCHESTRATION_ERROR = "orchestration_error"
    UNKNOWN = "unknown"


# Ordered (cause, pattern) rules. Order *is* the precedence documented in the
# module docstring: the first match wins, so never reorder without re-reading
# it. Patterns match case-insensitively against the error string, and are
# anchored on the literal prefixes the providers and strategies emit
# (``RateLimitError:``, ``EmptyResponse:``, ``invalid JSON:``, ...) rather
# than on loose keywords, so an unrelated message that merely mentions
# "timeout" in prose does not get mis-filed.
_RULES: tuple[tuple[FailureCause, re.Pattern[str]], ...] = (
    # 1. Infrastructure. These preempt everything: no usable output existed.
    (FailureCause.RATE_LIMIT, re.compile(r"\bRateLimitError\b")),
    (FailureCause.TIMEOUT, re.compile(r"\b(?:APITimeoutError|TimeoutError)\b")),
    (FailureCause.SAFETY_BLOCK, re.compile(r"\b(?:BlockedResponse|ContentFilter)\b")),
    # 2. Empty output, before the parse rules: nothing to parse.
    (
        FailureCause.EMPTY_OUTPUT,
        re.compile(r"\bEmptyResponse\b|\bempty output from provider\b"),
    ),
    # CrewAI surfaces an empty LLM reply as its own kickoff message rather
    # than an EmptyResponse; it is the same underlying cause.
    (
        FailureCause.EMPTY_OUTPUT,
        re.compile(r"Invalid response from LLM call\s*-\s*None or empty"),
    ),
    # 3. Schema violations. Checked before the generic parse rule because the
    # structural Mermaid checks (empty label bracket, unbalanced subgraph)
    # describe well-formed text that violates the output contract, which is a
    # different failure from text that is not parseable at all.
    (
        FailureCause.SCHEMA_VIOLATION,
        re.compile(r"empty node label bracket|unbalanced subgraph/end"),
    ),
    # 4. Parse errors: the model did not produce the requested format.
    (FailureCause.PARSE_ERROR, re.compile(r"\binvalid JSON\b|\bJSONDecodeError\b")),
    # 5. Orchestration: the framework misbehaved, not the model output.
    (
        FailureCause.ORCHESTRATION_ERROR,
        re.compile(r"Single-call invariant violated|kickoff raised"),
    ),
    # Generic API error last among the infrastructure family: APIError is the
    # SDKs' catch-all base class, so a more specific subclass above must win.
    (FailureCause.API_ERROR, re.compile(r"\bAPIError\b")),
)

# Signatures of a response cut off mid-token. An unterminated string or a
# structure that simply stops is what truncation looks like after the fact;
# the provider does not tell us the token limit was hit.
_TRUNCATION_PATTERN = re.compile(
    r"Unterminated string|Expecting value: line \d+ column \d+ \(char \d+\)"
)


def classify_failure(
    error: str | None,
    raw_response: str | None = None,
) -> FailureCause:
    """
    Resolve one failed run to its single primary cause.

    ``error`` is the run's error string; ``raw_response`` is the failing text
    from the sub-result that produced it, when available (see the module
    docstring on why it is a separate argument). The raw text is used only to
    separate truncation from an ordinary parse error, which the error string
    alone cannot distinguish.

    A ``None``/blank ``error`` classifies as ``EMPTY_OUTPUT``: that is the
    silent-empty shape, where a run has no error but no diagram either.
    Callers must only pass runs that actually failed, since a *successful*
    run also has ``error is None`` and would classify the same way.
    """
    if error is None or not error.strip():
        return FailureCause.EMPTY_OUTPUT

    for cause, pattern in _RULES:
        if pattern.search(error):
            # Truncation masquerades as a parse error: same message, different
            # cause. Promote it only with corroborating evidence, a long raw
            # response that stops mid-structure, so an ordinary malformed
            # short reply is not relabelled.
            if cause is FailureCause.PARSE_ERROR and _looks_truncated(
                error, raw_response
            ):
                return FailureCause.TRUNCATION
            return cause

    return FailureCause.UNKNOWN


def _looks_truncated(error: str, raw_response: str | None) -> bool:
    """
    Whether a parse failure is better explained by truncation.

    Requires both a truncation-shaped parser message and a raw response long
    enough that running out of tokens is plausible. Without the raw text we
    cannot tell the two apart, so we stay conservative and keep the parse-error
    label: under-reporting truncation is safer than inventing it.
    """
    if raw_response is None or len(raw_response) < _TRUNCATION_MIN_CHARS:
        return False
    return bool(_TRUNCATION_PATTERN.search(error))
