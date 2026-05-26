"""
MAESTRO — Retry-with-backoff helper shared by every provider.

Why this lives in its own module:
- The retry policy (max attempts, wait shape, what counts as retryable) is
  experimental infrastructure, not provider-specific logic. Centralising it
  here means tightening or loosening the policy is a single edit, not four.
- Each provider supplies its own ``is_retryable(exc)`` predicate because
  SDK exception hierarchies differ (anthropic has ``RateLimitError`` as a
  subclass of ``APIStatusError``, mistralai's ``SDKError`` carries the
  status code on ``.raw_response``, gemini's ``APIError`` exposes it on
  ``.status``). Asking each provider to classify its own errors keeps that
  knowledge co-located with the SDK import.
- ``call_with_retry`` returns ``(result, RetryStats)`` rather than just the
  result so the caller can record how many retries were needed. This signal
  is persisted to ``run_results.retry_count`` and is useful for "do high-
  retry calls correlate with worse quality?" analysis.

Failure semantics: if ``max_attempts`` is exhausted, the *last* exception is
re-raised so the caller's existing try/except in ``complete()`` builds a
failed RunResult exactly as before. No silent swallowing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, TypeVar

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


T = TypeVar("T")


# Policy constants — tuned for thesis-scale experiments (~5000 calls per run).
# `MAX_ATTEMPTS` includes the first call, so 5 total attempts = 4 retries.
# Backoff is capped at 60s so a long string of 429s on one provider doesn't
# stall the whole matrix for many minutes.
MAX_ATTEMPTS = 5
WAIT_INITIAL = 2.0
WAIT_MAX     = 60.0


@dataclass
class RetryStats:
    """How many attempts a call took, for persistence to run_results."""

    attempts: int = 1
    last_exception: str | None = None

    @property
    def retry_count(self) -> int:
        """Number of *retries* (i.e. attempts beyond the first)."""
        return max(0, self.attempts - 1)


def call_with_retry(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    provider_name: str,
    max_attempts: int = MAX_ATTEMPTS,
    wait_initial: float = WAIT_INITIAL,
    wait_max: float = WAIT_MAX,
) -> tuple[T, RetryStats]:
    """
    Run ``fn`` with exponential-backoff-with-jitter on retryable exceptions.

    ``is_retryable`` decides per exception whether to retry. Non-retryable
    exceptions are re-raised on the first occurrence so they reach the
    caller's existing exception handlers unchanged.

    ``provider_name`` is only used in the stderr log line so a long run is
    debuggable ("which provider is rate-limiting?").
    """
    stats = RetryStats()

    def _log_retry(retry_state) -> None:
        # tenacity calls this before each sleep; ``retry_state.attempt_number``
        # is the attempt that just failed.
        exc = retry_state.outcome.exception()
        stats.last_exception = f"{type(exc).__name__}: {exc}"
        sleep_for = retry_state.next_action.sleep if retry_state.next_action else 0
        print(
            f"  [retry] {provider_name} attempt {retry_state.attempt_number} "
            f"failed ({type(exc).__name__}); sleeping {sleep_for:.1f}s",
            file=sys.stderr,
        )

    for attempt in Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=wait_initial, max=wait_max),
        retry=retry_if_exception(is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    ):
        with attempt:
            stats.attempts = attempt.retry_state.attempt_number
            return fn(), stats

    # Unreachable: ``reraise=True`` propagates the last exception out of
    # the for-loop on failure; success returns from inside the with-block.
    raise RuntimeError("call_with_retry: unreachable")
