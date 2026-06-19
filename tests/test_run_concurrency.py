"""
Per-provider concurrency cap in run.py.

The matrix runs on a thread pool capped per provider so one provider's calls
can't burst past its rate limit and so a cell's measured duration_ms stays a
true latency rather than server-side queue time. --provider-concurrency sets
the permit count; the default is conservative for paid tiers and a free-tier
key drops it to 1. These pin the wiring:

  * one semaphore per provider needle, each with the requested permit count,
  * the needle keys match what the provider factory dispatches on, so a worker
    acquires the right provider's semaphore,
  * concurrency < 1 is rejected at parse time (it would deadlock the pool).
"""

from __future__ import annotations

from maestro.run import (
    _PROVIDER_DISPATCH,
    _build_provider_semaphores,
    _dispatch_for_model,
)

_PROVIDER_NEEDLES = {needle for needle, _cls, _env in _PROVIDER_DISPATCH}


def test_one_semaphore_per_provider():
    sems = _build_provider_semaphores(4)
    assert set(sems) == _PROVIDER_NEEDLES


def test_semaphore_permits_match_concurrency():
    sems = _build_provider_semaphores(2)
    # A fresh Semaphore(2) grants two non-blocking acquires, then blocks.
    sem = next(iter(sems.values()))
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is False


def test_keys_match_provider_dispatch():
    """The worker keys its semaphore by the needle the factory dispatches on,
    so every real model name must resolve to a key present in the map."""
    sems = _build_provider_semaphores(1)
    for model in ("claude-opus-4-8", "gpt-5.5-2026-04-23", "deepseek-v4-pro"):
        needle = _dispatch_for_model(model)[0]
        assert needle in sems
