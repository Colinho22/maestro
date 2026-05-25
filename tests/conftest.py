"""
Shared pytest fixtures for the MAESTRO test suite.

The cornerstone is ``RecordingProvider`` — a hand-rolled ``LLMProvider`` that
captures every ``complete()`` call into a list so tests can assert on the
arguments a strategy passed in (system_prompt, prompt, config). Using a real
subclass instead of ``unittest.mock.Mock`` keeps the contract honest: if
``LLMProvider.complete`` ever changes signature, the recording provider breaks
at import time, not silently in a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from maestro.providers.base import LLMProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult


@dataclass
class RecordedCall:
    """One captured ``provider.complete()`` invocation."""

    prompt: str
    config: RunConfig
    system_prompt: str | None


class RecordingProvider(LLMProvider):
    """
    Test double for ``LLMProvider`` that records every ``complete()`` call and
    returns canned outputs.

    Construct with ``outputs`` — a list of strings to return on successive
    calls. Each output becomes the ``output_diagram_code`` of a successful
    ``RunResult``. If the list is exhausted, the next call returns a failing
    ``RunResult`` so tests fail loudly rather than hang or loop.

    All captured calls are appended to ``calls`` in order.
    """

    def __init__(self, outputs: list[str] | None = None) -> None:
        pricing = ModelPricing(
            model="test-model",
            input_price_per_1m=0.0,
            output_price_per_1m=0.0,
        )
        super().__init__(api_key="test-key", pricing=pricing)
        self._outputs: list[str] = list(outputs) if outputs else []
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        self.calls.append(
            RecordedCall(prompt=prompt, config=config, system_prompt=system_prompt)
        )
        if not self._outputs:
            return RunResult(
                run_id=config.run_id,
                output_diagram_code=None,
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=0,
                cost_usd=0.0,
                error="RecordingProvider: no more canned outputs configured",
            )
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=self._outputs.pop(0),
            prompt_tokens=1,
            completion_tokens=1,
            duration_ms=0,
            cost_usd=0.0,
            error=None,
        )

    @property
    def system_prompts_seen(self) -> list[str | None]:
        """Convenience: just the system_prompt arg from each recorded call."""
        return [c.system_prompt for c in self.calls]


@pytest.fixture
def recording_provider_factory():
    """
    Returns a callable that builds a ``RecordingProvider`` with the given
    list of canned outputs. Use a factory (not a fixture instance) so tests
    can choose outputs that match the steps they exercise.
    """

    def _make(outputs: list[str] | None = None) -> RecordingProvider:
        return RecordingProvider(outputs=outputs)

    return _make
