"""
Request-parameter tests for AnthropicProvider.

Newer Anthropic models (Opus 4.7+/Fable) removed sampling parameters and return
400 if `temperature` is sent; older ones (Haiku 4.5) still accept it. The
provider decides per model via ModelPricing.supports_temperature. These tests
pin that the temperature kwarg is omitted/included accordingly, without making
a network call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from maestro.providers.anthropic import AnthropicProvider  # noqa: E402
from maestro.schemas import ModelPricing, RunConfig, Strategy, Tier  # noqa: E402


def _capture_create_kwargs(pricing: ModelPricing) -> dict:
    """Run complete() with a stubbed SDK client and return the create() kwargs."""
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        block = SimpleNamespace(text="graph TD\n a")
        return SimpleNamespace(usage=usage, content=[block])

    provider = AnthropicProvider(api_key="k", pricing=pricing)
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    cfg = RunConfig(
        strategy=Strategy.SINGLE_AGENT,
        model=pricing.model,
        example_id="e",
        tier=Tier.SIMPLE,
        run_number=1,
    )
    provider.complete("x", cfg)
    return captured


def test_omits_temperature_when_unsupported():
    """Opus 4.7+/Fable reject temperature: it must not be sent."""
    pricing = ModelPricing(
        model="claude-opus-4-8",
        input_price_per_1m=5.0,
        output_price_per_1m=25.0,
        supports_temperature=False,
    )
    kwargs = _capture_create_kwargs(pricing)
    assert "temperature" not in kwargs
    assert kwargs["max_tokens"] == AnthropicProvider.MAX_TOKENS


def test_includes_temperature_when_supported():
    """Haiku 4.5 still accepts temperature: it must be sent."""
    pricing = ModelPricing(
        model="claude-haiku-4-5-20251001",
        input_price_per_1m=1.0,
        output_price_per_1m=5.0,
    )  # supports_temperature defaults to True
    kwargs = _capture_create_kwargs(pricing)
    assert kwargs["temperature"] == AnthropicProvider.TEMPERATURE
