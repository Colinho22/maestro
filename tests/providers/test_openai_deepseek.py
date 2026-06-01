"""
Tests for the OpenAI provider and its DeepSeek subclass.

DeepSeekProvider is a deliberately thin subclass of OpenAIProvider: DeepSeek
exposes an OpenAI-compatible API, so the two share complete() / _is_retryable
/ _error_result / cost logic and differ only in the SDK client's base_url and
the API key source. These tests assert exactly that contract — that the two
providers stay in sync where they should and diverge only where intended —
so a future change to OpenAIProvider.__init__ that breaks the subclass is
caught here rather than at experiment time.

No network calls: construction, the inheritance relationship, base_url
divergence, and the run.py dispatch/preflight wiring are all verifiable
offline. The live API behavior (clean Mermaid, populated usage fields) is
covered by the pre-run smoke test, not pytest.
"""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")

from maestro.providers.deepseek import (  # noqa: E402
    DEEPSEEK_BASE_URL,
    DeepSeekProvider,
)
from maestro.providers.openai import OpenAIProvider  # noqa: E402
from maestro.schemas import ModelPricing  # noqa: E402

_OPENAI_PRICING = ModelPricing(
    model="gpt-4o-mini-2024-07-18",
    input_price_per_1m=0.15,
    output_price_per_1m=0.60,
)
_DEEPSEEK_PRICING = ModelPricing(
    model="deepseek-v4-flash",
    input_price_per_1m=0.14,
    output_price_per_1m=0.28,
)


# ---------------------------------------------------------------------------
# Construction — both providers, parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls, pricing",
    [
        (OpenAIProvider, _OPENAI_PRICING),
        (DeepSeekProvider, _DEEPSEEK_PRICING),
    ],
)
def test_provider_constructs_and_stores_fields(provider_cls, pricing):
    """Both providers store api_key + pricing and expose model_name."""
    p = provider_cls(api_key="test-key", pricing=pricing)
    assert p.api_key == "test-key"
    assert p.pricing is pricing
    assert p.model_name == pricing.model
    # The shared behavior both rely on is present.
    assert hasattr(p, "complete")
    assert hasattr(p, "_is_retryable")
    assert p.SYSTEM_PROMPT  # inherited default system prompt


# ---------------------------------------------------------------------------
# Inheritance contract — DeepSeek IS an OpenAIProvider, but points elsewhere
# ---------------------------------------------------------------------------


def test_deepseek_is_openai_subclass():
    """DeepSeek reuses OpenAIProvider's implementation by subclassing it."""
    assert issubclass(DeepSeekProvider, OpenAIProvider)
    # The methods are inherited, not re-defined: DeepSeek must NOT carry its
    # own copy of complete / _is_retryable (that would defeat the point and
    # let the two drift apart).
    assert "complete" not in DeepSeekProvider.__dict__
    assert "_is_retryable" not in DeepSeekProvider.__dict__


def test_base_url_diverges_between_providers():
    """The one intended difference: where the SDK client points."""
    openai_p = OpenAIProvider(api_key="k", pricing=_OPENAI_PRICING)
    deepseek_p = DeepSeekProvider(api_key="k", pricing=_DEEPSEEK_PRICING)

    deepseek_url = str(deepseek_p._client.base_url).rstrip("/")
    openai_url = str(openai_p._client.base_url).rstrip("/")

    assert deepseek_url == DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    # OpenAI's client points at OpenAI, not DeepSeek.
    assert "deepseek" not in openai_url
    assert deepseek_url != openai_url


# ---------------------------------------------------------------------------
# run.py wiring — dispatch + preflight resolve both providers
# ---------------------------------------------------------------------------


def test_dispatch_resolves_both_providers():
    from maestro.run import _dispatch_for_model

    _, openai_cls, openai_env = _dispatch_for_model("gpt-4o-mini-2024-07-18")
    assert openai_cls is OpenAIProvider
    assert openai_env == "OPENAI_API_KEY"

    needle, deepseek_cls, deepseek_env = _dispatch_for_model("deepseek-v4-flash")
    assert needle == "deepseek"
    assert deepseek_cls is DeepSeekProvider
    assert deepseek_env == "DEEPSEEK_API_KEY"


def test_create_provider_builds_deepseek_pointed_client(monkeypatch):
    from maestro.run import _create_provider

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = _create_provider(_DEEPSEEK_PRICING)
    assert isinstance(provider, DeepSeekProvider)
    assert str(provider._client.base_url).rstrip("/") == DEEPSEEK_BASE_URL


def test_preflight_flags_missing_deepseek_key(monkeypatch):
    """The shared dispatch table means preflight demands the DeepSeek key."""
    from maestro.run import preflight_check_env

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        preflight_check_env([_DEEPSEEK_PRICING])
