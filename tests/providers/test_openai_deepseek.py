"""
Tests for the OpenAI provider and its DeepSeek subclass.

DeepSeekProvider is a deliberately thin subclass of OpenAIProvider: DeepSeek
exposes an OpenAI-compatible API, so the two share complete() / _is_retryable
/ _error_result / cost logic and differ only in the SDK client's base_url and
the API key source. These tests assert exactly that contract: that the two
providers stay in sync where they should and diverge only where intended,
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
# Construction: both providers, parametrized
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
# Request params: token-param name + temperature omission per model/provider
# ---------------------------------------------------------------------------


def _capture_create_kwargs(provider: OpenAIProvider, prompt: str = "x") -> dict:
    """Run complete() with a stubbed SDK client and return the create() kwargs."""
    from types import SimpleNamespace

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        choice = SimpleNamespace(message=SimpleNamespace(content="graph TD\n a"))
        return SimpleNamespace(usage=usage, choices=[choice])

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    from maestro.schemas import RunConfig, Strategy, Tier

    cfg = RunConfig(
        strategy=Strategy.SINGLE_AGENT,
        model=provider.pricing.model,
        example_id="e",
        tier=Tier.SIMPLE,
        run_number=1,
    )
    provider.complete(prompt, cfg)
    return captured


def test_openai_uses_max_completion_tokens_and_omits_temperature():
    """GPT-5 family: max_completion_tokens param, no temperature."""
    pricing = ModelPricing(
        model="gpt-5.5-2026-04-23",
        input_price_per_1m=5.0,
        output_price_per_1m=30.0,
        supports_temperature=False,
    )
    kwargs = _capture_create_kwargs(OpenAIProvider(api_key="k", pricing=pricing))
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_deepseek_keeps_max_tokens_and_temperature():
    """DeepSeek's OpenAI-compatible endpoint keeps max_tokens + temperature."""
    kwargs = _capture_create_kwargs(
        DeepSeekProvider(api_key="k", pricing=_DEEPSEEK_PRICING)
    )
    assert "max_tokens" in kwargs
    assert "max_completion_tokens" not in kwargs
    # _DEEPSEEK_PRICING leaves supports_temperature at its True default.
    assert "temperature" in kwargs


def test_none_content_is_recorded_failure():
    """A None message.content (length/content-filter finish) is EmptyResponse.

    Without the guard this lands as a failed row with error=None, which is a
    silent failure: success is False but nothing explains why.
    """
    from types import SimpleNamespace

    from maestro.schemas import RunConfig, Strategy, Tier

    provider = OpenAIProvider(api_key="k", pricing=_OPENAI_PRICING)

    def fake_create(**kwargs):
        usage = SimpleNamespace(prompt_tokens=4, completion_tokens=0)
        choice = SimpleNamespace(message=SimpleNamespace(content=None))
        return SimpleNamespace(usage=usage, choices=[choice])

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    cfg = RunConfig(
        strategy=Strategy.SINGLE_AGENT,
        model=_OPENAI_PRICING.model,
        example_id="e",
        tier=Tier.SIMPLE,
        run_number=1,
    )
    result = provider.complete("x", cfg)
    assert result.success is False
    assert result.error is not None and "EmptyResponse" in result.error
    assert result.prompt_tokens == 4  # usage preserved


# ---------------------------------------------------------------------------
# Inheritance contract: DeepSeek IS an OpenAIProvider, but points elsewhere
# ---------------------------------------------------------------------------


def test_deepseek_exported_from_package():
    """DeepSeekProvider is importable from the package root like the others."""
    import maestro.providers as providers

    assert "DeepSeekProvider" in providers.__all__
    from maestro.providers import DeepSeekProvider as Exported

    assert Exported is DeepSeekProvider


def test_deepseek_is_openai_subclass():
    """DeepSeek reuses OpenAIProvider's implementation by subclassing it."""
    assert issubclass(DeepSeekProvider, OpenAIProvider)
    # The methods are inherited, not re-defined: DeepSeek must NOT carry its
    # own copy of complete / _is_retryable (that would defeat the point and
    # let the two drift apart).
    assert "complete" not in DeepSeekProvider.__dict__
    assert "_is_retryable" not in DeepSeekProvider.__dict__


def test_provider_name_overridden_for_retry_logs():
    """
    DeepSeek overrides _PROVIDER_NAME so retry log lines read 'deepseek',
    not the inherited 'openai', otherwise multi-provider run logs would
    misattribute DeepSeek failures to OpenAI.
    """
    assert OpenAIProvider._PROVIDER_NAME == "openai"
    assert DeepSeekProvider._PROVIDER_NAME == "deepseek"


def test_deepseek_init_does_not_run_openai_init():
    """
    DeepSeek calls LLMProvider.__init__ directly (grandparent), skipping
    OpenAIProvider.__init__, so its client points at DeepSeek not OpenAI.
    This pins the intent against an accidental super() change.
    """
    from maestro.providers.base import LLMProvider

    p = DeepSeekProvider(api_key="k", pricing=_DEEPSEEK_PRICING)
    # Grandparent stored the fields...
    assert p.api_key == "k"
    assert p.pricing is _DEEPSEEK_PRICING
    # ...and DeepSeek's own client wins (OpenAIProvider.__init__ did not run).
    assert isinstance(p, LLMProvider)
    assert "deepseek" in str(p._client.base_url)


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
# run.py wiring: dispatch + preflight resolve both providers
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
