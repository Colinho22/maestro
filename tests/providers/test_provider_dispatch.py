"""
Cross-provider tests: dispatch-table wiring and ``_is_retryable`` classification.

Two things break *silently* in this layer and are cheap to pin:

1. **Dispatch drift**: adding a model to ``experiment_config.MODELS`` but
   forgetting its ``_PROVIDER_DISPATCH`` entry in ``run.py`` passes preflight
   (unknown models are skipped) yet fails the cell at run time. The dispatch
   tests assert every non-control model resolves to a provider + env var.

2. **Retry misclassification**: each provider supplies its own
   ``_is_retryable`` predicate because SDK exception hierarchies differ
   (anthropic/openai ``APIStatusError.status_code``, mistralai
   ``SDKError.raw_response.status_code``, gemini ``APIError.code``). A wrong
   verdict means either no retry on a transient blip or pointless retries on a
   hard 4xx, across thousands of calls. These are pure logic, tested here by
   constructing the *real* SDK exceptions (no mocking of response shapes), so
   the production ``isinstance`` branch is genuinely exercised. If a future
   SDK bump changes an exception constructor, the relevant test fails loudly
   at the construction site rather than silently passing against a fake.

The full ``complete()`` path (usage parsing, cost, error capture) is
deliberately *not* covered here: that would require mocking four different
SDK client/response shapes, which is brittle and mostly re-tests the SDKs.
"""

from __future__ import annotations

import httpx
import pytest

from maestro.experiment_config import MODELS
from maestro.providers.anthropic import AnthropicProvider
from maestro.providers.deepseek import DeepSeekProvider
from maestro.providers.gemini import GeminiProvider
from maestro.providers.mistral import MistralProvider
from maestro.providers.openai import OpenAIProvider
from maestro.run import _PROVIDER_DISPATCH, _dispatch_for_model

# Synthetic control "model" never dispatches to a provider (controls bypass
# the LLM); exclude it from the all-models-dispatch assertion.
_CONTROL_MODEL_NAME = "control"


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


def test_every_configured_model_dispatches():
    """
    Every non-control model in MODELS resolves to a provider via the dispatch
    table. Guards against the add-model-forget-dispatch footgun.
    """
    for mp in MODELS:
        if mp.model == _CONTROL_MODEL_NAME:
            continue
        dispatch = _dispatch_for_model(mp.model)
        assert dispatch is not None, f"no provider dispatch for model '{mp.model}'"
        _, provider_cls, env_var = dispatch
        assert isinstance(provider_cls, type)
        assert env_var.endswith("_API_KEY")


def test_dispatch_needles_are_unique():
    """No two dispatch entries share a needle (would make order load-bearing)."""
    needles = [needle for needle, _, _ in _PROVIDER_DISPATCH]
    assert len(needles) == len(set(needles))


def test_unknown_model_does_not_dispatch():
    assert _dispatch_for_model("no-such-model-xyz") is None


@pytest.mark.parametrize(
    "model_substr, expected_cls, expected_env",
    [
        ("claude-haiku-4-5", AnthropicProvider, "ANTHROPIC_API_KEY"),
        ("gpt-5.4-mini-2026-03-17", OpenAIProvider, "OPENAI_API_KEY"),
        ("mistral-small", MistralProvider, "MISTRAL_API_KEY"),
        ("gemini-3.1-flash-lite", GeminiProvider, "GEMINI_API_KEY"),
        ("deepseek-v4-flash", DeepSeekProvider, "DEEPSEEK_API_KEY"),
    ],
)
def test_known_models_dispatch_to_expected_provider(
    model_substr, expected_cls, expected_env
):
    _, cls, env = _dispatch_for_model(model_substr)
    assert cls is expected_cls
    assert env == expected_env


# ---------------------------------------------------------------------------
# _is_retryable: transport-error classification
#
# Providers differ here BY DESIGN, and the difference is worth pinning:
#   - OpenAI / Anthropic wrap transport failures in their OWN SDK types
#     (APIConnectionError / APITimeoutError) before they reach _is_retryable,
#     so the predicate keys on those, not on bare httpx errors.
#   - Mistral / Gemini explicitly also catch raw httpx.ConnectError /
#     TimeoutException because those "leak through unwrapped on some failure
#     modes" (see each provider's _is_retryable docstring).
# A test asserting "all providers retry bare httpx errors" would be wrong:
# it would contradict the intended, SDK-specific design.
# ---------------------------------------------------------------------------

_ALL_PROVIDERS = [
    AnthropicProvider,
    OpenAIProvider,
    MistralProvider,
    GeminiProvider,
    DeepSeekProvider,
]


def test_openai_anthropic_retry_their_own_transport_types():
    """OpenAI/Anthropic (and DeepSeek via inheritance) retry SDK-wrapped
    connection/timeout errors, which is how transport failures actually
    reach them."""
    import anthropic
    import openai

    assert OpenAIProvider._is_retryable(openai.APIConnectionError(request=None))
    assert DeepSeekProvider._is_retryable(openai.APIConnectionError(request=None))
    assert AnthropicProvider._is_retryable(anthropic.APIConnectionError(request=None))


def test_mistral_gemini_retry_raw_httpx_transport_errors():
    """Mistral/Gemini additionally catch unwrapped httpx transport errors."""
    for cls in (MistralProvider, GeminiProvider):
        assert cls._is_retryable(httpx.ConnectError("connection refused"))
        assert cls._is_retryable(httpx.TimeoutException("read timed out"))


@pytest.mark.parametrize("provider_cls", _ALL_PROVIDERS)
def test_plain_value_error_not_retryable(provider_cls):
    """A non-SDK, non-transport exception is never retryable, for any provider."""
    assert provider_cls._is_retryable(ValueError("not an API error")) is False


# ---------------------------------------------------------------------------
# _is_retryable: per-SDK status-code classification (retryable + not)
# ---------------------------------------------------------------------------


def _httpx_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example"))


def test_openai_status_classification():
    from openai import APIStatusError

    def make(status):
        return APIStatusError("err", response=_httpx_response(status), body=None)

    # OpenAI + DeepSeek share the predicate.
    for cls in (OpenAIProvider, DeepSeekProvider):
        assert cls._is_retryable(make(503)) is True
        assert cls._is_retryable(make(400)) is False


def test_anthropic_status_classification():
    from anthropic import APIStatusError

    def make(status):
        return APIStatusError("err", response=_httpx_response(status), body=None)

    assert AnthropicProvider._is_retryable(make(429)) is True
    assert AnthropicProvider._is_retryable(make(401)) is False


def test_mistral_status_classification():
    from mistralai.client.errors import SDKError

    def make(status):
        return SDKError("err", raw_response=_httpx_response(status))

    assert MistralProvider._is_retryable(make(500)) is True
    assert MistralProvider._is_retryable(make(404)) is False


def test_gemini_status_classification():
    from google.genai import errors as genai_errors

    def make(status):
        # APIError(code, response_json, response=None)
        return genai_errors.APIError(status, {"error": {"message": "x"}}, None)

    assert GeminiProvider._is_retryable(make(502)) is True
    assert GeminiProvider._is_retryable(make(400)) is False


def test_mistral_no_response_error_is_retryable():
    from mistralai.client.errors import NoResponseError

    assert MistralProvider._is_retryable(NoResponseError("empty")) is True
