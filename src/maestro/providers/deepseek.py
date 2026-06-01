"""
MAESTRO — DeepSeek provider implementation.

DeepSeek exposes an OpenAI-compatible chat-completions API, so this provider
is a thin subclass of :class:`OpenAIProvider`: the request/response shape,
token-usage fields (``usage.prompt_tokens`` / ``usage.completion_tokens``),
retry classification, and error handling are all identical. The only
differences are the SDK client's ``base_url`` and the API key source — both
isolated to ``__init__`` here.

Why subclass rather than copy: keeping a single ``complete()`` /
``_is_retryable`` / ``_error_result`` implementation means the retry and
error-handling hardening lives in exactly one place. A change to that logic
in ``OpenAIProvider`` is inherited automatically instead of having to be
mirrored by hand.

Provenance note: DeepSeek also offers an Anthropic-compatible endpoint at
``https://api.deepseek.com/anthropic``. We deliberately use the OpenAI one —
it is DeepSeek's primary, best-documented surface and its usage fields match
the existing OpenAIProvider parsing exactly. See proposal §3.2 (model
providers as a replication dimension).
"""

from openai import OpenAI

from maestro.providers.openai import OpenAIProvider
from maestro.schemas import ModelPricing

# OpenAI-compatible base URL for the DeepSeek API. The Anthropic-compatible
# endpoint (``.../anthropic``) exists too but is intentionally not used here.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAIProvider):
    """
    Concrete provider for DeepSeek models (e.g. ``deepseek-v4-flash``).

    Reuses every method of :class:`OpenAIProvider` — only the SDK client is
    re-pointed at DeepSeek's OpenAI-compatible endpoint. ``config.model`` (the
    DeepSeek model id) is passed straight through to the chat-completions
    call by the inherited ``complete()``.
    """

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        # Skip OpenAIProvider.__init__ (which builds an api.openai.com client)
        # and call the grandparent to store api_key + pricing, then build the
        # DeepSeek-pointed client ourselves.
        super(OpenAIProvider, self).__init__(api_key, pricing)
        self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
