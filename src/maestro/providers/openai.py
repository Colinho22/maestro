"""
MAESTRO — OpenAI provider implementation
Wraps the OpenAI chat completions API into the LLMProvider interface.
"""

import time

import openai
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from maestro.providers._retry import RetryStats, call_with_retry
from maestro.providers.base import LLMProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost

# HTTP status codes that indicate a transient server-side problem worth
# retrying. 429 (rate limit) and 5xx (server errors). 4xx other than 429 are
# the caller's fault and will not heal themselves.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAIProvider(LLMProvider):
    """
    Concrete provider for OpenAI models (gpt-4o, gpt-4o-mini, etc.)
    Uses the official openai SDK — add 'openai>=1.0.0' to pyproject.toml.
    """

    # Name used in retry log lines. Defined as a class attribute (rather than
    # the literal "openai") so OpenAI-compatible subclasses like
    # DeepSeekProvider can override it — otherwise their retries would be
    # logged as "openai", misattributing failures in multi-provider runs.
    _PROVIDER_NAME = "openai"

    # Same role as AnthropicProvider.SYSTEM_PROMPT
    SYSTEM_PROMPT = (
        "You are a diagram generation assistant. "
        "Respond only with valid Mermaid diagram code. "
        "Do not include any explanation, markdown fencing, or additional text."
    )

    # Max tokens for the completion
    MAX_TOKENS = 4096

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        super().__init__(api_key, pricing)
        # Initialise the SDK client once — reused for all calls
        self._client = openai.OpenAI(api_key=api_key)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """
        OpenAI's exception hierarchy:
          APIError → APIStatusError → RateLimitError (+ others with .status_code)
          APIError → APIConnectionError → APITimeoutError
        Connection / timeout errors are always retryable. APIStatusError is
        retryable only for 429 + 5xx; 4xx other than 429 are programmer
        or auth errors and will not heal themselves.
        """
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in _RETRYABLE_STATUS
        return False

    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        """
        Call the OpenAI chat completions endpoint and return a RunResult.
        Never raises — all exceptions are captured into RunResult.error.
        Transient failures (429, 5xx, timeouts, connection errors) are
        retried with exponential backoff via ``call_with_retry``; the final
        attempt's exception falls through to the handlers below.
        """

        start_ms = time.monotonic()
        effective_system = (
            system_prompt if system_prompt is not None else self.SYSTEM_PROMPT
        )

        # Owned by the caller so retry_count survives an exhausted-retries
        # exception — the except blocks below read stats.retry_count to
        # record it on the failed RunResult.
        stats = RetryStats()

        def _do_call():
            """The SDK call ``call_with_retry`` re-runs on transient failures."""
            return self._client.chat.completions.create(
                model=config.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                messages=[
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": prompt},
                ],
            )

        try:
            response, _ = call_with_retry(
                _do_call,
                is_retryable=self._is_retryable,
                provider_name=self._PROVIDER_NAME,
                stats=stats,
            )

            duration_ms = int((time.monotonic() - start_ms) * 1000)

            # OpenAI returns usage on every non-streaming response
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            # Response content from the first choice
            output = response.choices[0].message.content

            return RunResult(
                run_id=config.run_id,
                output_diagram_code=output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                cost_usd=compute_cost(prompt_tokens, completion_tokens, self.pricing),
                retry_count=stats.retry_count,
            )

        except RateLimitError as e:
            return self._error_result(
                config, start_ms, f"RateLimitError: {e}", stats.retry_count
            )

        except APITimeoutError as e:
            return self._error_result(
                config, start_ms, f"TimeoutError: {e}", stats.retry_count
            )

        except APIError as e:
            return self._error_result(
                config, start_ms, f"APIError: {e}", stats.retry_count
            )

        except Exception as e:
            # Catch-all — unexpected failures should not crash the experiment
            return self._error_result(
                config, start_ms, f"UnexpectedError: {e}", stats.retry_count
            )

    def _error_result(
        self,
        config: RunConfig,
        start_ms: float,
        error: str,
        retry_count: int = 0,
    ) -> RunResult:
        """
        Build a failed RunResult with zero token counts and the error message.
        ``retry_count`` is propagated from ``RetryStats`` so an exhausted-
        retries failure still records how many attempts were made.
        """
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=int((time.monotonic() - start_ms) * 1000),
            cost_usd=0.0,
            error=error,
            retry_count=retry_count,
        )
