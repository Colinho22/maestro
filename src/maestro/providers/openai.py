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

from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost
from maestro.providers.base import LLMProvider
from maestro.providers._retry import call_with_retry


# HTTP status codes that indicate a transient server-side problem worth
# retrying. 429 (rate limit) and 5xx (server errors). 4xx other than 429 are
# the caller's fault and will not heal themselves.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAIProvider(LLMProvider):
    """
    Concrete provider for OpenAI models (gpt-4o, gpt-4o-mini, etc.)
    Uses the official openai SDK — add 'openai>=1.0.0' to pyproject.toml.
    """

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
        effective_system = system_prompt if system_prompt is not None else self.SYSTEM_PROMPT

        def _do_call():
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
            response, stats = call_with_retry(
                _do_call,
                is_retryable=self._is_retryable,
                provider_name="openai",
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
                cost_usd=compute_cost(
                    prompt_tokens, completion_tokens, self.pricing
                ),
                retry_count=stats.retry_count,
            )

        except RateLimitError as e:
            return self._error_result(config, start_ms, f"RateLimitError: {e}")

        except APITimeoutError as e:
            return self._error_result(config, start_ms, f"TimeoutError: {e}")

        except APIError as e:
            return self._error_result(config, start_ms, f"APIError: {e}")

        except Exception as e:
            # Catch-all — unexpected failures should not crash the experiment
            return self._error_result(config, start_ms, f"UnexpectedError: {e}")

    def _error_result(
        self, config: RunConfig, start_ms: float, error: str
    ) -> RunResult:
        """
        Build a failed RunResult with zero token counts and the error message.
        """
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=int((time.monotonic() - start_ms) * 1000),
            cost_usd=0.0,
            error=error,
        )
