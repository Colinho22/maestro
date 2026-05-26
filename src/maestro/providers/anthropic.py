"""
MAESTRO — Anthropic provider implementation
Wraps the Anthropic messages API into the LLMProvider interface.
"""

import time

import anthropic
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from maestro.providers._retry import RetryStats, call_with_retry
from maestro.providers.base import LLMProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost

# See providers/openai.py for the rationale on this status set.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class AnthropicProvider(LLMProvider):
    """
    Concrete provider for Anthropic models (claude-sonnet-4-5, etc.)
    Uses the official anthropic SDK — add 'anthropic>=0.25.0' to pyproject.toml.
    """

    # Instructs the model to output diagram code only — no prose or fencing
    SYSTEM_PROMPT = (
        "You are a diagram generation assistant. "
        "Respond only with valid Mermaid diagram code. "
        "Do not include any explanation, markdown fencing, or additional text."
    )

    # Max tokens for the completion — diagram code is rarely long
    MAX_TOKENS = 4096

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        super().__init__(api_key, pricing)
        # Initialise the SDK client once — reused for all calls
        self._client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Mirrors OpenAIProvider._is_retryable — same SDK exception shape."""
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
        Call the Anthropic messages endpoint and return a RunResult.
        Never raises — all exceptions are captured into RunResult.error.
        Transient failures are retried with exponential backoff via
        ``call_with_retry``; non-retryable errors fall through to the
        handlers below on the first attempt.
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
            return self._client.messages.create(
                model=config.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                system=effective_system,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

        try:
            response, _ = call_with_retry(
                _do_call,
                is_retryable=self._is_retryable,
                provider_name="anthropic",
                stats=stats,
            )

            duration_ms = int((time.monotonic() - start_ms) * 1000)

            # Anthropic returns usage on every non-streaming response
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            # Response content is a list of blocks — grab the first text block
            output = response.content[0].text

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
