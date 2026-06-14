"""
MAESTRO: Gemini provider implementation
Wraps the Google Gen AI generate_content API into the LLMProvider interface.
"""

import time

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from maestro.providers._retry import RetryStats, call_with_retry
from maestro.providers.base import LLMProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost

# See providers/openai.py for the rationale on this status set.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiProvider(LLMProvider):
    """
    Concrete provider for Google Gemini models (gemini-3.5-flash, etc.)
    Uses the official google-genai SDK; add 'google-genai>=1.0' to pyproject.toml.
    """

    # SYSTEM_PROMPT inherited from LLMProvider (maestro.prompts).

    MAX_TOKENS = 4096

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        super().__init__(api_key, pricing)
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """
        google-genai wraps HTTP errors in ``APIError`` (and its ``ClientError``
        / ``ServerError`` subclasses) which carry the HTTP status on ``.code``.
        Underlying transport errors may still leak through as raw httpx
        connect/timeout exceptions on connection failure.
        """
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        if isinstance(exc, genai_errors.APIError):
            return getattr(exc, "code", None) in _RETRYABLE_STATUS
        return False

    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        """
        Call the Gemini generate_content endpoint and return a RunResult.
        Never raises: all exceptions are captured into RunResult.error.
        Transient failures are retried with exponential backoff via
        ``call_with_retry``.
        """

        start_ms = time.monotonic()
        effective_system = (
            system_prompt if system_prompt is not None else self.SYSTEM_PROMPT
        )

        # Owned by the caller so retry_count survives an exhausted-retries
        # exception: the except blocks below read stats.retry_count to
        # record it on the failed RunResult.
        stats = RetryStats()

        def _do_call():
            """The SDK call ``call_with_retry`` re-runs on transient failures."""
            return self._client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=effective_system,
                    temperature=self.TEMPERATURE,
                    max_output_tokens=self.MAX_TOKENS,
                ),
            )

        try:
            response, _ = call_with_retry(
                _do_call,
                is_retryable=self._is_retryable,
                provider_name="gemini",
                stats=stats,
            )

            duration_ms = int((time.monotonic() - start_ms) * 1000)

            # usage_metadata can be absent on blocked / malformed responses
            usage = response.usage_metadata
            prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
            completion_tokens = (usage.candidates_token_count or 0) if usage else 0

            # response.text raises ValueError when the candidate has no text
            # parts (e.g. safety blocks). Capture that as a normal failure so
            # token usage from the prompt is still recorded.
            try:
                output = response.text
            except ValueError as e:
                return RunResult(
                    run_id=config.run_id,
                    output_diagram_code=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration_ms=duration_ms,
                    cost_usd=compute_cost(
                        prompt_tokens, completion_tokens, self.pricing
                    ),
                    error=f"BlockedResponse: {e}",
                    retry_count=stats.retry_count,
                )

            return RunResult(
                run_id=config.run_id,
                output_diagram_code=output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                cost_usd=compute_cost(prompt_tokens, completion_tokens, self.pricing),
                retry_count=stats.retry_count,
            )

        except genai_errors.APIError as e:
            return self._error_result(
                config, start_ms, f"APIError: {e}", stats.retry_count
            )

        except Exception as e:
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
