"""
MAESTRO: Mistral provider implementation
Wraps the Mistral chat completions API into the LLMProvider interface.
"""

from __future__ import annotations

import time

import httpx
from mistralai.client import Mistral
from mistralai.client.errors import NoResponseError, SDKError

from maestro.providers._retry import RetryStats, call_with_retry
from maestro.providers.base import LLMProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost

# See providers/openai.py for the rationale on this status set.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class MistralProvider(LLMProvider):
    """
    Concrete provider for Mistral models (mistral-small, mistral-large, etc.)
    Uses the official mistralai SDK (>=2.4.7). The 2.x SDK moved ``Mistral``
    and ``SDKError`` out of the package root into ``mistralai.client`` and
    ``mistralai.client.errors`` respectively.
    """

    # SYSTEM_PROMPT inherited from LLMProvider (maestro.prompts).

    _PROVIDER_NAME = "mistral"
    MAX_TOKENS = 4096

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        super().__init__(api_key, pricing)
        self._client = Mistral(api_key=api_key)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """
        mistralai exposes ``SDKError`` carrying ``raw_response: httpx.Response``;
        we read ``status_code`` from there. ``NoResponseError`` means the SDK
        got nothing back at all, always transient. Low-level httpx network
        errors (connect / timeout) also surface unwrapped on some failure
        modes and are equally transient.
        """
        if isinstance(
            exc, (NoResponseError, httpx.ConnectError, httpx.TimeoutException)
        ):
            return True
        if isinstance(exc, SDKError):
            status = getattr(getattr(exc, "raw_response", None), "status_code", None)
            return status in _RETRYABLE_STATUS
        return False

    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        """
        Call the Mistral chat completions endpoint and return a RunResult.
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
            return self._client.chat.complete(
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

            # usage / choices / message.content can be missing on truncated or
            # malformed responses; guard each access so token usage is still
            # recorded when content is absent.
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0

            choices = response.choices or []
            output = choices[0].message.content if choices else None
            if output is None:
                return RunResult(
                    run_id=config.run_id,
                    output_diagram_code=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration_ms=duration_ms,
                    cost_usd=compute_cost(
                        prompt_tokens, completion_tokens, self.pricing
                    ),
                    error=f"EmptyResponse: {self._PROVIDER_NAME} returned no content",
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

        except SDKError as e:
            return self._error_result(
                config, start_ms, f"SDKError: {e}", stats.retry_count
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
