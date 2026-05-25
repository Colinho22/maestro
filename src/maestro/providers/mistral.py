"""
MAESTRO — Mistral provider implementation
Wraps the Mistral chat completions API into the LLMProvider interface.
"""

import time

from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost
from maestro.providers.base import LLMProvider


class MistralProvider(LLMProvider):
    """
    Concrete provider for Mistral models (mistral-small, mistral-large, etc.)
    Uses the official mistralai SDK (>=2.4.7). The 2.x SDK moved ``Mistral``
    and ``SDKError`` out of the package root into ``mistralai.client`` and
    ``mistralai.client.errors`` respectively.
    """

    SYSTEM_PROMPT = (
        "You are a diagram generation assistant. "
        "Respond only with valid Mermaid diagram code. "
        "Do not include any explanation, markdown fencing, or additional text."
    )

    MAX_TOKENS = 4096

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        super().__init__(api_key, pricing)
        self._client = Mistral(api_key=api_key)

    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        """
        Call the Mistral chat completions endpoint and return a RunResult.
        Never raises — all exceptions are captured into RunResult.error.
        """

        start_ms = time.monotonic()
        effective_system = system_prompt if system_prompt is not None else self.SYSTEM_PROMPT

        try:
            response = self._client.chat.complete(
                model=config.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                messages=[
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": prompt},
                ],
            )

            duration_ms = int((time.monotonic() - start_ms) * 1000)

            # usage / choices / message.content can be missing on truncated or
            # malformed responses — guard each access so token usage is still
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
                    error="EmptyResponse: Mistral returned no content",
                )

            return RunResult(
                run_id=config.run_id,
                output_diagram_code=output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                cost_usd=compute_cost(
                    prompt_tokens, completion_tokens, self.pricing
                ),
            )

        except SDKError as e:
            return self._error_result(config, start_ms, f"SDKError: {e}")

        except Exception as e:
            return self._error_result(config, start_ms, f"UnexpectedError: {e}")

    def _error_result(
        self, config: RunConfig, start_ms: float, error: str
    ) -> RunResult:
        """Build a failed RunResult with zero token counts and the error message."""
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=int((time.monotonic() - start_ms) * 1000),
            cost_usd=0.0,
            error=error,
        )