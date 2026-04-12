"""
MAESTRO — Anthropic provider implementation
Wraps the Anthropic messages API into the LLMProvider interface.
"""

import time

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError

from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost
from maestro.providers.base import LLMProvider


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

    def complete(self, prompt: str, config: RunConfig) -> RunResult:
        """
        Call the Anthropic messages endpoint and return a RunResult.
        Never raises — all exceptions are captured into RunResult.error.
        """

        start_ms = time.monotonic()

        try:
            response = self._client.messages.create(
                model=config.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            duration_ms = int((time.monotonic() - start_ms) * 1000)

            # Anthropic returns usage on every non-streaming response
            prompt_tokens     = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            # Response content is a list of blocks — grab the first text block
            output = response.content[0].text

            return RunResult(
                run_id              = config.run_id,
                output_diagram_code = output,
                prompt_tokens       = prompt_tokens,
                completion_tokens   = completion_tokens,
                duration_ms         = duration_ms,
                cost_usd            = compute_cost(
                    prompt_tokens, completion_tokens, self.pricing
                ),
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
            run_id              = config.run_id,
            output_diagram_code = None,
            prompt_tokens       = 0,
            completion_tokens   = 0,
            duration_ms         = int((time.monotonic() - start_ms) * 1000),
            cost_usd            = 0.0,
            error               = error,
        )