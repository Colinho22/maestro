"""
MAESTRO: Abstract LLM provider interface
All concrete providers (OpenAI, Anthropic, etc.) must implement this.
"""

from abc import ABC, abstractmethod

from maestro.prompts import MERMAID_SYSTEM_IDENTITY
from maestro.schemas import ModelPricing, RunConfig, RunResult


class LLMProvider(ABC):
    """
    Base class for all LLM providers.
    One instance per provider, reused across multiple runs.
    """

    # Centralized temperature setting: 0 for reproducibility across all providers
    TEMPERATURE = 0

    # Default system identity for Mermaid generation, shared by every provider.
    # Subclasses inherit this; a strategy that needs a different identity for a
    # given call (e.g. SOP steps 1-2 requesting JSON) passes system_prompt
    # explicitly to complete().
    SYSTEM_PROMPT = MERMAID_SYSTEM_IDENTITY

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        # api_key stored on instance, never logged or serialised
        self.api_key = api_key
        self.pricing = pricing

    @abstractmethod
    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        """
        Send a prompt to the LLM and return a fully populated RunResult.

        If system_prompt is None, the provider falls back to its own default
        SYSTEM_PROMPT class attribute. Strategies that need a different system
        identity for a given call (e.g. SOP intermediate steps requesting JSON
        instead of Mermaid) pass an explicit string.

        Implementations must:
        - Measure wall-clock duration_ms
        - Populate prompt_tokens + completion_tokens from the API response
        - Call compute_cost() and set cost_usd
        - Catch all API exceptions and surface them via RunResult.error
        - Never raise: always return a RunResult (success or error)
        """
        ...

    @property
    def model_name(self) -> str:
        """
        The model identifier this provider was constructed for. Matches the
        string stored in ``RunConfig.model`` and ``ModelPricing.model``, so
        callers can correlate provider instances with rows in the run_configs
        table without reaching into ``self.pricing``.
        """
        return self.pricing.model
