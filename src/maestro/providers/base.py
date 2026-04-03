"""
MAESTRO — Abstract LLM provider interface
All concrete providers (OpenAI, Anthropic, etc.) must implement this.
"""

from abc import ABC, abstractmethod

from maestro.schemas import ModelPricing, RunConfig, RunResult


class LLMProvider(ABC):
    """
    Base class for all LLM providers.
    One instance per provider — reused across multiple runs.
    """

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        # api_key stored on instance — never logged or serialised
        self.api_key = api_key
        self.pricing = pricing

    @abstractmethod
    def complete(self, prompt: str, config: RunConfig) -> RunResult:
        """
        Send a prompt to the LLM and return a fully populated RunResult.

        Implementations must:
        - Measure wall-clock duration_ms
        - Populate prompt_tokens + completion_tokens from the API response
        - Call compute_cost() and set cost_usd
        - Catch all API exceptions and surface them via RunResult.error
        - Never raise — always return a RunResult (success or error)
        """
        ...

    @property
    def model_name(self) -> str:
        # Convenience accessor — matches RunConfig.model
        return self.pricing.model