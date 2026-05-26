"""
MAESTRO — Abstract strategy interface
All orchestration strategies (single agent, SOP, CrewAI, LangGraph) implement this.
"""

from abc import ABC, abstractmethod
from typing import Optional

from maestro.schemas import InputFile, RunConfig, RunResult, SubResult
from maestro.providers.base import LLMProvider


class BaseStrategy(ABC):
    """
    Base class for all orchestration strategies.
    Each strategy receives a provider and is responsible for:
    - Building the prompt(s) from the input file
    - Orchestrating one or more LLM calls
    - Returning a single RunResult

    ``provider`` is Optional because control strategies (NullControlStrategy,
    CopyInputControlStrategy, GroundTruthEchoControlStrategy) bypass the LLM
    entirely and have no use for one. Real strategies must still supply a
    provider — calling ``self.provider.complete(...)`` on None will error
    loudly, which is the right failure mode for a misconfigured run.
    """

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider

    @abstractmethod
    def run(self, input_file: InputFile, config: RunConfig) -> tuple[RunResult, list[SubResult]]:
        """
        Execute the strategy for one input and return the result including sub results.
        Must always return a RunResult.
        """
        ...

    @property
    def name(self) -> str:
        """
        Human-readable name of the strategy, used in log lines and progress
        output. Returns the concrete class name (e.g. ``SOPStrategy``); this
        is *not* the persisted ``Strategy`` enum value — that lives on
        ``RunConfig.strategy`` instead.
        """
        return self.__class__.__name__