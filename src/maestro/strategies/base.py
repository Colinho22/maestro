"""
MAESTRO — Abstract strategy interface
All orchestration strategies (single agent, SOP, CrewAI, LangGraph) implement this.
"""

from abc import ABC, abstractmethod

from maestro.schemas import InputFile, RunConfig, RunResult, SubResult
from maestro.providers.base import LLMProvider


class BaseStrategy(ABC):
    """
    Base class for all orchestration strategies.
    Each strategy receives a provider and is responsible for:
    - Building the prompt(s) from the input file
    - Orchestrating one or more LLM calls
    - Returning a single RunResult
    """

    def __init__(self, provider: LLMProvider) -> None:
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