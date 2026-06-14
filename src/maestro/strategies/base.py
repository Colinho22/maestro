"""
MAESTRO: Abstract strategy interface
All orchestration strategies (single agent, SOP, CrewAI, LangGraph) implement this.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

from maestro.providers.base import LLMProvider
from maestro.schemas import InputFile, RunConfig, RunResult, SubResult


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
    provider: calling ``self.provider.complete(...)`` on None will error
    loudly, which is the right failure mode for a misconfigured run.
    """

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider

    @abstractmethod
    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
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
        is *not* the persisted ``Strategy`` enum value, which lives on
        ``RunConfig.strategy`` instead.
        """
        return self.__class__.__name__

    def _error_result(
        self,
        config: RunConfig,
        message: str,
        *,
        start: Optional[float] = None,
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Build a failed ``(RunResult, [])`` for errors that prevent normal
        execution: file not found, JSON parse failure, control-strategy
        I/O failure, etc.

        Token counts and ``cost_usd`` are zero. ``duration_ms`` is the
        wall-clock since ``start`` (use ``time.monotonic()`` at the
        strategy entry point) when supplied, else ``0`` for errors that
        happen before any work was attempted.

        Two callers exist today:
        - SOP / CrewAI / LangGraph / SingleAgent abort *before* any
          monotonic-clock start, so they omit ``start`` and get
          ``duration_ms=0``. The error is structural, not measured work.
        - Control strategies pass ``start=`` so the (small but real)
          wall-clock cost of the failed file read is still recorded.
        """
        duration_ms = int((time.monotonic() - start) * 1000) if start is not None else 0
        result = RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=duration_ms,
            cost_usd=0.0,
            error=message,
        )
        return (result, [])
