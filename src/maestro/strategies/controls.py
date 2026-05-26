"""
MAESTRO — Control strategies.

Three deterministic strategies that bypass the LLM entirely. They serve two
distinct purposes in the experimental design:

1. **Metric-pipeline sanity checks.** A normalization bug that makes the
   empty string match anything would make every real strategy look better
   than it is. A control whose expected score is known catches that class
   of bug at experiment runtime (in addition to the unit tests in
   ``tests/analysis/test_metrics.py``).

2. **Interpretation anchors.** Reporting "single-agent F1 = 0.62" without
   context is hard to defend; reporting "single-agent F1 = 0.62, null
   control F1 = 0.00, ground-truth control F1 = 1.00" frames the result
   on a known scale.

Naming follows the DSR convention: these are *control conditions*, not
"baselines" (the issue used baseline loosely — ``SingleAgentStrategy``
already calls itself the baseline in the comparison sense). All class
names keep the ``Strategy`` suffix per ``coderabbit.yaml``.

Determinism: each control's output is a pure function of the input file
(NullControl ignores even that). The matrix builder collapses the
``model`` and ``run_number`` dimensions for controls — running them 5×
per model would just produce duplicate rows.
"""

from __future__ import annotations

import time

from maestro.schemas import InputFile, RunConfig, RunResult, SubResult
from maestro.strategies.base import BaseStrategy


class NullControlStrategy(BaseStrategy):
    """
    Floor control: returns a syntactically minimal but empty Mermaid diagram.

    Expected metric behavior: entity_f1 ≈ 0 and relationship_f1 ≈ 0 (the
    diagram contains no entities and no relationships to match). If a
    NullControl row ever scores meaningfully above zero, the metric
    pipeline is overcounting somewhere.
    """

    # Single-line valid Mermaid. Choosing ``flowchart LR`` (rather than
    # ``graph LR``) matches the ground-truth files in ``data/``, so the
    # parser path under test is the same one real strategies exercise.
    EMPTY_DIAGRAM = "flowchart LR\n"

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        start = time.monotonic()
        duration_ms = int((time.monotonic() - start) * 1000)
        result = RunResult(
            run_id=config.run_id,
            output_diagram_code=self.EMPTY_DIAGRAM,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=duration_ms,
            cost_usd=0.0,
        )
        return (result, [])


class CopyInputControlStrategy(BaseStrategy):
    """
    Floor control: returns the raw input file's contents as the "diagram".

    Expected metric behavior: ``parses_valid=False`` (the input is JSON,
    not Mermaid) and F1 ≈ 0 (the entity/relationship extractor won't find
    valid Mermaid syntax in a JSON document). If this scores meaningfully
    above zero, the parser is too permissive — e.g. regex-matching loose
    patterns that happen to appear in JSON strings.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        start = time.monotonic()
        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
        except Exception as e:
            return (self._file_error(config, start, f"Failed to read input file: {e}"), [])

        duration_ms = int((time.monotonic() - start) * 1000)
        result = RunResult(
            run_id=config.run_id,
            output_diagram_code=raw,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=duration_ms,
            cost_usd=0.0,
        )
        return (result, [])

    def _file_error(
        self, config: RunConfig, start: float, message: str
    ) -> RunResult:
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=int((time.monotonic() - start) * 1000),
            cost_usd=0.0,
            error=message,
        )


class GroundTruthEchoControlStrategy(BaseStrategy):
    """
    Ceiling control: returns the ground-truth file's contents verbatim.

    Expected metric behavior: every F1 = 1.0, ``parses_valid=True`` (or
    ``None`` if ``mmdc`` is not installed locally). If this *fails* to
    score 1.0 on any metric, the metric pipeline is over-strict in a way
    that would penalise real strategies even when they produce the
    correct answer — a louder bug than the floor controls catch.

    Reading the ground truth is the whole point of this control; there is
    no "leak" because no learning happens here. The control row exists
    precisely to verify the scoring code, not to compete on the metric.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        start = time.monotonic()
        try:
            truth = input_file.ground_truth_path.read_text(encoding="utf-8")
        except Exception as e:
            return (self._file_error(config, start, f"Failed to read ground truth: {e}"), [])

        duration_ms = int((time.monotonic() - start) * 1000)
        result = RunResult(
            run_id=config.run_id,
            output_diagram_code=truth,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=duration_ms,
            cost_usd=0.0,
        )
        return (result, [])

    def _file_error(
        self, config: RunConfig, start: float, message: str
    ) -> RunResult:
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=int((time.monotonic() - start) * 1000),
            cost_usd=0.0,
            error=message,
        )
