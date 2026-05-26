"""
MAESTRO — Control strategies.

**Controls are not baselines.** ``SingleAgentStrategy`` (in ``single.py``)
is the *comparison baseline* — the simplest real LLM-using approach that
the orchestration strategies are measured against. The classes here are
*control conditions* in the DSR / experimental-method sense: deterministic
non-strategies that bypass the LLM entirely and have a known expected
score, used to isolate the metric pipeline from the model under test.

The two words got conflated in early issue language; keep them distinct
in code, prose and (eventually) thesis text.

These three deterministic strategies serve two distinct purposes:

1. **Metric-pipeline sanity checks.** A normalization bug that makes the
   empty string match anything would make every real strategy look better
   than it is. A control whose expected score is known catches that class
   of bug at experiment runtime (in addition to the unit tests in
   ``tests/analysis/test_metrics.py``).

2. **Interpretation anchors.** Reporting "single-agent F1 = 0.62" without
   context is hard to defend; reporting "single-agent F1 = 0.62, null
   control F1 = 0.00, ground-truth control F1 = 1.00" frames the result
   on a known scale.

All class names keep the ``Strategy`` suffix per ``coderabbit.yaml``.

Determinism: each control's output is a pure function of the input file
(NullControl ignores even that). The matrix builder collapses the
``model`` and ``run_number`` dimensions for controls — running them 5×
per model would just produce duplicate rows.

Duration semantics: ``RunResult.duration_ms`` for control rows reflects
the wall-clock cost of the strategy itself, which is effectively zero
for NullControl (no I/O) and a single small file read for the other
two. Analysis code that filters on ``duration_ms > 0`` to "exclude
errored runs" would inadvertently exclude NullControl rows — filter on
``error IS NULL`` (or ``RunResult.success``) instead.
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
        """
        Return a RunResult carrying the empty diagram. ``input_file`` is
        accepted to satisfy the BaseStrategy contract but deliberately not
        read — the floor must be input-independent so the score reflects
        only the metric pipeline's response to "no entities, no edges".
        """
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
        """
        Read ``input_file.file_path`` and shove its bytes into
        ``output_diagram_code`` unmodified. The metric pipeline will see
        JSON where Mermaid was expected — exactly the input shape needed
        to test whether the parser is too permissive.
        """
        start = time.monotonic()
        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
        except Exception as e:
            return self._error_result(
                config, f"Failed to read input file: {e}", start=start
            )

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
        """
        Read the ground truth file and return its contents as the
        ``output_diagram_code``. The downstream metric pipeline will then
        compare ground truth to itself — anything less than F1 = 1.0
        indicates a bug in the scoring code, not in the "model".
        """
        start = time.monotonic()
        try:
            truth = input_file.ground_truth_path.read_text(encoding="utf-8")
        except Exception as e:
            return self._error_result(
                config, f"Failed to read ground truth: {e}", start=start
            )

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
