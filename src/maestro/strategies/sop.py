"""
MAESTRO — SOP (Standard Operating Procedure) strategy.

Orchestration shape: a *hand-coded sequential procedure*. The strategy walks
a fixed list of steps in a Python `for` loop, threading each step's output
into the next step's prompt by hand. There is no framework — the loop, the
state dict, the retry block and the abort logic are all written explicitly
in this file so the procedure is fully visible to the reader.

This is the procedural baseline that CrewAI and LangGraph are compared
against: same task, same prompts, same retry budget — different orchestrator.

Prompts and validation rules live in `_extraction.py` so all multi-step
strategies share them byte-for-byte.
"""

import json
import time

from maestro.schemas import (
    InputFile,
    RunConfig,
    RunResult,
    SubResult,
)
from maestro.strategies._extraction import (
    JSON_EXTRACTION_SYSTEM_PROMPT,
    MAX_RETRIES,
    STEP_1_PROMPT,
    STEP_2_PROMPT,
    STEP_3_PROMPT,
    strip_fences,
    validate_step_payload,
)
from maestro.strategies.base import BaseStrategy

# ---------------------------------------------------------------------------
# Step table — the procedure as data
# ---------------------------------------------------------------------------
# Steps 1 and 2 use the JSON system prompt; step 3 falls back to the
# provider's default Mermaid system prompt by passing system_prompt=None.

STEPS = [
    {
        "number": 1,
        "name": "extract_entities",
        "prompt": STEP_1_PROMPT,
        "system_prompt": JSON_EXTRACTION_SYSTEM_PROMPT,
    },
    {
        "number": 2,
        "name": "extract_relationships",
        "prompt": STEP_2_PROMPT,
        "system_prompt": JSON_EXTRACTION_SYSTEM_PROMPT,
    },
    {
        "number": 3,
        "name": "generate_mermaid",
        "prompt": STEP_3_PROMPT,
        "system_prompt": None,
    },
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SOPStrategy(BaseStrategy):
    """
    Standard Operating Procedure strategy.
    Decomposes diagram generation into 3 fixed sequential steps:
      1. Extract entities from input
      2. Extract relationships from input + entities
      3. Generate Mermaid from entities + relationships

    Each step is a separate LLM call. Results are chained.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Execute the three-step procedure for one input.

        Walks ``STEPS`` in order, threading each step's output into the
        next step's prompt via ``step_outputs``. A failed step (after its
        retry budget) aborts the whole run and the parent ``RunResult``
        carries the error; the ``SubResult`` for the failing step is still
        appended so partial-failure cost is observable in the database.
        """

        # Load input JSON
        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.dumps(json.loads(raw), indent=2)
        except FileNotFoundError:
            return self._error_result(
                config, f"Input file not found: {input_file.file_path}"
            )
        except json.JSONDecodeError as e:
            return self._error_result(config, f"Invalid JSON in input file: {e}")
        except Exception as e:
            return self._error_result(config, f"Failed to read input file: {e}")

        sub_results: list[SubResult] = []
        # Intermediate outputs passed between steps
        step_outputs: dict[str, str] = {}
        total_start = time.monotonic()

        for step in STEPS:
            step_num = step["number"]
            step_name = step["name"]
            step_system_prompt = step["system_prompt"]

            # Build prompt with outputs from previous steps
            prompt = self._build_prompt(step, input_data, step_outputs)

            # Execute with retry
            sub, output_text = self._execute_step(
                config,
                step_num,
                step_name,
                prompt,
                step_system_prompt,
            )
            sub_results.append(sub)

            # If step failed after retry, abort the whole run
            if sub.error is not None:
                return self._aggregate(
                    config,
                    sub_results,
                    total_start,
                    error=f"Step {step_num} ({step_name}) failed: {sub.error}",
                )

            # Store output for the next step
            step_outputs[step_name] = output_text

        # Success — step 3 output is the final Mermaid diagram
        return self._aggregate(
            config,
            sub_results,
            total_start,
            diagram_code=step_outputs["generate_mermaid"],
        )

    def _build_prompt(
        self,
        step: dict,
        input_data: str,
        step_outputs: dict[str, str],
    ) -> str:
        """
        Format the prompt template with available context.
        Each step gets different variables depending on what's available.
        """
        template = step["prompt"]
        fmt = {"input_data": input_data}

        if "extract_entities" in step_outputs:
            fmt["entities_json"] = step_outputs["extract_entities"]

        if "extract_relationships" in step_outputs:
            fmt["relationships_json"] = step_outputs["extract_relationships"]

        return template.format(**fmt)

    def _execute_step(
        self,
        config: RunConfig,
        step_number: int,
        step_name: str,
        prompt: str,
        system_prompt: str | None,
    ) -> tuple[SubResult, str | None]:
        """
        Run one step with retry logic.
        Returns (SubResult, output_text).
        output_text is None if the step failed.
        """
        last_error = None
        result = None

        # Accumulate metrics across all attempts (including failed ones)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_duration_ms = 0
        total_cost_usd = 0.0
        actual_retries = 0

        for attempt in range(MAX_RETRIES + 1):
            # provider.complete() returns a RunResult — we extract what we need
            result = self.provider.complete(prompt, config, system_prompt=system_prompt)

            # Accumulate metrics from every attempt
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_duration_ms += result.duration_ms
            total_cost_usd += result.cost_usd
            actual_retries = attempt

            if result.success:
                # For steps 1-2, validate JSON output
                output = strip_fences(result.output_diagram_code)
                if step_number < 3:
                    is_valid, validation_error = validate_step_payload(
                        output, step_number
                    )
                    if not is_valid:
                        last_error = (
                            f"Invalid {step_name} payload on attempt {attempt + 1}: "
                            f"{validation_error}"
                        )
                        continue

                return (
                    SubResult(
                        run_id=config.run_id,
                        step_number=step_number,
                        step_name=step_name,
                        output_text=output,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        duration_ms=total_duration_ms,
                        cost_usd=total_cost_usd,
                        error=None,
                        retry_count=actual_retries,
                    ),
                    output,
                )
            else:
                last_error = result.error

        # All attempts failed
        return (
            SubResult(
                run_id=config.run_id,
                step_number=step_number,
                step_name=step_name,
                output_text=None,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                duration_ms=total_duration_ms,
                cost_usd=total_cost_usd,
                error=last_error or "No attempts executed",
                retry_count=actual_retries,
            ),
            None,
        )

    def _aggregate(
        self,
        config: RunConfig,
        subs: list[SubResult],
        total_start: float,
        diagram_code: str | None = None,
        error: str | None = None,
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Sum all sub-call stats into one RunResult for the parent run.
        """
        result = RunResult(
            run_id=config.run_id,
            output_diagram_code=diagram_code,
            prompt_tokens=sum(s.prompt_tokens for s in subs),
            completion_tokens=sum(s.completion_tokens for s in subs),
            duration_ms=int((time.monotonic() - total_start) * 1000),
            cost_usd=sum(s.cost_usd for s in subs),
            error=error,
        )
        return (result, subs)
