"""
MAESTRO — SOP (Standard Operating Procedure) strategy
Decomposes diagram generation into 3 sequential sub-prompts.
Each step's output feeds the next step's input.
"""

import json
import time

from maestro.schemas import (
    InputFile,
    RunConfig,
    RunResult,
    SubResult,
)
from maestro.strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Prompt templates — one per step
# ---------------------------------------------------------------------------

STEP_1_PROMPT = """\
You are given a dataset describing entities and their relationships.
Your task is to extract all entities (nodes) and their hierarchy.

Rules:
- Output valid JSON only — no explanations, no markdown fencing
- Include every entity from the input
- Capture parent-child relationships (pools, lanes, subprocesses)
- Use this exact schema:
{{
  "entities": [
    {{
      "id": "string",
      "name": "string",
      "type": "string",
      "parent_id": "string or null"
    }}
  ]
}}

Input data:
{input_data}
"""

STEP_2_PROMPT = """\
You are given a list of entities and the original dataset.
Your task is to extract all relationships (edges) between entities.

Rules:
- Output valid JSON only — no explanations, no markdown fencing
- Include every sequence flow, message flow, and association
- Do not invent relationships not present in the data
- Use this exact schema:
{{
  "relationships": [
    {{
      "id": "string",
      "source": "string",
      "target": "string",
      "type": "string",
      "label": "string or null"
    }}
  ]
}}

Entities extracted in the previous step:
{entities_json}

Original input data:
{input_data}
"""

STEP_3_PROMPT = """\
You are given a set of entities and relationships extracted from a dataset.
Your task is to generate a Mermaid diagram that accurately represents them.

Rules:
- Output only valid Mermaid syntax
- Include all entities with correct hierarchy (subgraphs for pools/lanes/subprocesses)
- Include all relationships as edges
- Do not invent entities or relationships not provided
- Do not include explanations or markdown code fences
- Do not use realtionship IDs as edge labels

Entities:
{entities_json}

Relationships:
{relationships_json}
"""


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {"number": 1, "name": "extract_entities",       "prompt": STEP_1_PROMPT},
    {"number": 2, "name": "extract_relationships",  "prompt": STEP_2_PROMPT},
    {"number": 3, "name": "generate_mermaid",        "prompt": STEP_3_PROMPT},
]

# Max retries per step before aborting the whole run
MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str | None) -> str | None:
    """
    Remove markdown code fences if present.
    Models often wrap JSON in ```json ... ```
    """
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove first line (```json or ```)
        stripped = stripped.split("\n", 1)[-1]
    if stripped.endswith("```"):
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


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

        # Load input JSON
        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.dumps(json.loads(raw), indent=2)
        except FileNotFoundError:
            return self._abort(config, f"Input file not found: {input_file.file_path}")
        except json.JSONDecodeError as e:
            return self._abort(config, f"Invalid JSON in input file: {e}")
        except Exception as e:
            return self._abort(config, f"Failed to read input file: {e}")

        sub_results: list[SubResult] = []
        # Intermediate outputs passed between steps
        step_outputs: dict[str, str] = {}
        total_start = time.monotonic()

        for step in STEPS:
            step_num = step["number"]
            step_name = step["name"]

            # Build prompt with outputs from previous steps
            prompt = self._build_prompt(
                step, input_data, step_outputs
            )

            # Execute with retry
            sub, output_text = self._execute_step(
                config, step_num, step_name, prompt
            )
            sub_results.append(sub)

            # If step failed after retry, abort the whole run
            if sub.error is not None:
                return self._aggregate(
                    config, sub_results, total_start,
                    error=f"Step {step_num} ({step_name}) failed: {sub.error}"
                )

            # Store output for the next step
            step_outputs[step_name] = output_text

        # Success — step 3 output is the final Mermaid diagram
        return self._aggregate(
            config, sub_results, total_start,
            diagram_code=step_outputs["generate_mermaid"]
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
            result = self.provider.complete(prompt, config)

            # Accumulate metrics from every attempt
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_duration_ms += result.duration_ms
            total_cost_usd += result.cost_usd
            actual_retries = attempt

            if result.success:
                # For steps 1-2, validate JSON output
                output = result.output_diagram_code
                output = _strip_fences(output)
                if step_number < 3:
                    try:
                        json.loads(output)
                    except (json.JSONDecodeError, TypeError):
                        last_error = f"Invalid JSON output on attempt {attempt + 1}"
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

    def _abort(
        self, config: RunConfig, message: str
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Return a failed run for file-level errors before any LLM call.
        """
        result = RunResult(
            run_id=config.run_id,
            output_diagram_code=None,
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=0,
            cost_usd=0.0,
            error=message,
        )
        return (result, [])