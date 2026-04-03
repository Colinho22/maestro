"""
MAESTRO — Single Agent strategy (baseline)
One prompt, one LLM call, one diagram output.
This is the control condition all other strategies are compared against.
"""

import json

from maestro.schemas import InputFile, RunConfig, RunResult
from maestro.strategies.base import BaseStrategy


# Prompt template — kept here so it's easy to version and compare across strategies
PROMPT_TEMPLATE = """\
You are given a dataset describing entities and their relationships.
Your task is to generate a Mermaid diagram that accurately represents this data.

Rules:
- Output only valid Mermaid syntax
- Include all entities and relationships from the input
- Do not invent entities or relationships not present in the data
- Do not include explanations or markdown code fences

Input data:
{input_data}
"""


class SingleAgentStrategy(BaseStrategy):
    """
    Baseline strategy: one prompt → one LLM call → diagram code.

    No decomposition, no multi-step reasoning, no tool use.
    Establishes the lower bound for comparison with orchestrated strategies.
    """

    def run(self, input_file: InputFile, config: RunConfig) -> RunResult:
        """
        Load the input JSON, build a single prompt, call the provider.
        Returns the RunResult directly — no post-processing.
        """

        try:
            # Load the input JSON from disk
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.loads(raw)

        except FileNotFoundError:
            return self._file_error(config, f"Input file not found: {input_file.file_path}")
        except json.JSONDecodeError as e:
            return self._file_error(config, f"Invalid JSON in input file: {e}")

        # Serialise back to a clean indented string — helps the model parse structure
        formatted_input = json.dumps(input_data, indent=2)

        prompt = PROMPT_TEMPLATE.format(input_data=formatted_input)

        # Single call — provider handles timing, token counting, and error capture
        return self.provider.complete(prompt, config)

    def _file_error(self, config: RunConfig, message: str) -> RunResult:
        """
        Return a failed RunResult for file-level errors before any LLM call.
        duration_ms is 0 since no API call was made.
        """
        return RunResult(
            run_id              = config.run_id,
            output_diagram_code = None,
            prompt_tokens       = 0,
            completion_tokens   = 0,
            duration_ms         = 0,
            cost_usd            = 0.0,
            error               = message,
        )