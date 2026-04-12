"""
MAESTRO — Single Agent strategy (baseline)
One prompt, one LLM call, one diagram output.
This is the control condition all other strategies are compared against.
"""

import json

from maestro.schemas import InputFile, RunConfig, RunResult, SubResult
from maestro.strategies.base import BaseStrategy


PROMPT_TEMPLATE = """\
You are given a dataset describing entities and their relationships.
Your task is to generate a Mermaid diagram that accurately represents this data.

Rules:
- Output only valid Mermaid syntax
- Include all entities and relationships from the input
- Do not invent entities or relationships not present in the data
- Do not include explanations or markdown code fences
- Do not use internal IDs as edge labels

Input data:
{input_data}
"""


class SingleAgentStrategy(BaseStrategy):
    """
    Baseline strategy: one prompt → one LLM call → diagram code.

    No decomposition, no multi-step reasoning, no tool use.
    Establishes the lower bound for comparison with orchestrated strategies.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Load the input JSON, build a single prompt, call the provider.
        Returns (RunResult, []) — empty sub_results for single-agent.
        """

        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.loads(raw)

        except FileNotFoundError:
            return self._file_error(config, f"Input file not found: {input_file.file_path}")
        except json.JSONDecodeError as e:
            return self._file_error(config, f"Invalid JSON in input file: {e}")
        except Exception as e:
            return self._file_error(config, f"Failed to read input file: {e}")

        formatted_input = json.dumps(input_data, indent=2)
        prompt = PROMPT_TEMPLATE.format(input_data=formatted_input)

        # Single call — wrap result in tuple with empty sub_results
        result = self.provider.complete(prompt, config)
        return (result, [])

    def _file_error(
        self, config: RunConfig, message: str
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Return a failed RunResult for file-level errors before any LLM call.
        """
        result = RunResult(
            run_id              = config.run_id,
            output_diagram_code = None,
            prompt_tokens       = 0,
            completion_tokens   = 0,
            duration_ms         = 0,
            cost_usd            = 0.0,
            error               = message,
        )
        return (result, [])