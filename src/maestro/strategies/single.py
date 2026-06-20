"""
MAESTRO: Single Agent strategy (comparison baseline)

One prompt, one LLM call, one diagram output. This is the *comparison
baseline*: the simplest LLM-using approach that CrewAI / SOP / LangGraph
are measured against. It is NOT a control: controls (NullControl,
CopyInputControl, GroundTruthEchoControl in ``controls.py``) bypass the
LLM entirely and serve a different role (metric-pipeline sanity checks).

Vocabulary cheat sheet for this codebase:
- "Baseline" = the simplest *real* strategy in the comparison set
  (i.e. this file). Answers "does orchestration X beat no orchestration?".
- "Control"  = a non-strategy with a known expected score. Answers
  "is the metric pipeline correctly scoring things?".

These two roles got conflated in earlier iterations of the codebase
(and in the original issue language). They are distinct.
"""

from __future__ import annotations

import json

from maestro.prompts import render_rules
from maestro.schemas import InputFile, RunConfig, RunResult, SubResult
from maestro.strategies._extraction import extract_diagram_type
from maestro.strategies.base import BaseStrategy

# Rules come from the canonical contract (maestro.prompts) so single-agent and
# the multi-step strategies are given a byte-identical output contract. The
# runtime placeholders are escaped (``{{...}}``) so they survive this f-string
# and are filled later by ``.format(diagram_type=..., input_data=...)``. The
# diagram type is task context, stated explicitly to every strategy so the
# notation-dependent label rules are applied uniformly (the baseline would
# otherwise have to infer it from the input metadata on its own).
PROMPT_TEMPLATE = f"""\
You are given a dataset describing entities and their relationships.
Your task is to generate a Mermaid diagram that accurately represents this data.

The diagram notation is: {{diagram_type}}

Rules:
{render_rules()}

Input data:
{{input_data}}
"""


class SingleAgentStrategy(BaseStrategy):
    """
    Comparison baseline: one prompt -> one LLM call -> diagram code.

    No decomposition, no multi-step reasoning, no tool use. Establishes
    the *no-orchestration* reference point that CrewAI, SOP and LangGraph
    are compared against. Distinct from the control strategies in
    ``controls.py``, which bypass the LLM entirely.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Load the input JSON, build a single prompt, call the provider.
        Returns (RunResult, []): empty sub_results for single-agent.
        """

        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.loads(raw)

        except FileNotFoundError:
            return self._error_result(
                config, f"Input file not found: {input_file.file_path}"
            )
        except json.JSONDecodeError as e:
            return self._error_result(config, f"Invalid JSON in input file: {e}")
        except Exception as e:
            return self._error_result(config, f"Failed to read input file: {e}")

        formatted_input = json.dumps(input_data, indent=2)
        diagram_type = extract_diagram_type(raw)
        prompt = PROMPT_TEMPLATE.format(
            diagram_type=diagram_type, input_data=formatted_input
        )

        # Single call: wrap result in tuple with empty sub_results
        result = self.provider.complete(prompt, config)
        return (result, [])
