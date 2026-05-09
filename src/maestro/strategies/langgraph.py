"""
MAESTRO — LangGraph strategy.

Orchestration shape: *the procedure expressed as an explicit state graph*.
The same three steps as SOP and CrewAI — extract entities, extract
relationships, render Mermaid — but each step is a node in a
``StateGraph`` and the order is encoded as edges between nodes.

Reader's-eye view:
- A ``TypedDict`` (``GraphState``) declares the channels that flow through
  the graph: input data, the two intermediate JSON payloads, the final
  diagram, an accumulating list of ``SubResult``, and an ``error`` channel
  used to short-circuit.
- Three node functions — one per step — each take the current state, call
  ``provider.complete()``, validate (steps 1 and 2), retry on failure, and
  return a *partial state update*. LangGraph merges these into the running
  state.
- The graph itself is built up with explicit ``add_node`` /
  ``add_edge`` / ``add_conditional_edges`` calls. The reader can read those
  five lines and see the DAG: ``START → entities → relationships → mermaid
  → END``, with each step routed to ``END`` early if its predecessor failed.
- Unlike CrewAI there is no LLM adapter to write: LangGraph imposes no
  abstraction over the model — node functions call our provider directly.

Prompts, retry budget and JSON validation come from ``_extraction.py`` so
SOP, CrewAI and LangGraph share the experimental control variable.
"""

import json
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from maestro.providers.base import LLMProvider
from maestro.schemas import (
    InputFile,
    RunConfig,
    RunResult,
    SubResult,
)
from maestro.strategies.base import BaseStrategy
from maestro.strategies._extraction import (
    JSON_EXTRACTION_SYSTEM_PROMPT,
    MAX_RETRIES,
    STEP_1_PROMPT,
    STEP_2_PROMPT,
    STEP_3_PROMPT,
    strip_fences,
    validate_step_payload,
)


# ---------------------------------------------------------------------------
# Graph state — every channel the nodes can read or write
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    State channels that flow through the graph.

    LangGraph reads each node's return value as a *partial update*: only the
    keys present in the dict get merged into the running state. ``total=False``
    lets us declare every channel up-front while leaving them unset on entry.
    """
    input_data:         str
    entities_json:      str
    relationships_json: str
    diagram_code:       str
    sub_results:        list[SubResult]
    error:              str | None


# ---------------------------------------------------------------------------
# Per-step execution — shared by all three nodes
# ---------------------------------------------------------------------------

def _run_step(
    *,
    provider:      LLMProvider,
    config:        RunConfig,
    step_number:   int,
    step_name:     str,
    prompt:        str,
    system_prompt: str | None,
) -> SubResult:
    """
    Execute one step with the same retry / validation / metric-accumulation
    semantics as SOP and CrewAI. Returns a ``SubResult`` with the step output
    in ``output_text`` (or ``None`` + ``error`` set on failure).

    This helper is what each node delegates to. It is defined at module level
    rather than as a strategy method so the node functions can read like
    plain pure functions of ``state`` — keeping the graph-shape of the file
    unobstructed.
    """
    last_error: str | None = None

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_duration_ms = 0
    total_cost_usd = 0.0
    actual_retries = 0

    for attempt in range(MAX_RETRIES + 1):
        actual_retries = attempt
        result = provider.complete(prompt, config, system_prompt=system_prompt)

        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        total_duration_ms += result.duration_ms
        total_cost_usd += result.cost_usd

        if not result.success:
            last_error = result.error
            continue

        output = strip_fences(result.output_diagram_code)
        if step_number < 3:
            is_valid, validation_error = validate_step_payload(output, step_number)
            if not is_valid:
                last_error = (
                    f"Invalid {step_name} payload on attempt {attempt + 1}: "
                    f"{validation_error}"
                )
                continue

        return SubResult(
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
        )

    return SubResult(
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
    )


# ---------------------------------------------------------------------------
# Node factory — closes over (provider, config) and returns the three nodes
# ---------------------------------------------------------------------------

def _make_nodes(provider: LLMProvider, config: RunConfig):
    """
    Build the three node functions. Each closure captures the provider and
    run config so the function signature LangGraph sees is the canonical
    ``(state) -> partial_state``.

    The nodes return *partial state updates* — only the keys they wrote.
    LangGraph merges those into the running state for the next node.
    """

    def extract_entities_node(state: GraphState) -> GraphState:
        prompt = STEP_1_PROMPT.format(input_data=state["input_data"])
        sub = _run_step(
            provider=provider,
            config=config,
            step_number=1,
            step_name="extract_entities",
            prompt=prompt,
            system_prompt=JSON_EXTRACTION_SYSTEM_PROMPT,
        )
        update: GraphState = {"sub_results": [*state.get("sub_results", []), sub]}
        if sub.error is not None:
            update["error"] = f"Step 1 (extract_entities) failed: {sub.error}"
        else:
            update["entities_json"] = sub.output_text
        return update

    def extract_relationships_node(state: GraphState) -> GraphState:
        prompt = STEP_2_PROMPT.format(
            input_data=state["input_data"],
            entities_json=state["entities_json"],
        )
        sub = _run_step(
            provider=provider,
            config=config,
            step_number=2,
            step_name="extract_relationships",
            prompt=prompt,
            system_prompt=JSON_EXTRACTION_SYSTEM_PROMPT,
        )
        update: GraphState = {"sub_results": [*state.get("sub_results", []), sub]}
        if sub.error is not None:
            update["error"] = f"Step 2 (extract_relationships) failed: {sub.error}"
        else:
            update["relationships_json"] = sub.output_text
        return update

    def generate_mermaid_node(state: GraphState) -> GraphState:
        prompt = STEP_3_PROMPT.format(
            entities_json=state["entities_json"],
            relationships_json=state["relationships_json"],
        )
        sub = _run_step(
            provider=provider,
            config=config,
            step_number=3,
            step_name="generate_mermaid",
            prompt=prompt,
            system_prompt=None,
        )
        update: GraphState = {"sub_results": [*state.get("sub_results", []), sub]}
        if sub.error is not None:
            update["error"] = f"Step 3 (generate_mermaid) failed: {sub.error}"
        else:
            update["diagram_code"] = sub.output_text
        return update

    return extract_entities_node, extract_relationships_node, generate_mermaid_node


# ---------------------------------------------------------------------------
# Conditional edge — short-circuit to END if a step set state["error"]
# ---------------------------------------------------------------------------

def _route_after_step(next_step: str):
    """
    Build a router function that LangGraph calls after each node to decide
    the next destination. If the node populated ``state["error"]`` we route
    straight to END; otherwise we continue to ``next_step``.

    Defining this once and parameterising it keeps the graph wiring symmetric
    across the three steps.
    """
    def router(state: GraphState) -> str:
        if state.get("error") is not None:
            return END
        return next_step
    return router


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class LangGraphStrategy(BaseStrategy):
    """
    LangGraph strategy.

    Decomposes diagram generation into a 3-node ``StateGraph``. Each node is
    a step from ``_extraction`` (same prompts, same JSON validation, same
    retry budget as SOP and CrewAI). The graph is built fresh per ``run()``
    because each invocation needs to close over a different ``RunConfig``.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Execute the three-step procedure as a compiled ``StateGraph``.

        Builds the graph fresh per call so each invocation closes over its
        own ``RunConfig``, then ``compile().invoke(...)`` runs it. The graph
        wiring (``add_node`` × 3, one ``add_edge`` from START, two
        ``add_conditional_edges`` for short-circuit-on-error, one
        ``add_edge`` to END) is the readable description of the DAG; node
        bodies live in ``_make_nodes`` and reuse ``_run_step`` for retry
        and validation so behaviour matches SOP and CrewAI exactly.
        """

        # Load input JSON — same shape as SOP and CrewAI for parity
        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.dumps(json.loads(raw), indent=2)
        except FileNotFoundError:
            return self._abort(config, f"Input file not found: {input_file.file_path}")
        except json.JSONDecodeError as e:
            return self._abort(config, f"Invalid JSON in input file: {e}")
        except Exception as e:
            return self._abort(config, f"Failed to read input file: {e}")

        total_start = time.monotonic()

        # Build the graph — 3 nodes, linear pipeline with error short-circuits.
        # Each `add_*` call below contributes one piece of the DAG; reading
        # them in order reconstructs the full graph in the reader's head.
        extract_entities, extract_relationships, generate_mermaid = _make_nodes(
            self.provider, config
        )

        graph = StateGraph(GraphState)
        graph.add_node("extract_entities",      extract_entities)
        graph.add_node("extract_relationships", extract_relationships)
        graph.add_node("generate_mermaid",      generate_mermaid)

        graph.add_edge(START, "extract_entities")
        graph.add_conditional_edges(
            "extract_entities",
            _route_after_step("extract_relationships"),
            {END: END, "extract_relationships": "extract_relationships"},
        )
        graph.add_conditional_edges(
            "extract_relationships",
            _route_after_step("generate_mermaid"),
            {END: END, "generate_mermaid": "generate_mermaid"},
        )
        graph.add_edge("generate_mermaid", END)

        compiled = graph.compile()

        # Execute the graph. LangGraph collects partial updates from each
        # node into a single final state dict.
        final_state: GraphState = compiled.invoke(
            {"input_data": input_data, "sub_results": []}
        )

        sub_results: list[SubResult] = final_state.get("sub_results", [])
        error = final_state.get("error")
        diagram_code = final_state.get("diagram_code") if error is None else None

        return self._aggregate(
            config, sub_results, total_start,
            diagram_code=diagram_code,
            error=error,
        )

    # -----------------------------------------------------------------------
    # Aggregation — same shape as SOP and CrewAI
    # -----------------------------------------------------------------------

    def _aggregate(
        self,
        config: RunConfig,
        subs: list[SubResult],
        total_start: float,
        diagram_code: str | None = None,
        error: str | None = None,
    ) -> tuple[RunResult, list[SubResult]]:
        """Sum sub-call stats into a parent RunResult — identical to SOP / CrewAI."""
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
        """File-level error before any LLM call — same shape as SOP and CrewAI."""
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