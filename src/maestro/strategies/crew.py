"""
MAESTRO — CrewAI strategy.

Orchestration shape: *role-based agents executing a sequential crew*. The
same three steps as SOP — extract entities, extract relationships, render
Mermaid — but each step is reified as a CrewAI ``Agent`` + ``Task`` and
executed inside a ``Crew(process=Process.sequential)``.

Reader's-eye view:
- Each step gets its own Agent (role / goal / backstory) and its own Task
  (description / expected_output).
- Each step also gets its own Crew with a single Task. We deliberately do
  NOT use CrewAI's ``Task.context=[prev_task]`` chaining: we want the prompt
  content to be byte-identical to SOP, so harvested outputs from step N are
  passed into step N+1 via the same ``{entities_json}`` / ``{relationships_json}``
  template variables that SOP uses. The CrewAI machinery (agent personas,
  task framing, kickoff lifecycle) is what differs from SOP — not the prompt
  content.
- LLM traffic is routed through ``MaestroBackedLLM``, a thin ``BaseLLM``
  subclass that delegates every call to our ``LLMProvider``. This keeps
  token counts and cost in our pricing schema (``compute_cost``) instead of
  relying on CrewAI/LiteLLM's separate accounting.

Prompts, retry budget and JSON validation come from ``_extraction.py`` so
SOP, CrewAI and LangGraph share the experimental control variable.
"""

import json
import time
from dataclasses import dataclass, field

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

from maestro.providers.base import LLMProvider
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
# Per-call telemetry — captured by the LLM adapter, harvested by the strategy
# ---------------------------------------------------------------------------


@dataclass
class _CallRecord:
    """One LLM round-trip as observed by the adapter."""

    prompt_tokens: int
    completion_tokens: int
    duration_ms: int
    cost_usd: float
    output_text: str | None
    error: str | None


@dataclass
class _Recorder:
    """Mutable list of call records owned by the strategy, written by the adapter."""

    calls: list[_CallRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM adapter — bridges CrewAI's BaseLLM contract to our LLMProvider
# ---------------------------------------------------------------------------


class MaestroBackedLLM(BaseLLM):
    """
    CrewAI ``BaseLLM`` implementation that delegates every call to one of our
    ``LLMProvider`` instances.

    Why this exists:
    - CrewAI's default ``LLM`` routes through LiteLLM with its own credentials,
      retry logic and usage accounting. For a thesis-grade experiment we want
      a *single* code path computing tokens and cost across every strategy,
      so we hand CrewAI an LLM that is just a thin shim over our provider.

    One adapter instance is created per step, because each step needs a
    different ``system_prompt_override`` (steps 1-2 want JSON, step 3 wants
    Mermaid via the provider default).
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: RunConfig,
        system_prompt_override: str | None,
        recorder: _Recorder,
    ) -> None:
        # BaseLLM expects model + a few optionals — we pass through pricing.model
        super().__init__(model=provider.model_name, temperature=provider.TEMPERATURE)
        self._provider = provider
        self._config = config
        self._system_prompt_override = system_prompt_override
        self._recorder = recorder

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        **kwargs,
    ) -> str:
        """
        CrewAI calls this with either a string prompt or a list of role/content
        dicts. We collapse it into a single user-side prompt and forward to
        ``provider.complete()``.

        ``**kwargs`` swallows ``from_task`` / ``from_agent`` / ``response_model``
        and any future CrewAI additions to the BaseLLM contract — we don't
        need them, and accepting them keeps the adapter forward-compatible
        without a signature change.
        """
        prompt = self._collapse_messages(messages)

        result = self._provider.complete(
            prompt,
            self._config,
            system_prompt=self._system_prompt_override,
        )

        self._recorder.calls.append(
            _CallRecord(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                duration_ms=result.duration_ms,
                cost_usd=result.cost_usd,
                output_text=result.output_diagram_code,
                error=result.error,
            )
        )

        # CrewAI expects a string. On error we still return a string so the
        # crew lifecycle completes; the strategy detects the failure via the
        # recorder and surfaces it through SubResult.error.
        return result.output_diagram_code or ""

    @staticmethod
    def _collapse_messages(messages) -> str:
        """
        CrewAI's ``Agent`` + ``Task`` machinery composes a list of messages
        with role 'system' (agent persona) and 'user' (task description). We
        ignore CrewAI's system message — our system prompt is set explicitly
        per step via ``system_prompt_override`` — and concatenate the user
        portion as the prompt.
        """
        if isinstance(messages, str):
            return messages
        user_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        return "\n\n".join(p for p in user_parts if p)


# ---------------------------------------------------------------------------
# Step definition — three role/goal/prompt triples, one per step
# ---------------------------------------------------------------------------
#
# The agent personas (role/goal/backstory) are deliberately minimal: we are
# studying CrewAI's *orchestration overhead*, not prompt-engineering personas.
# The task description IS the SOP step prompt, byte-for-byte.

STEPS = [
    {
        "number": 1,
        "name": "extract_entities",
        "agent_role": "Entity Extractor",
        "agent_goal": (
            "Extract every entity and its hierarchy from the input dataset as JSON."
        ),
        "task_prompt": STEP_1_PROMPT,
        "expected_output": "A JSON object with an `entities` list.",
        "system_prompt": JSON_EXTRACTION_SYSTEM_PROMPT,
    },
    {
        "number": 2,
        "name": "extract_relationships",
        "agent_role": "Relationship Extractor",
        "agent_goal": "Extract every relationship between entities as JSON.",
        "task_prompt": STEP_2_PROMPT,
        "expected_output": "A JSON object with a `relationships` list.",
        "system_prompt": JSON_EXTRACTION_SYSTEM_PROMPT,
    },
    {
        "number": 3,
        "name": "generate_mermaid",
        "agent_role": "Mermaid Renderer",
        "agent_goal": (
            "Generate a valid Mermaid diagram from the provided entities "
            "and relationships."
        ),
        "task_prompt": STEP_3_PROMPT,
        "expected_output": "Valid Mermaid diagram code.",
        "system_prompt": None,
    },
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class CrewAIStrategy(BaseStrategy):
    """
    CrewAI strategy.

    Decomposes diagram generation into 3 sequential steps, each implemented
    as a single-task ``Crew`` with one role-specialised ``Agent``. Outputs
    are harvested between kickoffs and threaded into the next step's prompt
    template, so prompt content matches SOP exactly.
    """

    def run(
        self, input_file: InputFile, config: RunConfig
    ) -> tuple[RunResult, list[SubResult]]:
        """
        Execute the three-step procedure as three single-task Crew kickoffs.

        Each step builds a fresh ``MaestroBackedLLM`` + ``Agent`` + ``Task``
        + ``Crew`` (CrewAI mutates internal state on kickoff, so reuse is
        unsafe), runs ``crew.kickoff()``, and harvests exactly one call from
        the recorder. Outputs are threaded forward via the same prompt
        templates SOP uses, so prompt content is byte-identical across the
        two strategies — the difference the experiment captures is purely
        the orchestration overhead CrewAI's machinery adds.
        """

        # Load input JSON — same shape as SOP for parity
        try:
            raw = input_file.file_path.read_text(encoding="utf-8")
            input_data = json.dumps(json.loads(raw), indent=2)
        except FileNotFoundError:
            return self._error_result(
                config, f"Input file not found: {input_file.file_path}"
            )
        except json.JSONDecodeError as e:
            return self._error_result(config, f"Invalid JSON in input file: {e}")
        except Exception as e:  # noqa: BLE001
            return self._error_result(config, f"Failed to read input file: {e}")

        sub_results: list[SubResult] = []
        step_outputs: dict[str, str] = {}
        total_start = time.monotonic()

        for step in STEPS:
            # Build the prompt with outputs from previous steps (same logic as SOP)
            prompt = self._build_prompt(step, input_data, step_outputs)

            # Execute this step as a single-task Crew, with retries
            sub, output_text = self._execute_step(
                config=config,
                step_number=step["number"],
                step_name=step["name"],
                agent_role=step["agent_role"],
                agent_goal=step["agent_goal"],
                task_description=prompt,
                expected_output=step["expected_output"],
                system_prompt=step["system_prompt"],
            )
            sub_results.append(sub)

            if sub.error is not None:
                return self._aggregate(
                    config,
                    sub_results,
                    total_start,
                    error=f"Step {step['number']} ({step['name']}) failed: {sub.error}",
                )

            step_outputs[step["name"]] = output_text

        return self._aggregate(
            config,
            sub_results,
            total_start,
            diagram_code=step_outputs["generate_mermaid"],
        )

    # -----------------------------------------------------------------------
    # Step execution — one step = one Crew kickoff (with retry)
    # -----------------------------------------------------------------------

    def _execute_step(
        self,
        config: RunConfig,
        step_number: int,
        step_name: str,
        agent_role: str,
        agent_goal: str,
        task_description: str,
        expected_output: str,
        system_prompt: str | None,
    ) -> tuple[SubResult, str | None]:
        """
        Build a single-task Crew for this step and run it. Same retry budget
        as SOP. Token / duration / cost are accumulated across attempts, so a
        retried step is *more* expensive than a one-shot step — exactly as in
        SOP.
        """
        last_error: str | None = None
        recorder = _Recorder()

        # Accumulators across retry attempts — match SOP's metric model
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_duration_ms = 0
        total_cost_usd = 0.0
        actual_retries = 0

        for attempt in range(MAX_RETRIES + 1):
            actual_retries = attempt

            # Fresh adapter + crew per attempt — CrewAI mutates state on kickoff.
            llm = MaestroBackedLLM(
                provider=self.provider,
                config=config,
                system_prompt_override=system_prompt,
                recorder=recorder,
            )
            agent = Agent(
                role=agent_role,
                goal=agent_goal,
                backstory=(
                    "You execute one focused step in a multi-agent "
                    "diagram-generation pipeline."
                ),
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            task = Task(
                description=task_description,
                expected_output=expected_output,
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )

            # Snapshot call count before kickoff so we can measure the
            # per-attempt delta independently of earlier attempts/raises.
            start_calls = len(recorder.calls)

            # Kickoff. CrewAI's BaseLLM contract says return a string on error,
            # so kickoff itself shouldn't raise from a provider error — but
            # other CrewAI-internal failures could, so wrap it.
            try:
                crew.kickoff()
            except Exception as e:
                last_error = f"CrewAI kickoff raised on attempt {attempt + 1}: {e}"
                continue

            # Single-task crew + sequential process => exactly one LLM call
            # per kickoff. Assert the per-attempt delta (not a cumulative count)
            # so a raise on a previous attempt — with or without a provider call
            # — does not skew the check for this attempt.
            new_call_count = len(recorder.calls) - start_calls
            if new_call_count == 0:
                last_error = f"No LLM call recorded on attempt {attempt + 1}"
                continue
            if new_call_count != 1:
                last_error = (
                    f"Single-call invariant violated on attempt {attempt + 1}: "
                    f"expected 1 new call, got {new_call_count}"
                )
                continue
            call = recorder.calls[-1]

            total_prompt_tokens += call.prompt_tokens
            total_completion_tokens += call.completion_tokens
            total_duration_ms += call.duration_ms
            total_cost_usd += call.cost_usd

            if call.error is not None:
                last_error = call.error
                continue

            output = strip_fences(call.output_text)
            if step_number < 3:
                is_valid, validation_error = validate_step_payload(output, step_number)
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

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_prompt(
        self,
        step: dict,
        input_data: str,
        step_outputs: dict[str, str],
    ) -> str:
        """Like SOP._build_prompt — fill template vars from prior outputs."""
        template = step["task_prompt"]
        fmt = {"input_data": input_data}
        if "extract_entities" in step_outputs:
            fmt["entities_json"] = step_outputs["extract_entities"]
        if "extract_relationships" in step_outputs:
            fmt["relationships_json"] = step_outputs["extract_relationships"]
        return template.format(**fmt)

    def _aggregate(
        self,
        config: RunConfig,
        subs: list[SubResult],
        total_start: float,
        diagram_code: str | None = None,
        error: str | None = None,
    ) -> tuple[RunResult, list[SubResult]]:
        """Sum sub-call stats into a parent RunResult — same shape as SOP."""
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
