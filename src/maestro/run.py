"""
MAESTRO: Experiment runner
Iterates over the full experiment matrix: inputs x strategies x models x repeats.
Supports CLI filters to run subsets (e.g. only SOP, only tier 2).

Usage:
    # Full matrix (all inputs, all strategies, all models, 5 repeats)
    python -m maestro.run

    # Filter by strategy
    python -m maestro.run --strategy single_agent

    # Filter by tier
    python -m maestro.run --tier 2

    # Filter by model
    python -m maestro.run --model claude-haiku-4-5-20251001

    # Filter by input
    python -m maestro.run --example bpmn_collaboration_01

    # Override repeat count
    python -m maestro.run --repeats 3

    # Combine filters
    python -m maestro.run --strategy sop_based --tier 2 --repeats 3

    # Dry run: show matrix without executing
    python -m maestro.run --dry-run

    # Resume: default behaviour is to skip cells already successfully
    # recorded in the DB; previously-failed cells get another attempt.
    python -m maestro.run

    # Force full re-run (ignore DB)
    python -m maestro.run --no-resume

    # Re-run only previously-failed cells
    python -m maestro.run --rerun-failed
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

from dotenv import load_dotenv

# Load .env file so API keys are available via os.environ
load_dotenv()

# Silence CrewAI's interactive tracing prompt and telemetry so batch runs
# stay non-interactive on fresh checkouts (the user's preference file is
# machine-local and won't exist in CI / on a fresh clone).
#
# CREWAI_TRACING_ENABLED=false only governs the *enabled* path; it does NOT
# stop the first-execution "view your traces? [y/N]" prompt, which blocks on
# stdin for a 20s timeout per crew. In a headless container that file-based
# "declined" preference is wiped every `--rm`, so the prompt fires on every
# cell: ~1500 crew_ai cells x 20s is hours of dead waiting, and it corrupts
# the measured duration_ms. CREWAI_TESTING=true is the only flag that short
# circuits the prompt before any stdin read (crewai .../tracing/utils.py:
# _is_test_environment guards both the auto-collect check and the prompt). It
# disables trace prompts/telemetry only; it does not alter agent execution.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TESTING", "true")

from maestro.analysis.metrics import evaluate_run
from maestro.db.client import get_connection, init_db
from maestro.db.environment import capture_environment
from maestro.db.queries import (
    fetch_completed_cells,
    fetch_failed_cells,
    insert_metric_result,
    insert_run_config,
    insert_run_environment,
    insert_run_result,
    insert_sub_result,
)
from maestro.experiment_config import (
    CONTROL_MODEL,
    CONTROL_STRATEGIES,
    DB_PATH,
    DEFAULT_REPEATS,
    INPUTS,
    MODELS,
    STRATEGIES,
)
from maestro.providers.anthropic import AnthropicProvider
from maestro.providers.deepseek import DeepSeekProvider
from maestro.providers.gemini import GeminiProvider
from maestro.providers.mistral import MistralProvider
from maestro.providers.openai import OpenAIProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult, Strategy
from maestro.strategies.controls import (
    CopyInputControlStrategy,
    GroundTruthEchoControlStrategy,
    NullControlStrategy,
)
from maestro.strategies.crew import CrewAIStrategy
from maestro.strategies.langgraph import LangGraphStrategy
from maestro.strategies.single import SingleAgentStrategy
from maestro.strategies.sop import SOPStrategy

# ---------------------------------------------------------------------------
# Strategy factory: maps enum to class
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    Strategy.SINGLE_AGENT: SingleAgentStrategy,
    Strategy.SOP_BASED: SOPStrategy,
    Strategy.CREW_AI: CrewAIStrategy,
    Strategy.LANG_GRAPH: LangGraphStrategy,
    # Controls: see strategies/controls.py for rationale
    Strategy.NULL_CONTROL: NullControlStrategy,
    Strategy.COPY_CONTROL: CopyInputControlStrategy,
    Strategy.GROUND_TRUTH_CONTROL: GroundTruthEchoControlStrategy,
}


# ---------------------------------------------------------------------------
# Provider factory + pre-flight env check
# ---------------------------------------------------------------------------
#
# One source of truth for "which substring in a model name maps to which
# provider class + which env var". _create_provider and the pre-flight
# env check both consume this table so they can't drift out of sync.
#
# Order matters: the dispatch picks the first matching substring, so the
# control "model" (literal string "control") is handled by the caller
# (it's never created by _create_provider) and is intentionally absent here.

_PROVIDER_DISPATCH = (
    # (model-name substring, provider class, env var name)
    #
    # Order matters: dispatch picks the first matching substring. The needles
    # are mutually exclusive across the current model names, so the order is
    # not load-bearing today, but keep new needles specific enough not to
    # collide (e.g. "deepseek" must not be a prefix/suffix of another vendor's
    # model id).
    ("claude", AnthropicProvider, "ANTHROPIC_API_KEY"),
    ("gpt", OpenAIProvider, "OPENAI_API_KEY"),
    ("mistral", MistralProvider, "MISTRAL_API_KEY"),
    ("gemini", GeminiProvider, "GEMINI_API_KEY"),
    # DeepSeek uses an OpenAI-compatible endpoint (see providers/deepseek.py)
    # but a distinct API key + base URL, so it needs its own dispatch entry.
    ("deepseek", DeepSeekProvider, "DEEPSEEK_API_KEY"),
)


def _dispatch_for_model(model: str):
    """Return the dispatch tuple for a model name, or None if unknown."""
    model_lower = model.lower()
    for needle, cls, env_var in _PROVIDER_DISPATCH:
        if needle in model_lower:
            return (needle, cls, env_var)
    return None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
#
# Cells are independent network-bound work, so the matrix runs on a thread
# pool. Two invariants shape the design and must not be broken:
#
#   1. SQLite is single-writer. db/ is the only writer, and a lost write
#      violates "a failure is recorded, never silent." So workers do only the
#      LLM call + scoring; every insert_* runs on the main thread, which drains
#      finished cells one at a time. No worker ever touches the DB.
#
#   2. A request's measured duration_ms must stay a true latency, not a
#      server-side queue wait. Oversubscribing one provider would inflate it.
#      So concurrency is capped PER PROVIDER, not just globally: a semaphore per
#      provider needle (claude / gpt / mistral / gemini / deepseek) bounds how
#      many of its calls are in flight. The cap also keeps us clear of 429s.
#
# Controls make no LLM call (their model dispatches to None), so they acquire
# no semaphore and never contend.
#
# Default in-flight requests per provider. 4 is a conservative ceiling that
# stays under typical paid rate limits while still parallelising the network
# wait. Overridable with --provider-concurrency: a free-tier key sets 1, a
# high-limit account can raise it.
DEFAULT_PROVIDER_CONCURRENCY = 4


def _build_provider_semaphores(concurrency: int) -> dict[str, threading.Semaphore]:
    """
    One Semaphore per provider needle, each permitting ``concurrency`` calls.

    Keyed by the needle ``_dispatch_for_model`` returns ("claude", "gpt", ...),
    the same stable id the provider factory uses. Built once up-front from the
    CLI value and passed explicitly into the worker, so the concurrency limit
    is a run parameter rather than hidden global state.
    """
    return {
        needle: threading.Semaphore(concurrency)
        for needle, _cls, _env in _PROVIDER_DISPATCH
    }


def _create_provider(model_pricing):
    """
    Instantiate the correct LLM provider based on model name.
    API keys come from environment variables, never hardcoded.

    Raises ``RuntimeError`` on either failure mode (unknown model name,
    missing env var). Both are caught by the cell-level try/except in
    the main loop, so a transient env-var rotation or a typo in
    ``MODELS`` fails that one cell rather than aborting the whole
    experiment after potentially hours of work. ``preflight_check_env``
    should normally have caught both up-front; these raises are the
    defensive fallback.
    """
    dispatch = _dispatch_for_model(model_pricing.model)
    if dispatch is None:
        raise RuntimeError(f"No provider registered for model '{model_pricing.model}'")
    _, cls, env_var = dispatch
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(
            f"{env_var} not set in environment "
            f"(needed by model '{model_pricing.model}')"
        )
    return cls(api_key=api_key, pricing=model_pricing)


def preflight_check_env(models: list[ModelPricing]) -> None:
    """
    Verify every API key required by ``models`` exists in the environment.

    Aggregates *all* missing keys into one consolidated error message so
    a user with several missing keys learns about all of them at once,
    rather than the old per-cell pattern where the first missing key
    aborted before the others were even checked. Exits 1 if anything
    is missing; otherwise returns silently.

    Skips models whose name doesn't dispatch to any provider (e.g. the
    synthetic ``control`` ModelPricing used by control strategies):
    those don't make LLM calls and don't need keys.
    """
    # env_var -> set of distinct model names needing it. ``models`` is
    # the post-filter matrix, which has one entry per *cell*; the same
    # model name appears many times for a typical matrix. A set
    # de-duplicates so the error message isn't padded with repetition.
    missing: dict[str, set[str]] = {}
    for mp in models:
        dispatch = _dispatch_for_model(mp.model)
        if dispatch is None:
            # Unknown model or synthetic "control": skip silently. The
            # main loop handles unknown-model failures per-cell.
            continue
        _, _, env_var = dispatch
        if not os.environ.get(env_var):
            missing.setdefault(env_var, set()).add(mp.model)

    if not missing:
        return

    print("ERROR: Missing API keys for the following providers:", file=sys.stderr)
    for env_var, model_names in sorted(missing.items()):
        models_str = ", ".join(sorted(model_names))
        print(f"  - {env_var} (needed by: {models_str})", file=sys.stderr)
    print(
        "\nSet the missing keys in your environment (or .env file) and re-run.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for filtering the experiment matrix."""
    parser = argparse.ArgumentParser(
        description="MAESTRO experiment runner: iterate the full experiment matrix"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        help=(
            "Run only these strategies (default: all enabled). "
            "Comma-separated for several, e.g. --strategy single_agent,lang_graph"
        ),
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        help="Run only inputs of this tier (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "Run only these models (default: all registered). "
            "Comma-separated for several, e.g. "
            "--model gpt-5.4-mini-2026-03-17,deepseek-v4-flash"
        ),
    )
    parser.add_argument(
        "--example",
        type=str,
        help=(
            "Run only these example_ids (default: all registered). "
            "Comma-separated for several, e.g. --example bpmn_1_03,it_1_07"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeated runs per cell (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--provider-concurrency",
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
        help=(
            "Max concurrent requests per provider "
            f"(default: {DEFAULT_PROVIDER_CONCURRENCY}). Set 1 for free-tier "
            "rate limits; raise it if your account allows. Does not change "
            "results, only how fast the matrix runs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment matrix without executing any runs",
    )

    # Resume semantics: mutually exclusive.
    # Default (neither flag): skip cells whose RunResult is already
    # successful in the DB; re-run cells that previously failed
    # (transient errors usually deserve another attempt).
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore the DB and run the full matrix from scratch. "
            "Useful when you've nuked the DB or want every cell to "
            "execute regardless of prior runs."
        ),
    )
    resume_group.add_argument(
        "--rerun-failed",
        action="store_true",
        help=(
            "Only execute cells that previously failed (error IS NOT NULL "
            "or empty diagram). Skips both successful prior runs and any "
            "cells that have no row in the DB yet."
        ),
    )

    args = parser.parse_args()

    # A concurrency below 1 would deadlock the pool (no permits ever granted).
    if args.provider_concurrency < 1:
        parser.error("--provider-concurrency must be >= 1")

    return args


# ---------------------------------------------------------------------------
# Matrix builder: apply filters
# ---------------------------------------------------------------------------


def _split_csv(value: str | None) -> list[str] | None:
    """
    Parse a comma-separated filter value into a clean list, or None if the
    flag was absent. Empty/whitespace-only entries are dropped so trailing
    commas and stray spaces don't create phantom filter values.
    """
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_matrix(args: argparse.Namespace) -> list[dict]:
    """
    Build the experiment matrix as a list of dicts, each representing one run.
    Applies CLI filters to narrow the cross-product. The --strategy, --model
    and --example flags accept a comma-separated list (membership filter).
    """
    examples = _split_csv(args.example)
    model_names = _split_csv(args.model)
    strategy_names = _split_csv(args.strategy)

    # Validate filter values up front (argparse no longer does, now that the
    # flags accept lists). Catches a typo before any matrix work or API spend,
    # since a misspelled value in a list would otherwise silently shrink the matrix.
    def _reject_unknown(flag: str, given: list[str], valid: set[str]) -> None:
        unknown = [v for v in given if v not in valid]
        if unknown:
            print(
                f"ERROR: unknown {flag} value(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(valid))}",
                file=sys.stderr,
            )
            sys.exit(2)

    if strategy_names:
        _reject_unknown("--strategy", strategy_names, {s.value for s in Strategy})
    if examples:
        _reject_unknown("--example", examples, {i.example_id for i in INPUTS})
    # --model is validated below, after the strategy filter is known: an unknown
    # model only matters when a real (LLM) strategy is actually selected, so the
    # control-only no-op (--strategy null_control --model anything) is preserved.

    # Filter inputs
    inputs = INPUTS
    if args.tier:
        inputs = [i for i in inputs if i.tier.value == args.tier]
    if examples:
        inputs = [i for i in inputs if i.example_id in examples]

    # Filter strategies
    strategies = STRATEGIES
    if strategy_names:
        strategies = [s for s in strategies if s.value in strategy_names]

    # Filter models, applies only to real (LLM) strategies. Control rows
    # ignore --model because they don't use a model; a --model filter
    # should narrow which LLM rows run but should not silently drop the
    # sanity floor/ceiling rows the experiment needs.
    models = MODELS
    if model_names:
        models = [m for m in models if m.model in model_names]

    # Partition by strategy kind. Controls are deterministic in (model,
    # repeat), so collapsing both dimensions to a single row per
    # (input, control) avoids 40x duplicate rows per input that would
    # need to be filtered out at analysis time anyway.
    real_strategies = [s for s in strategies if s not in CONTROL_STRATEGIES]
    control_strategies = [s for s in strategies if s in CONTROL_STRATEGIES]

    # Fail fast on any unknown --model value, but only when a real (LLM)
    # strategy is selected. `--strategy null_control --model typo` stays a
    # no-op on --model (controls don't use any model), so it must not abort.
    # When a real strategy IS selected, a misspelled model would otherwise
    # silently shrink the matrix (e.g. `--model gpt-5.4-mini-2026-03-17,typo`
    # would quietly run only the valid one), so reject the typo loudly.
    if model_names and real_strategies:
        registered = {m.model for m in MODELS}
        unknown = [m for m in model_names if m not in registered]
        if unknown:
            print(
                f"ERROR: unknown --model value(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(registered))}",
                file=sys.stderr,
            )
            sys.exit(2)

    matrix = []

    # Real strategies: full inputs × strategies × models × repeats fan-out.
    # Order: run_number outermost so models are interleaved (no single
    # provider gets hammered back-to-back).
    for run_number in range(1, args.repeats + 1):
        for input_file, strategy, model_pricing in product(
            inputs, real_strategies, models
        ):
            matrix.append(
                {
                    "input_file": input_file,
                    "strategy": strategy,
                    "model_pricing": model_pricing,
                    "run_number": run_number,
                }
            )

    # Controls: one row per (input, control_strategy). Synthetic CONTROL_MODEL,
    # run_number=1. Not affected by --model or --repeats.
    for input_file, strategy in product(inputs, control_strategies):
        matrix.append(
            {
                "input_file": input_file,
                "strategy": strategy,
                "model_pricing": CONTROL_MODEL,
                "run_number": 1,
            }
        )

    return matrix


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def _apply_resume_filter(matrix: list[dict], args: argparse.Namespace) -> list[dict]:
    """
    Narrow ``matrix`` according to the user's resume flags.

    - default       : drop cells already successfully recorded in the DB.
                      Cells with prior failures stay in (retry transient).
    - --no-resume   : pass through unchanged.
    - --rerun-failed: keep only cells that have a prior *failed* row.

    Reads the DB once (lazy import of get_connection so unit tests that
    poke at the helpers don't need a real DB). Idempotent if the DB
    doesn't exist yet: init_db is called by ``main`` before this.
    """
    if args.no_resume:
        return matrix

    # Only fetch the set the active mode actually needs: on a big DB
    # the unused query would scan thousands of rows for nothing.
    with get_connection(DB_PATH) as conn:
        if args.rerun_failed:
            failed = fetch_failed_cells(conn)
            completed = set()  # unused in this branch
        else:
            completed = fetch_completed_cells(conn)
            failed = set()  # unused in this branch

    def cell_key(cell: dict) -> tuple[str, str, str, int]:
        return (
            cell["input_file"].example_id,
            cell["strategy"].value,
            cell["model_pricing"].model,
            cell["run_number"],
        )

    if args.rerun_failed:
        return [c for c in matrix if cell_key(c) in failed]
    return [c for c in matrix if cell_key(c) not in completed]


def _execute_cell(
    cell: dict,
    environment_id: str,
    semaphores: dict[str, threading.Semaphore],
) -> tuple[RunConfig, RunResult, list, object | None]:
    """
    Run one matrix cell off the main thread: build the provider, run the
    strategy, score the output. Returns everything the main thread needs to
    persist; this function never touches the DB, so SQLite keeps a single
    writer (see the Concurrency section above).

    The per-provider semaphore is held only around the strategy's network work,
    so at most ``--provider-concurrency`` of one provider's calls are in flight
    and a cell's measured ``duration_ms`` is a true latency, not queue time.

    Returns ``(config, result, sub_results, metric_or_None)``. ``metric`` is
    None for failed cells and for a metric pipeline that crashes on a cell's
    output; neither is fatal, both still record their RunResult row. Mirrors the
    old in-loop error isolation: any strategy/provider exception becomes a failed
    RunResult, never a crash, so one bad cell can't take down the pool.
    """
    input_file = cell["input_file"]
    strategy_enum = cell["strategy"]
    model_pricing = cell["model_pricing"]
    run_number = cell["run_number"]

    config = RunConfig(
        strategy=strategy_enum,
        model=model_pricing.model,
        example_id=input_file.example_id,
        tier=input_file.tier,
        run_number=run_number,
        environment_id=environment_id,
    )

    # KeyboardInterrupt / SystemExit deliberately propagate (not caught by the
    # bare Exception below), so Ctrl+C still tears the pool down.
    try:
        if strategy_enum in CONTROL_STRATEGIES:
            # Controls bypass the LLM, so no provider and no semaphore.
            strategy = STRATEGY_MAP[strategy_enum]()
            result, sub_results = strategy.run(input_file=input_file, config=config)
        else:
            provider = _create_provider(model_pricing)
            strategy = STRATEGY_MAP[strategy_enum](provider=provider)
            sem = semaphores.get(_dispatch_for_model(model_pricing.model)[0])
            with sem:
                result, sub_results = strategy.run(input_file=input_file, config=config)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return (
            config,
            RunResult(
                run_id=config.run_id,
                output_diagram_code=None,
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=0,
                cost_usd=0.0,
                error=f"unhandled: {type(exc).__name__}: {exc}",
            ),
            [],
            None,
        )

    # Score successful cells here (mmdc subprocess + comparison is CPU/IO work
    # with no DB dependency), so the main thread only inserts the result.
    metric = None
    if result.success:
        try:
            metric = evaluate_run(
                run_id=config.run_id,
                output_diagram_code=result.output_diagram_code,
                ground_truth_path=input_file.ground_truth_path,
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(
                f"  WARN - metric evaluation crashed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return config, result, sub_results, metric


def main():
    """Run the full experiment matrix with CLI filters applied."""
    args = parse_args()
    matrix = build_matrix(args)

    if not matrix:
        print(
            "No runs match the given filters. "
            "Check --strategy, --tier, --model, --example."
        )
        sys.exit(0)

    # Initialize DB before applying the resume filter: the filter reads
    # existing rows, and init_db is idempotent + cheap (CREATE IF NOT EXISTS).
    init_db(DB_PATH)

    # Resume filtering: skip already-successful cells (default), force
    # full re-run (--no-resume), or narrow to previously-failed cells
    # (--rerun-failed). Applied BEFORE the summary so the printed totals
    # match what will actually execute.
    matrix_before_resume = len(matrix)
    matrix = _apply_resume_filter(matrix, args)
    skipped = matrix_before_resume - len(matrix)

    if not matrix:
        if args.rerun_failed:
            print("No previously-failed cells to re-run - nothing to do.")
        else:
            print("All matrix cells already complete in the DB - nothing to do.")
        sys.exit(0)

    # Group summary for display
    n_inputs = len({m["input_file"].example_id for m in matrix})
    n_strategies = len({m["strategy"] for m in matrix})
    n_models = len({m["model_pricing"].model for m in matrix})
    total = len(matrix)

    print("=" * 60)
    print("MAESTRO - Experiment Runner")
    print("=" * 60)
    print(f"  Inputs:     {n_inputs}")
    print(f"  Strategies: {n_strategies}")
    print(f"  Models:     {n_models}")
    print(f"  Repeats:    {args.repeats}")
    print(f"  Total runs: {total}")
    if skipped:
        # In --rerun-failed mode the skipped set includes both
        # successful prior runs AND cells with no DB row yet, neither of which
        # is "already complete", so don't claim that.
        if args.rerun_failed:
            print(f"  Skipped:    {skipped} (no prior failure, rerun-failed mode)")
        else:
            print(f"  Skipped:    {skipped} (already complete, resume mode)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Matrix preview:\n")
        for i, cell in enumerate(matrix, 1):
            print(
                f"  {i:3d}. {cell['input_file'].example_id:30s} | "
                f"{cell['strategy'].value:15s} | "
                f"{cell['model_pricing'].model:30s} | "
                f"run {cell['run_number']}"
            )
        print(f"\n[DRY RUN] {total} runs would be executed.")
        return

    # Pre-flight: verify env vars for every provider in the post-filter
    # matrix. Fails fast with a consolidated error message before any
    # work starts, way better than discovering MISTRAL_API_KEY is
    # missing 80% of the way through a multi-hour run.
    matrix_models = [c["model_pricing"] for c in matrix]
    preflight_check_env(matrix_models)

    # Capture the runtime environment once per invocation. Every RunConfig
    # written below carries its environment_id so a future replication
    # attempt can diagnose diverging numbers against the exact stack that
    # produced the original data.
    environment = capture_environment()
    if environment.git_dirty:
        print(
            "⚠  Git working tree is dirty: uncommitted changes will not be "
            "reproducible from this commit hash."
        )
    with get_connection(DB_PATH) as conn:
        insert_run_environment(conn, environment)

    # Skip-not-an-error: an enum value with no class registered is a config
    # gap, not a failure. Filter these before the pool so a skipped strategy
    # never burns a worker or a row, and the printed totals match what runs.
    runnable = []
    for cell in matrix:
        if STRATEGY_MAP.get(cell["strategy"]) is None:
            print(f"  SKIP - strategy {cell['strategy'].value} not implemented")
        else:
            runnable.append(cell)

    # Cells are independent network-bound work, so they run on a thread pool
    # capped per provider (see the Concurrency section). Each worker does the
    # LLM call + scoring; the main thread is the ONLY DB writer, draining
    # finished cells as they complete. Pool size = providers x per-provider cap
    # is the natural ceiling on in-flight work; the semaphores enforce the real
    # per-provider limit underneath.
    semaphores = _build_provider_semaphores(args.provider_concurrency)
    max_workers = max(1, len(_PROVIDER_DISPATCH) * args.provider_concurrency)
    print(
        f"\nRunning {len(runnable)} cells, up to "
        f"{args.provider_concurrency} concurrent per provider.\n"
    )

    # Track totals for summary
    successes = 0
    failures = 0
    total_cost = 0.0
    done = 0
    total_runnable = len(runnable)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _execute_cell, cell, environment.environment_id, semaphores
            ): cell
            for cell in runnable
        }

        # as_completed yields cells in finish order, not submit order. Every
        # insert_* below runs here on the main thread: one writer, no SQLite
        # lock contention, the "db/ is the only writer" invariant preserved.
        for future in as_completed(futures):
            done += 1
            # _execute_cell catches strategy/provider errors itself, so a
            # future result is the normal path. A truly unexpected raise (a
            # bug in the worker harness) still shouldn't kill the run, so it
            # is caught and logged as a lost cell.
            try:
                config, result, sub_results, metric = future.result()
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                print(
                    f"  WARN - worker crashed: {type(exc).__name__}: {exc}; cell lost",
                    file=sys.stderr,
                )
                continue

            print(
                f"\n[{done}/{total_runnable}] "
                f"{config.example_id} | {config.strategy.value} | "
                f"{config.model} | run {config.run_number}"
            )

            # Persist. Wrapped because a DB failure here would otherwise lose
            # an already-paid-for LLM result; we at least log it for recovery.
            try:
                with get_connection(DB_PATH) as conn:
                    insert_run_config(conn, config)
                    insert_run_result(conn, result)
                    for sub in sub_results:
                        insert_sub_result(conn, sub)
                    if metric is not None:
                        insert_metric_result(conn, metric)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                print(
                    f"  WARN - DB persist crashed: {type(exc).__name__}: "
                    f"{exc}; result lost",
                    file=sys.stderr,
                )

            # Always track cost (partial responses may still consume tokens)
            total_cost += result.cost_usd

            if result.success:
                successes += 1
                print(
                    f"  OK - {result.duration_ms}ms, "
                    f"${result.cost_usd:.6f}, "
                    f"{result.total_tokens} tokens"
                )
            else:
                failures += 1
                print(f"  FAIL - {result.error} (cost: ${result.cost_usd:.6f})")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Successes:  {successes}/{total_runnable}")
    print(f"  Failures:   {failures}/{total_runnable}")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Database:   {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
