"""
MAESTRO — Experiment runner
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

    # Dry run — show matrix without executing
    python -m maestro.run --dry-run
"""

import argparse
import os
import sys
from itertools import product

from dotenv import load_dotenv

# Load .env file so API keys are available via os.environ
load_dotenv()

from maestro.experiment_config import (
    DB_PATH,
    DEFAULT_REPEATS,
    INPUTS,
    MODELS,
    STRATEGIES,
)
from maestro.schemas import RunConfig, Strategy, Tier
from maestro.db.client import get_connection, init_db
from maestro.db.queries import (
    insert_metric_result,
    insert_run_config,
    insert_run_result,
    insert_sub_result,
)
from maestro.analysis.metrics import evaluate_run
from maestro.providers.anthropic import AnthropicProvider
from maestro.providers.openai import OpenAIProvider
from maestro.strategies.single import SingleAgentStrategy
from maestro.strategies.sop import SOPStrategy


# ---------------------------------------------------------------------------
# Strategy factory — maps enum to class
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    Strategy.SINGLE_AGENT: SingleAgentStrategy,
    Strategy.SOP_BASED: SOPStrategy,
    # Enable once implemented:
    # Strategy.CREW_AI: CrewAIStrategy,
    # Strategy.LANG_GRAPH: LangGraphStrategy,
}


# ---------------------------------------------------------------------------
# Provider factory — maps model prefix to provider class
# Currently only Anthropic; extend when adding OpenAI etc.
# ---------------------------------------------------------------------------

def _create_provider(model_pricing):
    """
    Instantiate the correct LLM provider based on model name.
    API keys come from environment variables — never hardcoded.
    """
    model = model_pricing.model

    model_lower = model.lower()

    if "claude" in model_lower:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in environment")
            sys.exit(1)
        return AnthropicProvider(api_key=api_key, pricing=model_pricing)

    if "gpt" in model_lower:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in environment")
            sys.exit(1)
        return OpenAIProvider(api_key=api_key, pricing=model_pricing)

    print(f"ERROR: No provider registered for model '{model}'")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for filtering the experiment matrix."""
    parser = argparse.ArgumentParser(
        description="MAESTRO experiment runner — iterate the full experiment matrix"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in Strategy],
        help="Run only this strategy (default: all enabled)",
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
        help="Run only this model (default: all registered)",
    )
    parser.add_argument(
        "--example",
        type=str,
        help="Run only this example_id (default: all registered)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeated runs per cell (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment matrix without executing any runs",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Matrix builder — apply filters
# ---------------------------------------------------------------------------

def build_matrix(args: argparse.Namespace) -> list[dict]:
    """
    Build the experiment matrix as a list of dicts, each representing one run.
    Applies CLI filters to narrow the cross-product.
    """
    # Filter inputs
    inputs = INPUTS
    if args.tier:
        inputs = [i for i in inputs if i.tier.value == args.tier]
    if args.example:
        inputs = [i for i in inputs if i.example_id == args.example]

    # Filter strategies
    strategies = STRATEGIES
    if args.strategy:
        strategies = [s for s in strategies if s.value == args.strategy]

    # Filter models
    models = MODELS
    if args.model:
        models = [m for m in models if m.model == args.model]

    # Build cross-product
    # Order: run_number outermost, then input, strategy, model
    # This interleaves models so no single provider gets hammered back-to-back
    matrix = []
    for run_number in range(1, args.repeats + 1):
        for input_file, strategy, model_pricing in product(inputs, strategies, models):
            matrix.append({
                "input_file": input_file,
                "strategy": strategy,
                "model_pricing": model_pricing,
                "run_number": run_number,
            })

    return matrix


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """Run the full experiment matrix with CLI filters applied."""
    args = parse_args()
    matrix = build_matrix(args)

    if not matrix:
        print("No runs match the given filters. Check --strategy, --tier, --model, --example.")
        sys.exit(0)

    # Group summary for display
    n_inputs = len({m["input_file"].example_id for m in matrix})
    n_strategies = len({m["strategy"] for m in matrix})
    n_models = len({m["model_pricing"].model for m in matrix})
    total = len(matrix)

    print("=" * 60)
    print("MAESTRO — Experiment Runner")
    print("=" * 60)
    print(f"  Inputs:     {n_inputs}")
    print(f"  Strategies: {n_strategies}")
    print(f"  Models:     {n_models}")
    print(f"  Repeats:    {args.repeats}")
    print(f"  Total runs: {total}")
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

    # Initialize DB
    init_db(DB_PATH)

    # Track totals for summary
    successes = 0
    failures = 0
    total_cost = 0.0

    for i, cell in enumerate(matrix, 1):
        input_file = cell["input_file"]
        strategy_enum = cell["strategy"]
        model_pricing = cell["model_pricing"]
        run_number = cell["run_number"]

        # Progress header
        print(
            f"\n[{i}/{total}] "
            f"{input_file.example_id} | "
            f"{strategy_enum.value} | "
            f"{model_pricing.model} | "
            f"run {run_number}"
        )

        # Build RunConfig
        config = RunConfig(
            strategy=strategy_enum,
            model=model_pricing.model,
            example_id=input_file.example_id,
            tier=input_file.tier,
            run_number=run_number,
        )

        # Instantiate provider and strategy
        provider = _create_provider(model_pricing)
        strategy_cls = STRATEGY_MAP.get(strategy_enum)
        if strategy_cls is None:
            print(f"  SKIP — strategy {strategy_enum.value} not implemented")
            continue
        strategy = strategy_cls(provider=provider)

        # Execute
        result, sub_results = strategy.run(
            input_file=input_file,
            config=config,
        )

        # Persist to DB
        with get_connection(DB_PATH) as conn:
            insert_run_config(conn, config)
            insert_run_result(conn, result)
            for sub in sub_results:
                insert_sub_result(conn, sub)

            # Evaluate metrics if run succeeded
            if result.success:
                metric = evaluate_run(
                    run_id=config.run_id,
                    output_diagram_code=result.output_diagram_code,
                    ground_truth_path=input_file.ground_truth_path,
                )
                insert_metric_result(conn, metric)

        # Always track cost (partial responses may still consume tokens)
        total_cost += result.cost_usd

        # Log result
        if result.success:
            successes += 1
            print(
                f"  OK — {result.duration_ms}ms, "
                f"${result.cost_usd:.6f}, "
                f"{result.total_tokens} tokens"
            )
        else:
            failures += 1
            print(f"  FAIL — {result.error} (cost: ${result.cost_usd:.6f})")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Successes:  {successes}/{total}")
    print(f"  Failures:   {failures}/{total}")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Database:   {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
