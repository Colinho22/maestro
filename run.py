"""
MAESTRO — Execution script
Hardcoded config for POC. Run with: python -m maestro
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from maestro.schemas import InputFile, ModelPricing, RunConfig, Strategy, Tier
from maestro.providers.anthropic import AnthropicProvider
from maestro.strategies.single import SingleAgentStrategy
from maestro.strategies.sop import SOPStrategy
from maestro.analysis.metrics import evaluate_run
from maestro.db.client import init_db, get_connection
from maestro.db.queries import insert_run_config, insert_run_result, insert_sub_result, insert_metric_result

# ---------------------------------------------------------------------------
# Config — change these to run different experiment conditions
# ---------------------------------------------------------------------------

MODEL        = "claude-haiku-4-5-20251001"
STRATEGY     = Strategy.SOP_BASED
TIER         = Tier.INTERMEDIATE
EXAMPLE_ID   = "bpmn_collaboration_01"
RUN_NUMBER   = 1

# Paths — relative to project root
DATA_DIR     = Path("data")
DB_PATH      = Path("maestro.db")

# Anthropic pricing for claude-haiku-4-5 (USD per 1M tokens)
# Update when Anthropic changes pricing
PRICING = ModelPricing(
    model               = MODEL,
    input_price_per_1m  = 0.80,
    output_price_per_1m = 4.00,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load ANTHROPIC_API_KEY from .env
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    # Initialise DB — creates tables if they don't exist yet
    init_db(DB_PATH)

    # Build the input file descriptor
    input_file = InputFile(
        example_id        = EXAMPLE_ID,
        tier              = TIER,
        entity_count      = 17,
        file_path         = DATA_DIR / f"{EXAMPLE_ID}.json",
        ground_truth_path = DATA_DIR / f"{EXAMPLE_ID}_ground_truth.mmd",
    )

    # Build the run config — generates a UUID automatically
    config = RunConfig(
        strategy   = STRATEGY,
        model      = MODEL,
        example_id = EXAMPLE_ID,
        tier       = TIER,
        run_number = RUN_NUMBER,
    )

    print(f"Starting run {config.run_id}")
    print(f"  Strategy : {config.strategy.value}")
    print(f"  Model    : {config.model}")
    print(f"  Example  : {config.example_id} (Tier {config.tier.value})")

    # Initialise provider
    provider = AnthropicProvider(api_key=api_key, pricing=PRICING)

    # Select strategy based on config
    if STRATEGY == Strategy.SINGLE_AGENT:
        strategy = SingleAgentStrategy(provider=provider)
    elif STRATEGY == Strategy.SOP_BASED:
        strategy = SOPStrategy(provider=provider)
    else:
        raise ValueError(f"Strategy not implemented: {STRATEGY.value}")

    # Execute
    result, sub_result = strategy.run(input_file=input_file, config=config)

    # Persist to DB
    with get_connection(DB_PATH) as conn:
        insert_run_config(conn, config)
        insert_run_result(conn, result)
        for sub in sub_result:
            insert_sub_result(conn, sub)

    # Print summary
    if result.success:
        print(f"\nSuccess")
        print(f"  Tokens   : {result.prompt_tokens} in / {result.completion_tokens} out")
        print(f"  Duration : {result.duration_ms}ms")
        print(f"  Cost     : ${result.cost_usd:.6f}")
        print(f"\n--- Output ---\n{result.output_diagram_code}")
    else:
        print(f"\nFailed: {result.error}")

    # Print sub-call details if any
    if sub_result:
        print(f"\n--- Sub-calls ---")
        for sub in sub_result:
            status = "OK" if sub.error is None else f"FAIL: {sub.error}"
            print(f"  Step {sub.step_number} ({sub.step_name}): {status}")
            print(f"    Tokens: {sub.prompt_tokens} in / {sub.completion_tokens} out")
            print(f"    Cost: ${sub.cost_usd:.6f} | Retries: {sub.retry_count}")

    # Evaluate against ground truth (only if run succeeded)
    if result.success:
        metrics = evaluate_run(
            run_id=config.run_id,
            output_diagram_code=result.output_diagram_code,
            ground_truth_path=input_file.ground_truth_path,
        )
        with get_connection(DB_PATH) as conn:
            insert_metric_result(conn, metrics)

        print(f"\n--- Metrics ---")
        print(f"  Parse valid     : {metrics.parses_valid}")
        print(f"  Entity ID       : P={metrics.entity_id_precision} R={metrics.entity_id_recall} F1={metrics.entity_id_f1}")
        print(f"  Entity name     : P={metrics.entity_name_precision} R={metrics.entity_name_recall} F1={metrics.entity_name_f1}")
        print(f"  Entity lemma    : P={metrics.entity_lemma_precision} R={metrics.entity_lemma_recall} F1={metrics.entity_lemma_f1}")
        print(f"  Rel. relaxed    : P={metrics.relationship_relaxed_precision} R={metrics.relationship_relaxed_recall} F1={metrics.relationship_relaxed_f1}")
        print(f"  Rel. strict     : P={metrics.relationship_strict_precision} R={metrics.relationship_strict_recall} F1={metrics.relationship_strict_f1}")
        print(f"\n--- Taxonomy ---")
        print(f"  Entities  : {metrics.missing_entities} missing | {metrics.extra_entities} extra | {metrics.false_entities} false | {metrics.duplicate_entities} duplicate")
        print(f"  Relations : {metrics.missing_relationships} missing | {metrics.extra_relationships} extra | {metrics.false_relationships} false | {metrics.duplicate_relationships} duplicate")

if __name__ == "__main__":
    main()