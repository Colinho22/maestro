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
from maestro.db.client import init_db, get_connection
from maestro.db.queries import insert_run_config, insert_run_result

# ---------------------------------------------------------------------------
# Config — change these to run different experiment conditions
# ---------------------------------------------------------------------------

MODEL        = "claude-haiku-4-5-20251001"
STRATEGY     = Strategy.SINGLE_AGENT
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

    # Initialise provider and strategy
    provider = AnthropicProvider(api_key=api_key, pricing=PRICING)
    strategy = SingleAgentStrategy(provider=provider)

    # Execute
    result = strategy.run(input_file=input_file, config=config)

    # Persist to DB
    with get_connection(DB_PATH) as conn:
        insert_run_config(conn, config)
        insert_run_result(conn, result)

    # Print summary
    if result.success:
        print(f"\nSuccess")
        print(f"  Tokens   : {result.prompt_tokens} in / {result.completion_tokens} out")
        print(f"  Duration : {result.duration_ms}ms")
        print(f"  Cost     : ${result.cost_usd:.6f}")
        print(f"\n--- Output ---\n{result.output_diagram_code}")
    else:
        print(f"\nFailed: {result.error}")


if __name__ == "__main__":
    main()