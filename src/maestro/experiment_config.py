"""
MAESTRO — Experiment configuration
Central registry of inputs, model pricing, and available strategies.
Single source of truth for the experiment matrix.

To add a new input:   append to INPUTS
To add a new model:   append to MODELS
To enable a strategy: add to STRATEGIES (once implemented)
"""

from pathlib import Path

from maestro.schemas import InputFile, ModelPricing, Strategy, Tier


# ---------------------------------------------------------------------------
# Base path for all data files (relative to project root)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Input registry — each entry is one benchmark case + ground truth
# ---------------------------------------------------------------------------

INPUTS: list[InputFile] = [
    InputFile(
        example_id="bpmn_collaboration_01",
        tier=Tier.INTERMEDIATE,
        entity_count=17,
        file_path=DATA_DIR / "bpmn_collaboration_01.JSON",
        ground_truth_path=DATA_DIR / "bpmn_collaboration_01_ground_truth.MMD",
        description="BPMN collaboration diagram with pools, lanes, message flows",
    ),
    # --- Add new inputs below ---
    # InputFile(
    #     example_id="simple_flow_01",
    #     tier=Tier.SIMPLE,
    #     entity_count=6,
    #     file_path=DATA_DIR / "simple_flow_01.JSON",
    #     ground_truth_path=DATA_DIR / "simple_flow_01_ground_truth.MMD",
    #     description="Simple sequential flowchart, no subprocesses",
    # ),
]


# ---------------------------------------------------------------------------
# Model registry — pricing per model for cost calculation
# ---------------------------------------------------------------------------

MODELS: list[ModelPricing] = [
    ModelPricing(
        model="claude-haiku-4-5-20251001",
        input_price_per_1m=0.80,
        output_price_per_1m=4.00,
    ),
    ModelPricing(
        # Pinned to snapshot for reproducibility
        model="gpt-4o-mini-2024-07-18",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    # --- Add new models below ---
    # ModelPricing(
    #     model="claude-sonnet-4-20250514",
    #     input_price_per_1m=3.00,
    #     output_price_per_1m=15.00,
    # ),
]


# ---------------------------------------------------------------------------
# Strategy registry — only strategies with working implementations
# ---------------------------------------------------------------------------

STRATEGIES: list[Strategy] = [
    Strategy.SINGLE_AGENT,
    Strategy.SOP_BASED,
    # --- Enable once implemented ---
    # Strategy.CREW_AI,
    # Strategy.LANG_GRAPH,
]


# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------

# Number of repeated runs per (input, strategy, model) cell
DEFAULT_REPEATS = 5

# SQLite database path (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "maestro.db"
