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
        tier=Tier.COMPLEX,
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

# Synthetic "model" used only for control-strategy rows. Controls bypass the
# LLM entirely so no real model is involved; this entry exists so the
# ``RunConfig.model`` column has an honest value ("control") rather than
# borrowing the name of a real model and lying about what produced the row.
# Zero pricing means control rows never affect cost rollups.
CONTROL_MODEL = ModelPricing(
    model="control",
    input_price_per_1m=0.0,
    output_price_per_1m=0.0,
)

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
    ModelPricing(
        model="mistral-small-2603",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    ModelPricing(
        model="gemini-2.5-flash-lite",
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
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
    Strategy.CREW_AI,
    Strategy.LANG_GRAPH,
    # Control conditions (no LLM, deterministic). Included in the default
    # matrix so every full run produces a metric-pipeline sanity record;
    # ``build_matrix`` in run.py collapses their model/repeat fan-out to
    # one row per (input, control) cell since neither dimension varies.
    Strategy.NULL_CONTROL,
    Strategy.COPY_CONTROL,
    Strategy.GROUND_TRUTH_CONTROL,
]


# Set used by ``build_matrix`` and analysis code to special-case controls:
# - matrix builder uses CONTROL_MODEL and run_number=1 for these strategies
# - analysis can exclude them from ANOVA / cost rollups with
#   ``WHERE strategy NOT IN (SELECT value FROM control_strategies)`` or the
#   in-Python equivalent ``s not in CONTROL_STRATEGIES``.
CONTROL_STRATEGIES: set[Strategy] = {
    Strategy.NULL_CONTROL,
    Strategy.COPY_CONTROL,
    Strategy.GROUND_TRUTH_CONTROL,
}


# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------

# Number of repeated runs per (input, strategy, model) cell
DEFAULT_REPEATS = 5

# SQLite database path (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "maestro.db"
