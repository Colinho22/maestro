# ---------------------------------------------------------------------------
# Core data schema
# All models use Pydantic v2 for validation and serialization
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Enums — constrain experiment dimensions to valid values
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    SINGLE_AGENT = "single_agent"
    SOP_BASED    = "sop_based"
    CREW_AI      = "crew_ai"
    LANG_GRAPH   = "lang_graph"


class Tier(int, Enum):
    # Complexity tiers based on entity count
    SIMPLE       = 1   # < 10 entities
    INTERMEDIATE = 2   # 10-25 entities
    COMPLEX      = 3   # 25+ entities


# ---------------------------------------------------------------------------
# InputFile — describes one diagram generation task
# ---------------------------------------------------------------------------

class InputFile(BaseModel):
    """
    Represents a single benchmark input: a JSON file with relational data
    and its associated ground truth diagram code.
    """

    example_id:          str        # Human-readable ID, e.g. "er_diagram_01"
    tier:                Tier       # Complexity tier (1–3)
    entity_count:        int        # Number of entities in the input
    file_path:           Path       # Path to the JSON input file on disk
    ground_truth_path:   Path       # Path to the reference diagram code file
    description:         Optional[str] = None  # Optional human note about this input


# ---------------------------------------------------------------------------
# RunConfig — captures the full experimental context of one run
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """
    Groups all the dimensions of a single experiment run.
    run_id is the unique key; all other fields allow grouping/filtering.
    """

    run_id:      UUID     = Field(default_factory=uuid4)
    strategy:    Strategy
    model:       str                  # e.g. "gpt-4o", "claude-3-5-sonnet"
    example_id:  str                  # FK to InputFile.example_id
    tier:        Tier
    run_number:  int                  # Repeat index within same config (1–N)
    timestamp:   datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ModelPricing — lookup table for cost calculation
# ---------------------------------------------------------------------------

class ModelPricing(BaseModel):
    """
    Per-model token pricing in USD per 1M tokens.
    Used to compute cost_usd at write time.
    """

    model:                  str
    input_price_per_1m:     float   # USD per 1M prompt tokens
    output_price_per_1m:    float   # USD per 1M completion tokens


# ---------------------------------------------------------------------------
# RunResult — output and statistics of one LLM call
# ---------------------------------------------------------------------------

class RunResult(BaseModel):
    """
    Stores everything produced by a single LLM generation run.
    Links back to RunConfig via run_id.
    """

    run_id:               UUID   # FK to RunConfig.run_id

    # Output
    output_diagram_code:  Optional[str] = None  # Generated Mermaid / PlantUML / etc.

    # Token usage
    prompt_tokens:        int
    completion_tokens:    int

    # Performance
    duration_ms:          int    # Wall-clock time for the LLM call

    # Cost — computed from token counts + ModelPricing at write time
    cost_usd:             float

    # Error — None if successful, exception message otherwise
    error:                Optional[str] = None

    @computed_field
    @property
    def total_tokens(self) -> int:
        # Convenience field for quick cost/efficiency analysis
        return self.prompt_tokens + self.completion_tokens

    @computed_field
    @property
    def success(self) -> bool:
        # True if a diagram was produced without error
        return self.error is None and self.output_diagram_code is not None


class SubResult(BaseModel):
    """
    One sub-call within a multi-step strategy (e.g. SOP).
    Links to the parent run via run_id.
    """

    sub_id:            UUID = Field(default_factory=uuid4)
    run_id:            UUID          # FK to RunConfig.run_id
    step_number:       int           # 1, 2, 3…
    step_name:         str           # "extract_entities", "extract_relationships", etc.
    output_text:       Optional[str] = None
    prompt_tokens:     int
    completion_tokens: int
    duration_ms:       int
    cost_usd:          float
    error:             Optional[str] = None
    retry_count:       int = 0       # 0 = first attempt worked

# ---------------------------------------------------------------------------
# Helper — compute cost from token counts and pricing
# ---------------------------------------------------------------------------

def compute_cost(
    prompt_tokens:     int,
    completion_tokens: int,
    pricing:           ModelPricing,
) -> float:
    """
    Calculate USD cost for one LLM call.
    Prices are per 1M tokens — divide by 1_000_000.
    """
    input_cost  = (prompt_tokens     / 1_000_000) * pricing.input_price_per_1m
    output_cost = (completion_tokens / 1_000_000) * pricing.output_price_per_1m
    return round(input_cost + output_cost, 8)


# ---------------------------------------------------------------------------
# Metric Result — comparison to ground truth
# ---------------------------------------------------------------------------

class MetricResult(BaseModel):
    """
    Stores evaluation scores for one run against its ground truth.
    Links to run_configs via run_id.
    """

    metric_id:              UUID = Field(default_factory=uuid4)
    run_id:                 UUID

    # Structural validity (None = validation was skipped)
    parses_valid:           Optional[bool]
    parse_error:            Optional[str] = None

    # Entity metrics — exact ID match
    entity_id_precision:    float       # correct IDs / total IDs in output
    entity_id_recall:       float       # correct IDs / total IDs in ground truth
    entity_id_f1:           float

    # Entity metrics — fuzzy name match
    entity_name_precision:  float
    entity_name_recall:     float
    entity_name_f1:         float

    # Entity metrics — lemmatized name match
    entity_lemma_precision: float
    entity_lemma_recall:    float
    entity_lemma_f1:        float

    # Relationship metrics — relaxed (source + target match, ignores type)
    relationship_relaxed_precision: float
    relationship_relaxed_recall:    float
    relationship_relaxed_f1:        float

    # Relationship metrics — strict (source + target + type must all match)
    relationship_strict_precision:  float
    relationship_strict_recall:     float
    relationship_strict_f1:         float

    # Raw counts for transparency
    entities_in_output:     int
    entities_in_truth:      int
    relationships_in_output:        int
    relationships_in_truth:         int

    # Error taxonomy counts - entities
    missing_entities:       int
    extra_entities:         int
    false_entities:         int
    duplicate_entities:     int

    # Error taxonomy counts - relationships
    missing_relationships:  int
    extra_relationships:    int
    false_relationships:    int
    duplicate_relationships: int