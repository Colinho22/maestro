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
    """
    Orchestration strategy under test. The string values are persisted to
    the SQLite ``run_configs`` table and exposed as ``--strategy`` CLI
    choices, so they are part of the experiment's analysis interface and
    must not be renamed once data has been collected.
    """

    SINGLE_AGENT = "single_agent"
    SOP_BASED = "sop_based"
    CREW_AI = "crew_ai"
    LANG_GRAPH = "lang_graph"

    # Control conditions — bypass the LLM, deterministic, used as metric-
    # pipeline sanity checks and as interpretation anchors for absolute F1.
    # See maestro.strategies.controls.* for the implementations.
    NULL_CONTROL = "null_control"  # floor: empty diagram
    COPY_CONTROL = "copy_control"  # floor: raw input as diagram
    GROUND_TRUTH_CONTROL = "ground_truth_control"  # ceiling: ground truth verbatim


class Tier(int, Enum):
    """
    Complexity tier of an input dataset, bucketed by entity count. Used as
    a stratification dimension for the experiment matrix and as a filter
    via ``--tier``.

    Integer values are persisted to ``run_configs.tier``. Changing the
    enum (rename or bucket-shift) is therefore a breaking change for any
    pre-existing DB — start a fresh ``maestro.db`` after renaming.

    Tier names align with the thesis proposal (§3 / Table 3): Simple,
    Complex, Cross-layer. TODO: once the full input corpus is collected,
    re-run the bucket analysis and update the entity-count thresholds
    here (and/or replace them with structural criteria like pool count).
    """

    SIMPLE = 1  # < 10 entities
    COMPLEX = 2  # 10-25 entities
    CROSS_LAYER = 3  # 25+ entities or multi-pool / cross-layer flows


# ---------------------------------------------------------------------------
# InputFile — describes one diagram generation task
# ---------------------------------------------------------------------------


class InputFile(BaseModel):
    """
    Represents a single benchmark input: a JSON file with relational data
    and its associated ground truth diagram code.
    """

    example_id: str  # Human-readable ID, e.g. "er_diagram_01"
    tier: Tier  # Complexity tier (1-3)
    entity_count: int  # Number of entities in the input
    file_path: Path  # Path to the JSON input file on disk
    ground_truth_path: Path  # Path to the reference diagram code file
    description: Optional[str] = None  # Optional human note about this input


# ---------------------------------------------------------------------------
# RunConfig — captures the full experimental context of one run
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    """
    Groups all the dimensions of a single experiment run.
    run_id is the unique key; all other fields allow grouping/filtering.
    """

    run_id: UUID = Field(default_factory=uuid4)
    strategy: Strategy
    model: str  # e.g. "gpt-4o", "claude-3-5-sonnet"
    example_id: str  # FK to InputFile.example_id
    tier: Tier
    run_number: int  # Repeat index within same config (1-N)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # FK to run_environments.environment_id. Optional because rows written
    # before this column existed have NULL, and because env capture is best
    # effort — a run must still be persistable if the capture helper fails.
    environment_id: Optional[UUID] = None


# ---------------------------------------------------------------------------
# RunEnvironment — runtime stack snapshot, one row per CLI invocation
# ---------------------------------------------------------------------------


class RunEnvironment(BaseModel):
    """
    Snapshot of the runtime environment that produced a batch of runs.

    One row per CLI invocation, referenced by ``RunConfig.environment_id``.
    All fields except ``environment_id`` and ``captured_at`` are nullable
    because the underlying probe (subprocess, env var, package import) may
    legitimately fail — a missing field must be recorded as ``None`` rather
    than aborting the experiment.
    """

    environment_id: UUID = Field(default_factory=uuid4)

    # Host / runtime
    os: Optional[str] = None  # platform.platform()
    arch: Optional[str] = None  # platform.machine()
    python: Optional[str] = None  # sys.version
    hostname: Optional[str] = None  # platform.node()

    # Source control
    git_commit: Optional[str] = None  # git rev-parse HEAD
    git_dirty: Optional[bool] = None  # True/False/None (probe failed)

    # Dependency snapshot — JSON blob: {"anthropic": "1.2.3", "openai": None, ...}
    lib_versions: Optional[str] = None

    # Container provenance — set by CI/CD via env var, NULL when running locally
    docker_image_digest: Optional[str] = None

    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ModelPricing — lookup table for cost calculation
# ---------------------------------------------------------------------------


class ModelPricing(BaseModel):
    """
    Per-model token pricing in USD per 1M tokens.
    Used to compute cost_usd at write time.
    """

    model: str
    input_price_per_1m: float  # USD per 1M prompt tokens
    output_price_per_1m: float  # USD per 1M completion tokens


# ---------------------------------------------------------------------------
# RunResult — output and statistics of one LLM call
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """
    Stores everything produced by a single LLM generation run.
    Links back to RunConfig via run_id.
    """

    run_id: UUID  # FK to RunConfig.run_id

    # Output
    output_diagram_code: Optional[str] = None  # Generated Mermaid / PlantUML / etc.

    # Token usage
    prompt_tokens: int
    completion_tokens: int

    # Performance
    duration_ms: int  # Wall-clock time for the LLM call

    # Cost — computed from token counts + ModelPricing at write time
    cost_usd: float

    # Error — None if successful, exception message otherwise
    error: Optional[str] = None

    # Number of *retries* the underlying provider call needed (0 = first
    # attempt worked). Mirrors SubResult.retry_count for top-level runs.
    retry_count: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens for cost/efficiency analysis."""
        return self.prompt_tokens + self.completion_tokens

    @computed_field
    @property
    def success(self) -> bool:
        """True if a non-empty diagram was produced without error."""
        return (
            self.error is None
            and self.output_diagram_code is not None
            and self.output_diagram_code.strip() != ""
        )


class SubResult(BaseModel):
    """
    One sub-call within a multi-step strategy (e.g. SOP).
    Links to the parent run via run_id.
    """

    sub_id: UUID = Field(default_factory=uuid4)
    run_id: UUID  # FK to RunConfig.run_id
    step_number: int  # 1, 2, 3…
    step_name: str  # "extract_entities", "extract_relationships", etc.
    output_text: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    duration_ms: int
    cost_usd: float
    error: Optional[str] = None
    retry_count: int = 0  # 0 = first attempt worked


# ---------------------------------------------------------------------------
# Helper — compute cost from token counts and pricing
# ---------------------------------------------------------------------------


def compute_cost(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: ModelPricing,
) -> float:
    """
    Calculate USD cost for one LLM call.
    Prices are per 1M tokens — divide by 1_000_000.
    """
    input_cost = (prompt_tokens / 1_000_000) * pricing.input_price_per_1m
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

    metric_id: UUID = Field(default_factory=uuid4)
    run_id: UUID

    # Structural validity (None = validation was skipped)
    parses_valid: Optional[bool]
    parse_error: Optional[str] = None

    # Entity metrics — exact ID match
    entity_id_precision: float  # correct IDs / total IDs in output
    entity_id_recall: float  # correct IDs / total IDs in ground truth
    entity_id_f1: float

    # Entity metrics — fuzzy name match
    entity_name_precision: float
    entity_name_recall: float
    entity_name_f1: float

    # Entity metrics — lemmatized name match
    entity_lemma_precision: float
    entity_lemma_recall: float
    entity_lemma_f1: float

    # Relationship metrics — relaxed (source + target match, ignores type)
    relationship_relaxed_precision: float
    relationship_relaxed_recall: float
    relationship_relaxed_f1: float

    # Relationship metrics — strict (source + target + type must all match)
    relationship_strict_precision: float
    relationship_strict_recall: float
    relationship_strict_f1: float

    # Raw counts for transparency
    entities_in_output: int
    entities_in_truth: int
    relationships_in_output: int
    relationships_in_truth: int

    # Error taxonomy counts - entities
    missing_entities: int
    extra_entities: int
    false_entities: int
    duplicate_entities: int

    # Error taxonomy counts - relationships
    missing_relationships: int
    extra_relationships: int
    false_relationships: int
    duplicate_relationships: int

    # ------------------------------------------------------------------
    # Container metrics — pools / lanes / boundaries / expanded sub-processes
    # (subgraphs). Scored as a SEPARATE dimension from entities so swimlane
    # structure can be evaluated without polluting the entity metric/tiers.
    # P/R/F1 are None when the ground truth has no containers (metric N/A),
    # so aggregation can exclude those runs rather than averaging in a 0.
    # ------------------------------------------------------------------
    container_id_precision: Optional[float] = None
    container_id_recall: Optional[float] = None
    container_id_f1: Optional[float] = None
    container_name_precision: Optional[float] = None
    container_name_recall: Optional[float] = None
    container_name_f1: Optional[float] = None
    containers_in_output: int = 0
    containers_in_truth: int = 0

    # ------------------------------------------------------------------
    # Attachment metrics — BPMN boundary-event / compensation associations,
    # drawn as ``o--o`` edges. Undirected pairs (orientation-insensitive).
    # P/R/F1 are None when the ground truth has no attachments (metric N/A).
    # ------------------------------------------------------------------
    attachment_precision: Optional[float] = None
    attachment_recall: Optional[float] = None
    attachment_f1: Optional[float] = None
    attachments_in_output: int = 0
    attachments_in_truth: int = 0
