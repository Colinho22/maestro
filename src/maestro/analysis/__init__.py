"""MAESTRO analysis package (metrics + display helpers)."""

# Statistical analysis pipeline. Re-exported so callers can do
# ``from maestro.analysis import describe, anova_strategy`` without reaching
# into the submodule. The CLI lives in maestro.analysis.__main__.
from maestro.analysis.statistics import (
    DEFAULT_CONVENTION,
    INTENT_TO_TREAT,
    PRIMARY_DV,
    SCHEMA_VERSION,
    VALID_ONLY,
    aggregate_experimental,
    anova_strategy,
    anova_strategy_by_model,
    anova_strategy_by_tier,
    describe,
    effect_sizes,
    error_taxonomy_by_strategy,
    load_dataframe,
    mixed_effects_robustness,
    posthoc_strategy,
    tradeoff_correctness_efficiency,
)

# Two import blocks from the same module is intentional: ruff's isort
# rules sort `as`-aliased imports separately from plain ones and will
# split a merged block on every --fix. Leaving them split is the stable
# shape that survives `ruff check --fix`.
from maestro.analysis.timestamps import (
    ENV_VAR as DISPLAY_TZ_ENV_VAR,
)
from maestro.analysis.timestamps import (
    format_for_display,
    resolve_display_tz,
)

__all__ = [
    "DISPLAY_TZ_ENV_VAR",
    "format_for_display",
    "resolve_display_tz",
    # statistics
    "DEFAULT_CONVENTION",
    "INTENT_TO_TREAT",
    "PRIMARY_DV",
    "SCHEMA_VERSION",
    "VALID_ONLY",
    "aggregate_experimental",
    "anova_strategy",
    "anova_strategy_by_model",
    "anova_strategy_by_tier",
    "describe",
    "effect_sizes",
    "error_taxonomy_by_strategy",
    "load_dataframe",
    "mixed_effects_robustness",
    "posthoc_strategy",
    "tradeoff_correctness_efficiency",
]
