"""
MAESTRO viz — matplotlib theme and color palettes.

Implements the project Visualization Design Guide
(docs/visualization_design_guide.md). The palette dictionaries below are
copied verbatim from that guide (keyed by human display name, as the guide
specifies); the mapping helpers bridge from the values the database stores
(enum strings like ``single_agent``, model ids like
``claude-haiku-4-5-20251001``) to those display-keyed palettes.

Three categorical dimensions each get their own visual signature — strategy
(pink→magenta gradient), provider (one hue per provider, two lightness stops
for smaller-model vs flagship), tier (amber gradient) — plus a shared
sequential colormap for continuous metrics. The style itself (Arial, L-shaped
dark-gray axes, light grid, white background) is applied via
``apply_thesis_style``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palettes — verbatim from the design guide (display-name keys).
# ---------------------------------------------------------------------------

# Provider gradients. Each list is [smaller model, flagship].
PROVIDER_TIERS: dict[str, list[str]] = {
    "Claude": ["#F0997B", "#993C1D"],
    "ChatGPT": ["#6B7280", "#1F2937"],
    "Mistral": ["#A78BFA", "#5B21B6"],
    "Gemini": ["#60A5FA", "#1E40AF"],
    "DeepSeek": ["#14B8A6", "#115E59"],
}

# Strategy gradient. Ordered light = least explicit orchestration (the
# single-LLM-call baseline), dark = most explicit (the graph workflow). This
# assignment is frozen — never reassign a strategy's color across charts.
STRATEGY_COLORS: dict[str, str] = {
    "Single Agent": "#ED93B1",
    "SOP": "#D4537E",
    "Crew AI": "#993556",
    "LangGraph": "#72243E",
}

# Tier gradient. Light = simple input, dark = complex. Tier 3 is a reserved
# slot for the optional expansion analysis.
TIER_COLORS: dict[int, str] = {
    1: "#FAC775",
    2: "#BA7517",
    3: "#633806",
}

# Continuous-metric colormap (heatmaps etc.). Use with explicit vmin/vmax and
# a labelled colorbar.
HEAT_COLORMAP: str = "YlOrRd"

# Neutral gray for control conditions, which are reference floors/ceiling
# rather than a dimension to color distinctly. (Not part of the guide's named
# palettes; controls are excluded from most charts, and where they appear they
# should read as subordinate.)
CONTROL_COLOR: str = "#9CA3AF"

# Default categorical cycle for a dimension without its own palette: the
# strategy gradient, in its frozen order.
DEFAULT_CYCLE: list[str] = list(STRATEGY_COLORS.values())

# ---------------------------------------------------------------------------
# DB-value → guide-display mappings.
#
# The database stores enum values and full model ids; the guide keys palettes
# by display name. These dicts are the bridge. Keep them in sync with
# maestro.schemas.Strategy and maestro.experiment_config.MODELS.
# ---------------------------------------------------------------------------

# Strategy enum value (run_configs.strategy) → guide display name.
_STRATEGY_VALUE_TO_NAME: dict[str, str] = {
    "single_agent": "Single Agent",
    "sop_based": "SOP",
    "crew_ai": "Crew AI",
    "lang_graph": "LangGraph",
}

# Model id (run_configs.model) → (provider display name, slot) where slot is
# 0 for the smaller/efficiency model and 1 for the flagship. All currently
# configured models are the smaller (light) stop; flagships map to slot 1 when
# added.
_MODEL_TO_PROVIDER_SLOT: dict[str, tuple[str, int]] = {
    "claude-haiku-4-5-20251001": ("Claude", 0),
    "gpt-4o-mini-2024-07-18": ("ChatGPT", 0),
    "mistral-small-2603": ("Mistral", 0),
    "gemini-2.5-flash-lite": ("Gemini", 0),
    "deepseek-v4-flash": ("DeepSeek", 0),
}

# Module-level guard so the rcParams update runs once per process even if
# apply_thesis_style is called from several views in a single rerun.
_STYLE_APPLIED = False


def apply_thesis_style(*, force: bool = False) -> None:
    """
    Apply the house matplotlib style to ``plt.rcParams`` (idempotent).

    Mirrors the typography / axes / gridline / background rules of the design
    guide. Safe to call from every view; the work happens only on the first
    call unless ``force=True`` (useful in tests that mutate rcParams).
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return

    plt.rcParams.update(
        {
            # Typography — Arial with cross-platform fallbacks.
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Axes — L-shape, dark-gray spines.
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Ticks — outside, major only.
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Gridlines — light, solid, behind the data.
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            # Background — white everywhere (screen and saved files).
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Default categorical color cycle (strategy gradient order).
            "axes.prop_cycle": mpl.cycler(color=DEFAULT_CYCLE),
        }
    )
    _STYLE_APPLIED = True


# ---------------------------------------------------------------------------
# Color lookups — accept the values stored in the database.
# ---------------------------------------------------------------------------


def strategy_color(strategy_value: str) -> str:
    """
    Color for a strategy, given the value stored in ``run_configs.strategy``
    (e.g. ``"single_agent"``). Returns the frozen gradient color for the four
    orchestration strategies, and the neutral control gray for control
    conditions or any unmapped value (so a chart never crashes on a new
    strategy before its mapping is added).
    """
    name = _STRATEGY_VALUE_TO_NAME.get(strategy_value)
    if name is None:
        return CONTROL_COLOR
    return STRATEGY_COLORS[name]


def model_color(model_id: str) -> str:
    """
    Color for a model, given the id stored in ``run_configs.model`` (e.g.
    ``"claude-haiku-4-5-20251001"``). Resolves to the provider's hue at the
    model's lightness slot (smaller vs flagship). Neutral gray if unmapped.
    """
    mapping = _MODEL_TO_PROVIDER_SLOT.get(model_id)
    if mapping is None:
        return CONTROL_COLOR
    provider, slot = mapping
    return PROVIDER_TIERS[provider][slot]


def provider_of(model_id: str) -> str | None:
    """Provider display name for a model id, or None if unmapped."""
    mapping = _MODEL_TO_PROVIDER_SLOT.get(model_id)
    return mapping[0] if mapping else None


def tier_color(tier: int) -> str:
    """Color for a tier integer (1/2/3). Neutral gray if out of range."""
    return TIER_COLORS.get(tier, CONTROL_COLOR)
