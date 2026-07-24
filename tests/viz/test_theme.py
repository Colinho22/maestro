"""
Tests for the viz design system: theme palettes (per the Visualization Design
Guide), the DB-value->color mappings, style application, and the chart export
path.

These run without a Streamlit server. The export path (savefig to PNG/SVG
bytes) is the part of chart.py that matters for the thesis deliverable and is
verifiable headless; the st.pyplot/download_button display is exercised by
launching the app, not pytest.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("streamlit")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend: no display needed for tests

from maestro.experiment_config import CONTROL_STRATEGIES, MODELS  # noqa: E402
from maestro.schemas import Strategy  # noqa: E402
from maestro.viz import theme  # noqa: E402
from maestro.viz.chart import _savefig_bytes, new_figure  # noqa: E402

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_LLM_STRATEGIES = [s.value for s in Strategy if s not in CONTROL_STRATEGIES]


# ---------------------------------------------------------------------------
# Palettes match the guide
# ---------------------------------------------------------------------------


def test_strategy_palette_matches_guide():
    """The frozen strategy gradient: display names and hexes verbatim."""
    assert theme.STRATEGY_COLORS == {
        "Single Agent": "#ED93B1",
        "SOP": "#D4537E",
        "CrewAI": "#993556",
        "LangGraph": "#72243E",
    }


def test_provider_tiers_have_light_and_dark_stops():
    for provider, stops in theme.PROVIDER_TIERS.items():
        assert len(stops) == 2, f"{provider} needs [smaller, flagship]"
        assert all(_HEX_RE.match(c) for c in stops)


def test_tier_palette_keyed_by_int():
    assert set(theme.TIER_COLORS) == {1, 2, 3}
    assert all(_HEX_RE.match(c) for c in theme.TIER_COLORS.values())


def test_heat_colormap_is_ylorrd():
    assert theme.HEAT_COLORMAP == "YlOrRd"


# ---------------------------------------------------------------------------
# DB-value -> color mappings cover the real entities
# ---------------------------------------------------------------------------


def test_every_llm_strategy_maps_to_a_gradient_color():
    """Each non-control strategy value resolves to one of the frozen colors."""
    for value in _LLM_STRATEGIES:
        color = theme.strategy_color(value)
        assert color in theme.STRATEGY_COLORS.values(), value


def test_every_configured_model_maps_to_a_provider_color():
    """Each real model id resolves to a color from its provider's gradient."""
    for mp in MODELS:
        if mp.model == "control":
            continue
        color = theme.model_color(mp.model)
        provider = theme.provider_of(mp.model)
        assert provider is not None, f"{mp.model} has no provider mapping"
        assert color in theme.PROVIDER_TIERS[provider], mp.model


def test_strategy_color_falls_back_for_controls_and_unknown():
    assert theme.strategy_color("null_control") == theme.CONTROL_COLOR
    assert theme.strategy_color("nonexistent") == theme.CONTROL_COLOR


def test_model_color_falls_back_for_unknown():
    assert theme.model_color("no-such-model") == theme.CONTROL_COLOR
    assert theme.provider_of("no-such-model") is None


def test_tier_color_lookup_and_fallback():
    assert theme.tier_color(1) == theme.TIER_COLORS[1]
    assert theme.tier_color(99) == theme.CONTROL_COLOR


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


def test_apply_thesis_style_sets_rcparams():
    import matplotlib.pyplot as plt

    theme.apply_thesis_style(force=True)
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert "Arial" in plt.rcParams["font.sans-serif"]


def test_apply_thesis_style_pins_text_colors():
    """Text colors are pinned so labels never inherit an ambient (e.g. white,
    invisible) default. Guards the design-guide section 2 color spec: axis and
    chart titles pure black, body text #333333."""
    import matplotlib.pyplot as plt

    # Simulate a hostile dark-mode default where every text color is white.
    plt.rcParams.update(
        {
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
        }
    )
    theme.apply_thesis_style(force=True)
    assert plt.rcParams["axes.labelcolor"] == "#000000"
    assert plt.rcParams["axes.titlecolor"] == "#000000"
    assert plt.rcParams["text.color"] == "#333333"


# ---------------------------------------------------------------------------
# Export path: the thesis-figure deliverable
# ---------------------------------------------------------------------------


def test_export_produces_png_and_svg_bytes():
    import matplotlib.pyplot as plt

    fig, ax = new_figure()
    ax.bar(["a", "b"], [0.5, 0.9])

    png = _savefig_bytes(fig, "png")
    svg = _savefig_bytes(fig, "svg")

    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number
    assert b"<svg" in svg

    plt.close(fig)


def test_new_figure_applies_style():
    import matplotlib.pyplot as plt

    fig, ax = new_figure()
    assert plt.rcParams["axes.spines.top"] is False
    plt.close(fig)
