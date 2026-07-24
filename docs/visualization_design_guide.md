# Visualization Design Guide

Standards for every chart in the thesis. The goal: any chart in the document reads consistently, encodes the right dimension on the right channel, and passes accessibility checks.

This is a living document. Sections 1–4 are complete. Sections 5–6 are placeholders, to be filled in subsequent iterations.

The canonical machine-readable form of the palettes and style lives in
`src/maestro/viz/theme.py`. This document is the source of truth for the
*intent*; `theme.py` implements it (keyed to the values the database stores
via a mapping layer, since this guide names dimensions by their human display
labels).

---

## 1. Color system

Three independent dimensions need visual encoding: **provider** (which model family produces the diagram), **strategy** (which orchestration approach generates the diagram), and **tier** (input complexity). Each dimension gets its own visual signature so a reader can identify the encoded dimension at a glance. A separate sequential colormap handles continuous metrics.

| Dimension | Encoding | Levels |
|---|---|---|
| Provider | Distinct hue per provider | 5 |
| Provider × model | Lightness step within the provider hue | 2 per provider |
| Strategy | Pink/magenta gradient | 4 |
| Tier | Amber gradient | 3 (Tier 3 optional) |
| Continuous metric (F1, cost, latency, …) | YlOrRd colormap | sequential |

### 1.1 Provider palette

Each provider gets one hue. Two lightness stops within that hue encode the two model variants used in the extended-tier experiment (smaller model vs flagship).

| Provider | Light — smaller model | Dark — flagship |
|---|---|---|
| Claude | `#F0997B` | `#993C1D` |
| ChatGPT | `#6B7280` | `#1F2937` |
| Mistral | `#A78BFA` | `#5B21B6` |
| Gemini | `#60A5FA` | `#1E40AF` |
| DeepSeek | `#14B8A6` | `#115E59` |

**Rationale.** Hues anchor to brand identity where contrast allows: Claude coral, OpenAI slate, Gemini blue, DeepSeek teal. Mistral pivots from yellow to violet because pure yellow fails WCAG contrast on white. The light-dark pair within each provider uses OKLCH lightness steps, which keep the perceived progression even across hues.

**Known trade-off.** Mistral light (`#A78BFA`) and Gemini light (`#60A5FA`) sit close in color space. They appear next to each other only in charts that compare the smaller-model tier across providers, and alphabetical ordering places them adjacent. The dark stops separate cleanly, so flagship comparisons are unaffected. This affects at most one or two charts in the analysis.

### 1.2 Strategy palette

Four orchestration strategies on a single pink-to-magenta gradient. Lightness encodes structural explicitness — lighter for less explicit orchestration, darker for more explicit.

| Strategy | Hex |
|---|---|
| Single Agent | `#ED93B1` |
| SOP | `#D4537E` |
| CrewAI | `#993556` |
| LangGraph | `#72243E` |

**Rationale.** Pink occupies a region of color space that no provider uses, which removes any chance of a reader confusing a strategy for a provider. The gradient implies ordering. Lightest is the single-LLM-call baseline, darkest is the explicit graph workflow.

**The ordering is frozen.** Single Agent always gets `#ED93B1`, LangGraph always gets `#72243E`. Never reassign across charts.

### 1.3 Tier palette

Three input-complexity tiers on a light-to-dark amber gradient. Lightness encodes entity count.

| Tier | Definition | Hex |
|---|---|---|
| 1 — Simple | < 10 entities | `#FAC775` |
| 2 — Complex | 10–25 entities | `#BA7517` |
| 3 — Cross-layer | 25+ entities (optional) | `#633806` |

**Rationale.** Amber is the only major hue region not claimed by providers or strategies. The bronze-to-gold-to-brown progression maps intuitively to ascending tier. Tier 3 reserves a defined slot for the optional expansion analysis described in the proposal, so the palette never needs to shift if Tier 3 enters the analysis later.

### 1.4 Continuous metric colormap

Heatmaps and any other chart encoding a continuous metric (F1, cost, latency, hallucination rate) use a single shared colormap, separate from the categorical palettes above.

**Colormap.** `YlOrRd` (matplotlib built-in). Light yellow at the low end, deep red at the high end. Reads as "intensity" and matches the conventional heat-map aesthetic.

**Rules.**

- Always include a colorbar with the metric name as the label.
- Set `vmin` and `vmax` explicitly. For F1/accuracy use 0 and 1; for unbounded metrics use 0 and the maximum observed value.
- Cell value annotations: compute luminance per cell and switch text color — white on dark cells, `#333333` on light cells.

**Trade-off.** The dark-red zone of YlOrRd neighbors Claude's flagship coral (`#993C1D`). The colorbar disambiguates context, but avoid placing a YlOrRd heatmap directly next to a Claude-coral bar chart on the same page.

### 1.5 Co-occurrence safety

The 4×2 factorial design (strategy × tier) puts both palettes on the same chart often. Pink and amber sit on opposite sides of red in hue space. Even at their darkest stops — `#72243E` deep magenta vs `#633806` deep brown — they stay distinguishable under all common color-vision conditions.

Providers and tiers can also co-occur (provider as bar group, tier as color, or vice versa). The amber gradient does not overlap with any provider hue.

### 1.6 Accessibility

- Every dark stop meets WCAG AA contrast (≥4.5:1) on white.
- Every light stop meets AA Large (≥3:1) and the 3:1 non-text contrast threshold for chart fills.
- Lightness alone preserves the ordering in each gradient. Charts remain readable when printed in grayscale.
- Palettes were checked against deuteranopia, protanopia, and tritanopia simulations.

### 1.7 Python palette dictionaries

```python
# Provider gradients
# Each list: [smaller model, flagship]
PROVIDER_TIERS = {
    "Claude":   ["#F0997B", "#993C1D"],
    "ChatGPT":  ["#6B7280", "#1F2937"],
    "Mistral":  ["#A78BFA", "#5B21B6"],
    "Gemini":   ["#60A5FA", "#1E40AF"],
    "DeepSeek": ["#14B8A6", "#115E59"],
}

# Strategy gradient
# Ordered: light = least explicit orchestration, dark = most explicit
STRATEGY_COLORS = {
    "Single Agent": "#ED93B1",
    "SOP":          "#D4537E",
    "CrewAI":       "#993556",
    "LangGraph":    "#72243E",
}

# Tier gradient
# Light = simple input, dark = complex input
# Tier 3 reserved for optional expansion analysis
TIER_COLORS = {
    1: "#FAC775",
    2: "#BA7517",
    3: "#633806",
}

# Continuous metric colormap
HEAT_COLORMAP = "YlOrRd"
```

---

## 2. Typography

One font everywhere. Hierarchy comes from weight and size, not from switching fonts. Mixing typefaces in a thesis figure looks unprofessional.

**Font family.** Arial. On systems without Arial, fall back to Liberation Sans, then DejaVu Sans.

**Sizes.**

| Element | Size | Weight | Color |
|---|---|---|---|
| Body text (thesis prose) | 11pt | regular | `#000000` |
| Axis title | 10pt | bold | `#000000` |
| Tick labels | 9pt | regular | `#333333` |
| Legend text | 9pt | regular | `#333333` |
| In-chart annotations | 10pt | regular | `#333333` |
| Chart title (slides / repo only) | 12pt | bold | `#000000` |

**Chart titles.** Omit for figures inside the thesis PDF. The figure caption below the chart does the descriptive work. Include in-chart titles only for slide decks and the open-source repository's interactive views.

**Color choice rationale.** Axis titles use pure black for emphasis (titles outrank values in hierarchy). Tick labels, legend text, and annotations use `#333333` — softer than pure black, matches the axis line color, prints cleanly without sharp edges on Arial's thin strokes at 9pt.

---

## 3. Background and layout

**Default background.** White (`#FFFFFF`) for both the figure and the axes face. White prints cleanly and gives charts a neutral container regardless of where they end up — thesis PDF, slides, or the repo README.

**Transparent background.** Use as a per-chart override when the chart sits on a colored slide background or when the chart type benefits from showing through (donut, sunburst, mockup overlays). Apply by setting `facecolor='none'` on both the figure and the axes.

**Standard figure dimensions.**

| Chart type | Size (inches) | Aspect |
|---|---|---|
| Standard chart | 9 × 4.5 | 2:1 |
| Wide grouped bar (e.g., 5 providers × 2 models) | 10 × 4.5 | ~2.2:1 |
| Heatmap | 5.5 × 5 | square-ish, scales with data shape |
| Pareto scatter | 8 × 5 | 1.6:1 |
| Tall vertical bar (≤4 categories) | 7 × 4.5 | ~1.6:1 |

**Export resolution.** Save PNG at 200 DPI for slides and inline previews. Use SVG or PDF for thesis figures — vector formats scale without loss in the printed thesis.

---

## 4. Axes and gridlines

**Frame.** L-shape — bottom and left spines only. Top and right hidden. Open and modern, the standard for scientific publication figures.

**Spines and ticks.**

- Spine line: 0.75pt, color `#333333`
- Tick marks: outside the plot, 3pt long, major only
- Tick line width matches the spine (0.75pt)
- No minor ticks — they add visual noise at thesis-figure size without conveying useful detail

**Tick labels.** Arial 9pt regular, color `#333333`. Default rotation 0°. Rotate 45° only when adjacent labels would overlap.

**Axis titles.** Arial 10pt bold, color `#000000`. Always include units in parentheses when relevant — `"Latency (s)"`, `"Cost (USD)"`, `"Tokens (k)"`.

**Gridlines.**

- Color: `#D0D0D0`
- Weight: 0.5pt
- Style: solid (dashed reads as "uncertain" or "projected" — wrong signal for measured data)
- Always behind the data (`axes.axisbelow = True`)

**Gridline direction by chart type.**

| Chart type | Horizontal | Vertical |
|---|---|---|
| Vertical bars | yes | no |
| Horizontal bars | no | yes |
| Line | yes | optional |
| Area | yes | no |
| Scatter / Pareto | yes | yes |
| Heatmap | no | no — cells provide structure |

**Heatmap exception.** Heatmaps get a full box frame (all four spines visible) instead of the L-shape, because cells need clear boundaries on all sides.

**Linear vs log scale.**

- Linear by default.
- Log scale for metrics that span more than one order of magnitude — typically cost, latency, token count.
- Always linear for bounded metrics (F1, accuracy, hallucination rate, percentages).
- When using a log scale, annotate in the axis title: `"Cost in USD (log)"`.

**Zero baseline.** Bar charts must start at 0. Truncating the bar baseline distorts visual comparison. Line charts may start above 0 when the data range is narrow and 0 carries no meaning (e.g., F1 between 0.7 and 0.9). Mention any truncation in the figure caption.

**Number formatting.**

- Thousands separator: comma — `1,000`
- Currency: `$1.23` with 2 decimals
- Time: with units inline — `"1.2 s"`, `"150 ms"`
- Percentages: `87%` typically; `87.3%` only when small differences matter
- F1 / accuracy: 2 decimals — `0.87`
- Round consistently to avoid float artifacts

---

## 5. Legends

*To be defined.* Topics to cover: position, ordering rules, in-legend vs in-chart labels, marker style (square / line / dot), frame on/off, when to omit the legend entirely.

---

## 6. Chart-type recipes

*To be defined.* Topics to cover: which chart type fits which RQ (bar for category comparison, line for trends across input size, heatmap for the 4×2 factorial, Pareto scatter for RQ4 efficiency-vs-correctness), default configurations per recipe.

---

## Appendix — Reference Python configuration

The `apply_thesis_style()` function in `src/maestro/viz/theme.py` configures
every subsequent matplotlib chart from a single call. It mirrors the
typography, axes, gridline, and background rules above.
