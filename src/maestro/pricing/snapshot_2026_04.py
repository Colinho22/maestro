"""
Frozen pricing snapshot for the April 2026 main experiment run.

The ``VERSION`` string is stored verbatim in ``run_environments.pricing_version``
for every cell recorded under this snapshot, so historical rows stay
interpretable even after the vendor pages have moved on. A repricing is never
an edit to this file: a new dated snapshot (``snapshot_2026_10.py``,
``snapshot_2027_01.py``, ...) lands beside it with its own ``VERSION`` string,
and ``DEFAULT_VERSION`` in ``maestro.pricing`` bumps to point at the new file.
That way pre-change and post-change runs are pinned to whichever snapshot was
default the day they ran, and cross-snapshot mixing is a research decision
rather than a silent side effect.

Prices are USD per 1M tokens, verified against each provider's public pricing
page in April 2026. IDs match the canonical registry (``maestro.models``); the
one-to-one invariant is enforced by ``test_pricing_and_registry_are_one_to_one``
in the model-registry consistency suite.
"""

from __future__ import annotations

from maestro.schemas import ModelPricing

# ISO year-month, stored on every run captured under this snapshot. String is
# deliberately not a date: it identifies the file, not a specific day, and a
# stray ``date`` parse in a downstream consumer would silently drop context.
VERSION = "2026-04"


# One entry per active benchmarked model. Order mirrors
# ``maestro.models._REGISTRY_ENTRIES`` (alphabetical by provider, frontier
# then efficiency within each provider); the model-registry consistency test
# blocks silent drift on the id side.
PRICING: list[ModelPricing] = [
    # Anthropic
    ModelPricing(
        model="claude-opus-4-8",  # frontier
        input_price_per_1m=5.00,
        output_price_per_1m=25.00,
        # Opus 4.7+ removed sampling params; sending temperature returns 400.
        supports_temperature=False,
    ),
    ModelPricing(
        model="claude-haiku-4-5-20251001",  # efficiency
        input_price_per_1m=1.00,
        output_price_per_1m=5.00,
    ),
    # OpenAI (GPT-5 family: max_completion_tokens, no custom temperature)
    ModelPricing(
        model="gpt-5.5-2026-04-23",  # frontier
        input_price_per_1m=5.00,
        output_price_per_1m=30.00,
        supports_temperature=False,
    ),
    ModelPricing(
        model="gpt-5.4-mini-2026-03-17",  # efficiency
        input_price_per_1m=0.75,
        output_price_per_1m=4.50,
        supports_temperature=False,
    ),
    # Mistral
    ModelPricing(
        model="mistral-medium-3-5",  # frontier
        input_price_per_1m=1.50,
        output_price_per_1m=7.50,
    ),
    ModelPricing(
        model="mistral-small-2603",  # efficiency
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    # Gemini
    ModelPricing(
        model="gemini-3.5-flash",  # frontier
        input_price_per_1m=1.50,
        output_price_per_1m=9.00,
    ),
    ModelPricing(
        model="gemini-3.1-flash-lite",  # efficiency
        input_price_per_1m=0.25,
        output_price_per_1m=1.50,
    ),
    # DeepSeek: the cross-provider replication dimension's emerging-Chinese
    # entry (proposal section 3.2), consumed via the OpenAI-compatible endpoint
    # (see providers/deepseek.py). Pricing is the cache-MISS (standard) rate;
    # DeepSeek also offers a cheaper cache-hit input price, but ModelPricing has
    # a single input rate, so cache-miss makes the tracked cost an upper bound
    # on actual spend (never an under-count).
    ModelPricing(
        model="deepseek-v4-pro",  # frontier
        input_price_per_1m=0.435,
        output_price_per_1m=0.87,
    ),
    ModelPricing(
        model="deepseek-v4-flash",  # efficiency
        input_price_per_1m=0.14,
        output_price_per_1m=0.28,
    ),
]
