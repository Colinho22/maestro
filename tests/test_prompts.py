"""
Guards for the single-source-of-truth Mermaid output contract (maestro.prompts).

These tests pin the contract so it cannot silently drift back into the
per-provider / per-strategy copies it was consolidated from:

  * snapshot           — the canonical identity + rules text is intentional,
                         so a change must be a deliberate edit to this file.
  * no-duplication     — every provider's SYSTEM_PROMPT *is* the shared
                         identity object (catches anyone re-inlining a literal),
                         and the identity actually reaches provider.complete().
  * identical-injection — the rules block embedded in the single-agent prompt
                         equals the one in multi-step step 3 (the drift this
                         refactor fixed).
"""

from __future__ import annotations

from maestro.prompts import (
    MERMAID_RULES,
    MERMAID_SYSTEM_IDENTITY,
    render_rules,
)
from maestro.providers.anthropic import AnthropicProvider
from maestro.providers.base import LLMProvider
from maestro.providers.deepseek import DeepSeekProvider
from maestro.providers.gemini import GeminiProvider
from maestro.providers.mistral import MistralProvider
from maestro.providers.openai import OpenAIProvider
from maestro.strategies._extraction import STEP_3_PROMPT
from maestro.strategies.single import PROMPT_TEMPLATE

# Concrete providers whose SYSTEM_PROMPT must resolve to the shared identity.
# deepseek defines no literal of its own — it must inherit via OpenAIProvider.
_PROVIDERS = [
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    MistralProvider,
    DeepSeekProvider,
]


# ---------------------------------------------------------------------------
# Snapshot — canonical text is pinned; edits must be deliberate.
# ---------------------------------------------------------------------------


def test_system_identity_snapshot():
    assert MERMAID_SYSTEM_IDENTITY == (
        "You are a diagram generation assistant. "
        "Respond only with valid Mermaid diagram code. "
        "Do not include any explanation, markdown fencing, or additional text."
    )


def test_rules_snapshot():
    assert MERMAID_RULES == (
        "- Begin the diagram with a flowchart header, `flowchart LR`; "
        "do not use C4, sequence, class, or other diagram types\n"
        "- Output only valid Mermaid syntax\n"
        '- Wrap node labels in double quotes, e.g. node_id["My Label"], so '
        "labels with spaces, parentheses, slashes, or line breaks stay "
        "parseable\n"
        "- If a node has no label, write just its id (e.g. gw_result) — "
        'never an empty bracket like node_id[""]\n'
        "- Quote edge labels the same way, with no spaces inside the pipes, "
        'e.g. a -->|"My edge"| b; for an unlabelled edge use a plain arrow '
        "a --> b and never an empty label like -->|| or -->| |\n"
        "- Include every entity and relationship from the input\n"
        "- Preserve hierarchy using subgraphs for pools, lanes, and subprocesses\n"
        "- Do not invent entities or relationships not present in the input\n"
        "- Do not include explanations or markdown code fences\n"
        "- Do not use internal or relationship IDs as edge labels"
    )


def test_rules_are_brace_free():
    """Braces would break the later .format() call on the strategy templates."""
    assert "{" not in MERMAID_RULES and "}" not in MERMAID_RULES


# ---------------------------------------------------------------------------
# No-duplication — providers share the one identity object, not copies.
# ---------------------------------------------------------------------------


def test_base_defines_the_identity():
    assert LLMProvider.SYSTEM_PROMPT is MERMAID_SYSTEM_IDENTITY


def test_no_provider_reinlines_system_prompt():
    for provider_cls in _PROVIDERS:
        assert provider_cls.SYSTEM_PROMPT is MERMAID_SYSTEM_IDENTITY, (
            f"{provider_cls.__name__} re-inlined SYSTEM_PROMPT instead of "
            "inheriting the shared identity"
        )


def test_fallback_identity_resolves_to_shared(recording_provider_factory):
    """
    When a caller passes system_prompt=None, real providers fall back to
    self.SYSTEM_PROMPT (the ``system_prompt if system_prompt is not None else
    self.SYSTEM_PROMPT`` expression in every complete()). This exercises that
    fallback expression against an LLMProvider subclass and asserts it resolves
    to the shared identity object — so a re-inlined per-provider literal would
    be caught here, not just at the class-attribute level.
    """
    provider = recording_provider_factory(outputs=["graph TD\n  a"])
    system_prompt = None
    effective_system = (
        system_prompt if system_prompt is not None else provider.SYSTEM_PROMPT
    )
    assert effective_system is MERMAID_SYSTEM_IDENTITY


# ---------------------------------------------------------------------------
# Identical-injection — single-agent and step 3 carry the same rules block.
# ---------------------------------------------------------------------------


def _rules_block(template: str) -> str:
    """Extract the block under 'Rules:' up to the next blank line."""
    return template.split("Rules:\n", 1)[1].split("\n\n", 1)[0]


def test_single_and_step3_inject_identical_rules():
    single_rules = _rules_block(PROMPT_TEMPLATE)
    step3_rules = _rules_block(STEP_3_PROMPT)
    assert single_rules == step3_rules == render_rules()


def test_templates_keep_runtime_placeholders():
    """The .format() placeholders must survive the f-string composition."""
    assert "{input_data}" in PROMPT_TEMPLATE
    assert "{entities_json}" in STEP_3_PROMPT
    assert "{relationships_json}" in STEP_3_PROMPT


def test_templates_format_without_stray_braces():
    PROMPT_TEMPLATE.format(input_data="{}")
    STEP_3_PROMPT.format(entities_json="[]", relationships_json="[]")


# ---------------------------------------------------------------------------
# render_rules — baseline vs append-only skills layer.
# ---------------------------------------------------------------------------


def test_render_rules_baseline_is_canonical():
    assert render_rules() == MERMAID_RULES
    assert render_rules(None) == MERMAID_RULES


def test_render_rules_skill_is_append_only():
    rendered = render_rules("- Prefer graph LR")
    assert rendered.startswith(MERMAID_RULES)
    assert rendered.endswith("- Prefer graph LR")
