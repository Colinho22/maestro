"""
MAESTRO: Canonical Mermaid output contract (single source of truth).

The rules that tell a model how to emit Mermaid live HERE and nowhere else.
Providers supply the system-message identity from ``MERMAID_SYSTEM_IDENTITY``;
the single-agent baseline and multi-step step 3 build their user prompts from
``render_rules()``. Defining the contract once keeps every provider and every
orchestration strategy on a byte-identical output contract, so quality
differences are attributable to orchestration (the independent variable), not
to drifting prompt wording.

Why this module imports nothing from ``maestro``: ``providers`` and
``strategies`` both depend on it, so any back-import would create a cycle.
Keep it dependency-free (plain strings + one helper).

The optional ``skill`` layer in ``render_rules`` is the future
prompt-enhancement variable: an append-only block, never an edit to the
baseline rules, so the enhancement stays an isolatable condition.
"""

from __future__ import annotations

# System-message identity. Was duplicated verbatim as ``SYSTEM_PROMPT`` in
# every provider subclass; now defined once and assigned on ``LLMProvider``.
MERMAID_SYSTEM_IDENTITY = (
    "You are a diagram generation assistant. "
    "Respond only with valid Mermaid diagram code. "
    "Do not include any explanation, markdown fencing, or additional text."
)

# Unified user-prompt output rules. Was hand-copied (with drift) in
# ``single.py`` and step 3 of ``_extraction.py``; now one contract applied
# identically to both. Brace-free on purpose so it can be embedded into a
# template string ahead of any later ``.format()`` call without escaping.
MERMAID_RULES = """\
- Begin the diagram with a flowchart header, `flowchart LR`; do not use C4, sequence, class, or other diagram types
- Output only valid Mermaid syntax
- Wrap node labels in double quotes, e.g. node_id["My Label"], so labels with spaces, parentheses, slashes, or line breaks stay parseable
- If a node has no label, write just its id (e.g. gw_result): never an empty bracket like node_id[""]
- Quote edge labels the same way, with no spaces inside the pipes, e.g. a -->|"My edge"| b; for an unlabelled edge use a plain arrow a --> b and never an empty label like -->|| or -->| |
- Include every entity and relationship from the input
- Preserve hierarchy using subgraphs for pools, lanes, and subprocesses
- Do not invent entities or relationships not present in the input
- Do not include explanations or markdown code fences
- Do not use internal or relationship IDs as edge labels"""


def render_rules(skill: str | None = None) -> str:
    """
    Return the canonical Mermaid rules, optionally extended by a skills block.

    The skills block is APPEND-ONLY: it is concatenated after the baseline
    rules and must never edit them, so a baseline run (``skill=None``) and an
    enhanced run differ only by the added text. Baseline callers pass ``None``.
    """
    if skill is None:
        return MERMAID_RULES
    return MERMAID_RULES + "\n" + skill
