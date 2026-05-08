"""
MAESTRO — Shared task definition for multi-step strategies.

This module is the experimental *control variable*: the prompts, the JSON
schemas, the validation rules and the retry budget are byte-identical across
every multi-step strategy (SOP, CrewAI, LangGraph). Only the *orchestration*
differs between strategies — what each strategy file expresses in its own
shape.

Single-agent does not import from here: its baseline prompt is intentionally
distinct (one shot → diagram, no decomposition) and lives in `single.py`.

Adding a new multi-step strategy?
- Reuse the prompts, schemas and validators from this module unchanged.
- Write the orchestration longhand in your strategy file so a reader can see
  how that framework wires the same three steps.
"""

import json


# ---------------------------------------------------------------------------
# Step 1 — Extract entities (nodes) from the input dataset
# ---------------------------------------------------------------------------

STEP_1_PROMPT = """\
You are given a dataset describing entities and their relationships.
Your task is to extract all entities (nodes) and their hierarchy.

Rules:
- Output valid JSON only — no explanations, no markdown fencing
- Include every entity from the input
- Capture parent-child relationships (pools, lanes, subprocesses)
- Use this exact schema:
{{
  "entities": [
    {{
      "id": "string",
      "name": "string",
      "type": "string",
      "parent_id": "string or null"
    }}
  ]
}}

Input data:
{input_data}
"""


# ---------------------------------------------------------------------------
# Step 2 — Extract relationships (edges) from input + entities
# ---------------------------------------------------------------------------

STEP_2_PROMPT = """\
You are given a list of entities and the original dataset.
Your task is to extract all relationships (edges) between entities.

Rules:
- Output valid JSON only — no explanations, no markdown fencing
- Include every sequence flow, message flow, and association
- Do not invent relationships not present in the data
- Use this exact schema:
{{
  "relationships": [
    {{
      "id": "string",
      "source": "string",
      "target": "string",
      "type": "string",
      "label": "string or null"
    }}
  ]
}}

Entities extracted in the previous step:
{entities_json}

Original input data:
{input_data}
"""


# ---------------------------------------------------------------------------
# Step 3 — Render Mermaid diagram from entities + relationships
# ---------------------------------------------------------------------------

STEP_3_PROMPT = """\
You are given a set of entities and relationships extracted from a dataset.
Your task is to generate a Mermaid diagram that accurately represents them.

Rules:
- Output only valid Mermaid syntax
- Include all entities with correct hierarchy (subgraphs for pools/lanes/subprocesses)
- Include all relationships as edges
- Do not invent entities or relationships not provided
- Do not include explanations or markdown code fences
- Do not use relationship IDs as edge labels

Entities:
{entities_json}

Relationships:
{relationships_json}
"""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Steps 1 and 2 expect strict JSON. Providers' default Mermaid-only system
# prompt would mislead the model (smaller models follow it literally and
# return Mermaid for an extraction step). Step 3 generates the diagram and
# uses the provider's default system prompt (system_prompt=None at call site).
JSON_EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extraction assistant. "
    "Respond only with valid JSON matching the schema in the user message. "
    "Do not include any explanation, markdown fencing, or additional text."
)


# ---------------------------------------------------------------------------
# Retry budget — identical across all multi-step strategies
# ---------------------------------------------------------------------------

# Number of *additional* attempts after the first one (1 → up to 2 calls per step).
MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_fences(text: str | None) -> str | None:
    """
    Remove markdown code fences if present.
    Models often wrap JSON in ```json ... ``` despite the instruction not to.
    """
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove first line (```json or ```)
        stripped = stripped.split("\n", 1)[-1]
    if stripped.endswith("```"):
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def validate_step_payload(text: str | None, step_number: int) -> tuple[bool, str | None]:
    """
    Validate a step-1 or step-2 JSON payload against its expected schema shape.

    Returns (is_valid, error_message). On success, error_message is None.
    Step 3 is Mermaid (free-form text) and is not validated here.

    Schema rule:
    - Step 1 → top-level dict with key "entities" → list
    - Step 2 → top-level dict with key "relationships" → list

    Field-level shape (id, name, type, …) is intentionally NOT enforced —
    we want to capture model errors as data, not reject borderline outputs.
    """
    expected_key = "entities" if step_number == 1 else "relationships"
    try:
        parsed = json.loads(text or "")
    except (json.JSONDecodeError, TypeError) as e:
        return False, f"invalid JSON: {e}"

    if not isinstance(parsed, dict) or not isinstance(parsed.get(expected_key), list):
        return False, f"missing `{expected_key}` list"

    return True, None