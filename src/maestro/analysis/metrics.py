"""
MAESTRO — Metrics module
Compares generated Mermaid output against ground truth.
Evaluation dimensions:
  1. Structural validity (mmdc parse check)
  2. Entity precision/recall (exact ID, fuzzy name, lemmatized name)
  3. Relationship precision/recall (relaxed + strict)
  4. Error taxonomy counts
"""

import re
import subprocess
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from uuid import UUID

from maestro.schemas import MetricResult


# ---------------------------------------------------------------------------
# Mermaid parsing validation via mmdc CLI
# ---------------------------------------------------------------------------

def check_mermaid_valid(diagram_code: str) -> tuple[bool, str | None]:
    """
    Validate Mermaid syntax using mmdc CLI.
    Returns (is_valid, error_message_or_none).
    Requires: npm install -g @mermaid-js/mermaid-cli
    """
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        return (True, "mmdc not found — validation skipped")

    try:
        result = subprocess.run(
            [mmdc, "-i", "/dev/stdin", "-o", "/dev/null", "-e", "png"],
            input=diagram_code,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return (True, None)
        else:
            return (False, result.stderr.strip()[:500])
    except subprocess.TimeoutExpired:
        return (False, "mmdc timed out after 15s")
    except Exception as e:
        return (False, str(e)[:500])


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_label(label: str) -> str:
    """
    Basic normalization: lowercase, strip whitespace.
    Used for raw fuzzy matching — no linguistic processing.
    """
    return label.strip().lower()


def _lemmatize_label(label: str) -> str:
    """
    Normalize + lemmatize: lowercase, strip plurals, collapse separators.
    Catches 'Tasks' → 'task', 'start_event_1' → 'start event 1'.
    """
    text = label.strip().lower()
    # Replace underscores and hyphens with spaces
    text = re.sub(r"[_\-]", " ", text)
    # Strip trailing 's' for basic plural handling
    # (avoids nltk dependency for now — can upgrade later)
    words = text.split()
    lemmatized = []
    for w in words:
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        lemmatized.append(w)
    return " ".join(lemmatized)


# ---------------------------------------------------------------------------
# Mermaid text extraction — regex-based
# ---------------------------------------------------------------------------

def extract_nodes(mermaid_code: str) -> list[dict]:
    """
    Extract node definitions from Mermaid code.
    Returns list of {"id": str, "label": str}.
    """
    nodes = []
    seen_ids = set()

    # Node pattern: id followed by brackets with label
    pattern = r'^\s*(\w+)\s*[\[\(\{]+["\'/]?\s*"?([^"\]\)\}]+)"?\s*[\]\)\}]+'
    for line in mermaid_code.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            node_id = match.group(1)
            label = match.group(2).strip().strip('"').strip("'")
            if node_id.lower() not in (
                "graph", "flowchart", "subgraph", "end", "direction"
            ):
                if node_id not in seen_ids:
                    nodes.append({"id": node_id, "label": label})
                    seen_ids.add(node_id)

    # Subgraph definitions
    subgraph_pattern = r'subgraph\s+(\w+)\s*\["?([^"\]]*)"?\]'
    for match in re.finditer(subgraph_pattern, mermaid_code):
        sg_id = match.group(1)
        if sg_id not in seen_ids:
            nodes.append({"id": sg_id, "label": match.group(2).strip()})
            seen_ids.add(sg_id)

    return nodes


def extract_relationships(mermaid_code: str) -> list[dict]:
    """
    Extract relationship definitions from Mermaid code.
    Returns list of {"source": str, "target": str, "type": str}.
    Handles multiple arrow and label formats.
    """
    relationships = []

    patterns = [
        # Format: source -.->|label| target  OR  source -.-> target
        r'(\w+)\s+(-->|-.->)\s*(?:\|[^|]*\|)?\s*(\w+)',
        # Format: source -.label.-> target (inline dot-delimited label)
        r'(\w+)\s+-\..*?\.->?\s*(\w+)',
    ]

    seen = set()

    # Pattern 1: standard arrows with optional pipe labels
    for match in re.finditer(patterns[0], mermaid_code):
        source = match.group(1)
        arrow = match.group(2)
        target = match.group(3)
        rel_type = "message_flow" if "-." in arrow else "sequence_flow"
        key = (source, target)
        if key not in seen:
            relationships.append({
                "source": source,
                "target": target,
                "type": rel_type,
            })
            seen.add(key)

    # Pattern 2: inline label between dots  e.g. -.Message Flow 1.->
    for match in re.finditer(patterns[1], mermaid_code):
        source = match.group(1)
        target = match.group(2)
        key = (source, target)
        if key not in seen:
            relationships.append({
                "source": source,
                "target": target,
                "type": "message_flow",
            })
            seen.add(key)

    return relationships


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _f1(precision: float, recall: float) -> float:
    """Compute F1 score. Returns 0.0 if both inputs are 0."""
    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)


def _fuzzy_score(a: str, b: str) -> float:
    """String similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


FUZZY_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Entity metrics
# ---------------------------------------------------------------------------

def compute_entity_metrics_exact(
    output_nodes: list[dict], truth_nodes: list[dict]
) -> tuple[float, float, float]:
    """Exact ID match: precision, recall, F1."""
    output_ids = {n["id"] for n in output_nodes}
    truth_ids = {n["id"] for n in truth_nodes}

    if not output_ids:
        return (0.0, 0.0, 0.0)

    correct = output_ids & truth_ids
    precision = round(len(correct) / len(output_ids), 4) if output_ids else 0.0
    recall = round(len(correct) / len(truth_ids), 4) if truth_ids else 0.0
    return (precision, recall, _f1(precision, recall))


def _fuzzy_match(
    output_nodes: list[dict],
    truth_nodes: list[dict],
    normalizer,
) -> tuple[float, float, float]:
    """
    Fuzzy name matching with a configurable normalizer function.
    Used for both raw and lemmatized matching.
    """
    if not output_nodes or not truth_nodes:
        return (0.0, 0.0, 0.0)

    matched_truth = set()
    correct = 0

    for out_node in output_nodes:
        out_label = normalizer(out_node["label"])
        best_score = 0.0
        best_idx = None

        for i, truth_node in enumerate(truth_nodes):
            if i in matched_truth:
                continue
            truth_label = normalizer(truth_node["label"])
            score = _fuzzy_score(out_label, truth_label)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= FUZZY_THRESHOLD and best_idx is not None:
            correct += 1
            matched_truth.add(best_idx)

    precision = round(correct / len(output_nodes), 4)
    recall = round(correct / len(truth_nodes), 4)
    return (precision, recall, _f1(precision, recall))


def compute_entity_metrics_fuzzy(
    output_nodes: list[dict], truth_nodes: list[dict]
) -> tuple[float, float, float]:
    """Fuzzy name match with basic normalization (lowercase only)."""
    return _fuzzy_match(output_nodes, truth_nodes, _normalize_label)


def compute_entity_metrics_lemma(
    output_nodes: list[dict], truth_nodes: list[dict]
) -> tuple[float, float, float]:
    """Fuzzy name match with lemmatization (lowercase + strip plurals)."""
    return _fuzzy_match(output_nodes, truth_nodes, _lemmatize_label)


# ---------------------------------------------------------------------------
# Relationship metrics
# ---------------------------------------------------------------------------

def compute_relationship_metrics_relaxed(
    output_relationships: list[dict], truth_relationships: list[dict]
) -> tuple[float, float, float]:
    """Relaxed: match by (source, target) pair only, ignore type."""
    output_pairs = {(e["source"], e["target"]) for e in output_relationships}
    truth_pairs = {(e["source"], e["target"]) for e in truth_relationships}

    if not output_pairs:
        return (0.0, 0.0, 0.0)

    correct = output_pairs & truth_pairs
    precision = round(len(correct) / len(output_pairs), 4) if output_pairs else 0.0
    recall = round(len(correct) / len(truth_pairs), 4) if truth_pairs else 0.0
    return (precision, recall, _f1(precision, recall))


def compute_relationship_metrics_strict(
    output_relationships: list[dict], truth_relationships: list[dict]
) -> tuple[float, float, float]:
    """Strict: match by (source, target, type) — all three must match."""
    output_tuples = {(e["source"], e["target"], e["type"]) for e in output_relationships}
    truth_tuples = {(e["source"], e["target"], e["type"]) for e in truth_relationships}

    if not output_tuples:
        return (0.0, 0.0, 0.0)

    correct = output_tuples & truth_tuples
    precision = round(len(correct) / len(output_tuples), 4) if output_tuples else 0.0
    recall = round(len(correct) / len(truth_tuples), 4) if truth_tuples else 0.0
    return (precision, recall, _f1(precision, recall))


# ---------------------------------------------------------------------------
# Error taxonomy counts
# ---------------------------------------------------------------------------

def compute_entity_taxonomy(
    output_nodes: list[dict], truth_nodes: list[dict]
) -> dict:
    """
    Count entity-level errors by taxonomy category.
    Returns: {"missing": int, "extra": int, "false": int}
    """
    output_ids = {n["id"] for n in output_nodes}
    truth_ids = {n["id"] for n in truth_nodes}

    # Missing: in truth but not in output
    missing = len(truth_ids - output_ids)

    # Extra: in output but not in truth
    extra = len(output_ids - truth_ids)

    # False: ID matches but label is significantly different
    # (entity exists but represents something wrong)
    shared_ids = output_ids & truth_ids
    output_labels = {n["id"]: n["label"] for n in output_nodes}
    truth_labels = {n["id"]: n["label"] for n in truth_nodes}

    false_count = 0
    for nid in shared_ids:
        similarity = _fuzzy_score(
            _normalize_label(output_labels[nid]),
            _normalize_label(truth_labels[nid]),
        )
        if similarity < FUZZY_THRESHOLD:
            false_count += 1

    return {"missing": missing, "extra": extra, "false": false_count}


def compute_entity_taxonomy(
    output_nodes: list[dict], truth_nodes: list[dict]
) -> dict:
    """
    Count entity-level errors by taxonomy category.
    Returns: {"missing": int, "extra": int, "false": int, "duplicate": int}
    """
    output_ids = [n["id"] for n in output_nodes]
    truth_ids = {n["id"] for n in truth_nodes}

    # Duplicate: same ID appears more than once in output
    id_counts = Counter(output_ids)
    duplicate = sum(c - 1 for c in id_counts.values() if c > 1)

    output_ids_set = set(output_ids)

    # Missing: in truth but not in output
    missing = len(truth_ids - output_ids_set)

    # Extra: in output but not in truth
    extra = len(output_ids_set - truth_ids)

    # False: ID matches but label is significantly different
    shared_ids = output_ids_set & truth_ids
    output_labels = {n["id"]: n["label"] for n in output_nodes}
    truth_labels = {n["id"]: n["label"] for n in truth_nodes}

    false_count = 0
    for nid in shared_ids:
        similarity = _fuzzy_score(
            _normalize_label(output_labels[nid]),
            _normalize_label(truth_labels[nid]),
        )
        if similarity < FUZZY_THRESHOLD:
            false_count += 1

    return {
        "missing": missing,
        "extra": extra,
        "false": false_count,
        "duplicate": duplicate,
    }


def compute_relationship_taxonomy(
    output_relationships: list[dict], truth_relationships: list[dict]
) -> dict:
    """
    Count relationship-level errors by taxonomy category.
    Returns: {"missing": int, "extra": int, "false": int, "duplicate": int}
    """
    output_pairs = [(e["source"], e["target"]) for e in output_relationships]
    truth_pairs = {(e["source"], e["target"]) for e in truth_relationships}

    # Duplicate: same (source, target) pair appears more than once in output
    pair_counts = Counter(output_pairs)
    duplicate = sum(c - 1 for c in pair_counts.values() if c > 1)

    output_pairs_set = set(output_pairs)

    # Missing: in truth but not in output
    missing = len(truth_pairs - output_pairs_set)

    # Extra: in output but not in truth
    extra = len(output_pairs_set - truth_pairs)

    # False: (source, target) matches but type is different
    shared_pairs = output_pairs_set & truth_pairs
    output_types = {(e["source"], e["target"]): e["type"] for e in output_relationships}
    truth_types = {(e["source"], e["target"]): e["type"] for e in truth_relationships}

    false_count = 0
    for pair in shared_pairs:
        if output_types.get(pair) != truth_types.get(pair):
            false_count += 1

    return {
        "missing": missing,
        "extra": extra,
        "false": false_count,
        "duplicate": duplicate,
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_run(
    run_id: UUID,
    output_diagram_code: str,
    ground_truth_path: Path,
) -> MetricResult:
    """
    Full evaluation pipeline for one run.
    Compares generated diagram against ground truth file.
    """
    truth_code = ground_truth_path.read_text(encoding="utf-8")


    # 1. Structural validity
    parses_valid, parse_error = check_mermaid_valid(output_diagram_code)


    # 2. Extract nodes and relationships
    output_nodes = extract_nodes(output_diagram_code)
    truth_nodes = extract_nodes(truth_code)
    output_relationships = extract_relationships(output_diagram_code)
    truth_relationships = extract_relationships(truth_code)


    # 3. Entity metrics — three levels
    id_p, id_r, id_f1 = compute_entity_metrics_exact(output_nodes, truth_nodes)
    name_p, name_r, name_f1 = compute_entity_metrics_fuzzy(output_nodes, truth_nodes)
    lemma_p, lemma_r, lemma_f1 = compute_entity_metrics_lemma(output_nodes, truth_nodes)


    # 4. Relationship metrics — two levels
    rel_p, rel_r, rel_f1 = compute_relationship_metrics_relaxed(output_relationships, truth_relationships)
    str_p, str_r, str_f1 = compute_relationship_metrics_strict(output_relationships, truth_relationships)


    # 5. Duplicate detection
    def compute_entity_taxonomy(
            output_nodes: list[dict], truth_nodes: list[dict]
    ) -> dict:
        """
        Count entity-level errors by taxonomy category.
        Returns: {"missing": int, "extra": int, "false": int, "duplicate": int}
        """
        output_ids = [n["id"] for n in output_nodes]
        truth_ids = {n["id"] for n in truth_nodes}

        # Duplicate: same ID appears more than once in output
        id_counts = Counter(output_ids)
        duplicate = sum(c - 1 for c in id_counts.values() if c > 1)

        output_ids_set = set(output_ids)

        # Missing: in truth but not in output
        missing = len(truth_ids - output_ids_set)

        # Extra: in output but not in truth
        extra = len(output_ids_set - truth_ids)

        # False: ID matches but label is significantly different
        shared_ids = output_ids_set & truth_ids
        output_labels = {n["id"]: n["label"] for n in output_nodes}
        truth_labels = {n["id"]: n["label"] for n in truth_nodes}

        false_count = 0
        for nid in shared_ids:
            similarity = _fuzzy_score(
                _normalize_label(output_labels[nid]),
                _normalize_label(truth_labels[nid]),
            )
            if similarity < FUZZY_THRESHOLD:
                false_count += 1

        return {
            "missing": missing,
            "extra": extra,
            "false": false_count,
            "duplicate": duplicate,
        }

    # 5. Error taxonomy
    entity_tax = compute_entity_taxonomy(output_nodes, truth_nodes)
    relationship_tax = compute_relationship_taxonomy(output_relationships, truth_relationships)

    return MetricResult(
        run_id=run_id,
        parses_valid=parses_valid,
        parse_error=parse_error,
        entity_id_precision=id_p,
        entity_id_recall=id_r,
        entity_id_f1=id_f1,
        entity_name_precision=name_p,
        entity_name_recall=name_r,
        entity_name_f1=name_f1,
        entity_lemma_precision=lemma_p,
        entity_lemma_recall=lemma_r,
        entity_lemma_f1=lemma_f1,
        relationship_relaxed_precision=rel_p,
        relationship_relaxed_recall=rel_r,
        relationship_relaxed_f1=rel_f1,
        relationship_strict_precision=str_p,
        relationship_strict_recall=str_r,
        relationship_strict_f1=str_f1,
        entities_in_output=len(output_nodes),
        entities_in_truth=len(truth_nodes),
        relationships_in_output=len(output_relationships),
        relationships_in_truth=len(truth_relationships),
        missing_entities=entity_tax["missing"],
        extra_entities=entity_tax["extra"],
        false_entities=entity_tax["false"],
        duplicate_entities=entity_tax["duplicate"],
        missing_relationships=relationship_tax["missing"],
        extra_relationships=relationship_tax["extra"],
        false_relationships=relationship_tax["false"],
        duplicate_relationships=relationship_tax["duplicate"],
    )