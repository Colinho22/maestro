"""
MAESTRO Metrics module
Compares generated Mermaid output against ground truth.
Evaluation dimensions:
  1. Structural validity (mmdc parse check)
  2. Entity precision/recall (exact ID, fuzzy name, lemmatized name)
  3. Relationship precision/recall (relaxed + strict)
  4. Error taxonomy counts
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from uuid import UUID

from maestro.schemas import MetricResult

# ---------------------------------------------------------------------------
# Mermaid parsing validation via mmdc CLI
# ---------------------------------------------------------------------------


def check_mermaid_valid(diagram_code: str) -> tuple[bool | None, str | None]:
    """
    Validate Mermaid syntax using mmdc CLI.
    Returns (is_valid, error_message_or_none).
    Returns (None, skip_message) when mmdc is not installed.
    Requires: npm install -g @mermaid-js/mermaid-cli

    Why the temp-file dance: mmdc *renders* its input to an output file
    and requires the path to end with a known extension (.md, .svg,
    .png, .pdf). Passing ``-o /dev/null`` worked-around the rendering
    only on systems where mmdc happened not to validate the suffix; the
    current SDK rejects it with "Output file must end with...". We hand
    it a real temp PNG path, throw the file away, and only inspect the
    return code, the validity signal we actually want.

    Both the input and output use explicit temp files from ``mkstemp`` so the
    function is cross-platform: ``/dev/stdin`` does not exist on Windows, and
    ``NamedTemporaryFile`` holds an exclusive lock there that would block mmdc
    from writing the output. The descriptors are closed immediately (mmdc
    opens the paths itself), the diagram is written via ``Path.write_text``,
    and both files are removed in a ``finally`` block.
    """
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        return (None, "mmdc not found: validation skipped")

    # mmdc renders via Puppeteer/Chromium. In a container running as root,
    # Chromium refuses to start without --no-sandbox, which mmdc only picks up
    # from a config file passed with -p (it does NOT read a PUPPETEER_* env).
    # MERMAID_PUPPETEER_CONFIG points at that file when set (see the Docker
    # image); locally it is unset and mmdc uses its working default.
    puppeteer_args: list[str] = []
    puppeteer_config = os.environ.get("MERMAID_PUPPETEER_CONFIG")
    if puppeteer_config and Path(puppeteer_config).is_file():
        puppeteer_args = ["-p", puppeteer_config]

    # Initialised before the try so the finally cleanup is safe even if the
    # mkstemp calls below raise (e.g. no temp dir / disk full).
    in_path: str | None = None
    out_path: str | None = None
    try:
        # mkstemp creates each file and returns (fd, path). Close the fds at
        # once: we write the input via Path.write_text and mmdc opens both
        # paths itself, which sidesteps Windows' exclusive-lock behaviour.
        in_fd, in_path = tempfile.mkstemp(suffix=".mmd")
        out_fd, out_path = tempfile.mkstemp(suffix=".png")
        os.close(in_fd)
        os.close(out_fd)
        Path(in_path).write_text(diagram_code, encoding="utf-8")
        result = subprocess.run(
            [mmdc, *puppeteer_args, "-i", in_path, "-o", out_path, "-e", "png"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return (True, None)
        return (False, result.stderr.strip()[:500])
    except subprocess.TimeoutExpired:
        return (False, "mmdc timed out after 15s")
    except Exception as e:
        return (False, str(e)[:500])
    finally:
        # Best-effort cleanup; teardown errors must not mask the result. Paths
        # may be None if mkstemp itself failed.
        for p in (in_path, out_path):
            if p is None:
                continue
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def extract_input_unnamed_ids(input_path: Path | None) -> set[str]:
    """
    Ids of input elements with an empty ``name`` (e.g. BPMN gateways/events the
    source leaves unnamed). Their ground-truth label is authoring convention the
    input does not provide, so the entity-name metric scores them by id only.

    Returns an empty set when no input path is given or the input cannot be read
    or parsed, so scoring degrades to the strict label comparison rather than
    crashing (observability fails soft).
    """
    if input_path is None:
        return set()
    try:
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    elements = data.get("elements") or data.get("nodes") or []
    if not isinstance(elements, list):
        return set()
    return {
        e["id"]
        for e in elements
        if isinstance(e, dict) and e.get("id") and not (e.get("name") or "").strip()
    }


def _label_core(label: str) -> str:
    r"""
    Keep the scored part of a multi-line node label: the name and the bracketed
    ``[Type]`` line, dropping any trailing descriptor line.

    Labels are ``name`` (BPMN), ``name\n[Type]`` or ``name\n[Type]\ndescriptor``
    (C4 / network), with ``\n`` as a literal two-character separator. The third
    descriptor line is authored inconsistently in the ground truth (network
    topology includes it for some nodes and not others, though the input always
    carries the field), so no model can predict it. Scoring it would penalise a
    correct name and type for an unpredictable authoring choice, so the entity
    name metric compares on name + type only. The descriptor is out of the
    scored contract by design; this is applied to output and truth identically.
    """
    parts = label.split("\\n")
    kept = [parts[0]]
    for p in parts[1:]:
        if p.strip().startswith("["):  # the [Type] line; descriptor follows
            kept.append(p)
            break
    return "\\n".join(kept)


def _normalize_label(label: str) -> str:
    """
    Basic normalization: drop the descriptor line, lowercase, strip whitespace.
    Used for raw fuzzy matching: no linguistic processing.
    """
    return _label_core(label).strip().lower()


def _lemmatize_label(label: str) -> str:
    """
    Normalize + lemmatize: drop the descriptor line, lowercase, strip plurals,
    collapse separators. Catches 'Tasks' -> 'task', 'start_event_1' -> 'start
    event 1'.
    """
    text = _label_core(label).strip().lower()
    # Replace underscores and hyphens with spaces
    text = re.sub(r"[_\-]", " ", text)
    # Strip trailing 's' for basic plural handling
    # (avoids nltk dependency for now: can upgrade later)
    words = text.split()
    lemmatized = []
    for w in words:
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        lemmatized.append(w)
    return " ".join(lemmatized)


# ---------------------------------------------------------------------------
# Mermaid text extraction: regex-based
# ---------------------------------------------------------------------------


# Mermaid keywords that are syntax, never node ids.
_SKIP = {
    "graph",
    "flowchart",
    "subgraph",
    "end",
    "direction",
    "style",
    "classdef",
    "class",
    "linkstyle",
    "click",
}

# A node definition: an id, an opening shape bracket, a label that is EITHER a
# quoted string (consumed whole, so brackets/newlines INSIDE a label such as
# "Web App\n[Device]\nLaptops (WiFi)" cannot spawn phantom nodes) or unquoted
# text up to the closing bracket, then the closing bracket(s). An empty label
# ("" or '') is allowed, so nodes like gw{""} are still captured.
_NODE_DEF = re.compile(
    r"(\w+)\s*"  # 1: node id
    r"[\[\(\{]+"  # opening bracket(s): [ ( { ([ [[ (( {{ [( {{
    r'(?:"([^"]*)"|\'([^\']*)\'|([^"\'\]\)\}|]*?))'  # 2/3/4: quoted or unquoted label
    r"\s*[\]\)\}]+"  # closing bracket(s)
)


def _strip_inline_labels(line: str) -> str:
    """
    Replace every inline node definition (``id["Label"]``) with its bare ``id``.

    Edge lines may redeclare a node's label on one or both endpoints, e.g.
    ``a["A"] --> b["B"]``. The edge regexes expect the id to sit directly
    against the operator, so a labelled *source* would otherwise break edge
    extraction. Collapsing each ``id[...]`` to ``id`` leaves a clean
    ``a --> b`` for the operator scan; node labels are captured separately by
    ``extract_nodes`` so nothing is lost here.
    """
    return _NODE_DEF.sub(lambda m: m.group(1), line)


# Edge label between pipes, e.g. -->|"Green (no risk)"|: stripped before node
# scanning so its text is never mistaken for a node definition.
_PIPE_LABEL = re.compile(r"\|[^|]*\|")

# Subgraph (container) header: subgraph id  OR  subgraph id["Label"]
_SUBGRAPH = re.compile(
    r'^\s*subgraph\s+(\w+)\s*(?:\[\s*"?([^"\]]*)"?\s*\])?', re.MULTILINE
)

# Edge operators. Order in the alternation matters (longest / bidirectional
# first). o--o / --o / --x are association/attachment edges, NOT flow edges.
_EDGE = re.compile(
    r"(\w+)\s*"
    r"(<-\.->|<-->|-\.->|o--o|--o|--x|-->)"
    r"\s*(?:\|[^|]*\|)?\s*"
    r"(\w+)"
)
# Inline dot-delimited label form: source -. some text .-> target  (message flow)
_EDGE_DOTLABEL = re.compile(r"(\w+)\s+-\.[^.|>]*\.->\s*(\w+)")
# Attachment / association edge: host o--o event  (undirected, o-ended)
_ATTACH = re.compile(r"(\w+)\s*o--o\s*(?:\|[^|]*\|)?\s*(\w+)")


def _iter_node_defs(mermaid_code: str):
    """
    Yield (id, label) for every node definition. Robust to labels that contain
    brackets/newlines and to nodes defined inline on an edge line (e.g.
    ``host o--o evt(("Label"))``). Skips comment lines and edge-label text.
    """
    for raw in mermaid_code.splitlines():
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        # remove |edge labels| so their words can't be read as node defs
        line = _PIPE_LABEL.sub(" ", line)
        for m in _NODE_DEF.finditer(line):
            nid = m.group(1)
            if nid.lower() in _SKIP:
                continue
            label = (m.group(2) or m.group(3) or m.group(4) or "").strip()
            yield nid, label


def extract_containers(mermaid_code: str) -> list[dict]:
    """
    Extract subgraph containers (pools / lanes / boundaries / expanded
    sub-processes). Scored as a separate dimension from entities.
    Returns list of {"id": str, "label": str}.
    """
    containers = []
    seen = set()
    for m in _SUBGRAPH.finditer(mermaid_code):
        cid = m.group(1)
        if cid not in seen:
            containers.append({"id": cid, "label": (m.group(2) or "").strip()})
            seen.add(cid)
    return containers


def extract_nodes(mermaid_code: str) -> list[dict]:
    """
    Extract ENTITY definitions (inline nodes) from Mermaid code.
    Returns list of {"id": str, "label": str}.

    Per the scoring contract, an entity is a node drawn inline; a node drawn as
    a ``subgraph`` is a container (see ``extract_containers``) and is excluded
    here so it does not inflate the entity metric or the complexity tiers.
    """
    container_ids = {c["id"] for c in extract_containers(mermaid_code)}
    nodes = []
    seen = set()
    for nid, label in _iter_node_defs(mermaid_code):
        if nid in container_ids or nid in seen:
            continue
        nodes.append({"id": nid, "label": label})
        seen.add(nid)
    return nodes


def extract_relationships(mermaid_code: str) -> list[dict]:
    """
    Extract flow relationships from Mermaid code.
    Returns list of {"source": str, "target": str, "type": str}.

    Rules:
      - ``-->``   directed sequence_flow.
      - ``-.->``  directed message_flow (dotted), also ``-. label .->``.
      - ``<-->`` / ``<-.->``  one UNDIRECTED relationship: endpoints are
        canonicalised (sorted) so orientation does not matter when matching.
      - ``o--o`` / ``--o`` / ``--x``  attachment / association edges: NOT
        relationships; excluded here and scored via ``extract_attachments``.
    """
    relationships = []
    seen = set()

    def _add(src: str, tgt: str, rel_type: str, undirected: bool = False) -> None:
        if undirected:
            src, tgt = sorted((src, tgt))
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            relationships.append({"source": src, "target": tgt, "type": rel_type})

    for raw in mermaid_code.splitlines():
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        # Collapse any inline node labels (``a["A"] --> b["B"]``) to bare ids
        # so a labelled source endpoint can't hide the edge from the operator
        # scan. Node labels themselves are captured by ``extract_nodes``.
        line = _strip_inline_labels(line)
        for m in _EDGE.finditer(line):
            src, op, tgt = m.group(1), m.group(2), m.group(3)
            if op in ("o--o", "--o", "--x"):
                continue  # attachment / association, not a flow relationship
            undirected = op.startswith("<")
            dotted = "." in op
            _add(src, tgt, "message_flow" if dotted else "sequence_flow", undirected)
        for m in _EDGE_DOTLABEL.finditer(line):
            _add(m.group(1), m.group(2), "message_flow")

    return relationships


def extract_attachments(mermaid_code: str) -> list[dict]:
    """
    Extract attachment / compensation-association edges (``host o--o event``).
    Undirected: endpoints are canonicalised (sorted). Scored as its own
    dimension, separate from flow relationships.
    Returns list of {"a": str, "b": str}.
    """
    attachments = []
    seen = set()
    for raw in mermaid_code.splitlines():
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        # Same inline-label collapse as extract_relationships: an attachment
        # written ``task["T"] o--o evt(("E"))`` must still match _ATTACH.
        line = _strip_inline_labels(line)
        for m in _ATTACH.finditer(line):
            a, b = sorted((m.group(1), m.group(2)))
            if (a, b) not in seen:
                seen.add((a, b))
                attachments.append({"a": a, "b": b})
    return attachments


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
    precision = round(len(correct) / len(output_ids), 4)
    recall = round(len(correct) / len(truth_ids), 4) if truth_ids else 0.0
    return (precision, recall, _f1(precision, recall))


def _fuzzy_match(
    output_nodes: list[dict],
    truth_nodes: list[dict],
    normalizer,
    input_unnamed_ids: set[str] | None = None,
) -> tuple[float, float, float]:
    """
    Fuzzy name matching with a configurable normalizer function.
    Used for both raw and lemmatized matching.

    ``input_unnamed_ids`` are node ids the *input* left without a name (e.g. a
    BPMN gateway with name ""). The ground truth labels these from convention
    (type name, unicode symbols, split/join) that the input does not provide, so
    the model cannot derive the label. For such a node, an id match counts as a
    name match regardless of the produced label, including a blank one. This is
    conditional on the input: a node the input *did* name is still scored on its
    label, so a model that blanks a nameable node is still penalised.
    """
    if not output_nodes or not truth_nodes:
        return (0.0, 0.0, 0.0)

    unnamed = input_unnamed_ids or set()
    truth_ids = {t["id"]: i for i, t in enumerate(truth_nodes)}
    matched_truth = set()
    correct = 0

    for out_node in output_nodes:
        # Input-unnamed node: an id match is a name match, label not scored.
        if out_node["id"] in unnamed and out_node["id"] in truth_ids:
            idx = truth_ids[out_node["id"]]
            if idx not in matched_truth:
                correct += 1
                matched_truth.add(idx)
                continue

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
    output_nodes: list[dict],
    truth_nodes: list[dict],
    input_unnamed_ids: set[str] | None = None,
) -> tuple[float, float, float]:
    """Fuzzy name match with basic normalization (lowercase only)."""
    return _fuzzy_match(output_nodes, truth_nodes, _normalize_label, input_unnamed_ids)


def compute_entity_metrics_lemma(
    output_nodes: list[dict],
    truth_nodes: list[dict],
    input_unnamed_ids: set[str] | None = None,
) -> tuple[float, float, float]:
    """Fuzzy name match with lemmatization (lowercase + strip plurals)."""
    return _fuzzy_match(output_nodes, truth_nodes, _lemmatize_label, input_unnamed_ids)


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
    """Strict: match by (source, target, type), all three must match."""
    output_tuples = {
        (e["source"], e["target"], e["type"]) for e in output_relationships
    }
    truth_tuples = {(e["source"], e["target"], e["type"]) for e in truth_relationships}

    if not output_tuples:
        return (0.0, 0.0, 0.0)

    correct = output_tuples & truth_tuples
    precision = round(len(correct) / len(output_tuples), 4) if output_tuples else 0.0
    recall = round(len(correct) / len(truth_tuples), 4) if truth_tuples else 0.0
    return (precision, recall, _f1(precision, recall))


# ---------------------------------------------------------------------------
# Container metrics (subgraphs: pools / lanes / boundaries / expanded subprocs)
# ---------------------------------------------------------------------------


def compute_container_metrics(
    output_containers: list[dict], truth_containers: list[dict]
) -> tuple | None:
    """
    Score containers as a separate dimension. Returns
    (id_p, id_r, id_f1, name_p, name_r, name_f1) or ``None`` when the ground
    truth has no containers (metric not applicable for this diagram).

    Reuses the entity matchers: containers are {"id", "label"} dicts, so exact
    ID and fuzzy name matching apply unchanged.
    """
    if not truth_containers:
        return None
    id_p, id_r, id_f1 = compute_entity_metrics_exact(
        output_containers, truth_containers
    )
    nm_p, nm_r, nm_f1 = compute_entity_metrics_fuzzy(
        output_containers, truth_containers
    )
    return (id_p, id_r, id_f1, nm_p, nm_r, nm_f1)


# ---------------------------------------------------------------------------
# Attachment metrics (o--o edges: boundary attachments + compensation assocs)
# ---------------------------------------------------------------------------


def compute_attachment_metrics(
    output_attachments: list[dict], truth_attachments: list[dict]
) -> tuple | None:
    """
    Score attachment edges as undirected pairs. Returns (precision, recall, f1)
    or ``None`` when the ground truth has no attachments (metric N/A).
    """
    truth_pairs = {tuple(sorted((a["a"], a["b"]))) for a in truth_attachments}
    if not truth_pairs:
        return None
    output_pairs = {tuple(sorted((a["a"], a["b"]))) for a in output_attachments}
    correct = len(output_pairs & truth_pairs)
    precision = round(correct / len(output_pairs), 4) if output_pairs else 0.0
    recall = round(correct / len(truth_pairs), 4)
    return (precision, recall, _f1(precision, recall))


# ---------------------------------------------------------------------------
# Error taxonomy counts
# ---------------------------------------------------------------------------


def compute_entity_taxonomy(
    output_nodes: list[dict],
    truth_nodes: list[dict],
    input_unnamed_ids: set[str] | None = None,
) -> dict:
    """
    Count entity-level errors by taxonomy category.
    Returns: {"missing": int, "extra": int, "false": int, "duplicate": int}

    A "false" entity is an id match with a mismatched label. Nodes the input
    left unnamed (``input_unnamed_ids``) are exempt: their ground-truth label is
    convention the input does not provide, so a label mismatch there is not a
    model error (see ``_fuzzy_match``).
    """
    unnamed = input_unnamed_ids or set()
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
        if nid in unnamed:
            continue  # input gave no name; label is not the model's to get right
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
    input_path: Path | None = None,
) -> MetricResult:
    """
    Full evaluation pipeline for one run.
    Compares generated diagram against ground truth file.

    ``input_path`` is the source JSON. When given, elements it leaves unnamed
    are scored by id only for the entity-name metric, since their ground-truth
    label is convention the input does not supply. Optional and backward
    compatible: omitted means strict label scoring for every node.
    """
    try:
        truth_code = ground_truth_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Return zeroed metrics with error instead of crashing the runner
        return MetricResult(
            run_id=run_id,
            parses_valid=None,
            parse_error=f"Ground truth file not found: {ground_truth_path}",
            entity_id_precision=0.0,
            entity_id_recall=0.0,
            entity_id_f1=0.0,
            entity_name_precision=0.0,
            entity_name_recall=0.0,
            entity_name_f1=0.0,
            entity_lemma_precision=0.0,
            entity_lemma_recall=0.0,
            entity_lemma_f1=0.0,
            relationship_relaxed_precision=0.0,
            relationship_relaxed_recall=0.0,
            relationship_relaxed_f1=0.0,
            relationship_strict_precision=0.0,
            relationship_strict_recall=0.0,
            relationship_strict_f1=0.0,
            entities_in_output=0,
            entities_in_truth=0,
            relationships_in_output=0,
            relationships_in_truth=0,
            missing_entities=0,
            extra_entities=0,
            false_entities=0,
            duplicate_entities=0,
            missing_relationships=0,
            extra_relationships=0,
            false_relationships=0,
            duplicate_relationships=0,
        )

    # 1. Structural validity
    parses_valid, parse_error = check_mermaid_valid(output_diagram_code)

    # 2. Extract nodes, containers, relationships, attachments
    output_nodes = extract_nodes(output_diagram_code)
    truth_nodes = extract_nodes(truth_code)
    output_containers = extract_containers(output_diagram_code)
    truth_containers = extract_containers(truth_code)
    output_relationships = extract_relationships(output_diagram_code)
    truth_relationships = extract_relationships(truth_code)
    output_attachments = extract_attachments(output_diagram_code)
    truth_attachments = extract_attachments(truth_code)

    # Ids the input left unnamed: their label is GT convention, not derivable,
    # so the name/lemma metrics and the false-entity count score them by id.
    unnamed_ids = extract_input_unnamed_ids(input_path)

    # 3. Entity metrics: three levels
    id_p, id_r, id_f1 = compute_entity_metrics_exact(output_nodes, truth_nodes)
    name_p, name_r, name_f1 = compute_entity_metrics_fuzzy(
        output_nodes, truth_nodes, unnamed_ids
    )
    lemma_p, lemma_r, lemma_f1 = compute_entity_metrics_lemma(
        output_nodes, truth_nodes, unnamed_ids
    )

    # 4. Relationship metrics: two levels
    rel_p, rel_r, rel_f1 = compute_relationship_metrics_relaxed(
        output_relationships, truth_relationships
    )
    str_p, str_r, str_f1 = compute_relationship_metrics_strict(
        output_relationships, truth_relationships
    )

    # 5. Error taxonomy
    entity_tax = compute_entity_taxonomy(output_nodes, truth_nodes, unnamed_ids)
    relationship_tax = compute_relationship_taxonomy(
        output_relationships, truth_relationships
    )

    # 6. Container + attachment dimensions (None when truth has none -> N/A)
    container = compute_container_metrics(output_containers, truth_containers)
    c_id_p, c_id_r, c_id_f1, c_nm_p, c_nm_r, c_nm_f1 = (
        container if container is not None else (None,) * 6
    )
    attach = compute_attachment_metrics(output_attachments, truth_attachments)
    a_p, a_r, a_f1 = attach if attach is not None else (None, None, None)

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
        container_id_precision=c_id_p,
        container_id_recall=c_id_r,
        container_id_f1=c_id_f1,
        container_name_precision=c_nm_p,
        container_name_recall=c_nm_r,
        container_name_f1=c_nm_f1,
        containers_in_output=len(output_containers),
        containers_in_truth=len(truth_containers),
        attachment_precision=a_p,
        attachment_recall=a_r,
        attachment_f1=a_f1,
        attachments_in_output=len(output_attachments),
        attachments_in_truth=len(truth_attachments),
    )
