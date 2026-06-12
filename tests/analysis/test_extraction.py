"""
MAESTRO — Mermaid extraction unit tests (Phase 3a).

Pure-function tests for the rewritten extractors in ``metrics.py``. They pin the
four bug-fixes from the scoring-pipeline audit and the entity/container/
relationship/attachment split defined by the scoring contract:

  A1  ``<-->`` bidirectional edges are captured (one undirected pair).
  A2  empty-label nodes (``gw{""}``) are captured, not silently dropped.
  A3  phantom nodes from edge-label / multi-line-label / comment text are NOT
      produced.
  A4  ``o--o`` attachment edges are excluded from relationships and surfaced by
      ``extract_attachments`` instead.

Entities = inline nodes; containers = ``subgraph`` headers. These are unit tests
on the extractors only — no DB, no pydantic, no mmdc.
"""

from __future__ import annotations

from maestro.analysis.metrics import (
    compute_attachment_metrics,
    compute_container_metrics,
    extract_attachments,
    extract_containers,
    extract_nodes,
    extract_relationships,
)


def _ids(nodes):
    return {n["id"] for n in nodes}


def _pairs(rels):
    return {(r["source"], r["target"]) for r in rels}


# ---------------------------------------------------------------------------
# Entity / container split
# ---------------------------------------------------------------------------


def test_subgraph_is_container_not_entity():
    code = """flowchart LR
    subgraph pool_a["Team A"]
        task_1["Do Thing"]
    end
    task_2["Other Thing"]
    """
    assert _ids(extract_nodes(code)) == {"task_1", "task_2"}
    assert _ids(extract_containers(code)) == {"pool_a"}


def test_collapsed_subprocess_inline_is_entity():
    # A collapsed sub-process renders inline [[ ]] -> entity (per contract B1).
    code = 'flowchart LR\n    sub_x[["Collapsed"]]\n    a["A"]\n    a --> sub_x\n'
    assert "sub_x" in _ids(extract_nodes(code))
    assert _ids(extract_containers(code)) == set()


# ---------------------------------------------------------------------------
# A2 — empty / whitespace labels
# ---------------------------------------------------------------------------


def test_empty_label_node_extracted():
    code = 'flowchart LR\n    gw{""}\n    ev([""])\n    a["A"]\n    a --> gw\n'
    ids = _ids(extract_nodes(code))
    assert {"gw", "ev", "a"} <= ids


def test_whitespace_label_node_extracted():
    code = 'flowchart LR\n    gw{" "}\n    a["A"]\n'
    assert "gw" in _ids(extract_nodes(code))


# ---------------------------------------------------------------------------
# A3 — phantom suppression
# ---------------------------------------------------------------------------


def test_no_phantom_from_pipe_edge_label():
    # "Green (no risk)" inside an edge label must not become a node "Green".
    code = (
        'flowchart LR\n'
        '    gw{"Risk?"}\n'
        '    deliver["Deliver"]\n'
        '    gw -->|"Green (no risk)"| deliver\n'
    )
    ids = _ids(extract_nodes(code))
    assert ids == {"gw", "deliver"}
    assert "Green" not in ids


def test_no_phantom_from_multiline_bracketed_label():
    # A quoted label containing [Device] and (WiFi) must be consumed whole.
    code = (
        'flowchart LR\n'
        '    user_clients["User Clients\\n[Device]\\nLaptops (WiFi)"]\n'
    )
    ids = _ids(extract_nodes(code))
    assert ids == {"user_clients"}
    assert "nLaptops" not in ids and "Laptops" not in ids


def test_comment_lines_ignored():
    code = (
        'flowchart LR\n'
        '    %% Fraud path (expanded) with routing\n'
        '    a["A"]\n'
        '    b["B"]\n'
        '    a --> b\n'
    )
    ids = _ids(extract_nodes(code))
    assert ids == {"a", "b"}
    assert "routing" not in ids and "expanded" not in ids


def test_inline_on_edge_node_is_extracted():
    # A node defined inline on an edge line must still be captured.
    code = 'flowchart LR\n    host["Host"]\n    host o--o evt(("Boundary"))\n'
    assert {"host", "evt"} <= _ids(extract_nodes(code))


# ---------------------------------------------------------------------------
# A1 — bidirectional edges
# ---------------------------------------------------------------------------


def test_bidirectional_edge_is_one_undirected_pair():
    code = 'flowchart LR\n    a["A"]\n    b["B"]\n    a <-->|"IPsec"| b\n'
    pairs = _pairs(extract_relationships(code))
    # canonicalised (sorted) — exactly one pair, orientation-independent
    assert pairs == {("a", "b")}


def test_bidirectional_dotted_edge_is_message_flow():
    code = 'flowchart LR\n    a["A"]\n    b["B"]\n    a <-.-> b\n'
    rels = extract_relationships(code)
    assert len(rels) == 1
    assert rels[0]["type"] == "message_flow"
    assert (rels[0]["source"], rels[0]["target"]) == ("a", "b")


# ---------------------------------------------------------------------------
# A4 — o--o excluded from relationships, surfaced as attachments
# ---------------------------------------------------------------------------


def test_o_o_attachment_excluded_from_relationships():
    code = (
        'flowchart LR\n'
        '    host["Host"]\n'
        '    evt(("Boundary"))\n'
        '    nxt["Next"]\n'
        '    host o--o evt\n'      # attachment, NOT a relationship
        '    evt --> nxt\n'        # real outgoing sequence flow
    )
    pairs = _pairs(extract_relationships(code))
    assert ("host", "evt") not in pairs and ("evt", "host") not in pairs
    assert ("evt", "nxt") in pairs


def test_extract_attachments_is_undirected_and_deduped():
    code = (
        'flowchart LR\n'
        '    host["Host"]\n'
        '    evt(("Boundary"))\n'
        '    host o--o evt\n'
    )
    atts = extract_attachments(code)
    assert len(atts) == 1
    assert tuple(sorted((atts[0]["a"], atts[0]["b"]))) == ("evt", "host")


def test_message_flow_dotted_arrow():
    code = 'flowchart LR\n    a["A"]\n    b["B"]\n    a -.-> b\n'
    rels = extract_relationships(code)
    assert rels[0]["type"] == "message_flow"


def test_sequence_flow_solid_arrow():
    code = 'flowchart LR\n    a["A"]\n    b["B"]\n    a --> b\n'
    rels = extract_relationships(code)
    assert rels[0]["type"] == "sequence_flow"


# ---------------------------------------------------------------------------
# 3b — container metrics (reuse entity matchers)
# ---------------------------------------------------------------------------


def test_container_metrics_none_when_no_truth_containers():
    assert compute_container_metrics([], []) is None
    assert compute_container_metrics([{"id": "x", "label": "X"}], []) is None


def test_container_metrics_perfect_match():
    truth = [{"id": "pool_a", "label": "Team A"}]
    result = compute_container_metrics(truth, truth)
    assert result is not None
    id_p, id_r, id_f1, nm_p, nm_r, nm_f1 = result
    assert id_f1 == 1.0 and nm_f1 == 1.0


def test_container_metrics_partial_recall():
    truth = [{"id": "p1", "label": "A"}, {"id": "p2", "label": "B"}]
    out = [{"id": "p1", "label": "A"}]
    _, id_r, _, _, _, _ = compute_container_metrics(out, truth)
    assert id_r == 0.5


# ---------------------------------------------------------------------------
# 3b — attachment metrics (undirected pairs)
# ---------------------------------------------------------------------------


def test_attachment_metrics_none_when_no_truth_attachments():
    assert compute_attachment_metrics([], []) is None
    assert compute_attachment_metrics([{"a": "x", "b": "y"}], []) is None


def test_attachment_metrics_perfect_match():
    truth = [{"a": "host", "b": "evt"}]
    p, r, f1 = compute_attachment_metrics(truth, truth)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_attachment_metrics_orientation_insensitive():
    truth = [{"a": "host", "b": "evt"}]
    out = [{"a": "evt", "b": "host"}]  # reversed
    p, r, f1 = compute_attachment_metrics(out, truth)
    assert f1 == 1.0


def test_attachment_metrics_partial_and_spurious():
    truth = [{"a": "h1", "b": "e1"}, {"a": "h2", "b": "e2"}]
    out = [{"a": "h1", "b": "e1"}, {"a": "h9", "b": "e9"}]  # 1 correct, 1 spurious
    p, r, f1 = compute_attachment_metrics(out, truth)
    assert p == 0.5 and r == 0.5
