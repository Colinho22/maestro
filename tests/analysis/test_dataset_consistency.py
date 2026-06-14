"""
MAESTRO: Dataset JSON <-> ground-truth MMD consistency (Phase 5).

For every registered input this asserts that the structured JSON (the model's
INPUT) and the reference Mermaid (the expected OUTPUT) agree on:

  * entities      : JSON leaf nodes/elements  == MMD inline nodes
  * containers    : JSON-derived groupings     == MMD subgraphs
  * flows         : JSON sequence/message/rels  == MMD flow edges (undirected)
  * attachments   : JSON attached_to + compensation == MMD ``o--o`` edges
  * metadata      : entity_count / container_count / attachment_count match

Why this matters beyond data hygiene: container-ness is *derived from the
JSON's own containment fields* (``lane`` / ``pool`` / ``parent_subprocess`` for
BPMN, ``boundary`` for IT) plus the rule "a grouping is drawn iff something
nests inside it". If this test passes, every container in the expected output is
inferable from the input, i.e. the benchmark never asks a model to produce
structure the input didn't specify. A failure means either a ground-truth bug
or an under-specified input, both of which would silently penalise models.

Conventions encoded here (the scoring contract):
  * A pool's *sole* lane is subsumed by the pool and not drawn (a lane is a
    container only if its pool has more than one lane).
  * The outermost ``system_boundary`` is always drawn; it is a derived
    container because its top-level zones reference it via ``boundary``.
  * ``<-->`` and ``o--o`` are compared as undirected pairs.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from maestro.analysis.metrics import (
    extract_attachments,
    extract_containers,
    extract_nodes,
    extract_relationships,
)
from maestro.experiment_config import INPUTS


def _json_truth(d: dict) -> tuple[set, set, set, set]:
    """Derive (entities, containers, flow_pairs, attachment_pairs) from the JSON."""
    dt = d["metadata"]["diagram_type"]
    if dt.startswith("bpmn"):
        node_ids = {n["id"] for n in d["nodes"]}
        pool_ids = {p["id"] for p in d.get("participants", d.get("pools", []))}
        lane_ids = {lane["id"] for lane in d.get("lanes", [])}
        parents: set = set()
        for n in d["nodes"]:
            for field in ("lane", "pool", "parent_subprocess"):
                if n.get(field):
                    parents.add(n[field])
        for lane in d.get("lanes", []):
            if lane.get("pool"):
                parents.add(lane["pool"])
        # A pool's sole lane is not drawn (subsumed by the pool).
        lanes_by_pool: dict = defaultdict(list)
        for lane in d.get("lanes", []):
            lanes_by_pool[lane.get("pool")].append(lane["id"])
        sole_lanes = {ls[0] for ls in lanes_by_pool.values() if len(ls) == 1}
        containers = (parents & (pool_ids | lane_ids | node_ids)) - sole_lanes
        entities = node_ids - containers
        rel = {
            tuple(sorted((f["source"], f["target"])))
            for f in d.get("sequence_flows", []) + d.get("message_flows", [])
        }
        att = {
            tuple(sorted((n["id"], n["attached_to"])))
            for n in d["nodes"]
            if n.get("attached_to")
        }
        att |= {
            tuple(sorted((c["source"], c["target"])))
            for c in d.get("compensation_associations", [])
        }
    else:  # c4_container / network_topology
        elem_ids = {e["id"] for e in d["elements"]}
        # A grouping is a container iff some element nests inside it via
        # ``boundary``. The outermost system_boundary is included because the
        # top-level zones reference it.
        containers = {e["boundary"] for e in d["elements"] if e.get("boundary")}
        entities = elem_ids - containers
        rel = {
            tuple(sorted((r["source"], r["target"])))
            for r in d.get("relationships", [])
        }
        att = set()
    return entities, containers, rel, att


def _mmd_truth(code: str) -> tuple[set, set, set, set]:
    rels = extract_relationships(code)
    return (
        {n["id"] for n in extract_nodes(code)},
        {c["id"] for c in extract_containers(code)},
        {tuple(sorted((r["source"], r["target"]))) for r in rels},
        {tuple(sorted((a["a"], a["b"]))) for a in extract_attachments(code)},
    )


# One parametrised case per registered input: failures name the diagram.
@pytest.mark.parametrize("inp", INPUTS, ids=lambda i: i.example_id)
def test_json_mmd_structurally_consistent(inp):
    d = json.loads(inp.file_path.read_text(encoding="utf-8"))
    code = inp.ground_truth_path.read_text(encoding="utf-8")
    je, jc, jr, ja = _json_truth(d)
    me, mc, mr, ma = _mmd_truth(code)

    assert je == me, (
        f"{inp.example_id} ENTITIES differ: "
        f"in JSON not MMD: {sorted(je - me)}; in MMD not JSON: {sorted(me - je)}"
    )
    assert jc == mc, (
        f"{inp.example_id} CONTAINERS differ: "
        f"in JSON not MMD: {sorted(jc - mc)}; in MMD not JSON: {sorted(mc - jc)}"
    )
    assert jr == mr, (
        f"{inp.example_id} FLOWS differ: "
        f"in JSON not MMD: {sorted(jr - mr)}; in MMD not JSON: {sorted(mr - jr)}"
    )
    assert ja == ma, (
        f"{inp.example_id} ATTACHMENTS differ: "
        f"in JSON not MMD: {sorted(ja - ma)}; in MMD not JSON: {sorted(ma - ja)}"
    )


@pytest.mark.parametrize("inp", INPUTS, ids=lambda i: i.example_id)
def test_metadata_counts_match_extractors(inp):
    d = json.loads(inp.file_path.read_text(encoding="utf-8"))
    code = inp.ground_truth_path.read_text(encoding="utf-8")
    meta = d["metadata"]
    assert meta["entity_count"] == len(extract_nodes(code)), (
        f"{inp.example_id} entity_count={meta['entity_count']} "
        f"!= {len(extract_nodes(code))} inline nodes"
    )
    assert meta.get("container_count") == len(extract_containers(code)), (
        f"{inp.example_id} container_count mismatch"
    )
    assert meta.get("attachment_count") == len(extract_attachments(code)), (
        f"{inp.example_id} attachment_count mismatch"
    )
