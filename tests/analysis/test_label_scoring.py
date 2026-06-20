"""
Label-scoring policy: what the entity-name metric does and does not score.

Two ground-truth authoring conventions are not derivable from the input, so the
metric excludes them rather than penalising a model for an unreachable target:

  * the optional third descriptor line of a C4 / network label, authored
    inconsistently across nodes, is dropped before comparison;
  * nodes the input leaves unnamed (BPMN gateways/events with name "") are
    scored by id only, since their ground-truth label is pure convention.

The second is conditional on the input: a node the input *did* name is still
scored on its label, so a model that blanks a nameable node is still penalised.
"""

from __future__ import annotations

from maestro.analysis.metrics import (
    _label_core,
    compute_entity_metrics_fuzzy,
    extract_input_unnamed_ids,
)


def test_label_core_keeps_name_and_type_drops_descriptor():
    assert _label_core("Task 1") == "Task 1"
    assert _label_core("User\\n[Person]") == "User\\n[Person]"
    # third descriptor line dropped; name + [Type] kept
    assert _label_core("SomeApp\\n[Container]\\nWeb Application") == (
        "SomeApp\\n[Container]"
    )


def test_descriptor_line_not_scored():
    """A mismatched third line must not sink an otherwise correct label."""
    out = [{"id": "a", "label": "Router\\n[Device]\\nNetwork Switch"}]
    truth = [{"id": "a", "label": "Router\\n[Device]"}]
    _, _, f1 = compute_entity_metrics_fuzzy(out, truth)
    assert f1 == 1.0


def test_input_unnamed_node_scored_by_id_only():
    """A gateway the input left unnamed: id match counts even though the model's
    label cannot match the convention-authored ground-truth label."""
    out = [{"id": "pgw_3", "label": ""}]
    truth = [{"id": "pgw_3", "label": "+ Parallel Split"}]
    # Without the exemption, the empty label would never reach threshold.
    _, _, f1_strict = compute_entity_metrics_fuzzy(out, truth)
    assert f1_strict < 1.0
    # With the input marking pgw_3 unnamed, the id match is a name match.
    _, _, f1_exempt = compute_entity_metrics_fuzzy(out, truth, {"pgw_3"})
    assert f1_exempt == 1.0


def test_exemption_does_not_excuse_a_nameable_node():
    """The guardrail: a node the input *named* is still scored on its label, so a
    model that blanks it is still penalised even if other nodes are exempt."""
    out = [{"id": "task_1", "label": ""}]
    truth = [{"id": "task_1", "label": "Receive Travel Request"}]
    # task_1 is NOT in the unnamed set, so the blank label is still wrong.
    _, _, f1 = compute_entity_metrics_fuzzy(out, truth, {"pgw_3"})
    assert f1 < 1.0


def test_extract_input_unnamed_ids_handles_missing_and_malformed():
    assert extract_input_unnamed_ids(None) == set()
    # nonexistent path fails soft to an empty set
    from pathlib import Path

    assert extract_input_unnamed_ids(Path("/no/such/input.json")) == set()
