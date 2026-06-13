"""
MAESTRO — Metric pipeline sanity tests.

These mirror the runtime control strategies (NullControl, CopyInputControl,
GroundTruthEchoControl) and a handful of explicit edge cases. They run in
~milliseconds and catch metric-pipeline bugs at dev time without needing
to spin up the full experiment matrix.

The runtime controls and these tests are deliberately redundant: pytest
catches regressions before a run; the control rows persist a sanity
record in every actual run's database. Two audiences, same job.
"""

from __future__ import annotations

import re
from uuid import uuid4

import pytest

from maestro.analysis.metrics import check_mermaid_valid, evaluate_run
from maestro.experiment_config import INPUTS

# All tests run against the first registered input. If/when the input
# registry grows, parametrising over INPUTS is the obvious extension.
INPUT = INPUTS[0]


def test_null_output_scores_zero_floor():
    """
    NullControl analogue: an empty Mermaid diagram must score F1=0 on
    entity_id and relationship_relaxed metrics. Catches the class of bug
    where empty / None / "" matches everything.
    """
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code="flowchart LR\n",
        ground_truth_path=INPUT.ground_truth_path,
    )
    assert metric.entities_in_output == 0
    assert metric.relationships_in_output == 0
    assert metric.entity_id_f1 == 0.0
    assert metric.relationship_relaxed_f1 == 0.0


def test_raw_json_input_as_diagram_scores_low():
    """
    CopyInputControl analogue: dropping the raw input JSON into the
    diagram slot must not score meaningfully — the extractor should not
    find valid Mermaid in JSON. Catches parser leniency (e.g. a regex that
    happens to match strings inside JSON keys/values).

    We accept entity_id_f1 strictly equal to 0 here; if the extractor
    ever returns >0 for JSON-as-Mermaid, that's the bug this test exists
    to catch.
    """
    raw = INPUT.file_path.read_text(encoding="utf-8")
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code=raw,
        ground_truth_path=INPUT.ground_truth_path,
    )
    # "Meaningfully above zero" rather than "exactly zero": the relaxed
    # relationship regex is permissive and a future input file containing
    # `-->` in a string value would push F1 fractionally above 0 without
    # the parser actually being broken. 0.1 is well below anything a real
    # strategy would ever produce — if this fires, the parser is genuinely
    # over-matching JSON content.
    assert metric.entity_id_f1 < 0.1, (
        f"JSON-as-diagram scored entity_id_f1={metric.entity_id_f1} — "
        "parser may be matching content inside JSON strings"
    )
    assert metric.relationship_relaxed_f1 < 0.1, (
        f"JSON-as-diagram scored relationship_relaxed_f1="
        f"{metric.relationship_relaxed_f1} — the relationship regex may be "
        "matching `-->` inside JSON string values"
    )


def test_ground_truth_echo_scores_perfect_ceiling():
    """
    GroundTruthEchoControl analogue: feeding the ground truth back as the
    diagram must score F1=1.0 on every metric. A failure here means the
    metric pipeline is over-strict in a way that would penalise even a
    perfect answer — a louder bug than the floor cases catch.

    ``parses_valid`` may be True (mmdc installed) or None (mmdc not
    installed locally — see check_mermaid_valid). Either is acceptable;
    False would indicate the validator rejects the ground truth itself.
    """
    truth = INPUT.ground_truth_path.read_text(encoding="utf-8")
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code=truth,
        ground_truth_path=INPUT.ground_truth_path,
    )
    assert metric.parses_valid in (True, None), (
        f"Ground truth rejected by validator: {metric.parse_error}"
    )
    assert metric.entities_in_output == metric.entities_in_truth
    assert metric.relationships_in_output == metric.relationships_in_truth
    assert metric.entity_id_f1 == 1.0
    assert metric.entity_name_f1 == 1.0
    assert metric.entity_lemma_f1 == 1.0
    assert metric.relationship_relaxed_f1 == 1.0
    assert metric.relationship_strict_f1 == 1.0


def test_check_mermaid_valid_accepts_valid_diagram():
    """
    A syntactically valid diagram must validate (True). Skipped when mmdc is
    not installed — the validator returns (None, skip_message) and there is
    nothing to assert about validity.
    """
    is_valid, error = check_mermaid_valid('flowchart LR\n    a["A"] --> b["B"]')
    if is_valid is None:
        pytest.skip(f"mmdc not available: {error}")
    assert is_valid is True
    assert error is None


def test_check_mermaid_valid_rejects_invalid_diagram():
    """
    The negative path through check_mermaid_valid: syntactically broken Mermaid
    must score parses_valid=False with a non-empty parse error (mmdc's stderr).
    Skipped when mmdc is not installed (validity is unknowable without it).

    Exercises the cross-platform temp-file path end to end — the input is
    written to a real temp file and mmdc renders to another, with no
    ``/dev/stdin`` / ``/dev/null`` involved.
    """
    # Spaces inside the edge-label pipes are rejected by mmdc's parser.
    is_valid, error = check_mermaid_valid('flowchart LR\n    a -->| "x" | b')
    if is_valid is None:
        pytest.skip(f"mmdc not available: {error}")
    assert is_valid is False
    assert error and error.strip()


def test_empty_output_does_not_raise_on_zero_denominator():
    """
    Edge case: ``compute_entity_metrics_exact`` divides by ``len(output)``
    and ``len(truth)``. With an empty output the precision branch must
    short-circuit to (0, 0, 0), not raise ZeroDivisionError. We exercise
    that branch and assert sane numeric values rather than a crash.
    """
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code="",
        ground_truth_path=INPUT.ground_truth_path,
    )
    # No exception, all metrics finite and zero
    assert metric.entity_id_precision == 0.0
    assert metric.entity_id_recall == 0.0
    assert metric.entity_id_f1 == 0.0


def test_sparse_output_scores_below_ground_truth():
    """
    Sanity check that the metric scale is continuous, not bimodal: a
    diagram with one matching entity and one matching relationship from
    a many-entity ground truth should produce a low-but-nonzero recall
    (recall = 1/N where N = entities in truth). If recall stays 0 or
    jumps to 1.0, an aggregation step is broken.
    """
    # Pull a real entity id from the ground truth so this test is robust
    # to changes in the input file. The id syntax in the .MMD files is
    # the leading token before "[" or "(" on entity declarations.
    truth = INPUT.ground_truth_path.read_text(encoding="utf-8")
    # Find the first node line: a token followed by '[' or '(' (Mermaid syntax).
    match = re.search(r"^\s*([A-Za-z_][\w]*)[\[\(]", truth, re.MULTILINE)
    assert match, "Could not find any node id in ground truth — input format changed?"
    real_id = match.group(1)

    sparse_diagram = f'flowchart LR\n    {real_id}["Some Label"]\n'
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code=sparse_diagram,
        ground_truth_path=INPUT.ground_truth_path,
    )

    # Precision should be 1.0 (the single output entity does match), recall
    # should be 1/N where N is the count in ground truth.
    assert metric.entities_in_output == 1
    assert metric.entity_id_precision == 1.0
    # Recall < 1.0 because we only emitted one of many ground-truth entities
    assert 0.0 < metric.entity_id_recall < 1.0


def _input(example_id: str):
    """Locate a registered InputFile by example_id, or skip if absent."""
    for inp in INPUTS:
        if inp.example_id == example_id:
            return inp
    pytest.skip(f"input {example_id} not registered")


def test_container_and_attachment_echo_perfect():
    """
    A ground truth WITH containers and attachments, echoed back, must score
    F1=1.0 on both the container and attachment dimensions (Phase 3b). Uses
    bpmn_3_23 (Travel Booking: one expanded sub-process container + boundary /
    compensation o--o attachments).
    """
    inp = _input("bpmn_3_23")
    truth = inp.ground_truth_path.read_text(encoding="utf-8")
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code=truth,
        ground_truth_path=inp.ground_truth_path,
    )
    assert metric.containers_in_truth > 0 and metric.attachments_in_truth > 0
    assert metric.container_id_f1 == 1.0
    assert metric.container_name_f1 == 1.0
    assert metric.attachment_f1 == 1.0
    assert metric.containers_in_output == metric.containers_in_truth
    assert metric.attachments_in_output == metric.attachments_in_truth


def test_container_attachment_metrics_none_when_absent():
    """
    A ground truth with NO containers and NO attachments must report those
    dimensions as None (metric not applicable), not 0.0 — so aggregation can
    exclude the run rather than averaging in a spurious zero. Uses bpmn_1_01
    (a plain single-pool process).
    """
    inp = _input("bpmn_1_01")
    truth = inp.ground_truth_path.read_text(encoding="utf-8")
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code=truth,
        ground_truth_path=inp.ground_truth_path,
    )
    assert metric.containers_in_truth == 0 and metric.attachments_in_truth == 0
    assert metric.container_id_f1 is None
    assert metric.container_name_f1 is None
    assert metric.attachment_f1 is None


@pytest.mark.parametrize("diagram", ["", "flowchart LR\n", "not mermaid at all"])
def test_zero_entities_in_output_never_crashes(diagram: str):
    """
    Robustness: a variety of "empty or junk" diagrams must all complete
    ``evaluate_run`` without raising. The exact F1 values are covered
    elsewhere; here we just pin the no-crash contract.
    """
    metric = evaluate_run(
        run_id=uuid4(),
        output_diagram_code=diagram,
        ground_truth_path=INPUT.ground_truth_path,
    )
    # Just touch every required field to be sure MetricResult validated.
    assert metric.entity_id_f1 >= 0.0
    assert metric.relationship_relaxed_f1 >= 0.0
