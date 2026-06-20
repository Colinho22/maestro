"""
Tests for the shared multi-step output contract in strategies/_extraction.py.

validate_step_output is the single acceptance check every multi-step strategy
(SOP, CrewAI, LangGraph) runs on a fenced-stripped step output, so the rules
that decide "is this step a pass or a recorded failure" live in one place.
"""

from __future__ import annotations

from maestro.strategies._extraction import (
    validate_step_output,
    validate_step_payload,
)


def test_empty_output_rejected_on_every_step():
    """Empty or whitespace output is a failure on all three steps.

    strip_fences can reduce a non-empty response (bare ``` fences) to "",
    which would otherwise pass the provider success check and leave step 3
    with a silent empty diagram.
    """
    for step in (1, 2, 3):
        ok, err = validate_step_output("", step)
        assert ok is False and err == "empty output"
        ok, err = validate_step_output("   \n  ", step)
        assert ok is False and err == "empty output"
        ok, err = validate_step_output(None, step)
        assert ok is False and err == "empty output"


def test_step3_accepts_wellformed_mermaid():
    """Step 3 passes any non-empty, structurally sound Mermaid."""
    ok, err = validate_step_output("flowchart LR\n  a --> b", 3)
    assert ok is True and err is None
    # Multi-line labels, cylinders, and subgraphs must not trip the shape check.
    diagram = (
        "flowchart LR\n"
        '  user["User\\n[Person]"]\n'
        '  store[("Object Storage\\n[Container]")]\n'
        '  subgraph infomaniak["Infomaniak"]\n'
        '    web["SomeApp"]\n'
        "  end\n"
        "  user --> web"
    )
    ok, err = validate_step_output(diagram, 3)
    assert ok is True and err is None


def test_step3_rejects_empty_label_bracket():
    """The empty-label malformation a weak model emitted must consume a retry,
    not pass as non-empty and land as a scored parse failure."""
    for bad in ('a[""]', "a['']", 'gw{""}', 'n(("")) '):
        ok, err = validate_step_output(f"flowchart LR\n  {bad}", 3)
        assert ok is False and "empty node label" in err


def test_step3_rejects_concatenated_nodes():
    """Two node defs concatenated without a separator (the other observed
    failure, e.g. ``a[""]b["B"]``) is rejected."""
    ok, err = validate_step_output('flowchart LR\n  pool[""]web["W"]', 3)
    assert ok is False
    # Either malformation may match first; both are real defects in this line.
    assert "empty node label" in err or "concatenated" in err


def test_steps_1_2_still_apply_json_shape():
    """Non-empty step 1/2 output must still pass the JSON payload check."""
    ok, _ = validate_step_output('{"entities": []}', 1)
    assert ok is True
    ok, err = validate_step_output("not json", 1)
    assert ok is False and "invalid JSON" in err
    ok, err = validate_step_output('{"wrong": []}', 2)
    assert ok is False and "relationships" in err


def test_step3_bypasses_json_check():
    """A step-3 payload that is not JSON is fine; the legacy helper would reject it."""
    payload = "flowchart LR"
    assert validate_step_output(payload, 3)[0] is True
    # The narrower helper only knows about steps 1-2 and would reject this.
    assert validate_step_payload(payload, 3)[0] is False
