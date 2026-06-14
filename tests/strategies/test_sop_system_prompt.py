"""
Regression test for SOPStrategy per-step ``system_prompt`` plumbing (PR #10).

The bug fixed in #10 was that ``SOPStrategy._execute_step`` was not forwarding
the per-step ``system_prompt`` to ``provider.complete()``, so steps 1 and 2
(which need a JSON-extraction system identity) silently fell back to the
provider's default Mermaid system prompt. This test pins the corrected wiring.

CodeRabbit originally suggested asserting on the *built user prompt*, but the
system prompt is a separate kwarg: the assertion has to be made on the call
captured at the provider boundary, which is what ``RecordingProvider`` records.
"""

from __future__ import annotations

import json

from maestro.schemas import RunConfig, Strategy, Tier
from maestro.strategies._extraction import JSON_EXTRACTION_SYSTEM_PROMPT
from maestro.strategies.sop import SOPStrategy


def _valid_step1_output() -> str:
    return json.dumps(
        {"entities": [{"id": "e1", "name": "E1", "type": "t", "parent_id": None}]}
    )


def _valid_step2_output() -> str:
    return json.dumps(
        {
            "relationships": [
                {"id": "r1", "source": "e1", "target": "e1", "type": "t", "label": None}
            ]
        }
    )


def test_sop_forwards_per_step_system_prompt(tmp_path, recording_provider_factory):
    """
    Each of the three SOP steps must pass the right ``system_prompt`` through
    to ``provider.complete()``:

    - Step 1 (extract_entities)      -> JSON_EXTRACTION_SYSTEM_PROMPT
    - Step 2 (extract_relationships) -> JSON_EXTRACTION_SYSTEM_PROMPT
    - Step 3 (generate_mermaid)      -> None  (provider falls back to its default)

    The mock provider returns schema-valid JSON for steps 1 and 2 so
    ``validate_step_payload`` accepts them and there is exactly one call per
    step (no retries).
    """

    # Minimal input JSON file the strategy can read.
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"entities": [], "relationships": []}))

    # Construct an InputFile that satisfies pydantic but is otherwise unused
    # by the strategy beyond reading ``file_path``.
    from maestro.schemas import InputFile

    input_file = InputFile(
        example_id="test_example",
        tier=Tier.SIMPLE,
        entity_count=0,
        file_path=input_path,
        ground_truth_path=input_path,  # not read in this test
    )

    config = RunConfig(
        strategy=Strategy.SOP_BASED,
        model="test-model",
        example_id="test_example",
        tier=Tier.SIMPLE,
        run_number=1,
    )

    provider = recording_provider_factory(
        outputs=[
            _valid_step1_output(),
            _valid_step2_output(),
            "graph TD\n  e1",  # step 3: any non-empty string is accepted
        ]
    )
    strategy = SOPStrategy(provider)

    run_result, sub_results = strategy.run(input_file, config)

    # Sanity: all three steps ran and the run succeeded end-to-end.
    assert run_result.error is None, f"run errored: {run_result.error}"
    assert len(sub_results) == 3
    assert provider.system_prompts_seen == [
        JSON_EXTRACTION_SYSTEM_PROMPT,
        JSON_EXTRACTION_SYSTEM_PROMPT,
        None,
    ]
