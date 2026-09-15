"""
Tests for failure-mode classification (src/maestro/analysis/failures.py) and
the failure analyses in statistics.py.

Two layers, deliberately separated:

  1. ``classify_failure`` is pure, so it is tested directly against the error
     strings the codebase actually emits. The strings are copied from the
     provider/strategy error paths and from the production database, not
     invented, so a reworded error message fails a test instead of silently
     becoming ``UNKNOWN`` in a live run.
  2. The rate and survivor-bias functions are exercised through the *real*
     schema (db.client SCHEMA + db.queries inserts), matching the approach in
     test_statistics.py: the correlated sub_results subquery in
     fetch_failure_rows is exactly the part worth testing against real SQL.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

pytest.importorskip("pandas")

from maestro.analysis.failures import (  # noqa: E402
    FailureCause,
    classify_failure,
)
from maestro.analysis.statistics import (  # noqa: E402
    failure_rates,
    load_dataframe,
    load_failure_dataframe,
    survivor_bias,
)
from maestro.db.client import SCHEMA  # noqa: E402
from maestro.db.queries import (  # noqa: E402
    fetch_failure_rows,
    insert_run_config,
    insert_run_result,
    insert_sub_result,
)
from maestro.schemas import (  # noqa: E402
    RunConfig,
    RunResult,
    Strategy,
    SubResult,
    Tier,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

# Error strings taken verbatim from the provider error paths and from the
# production database, paired with the cause each must resolve to.
_REAL_ERRORS = [
    (
        "Step 3 (generate_mermaid) failed: Invalid generate_mermaid output on "
        'attempt 2: empty node label bracket (e.g. node_id[""])',
        FailureCause.SCHEMA_VIOLATION,
    ),
    (
        "Step 3 (generate_mermaid) failed: Invalid generate_mermaid output on "
        "attempt 2: unbalanced subgraph/end (6 subgraph, 5 end)",
        FailureCause.SCHEMA_VIOLATION,
    ),
    (
        "Step 1 (extract_entities) failed: Invalid extract_entities output on "
        "attempt 2: invalid JSON: Expecting property name enclosed in double "
        "quotes: line 26 column 5 (char 500)",
        FailureCause.PARSE_ERROR,
    ),
    (
        "Step 3 (generate_mermaid) failed: empty output from provider",
        FailureCause.EMPTY_OUTPUT,
    ),
    (
        "Step 3 (generate_mermaid) failed: CrewAI kickoff raised on attempt 2: "
        "Invalid response from LLM call - None or empty.",
        FailureCause.EMPTY_OUTPUT,
    ),
    (
        "Step 3 (generate_mermaid) failed: Single-call invariant violated on "
        "attempt 2: expected 1 new call, got 3",
        FailureCause.ORCHESTRATION_ERROR,
    ),
    ("RateLimitError: 429 rate limit exceeded", FailureCause.RATE_LIMIT),
    ("TimeoutError: request timed out", FailureCause.TIMEOUT),
    ("APITimeoutError: deadline exceeded", FailureCause.TIMEOUT),
    ("APIError: 500 internal server error", FailureCause.API_ERROR),
    ("BlockedResponse: response blocked by safety filter", FailureCause.SAFETY_BLOCK),
    ("EmptyResponse: openai returned no content", FailureCause.EMPTY_OUTPUT),
    ("EmptyResponse: anthropic returned no text content", FailureCause.EMPTY_OUTPUT),
]


@pytest.mark.parametrize("error,expected", _REAL_ERRORS)
def test_classifies_real_error_strings(error: str, expected: FailureCause) -> None:
    assert classify_failure(error) is expected


def test_no_error_is_empty_output() -> None:
    """The silent-empty shape: no error string, but no diagram either."""
    assert classify_failure(None) is FailureCause.EMPTY_OUTPUT
    assert classify_failure("   ") is FailureCause.EMPTY_OUTPUT


def test_unrecognized_error_is_unknown_not_misfiled() -> None:
    """
    An unmatched string must surface as UNKNOWN. Silently absorbing it into a
    real category is the failure mode the ordered rules exist to prevent.
    """
    assert classify_failure("KrakenError: the kraken woke up") is FailureCause.UNKNOWN


def test_specific_api_error_beats_generic_api_error() -> None:
    """
    APIError is the SDKs' catch-all base class, so a message naming a more
    specific subclass must not be filed under the generic cause.
    """
    assert classify_failure("RateLimitError: 429") is FailureCause.RATE_LIMIT
    assert classify_failure("APITimeoutError: slow") is FailureCause.TIMEOUT


def test_truncation_promoted_over_parse_error_with_long_raw() -> None:
    """
    A long response that stops mid-string is truncation, not a parse error:
    the parse failure is the symptom, the token limit is the cause.
    """
    error = (
        "Step 1 (extract_entities) failed: Invalid extract_entities output on "
        "attempt 2: invalid JSON: Unterminated string starting at: line 194 "
        "column 15 (char 4400)"
    )
    assert classify_failure(error, "x" * 900) is FailureCause.TRUNCATION


def test_truncation_not_promoted_without_corroborating_raw() -> None:
    """
    Without a long raw response there is no evidence of truncation, so the
    conservative parse-error label stands. Under-reporting truncation is
    safer than inventing it.
    """
    error = (
        "Invalid extract_entities output on attempt 2: invalid JSON: "
        "Unterminated string starting at: line 4 column 1 (char 40)"
    )
    assert classify_failure(error, "short") is FailureCause.PARSE_ERROR
    assert classify_failure(error, None) is FailureCause.PARSE_ERROR


def test_empty_output_precedes_parse_rules() -> None:
    """Nothing came back, so there was nothing to parse."""
    assert (
        classify_failure("EmptyResponse: no content", "x" * 900)
        is FailureCause.EMPTY_OUTPUT
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        ("ratelimiterror: 429", FailureCause.RATE_LIMIT),
        ("RATELIMITERROR: 429", FailureCause.RATE_LIMIT),
        ("apierror: 500", FailureCause.API_ERROR),
        ("ApiError: 500", FailureCause.API_ERROR),
        ("emptyresponse: no content", FailureCause.EMPTY_OUTPUT),
        ("Invalid Json: bad", FailureCause.PARSE_ERROR),
        ("Empty Node Label Bracket", FailureCause.SCHEMA_VIOLATION),
    ],
)
def test_classification_is_case_insensitive(error: str, expected: FailureCause) -> None:
    """
    Error text originates in vendor SDKs and third-party frameworks, so its
    casing is not ours to rely on: a provider rewording ``APIError`` to
    ``ApiError`` must not silently push a whole category into UNKNOWN.
    """
    assert classify_failure(error) is expected


def test_truncation_detection_is_case_insensitive() -> None:
    """The truncation signatures come from the json module, not from us."""
    error = "invalid json: unterminated string starting at: line 9 column 2"
    assert classify_failure(error, "x" * 900) is FailureCause.TRUNCATION


# ---------------------------------------------------------------------------
# fetch_failure_rows / failure_rates
# ---------------------------------------------------------------------------


def _insert_run(
    conn: sqlite3.Connection,
    *,
    strategy: Strategy,
    error: str | None,
    diagram: str | None = "graph TD; a-->b",
    sub_errors: list[tuple[int, str | None, str | None]] | None = None,
    model: str = "model-a",
    tier: Tier = Tier.SIMPLE,
    run_number: int = 1,
) -> uuid.UUID:
    """
    Insert one config+result, plus optional sub_results.

    ``sub_errors`` entries are (step_number, error, raw_response), letting a
    test build the multi-step shape where the failing text lives on the
    sub-result rather than the run row.
    """
    run_id = uuid.uuid4()
    insert_run_config(
        conn,
        RunConfig(
            run_id=run_id,
            strategy=strategy,
            model=model,
            example_id="ex_01",
            tier=tier,
            run_number=run_number,
        ),
    )
    insert_run_result(
        conn,
        RunResult(
            run_id=run_id,
            output_diagram_code=diagram,
            prompt_tokens=10,
            completion_tokens=10,
            duration_ms=100,
            cost_usd=0.001,
            error=error,
        ),
    )
    for step, sub_error, raw in sub_errors or []:
        insert_sub_result(
            conn,
            SubResult(
                run_id=run_id,
                step_number=step,
                step_name=f"step_{step}",
                output_text=None if sub_error else "ok",
                raw_response=raw,
                prompt_tokens=5,
                completion_tokens=5,
                duration_ms=50,
                cost_usd=0.0005,
                error=sub_error,
            ),
        )
    return run_id


def test_fetch_failure_rows_covers_both_failure_shapes() -> None:
    """
    An errored run and a no-error-but-blank-diagram run are both failures;
    a successful run is neither.
    """
    conn = _conn()
    _insert_run(conn, strategy=Strategy.SOP_BASED, error="APIError: 500")
    _insert_run(conn, strategy=Strategy.SINGLE_AGENT, error=None, diagram="   ")
    _insert_run(conn, strategy=Strategy.SINGLE_AGENT, error=None, diagram=None)
    _insert_run(conn, strategy=Strategy.SINGLE_AGENT, error=None)

    rows = fetch_failure_rows(conn)
    assert len(rows) == 3


def test_failing_raw_response_comes_from_first_failed_step() -> None:
    """
    The raw text is pulled from the earliest failed sub-result: that is the
    step that broke the run, and later steps never ran.
    """
    conn = _conn()
    _insert_run(
        conn,
        strategy=Strategy.SOP_BASED,
        error="Step 1 (extract_entities) failed: invalid JSON: Expecting value",
        diagram=None,
        sub_errors=[
            (1, "invalid JSON: Expecting value", "the raw step-1 text"),
            (2, "downstream failure", "later text"),
        ],
    )
    rows = fetch_failure_rows(conn)
    assert len(rows) == 1
    assert rows[0]["failing_raw_response"] == "the raw step-1 text"


def test_one_row_per_run_despite_many_sub_results() -> None:
    """
    The correlated subquery must not multiply a run into one row per
    sub-result, which would inflate every failure count.
    """
    conn = _conn()
    _insert_run(
        conn,
        strategy=Strategy.CREW_AI,
        error="Step 1 failed: invalid JSON: Expecting value",
        diagram=None,
        sub_errors=[
            (i, "invalid JSON: Expecting value", f"raw {i}") for i in range(1, 6)
        ],
    )
    assert len(fetch_failure_rows(conn)) == 1


def test_failure_rates_are_pooled_counts() -> None:
    """
    Rate is failures over runs attempted in the group, not a mean of per-cell
    rates: 1 failure in 4 runs is 0.25 regardless of how the runs split across
    cells.
    """
    conn = _conn()
    for i in range(3):
        _insert_run(conn, strategy=Strategy.SOP_BASED, error=None, run_number=i + 1)
    _insert_run(
        conn,
        strategy=Strategy.SOP_BASED,
        error="APIError: 500",
        diagram=None,
        run_number=4,
    )

    payload = failure_rates(load_dataframe(conn), load_failure_dataframe(conn))
    assert payload["status"] == "ok"
    assert payload["overall"]["n_runs"] == 4
    assert payload["overall"]["n_failed"] == 1
    assert payload["overall"]["failure_rate"] == pytest.approx(0.25)
    assert payload["overall"]["causes"]["api_error"] == 1


def test_failure_rates_report_every_cause_including_zeros() -> None:
    """
    A cause absent from the output would be ambiguous between "never happened"
    and "not measured"; the zeros are what make groups comparable.
    """
    conn = _conn()
    _insert_run(conn, strategy=Strategy.SOP_BASED, error="APIError: 500", diagram=None)
    payload = failure_rates(load_dataframe(conn), load_failure_dataframe(conn))
    assert set(payload["overall"]["causes"]) == {c.value for c in FailureCause}
    assert payload["overall"]["causes"]["rate_limit"] == 0


def test_failure_rates_exclude_controls() -> None:
    """Controls never call a model, so they must not dilute the denominator."""
    conn = _conn()
    _insert_run(conn, strategy=Strategy.SOP_BASED, error="APIError: 500", diagram=None)
    for i in range(5):
        _insert_run(conn, strategy=Strategy.NULL_CONTROL, error=None, run_number=i + 1)

    payload = failure_rates(load_dataframe(conn), load_failure_dataframe(conn))
    assert payload["overall"]["n_runs"] == 1
    assert payload["overall"]["failure_rate"] == pytest.approx(1.0)
    strategies = {row["strategy"] for row in payload["by_strategy"]}
    assert Strategy.NULL_CONTROL.value not in strategies


def test_failure_rates_empty_db_is_status_empty_not_crash() -> None:
    conn = _conn()
    payload = failure_rates(load_dataframe(conn), load_failure_dataframe(conn))
    assert payload["status"] == "empty"


def test_load_failure_dataframe_empty_when_nothing_failed() -> None:
    """A perfect run is a legitimate result, not an error."""
    conn = _conn()
    _insert_run(conn, strategy=Strategy.SOP_BASED, error=None)
    assert load_failure_dataframe(conn).empty


# ---------------------------------------------------------------------------
# survivor_bias
# ---------------------------------------------------------------------------


def test_survivor_bias_is_zero_without_failures() -> None:
    """
    With nothing dropped, the two conventions see the same runs, so the gap
    must be exactly zero rather than merely small.
    """
    pytest.importorskip("statsmodels")
    conn = _conn()
    for i in range(3):
        _insert_run_with_metric(conn, f1=0.8, run_number=i + 1)

    payload = survivor_bias(load_dataframe(conn))
    assert payload["status"] == "ok"
    row = payload["by_strategy"][0]
    assert row["survivor_bias"] == pytest.approx(0.0)
    assert row["n_cells_dropped"] == 0


def test_survivor_bias_positive_when_failures_dropped() -> None:
    """
    Dropping a failed run raises the surviving mean above the all-runs mean:
    that gap is precisely how much valid_only flatters the strategy.
    """
    pytest.importorskip("statsmodels")
    conn = _conn()
    _insert_run_with_metric(conn, f1=0.9, run_number=1)
    # A failed run: no metric row, so intent_to_treat scores it 0.0 and
    # valid_only drops it entirely.
    _insert_run(
        conn,
        strategy=Strategy.SOP_BASED,
        error="APIError: 500",
        diagram=None,
        run_number=2,
    )

    payload = survivor_bias(load_dataframe(conn))
    row = payload["by_strategy"][0]
    assert row["mean_all_runs"] == pytest.approx(0.45)  # (0.9 + 0.0) / 2
    assert row["mean_survivors"] == pytest.approx(0.9)
    assert row["survivor_bias"] == pytest.approx(0.45)


def _insert_run_with_metric(
    conn: sqlite3.Connection,
    *,
    f1: float,
    run_number: int,
    strategy: Strategy = Strategy.SOP_BASED,
) -> None:
    """Insert a successful run carrying a metric row with the given F1."""
    from maestro.db.queries import insert_metric_result
    from maestro.schemas import MetricResult

    run_id = _insert_run(conn, strategy=strategy, error=None, run_number=run_number)
    insert_metric_result(
        conn,
        MetricResult(
            run_id=run_id,
            parses_valid=True,
            entity_id_precision=f1,
            entity_id_recall=f1,
            entity_id_f1=f1,
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
        ),
    )
