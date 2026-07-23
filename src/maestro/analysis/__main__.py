"""
MAESTRO analysis CLI entry point.

    python -m maestro.analysis [--db PATH] [--out DIR] [--display-tz TZ]

Runs the statistical analysis pipeline against the experiment database
(read-only) and writes canonical JSON outputs plus an assembled
``report.md`` to ``<out>/<timestamp>/``. The visualizer consumes these JSON
files; it does not recompute the statistics.

Single-command by design: the same invocation regenerates thesis figures'
source data, defense-slide numbers, and conference-demo output with no
manual prep (no notebooks).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from maestro.analysis.statistics import (
    INTENT_TO_TREAT,
    SCHEMA_VERSION,
    VALID_ONLY,
    anova_strategy,
    anova_strategy_by_model,
    anova_strategy_by_tier,
    describe,
    effect_sizes,
    error_taxonomy_by_strategy,
    load_dataframe,
    mixed_effects_robustness,
    posthoc_strategy,
    tradeoff_correctness_efficiency,
)
from maestro.analysis.timestamps import format_for_display
from maestro.db.client import get_readonly_connection
from maestro.experiment_config import DB_PATH

# Default output root, relative to project root (two parents up from this
# package: src/maestro/analysis -> src/maestro -> src -> <root>).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _PROJECT_ROOT / "output" / "analysis"

# The scoring conventions to emit, primary first. Every convention-dependent
# analysis is written once per convention, so the results chapter can report
# intent_to_treat as the headline and valid_only as a sensitivity check.
_CONVENTIONS = (INTENT_TO_TREAT, VALID_ONLY)

# Convention-independent analyses: (filename, callable). These take only the
# DataFrame and do not depend on a scoring convention. Descriptives keep the
# raw per-run grain and include controls; the taxonomy and trade-off outputs
# are descriptive summaries, not inferential tests.
_ANALYSES: list[tuple[str, Callable]] = [
    ("descriptive.json", describe),
    ("error_taxonomy_by_strategy.json", error_taxonomy_by_strategy),
    ("tradeoff_correctness_efficiency.json", tradeoff_correctness_efficiency),
]

# Convention-dependent analyses: (stem, callable). Each is emitted once per
# scoring convention as ``<stem>__<convention>.json`` (content-based naming:
# the filename states both the test and the convention, so a file is never
# ambiguous about which numbers it holds). The callable takes
# (DataFrame, convention).
_CONVENTION_ANALYSES: list[tuple[str, Callable]] = [
    ("anova_strategy", anova_strategy),
    ("anova_strategy_by_tier", anova_strategy_by_tier),
    ("anova_strategy_by_model", anova_strategy_by_model),
    ("posthoc_strategy", posthoc_strategy),
    ("effect_sizes", effect_sizes),
    ("mixed_effects_robustness", mixed_effects_robustness),
]

# RQ -> file mapping, surfaced in report.md. This is the *interpretation*
# layer: it lives in the human-facing report, never in the data files (so
# the JSON stays reusable if the RQs are reframed). Keep in sync with the
# thesis RQ definitions. Inferential outputs are emitted per scoring convention
# as ``<stem>__<convention>.json``; the map names the intent-to-treat file (the
# primary convention), and the parallel ``__valid_only`` file is the
# sensitivity check for the same question.
_RQ_MAP: list[tuple[str, str, str]] = [
    (
        "RQ1",
        "anova_strategy__intent_to_treat.json + posthoc_strategy__intent_to_treat.json",
        "Multi-agent strategies vs. single-agent baseline correctness.",
    ),
    (
        "RQ2",
        "anova_strategy_by_tier__intent_to_treat.json",
        "Does input complexity moderate the multi-agent advantage "
        "(strategy×tier interaction)?",
    ),
    (
        "RQ3",
        "error_taxonomy_by_strategy.json",
        "Exploratory: hallucination / error patterns per strategy "
        "(descriptive, no inferential test).",
    ),
    (
        "RQ4",
        "tradeoff_correctness_efficiency.json + effect_sizes__intent_to_treat.json",
        "Correctness vs. efficiency trade-off across strategies.",
    ),
    (
        "robustness",
        "anova_strategy_by_model__intent_to_treat.json + "
        "mixed_effects_robustness__intent_to_treat.json",
        "Cross-cutting: does the strategy effect hold across models / "
        "providers, and does it survive a mixed-effects model?",
    ),
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments (``--db``, ``--out``, ``--display-tz``)."""
    parser = argparse.ArgumentParser(
        prog="python -m maestro.analysis",
        description="Run the MAESTRO statistical analysis pipeline (compute only).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to the experiment SQLite database (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Output root directory; a timestamped subdirectory is created "
            f"inside it (default: {DEFAULT_OUT_DIR})."
        ),
    )
    parser.add_argument(
        "--display-tz",
        default=None,
        help=(
            "IANA timezone for human-readable timestamps in report.md "
            "(e.g. Europe/Zurich). Falls back to $MAESTRO_DISPLAY_TZ, then "
            "system local. Storage stays UTC regardless."
        ),
    )
    return parser.parse_args(argv)


def _run_dir(out_root: Path, now_utc: datetime) -> Path:
    """
    Create and return a unique timestamped output subdirectory.

    The name uses a UTC, filesystem-safe, second-precision stamp (sortable,
    unambiguous); the human-readable display-tz rendering goes inside
    report.md. Two invocations within the same second would otherwise map to
    the same directory and silently overwrite each other's outputs, so the
    directory is created with ``exist_ok=False`` and, on collision, a short
    numeric suffix (``-1``, ``-2``, ...) is appended until an unused name is
    found. The first run of any given second keeps the clean stamp.
    """
    stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    candidate = out_root / stamp
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = out_root / f"{stamp}-{suffix}"
            suffix += 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize ``payload`` to ``path`` as strict JSON (fails on NaN/inf)."""
    # allow_nan=False makes non-finite floats (NaN / inf) raise instead of
    # emitting the non-standard ``NaN`` / ``Infinity`` tokens that strict
    # JSON parsers (and the visualizer) reject. The statistics layer
    # already coerces non-finite values to null via _to_native, so this is
    # a fail-fast guard: if something slips through, we want a loud error
    # here, not a silently invalid file.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_report(
    results: dict[str, dict[str, Any]],
    *,
    db_path: Path,
    now_utc: datetime,
    display_tz: str | None,
) -> str:
    """Assemble a human-readable markdown summary embeddable into the thesis."""
    when = format_for_display(now_utc, display_tz)
    lines: list[str] = []
    lines.append("# MAESTRO Statistical Analysis Report")
    lines.append("")
    lines.append(f"- Generated: {when}")
    lines.append(f"- Database: `{db_path}`")
    lines.append(f"- Output schema version: {SCHEMA_VERSION}")
    lines.append("")

    # RQ -> file mapping (interpretation layer, lives here, not in the JSON).
    lines.append("## Research question -> output mapping")
    lines.append("")
    lines.append("| RQ | Output file(s) | What it answers |")
    lines.append("| --- | --- | --- |")
    for rq, files, desc in _RQ_MAP:
        lines.append(f"| {rq} | `{files}` | {desc} |")
    lines.append("")

    # Headline numbers, pulled defensively (every analysis may be a skip-stub).
    lines.append("## Summary")
    lines.append("")

    desc = results.get("descriptive.json", {})
    n_cells = len(desc.get("cells", []))
    lines.append(f"- Descriptive cells: {n_cells}")
    lines.append(
        f"- Primary scoring convention: `{INTENT_TO_TREAT}` "
        f"(sensitivity check: `{VALID_ONLY}`)"
    )
    lines.append(
        "- Inferential unit of analysis: mean over repeats per (strategy, model, input)"
    )

    # Headline numbers use the primary (intent_to_treat) convention; the
    # __valid_only files carry the sensitivity check for the same tests.
    _summarize_anova(
        lines,
        "RQ1: strategy",
        results.get("anova_strategy__intent_to_treat.json", {}),
    )
    _summarize_anova(
        lines,
        "RQ2: strategy×tier",
        results.get("anova_strategy_by_tier__intent_to_treat.json", {}),
    )
    _summarize_anova(
        lines,
        "robustness: strategy×model",
        results.get("anova_strategy_by_model__intent_to_treat.json", {}),
    )

    # Pairwise effect sizes for both conventions, not just the primary one:
    # the results chapter reports the |d| range next to each convention's
    # ANOVA row, so both belong in the report the numbers are read from.
    for convention in _CONVENTIONS:
        _summarize_effect_sizes(
            lines,
            f"effect sizes ({convention})",
            results.get(f"effect_sizes__{convention}.json", {}),
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Inferential tests run on per-cell means (one observation per "
        "strategy x model x input, repeats averaged), not the raw runs. "
        "Fitting the raw repeats would be pseudoreplication and overstate "
        "significance."
    )
    lines.append(
        f"- Two scoring conventions are reported. `{INTENT_TO_TREAT}` "
        "(primary) scores every run, with a failed or unrenderable diagram "
        "counting 0.0: it measures what a user receives, and does not let a "
        "strategy look good by failing and being dropped. `{valid}` "
        "(sensitivity) keeps only renderable diagrams. Failures are not "
        "random (they concentrate in the more complex orchestrations), so the "
        "two conventions can disagree, and that is the point of reporting "
        "both.".format(valid=VALID_ONLY)
    )
    lines.append(
        "- `mixed_effects_robustness__*.json` refits the strategy x tier "
        "effect on the un-aggregated runs with crossed random intercepts for "
        "model and input, as a check that the aggregated ANOVA conclusion "
        "survives a model of the grouping structure. It self-skips if the "
        "mixed model does not converge."
    )
    lines.append(
        "- Analyses reported as *skipped* lacked a factor with >= 2 observed "
        "levels in the current corpus (e.g. a single input tier). They "
        "recompute automatically once the corpus grows: no code change."
    )
    lines.append(
        "- Control conditions are excluded from ANOVA (their F1 is 0 or 1 "
        "by construction) but retained in `descriptive.json` and "
        "`error_taxonomy_by_strategy.json` as sanity anchors."
    )
    lines.append("- `single_agent` is the ANOVA reference (comparison baseline).")
    lines.append("")
    return "\n".join(lines)


def _summarize_anova(lines: list[str], label: str, payload: dict[str, Any]) -> None:
    """Append a one-line summary of an ANOVA result (or its skip reason)."""
    status = payload.get("status")
    if status == "skipped":
        lines.append(f"- {label}: skipped: {payload.get('reason', 'n/a')}")
        return
    if status != "ok":
        lines.append(f"- {label}: not computed")
        return
    toi = payload.get("term_of_interest")
    terms = payload.get("terms", {})
    term = terms.get(toi) if toi else None
    if term is None:
        # Fall back to the first term if the named one is absent.
        term = next(iter(terms.values()), None)
    if term is None:
        lines.append(f"- {label}: ok (no terms)")
        return
    lines.append(
        f"- {label}: F={_fmt(term.get('F'))}, p={_fmt(term.get('p'))}, "
        f"partial η²={_fmt(term.get('partial_eta_sq'))} (n={payload.get('n')})"
    )


def _summarize_effect_sizes(
    lines: list[str], label: str, payload: dict[str, Any]
) -> None:
    """
    Append the absolute Cohen's d range and the widest contrast (or the skip
    reason). This is the line the results chapter reads its effect-size column
    off, so incomplete coverage is stated inline rather than left to a reader
    who would otherwise take the range for all of the contrasts.
    """
    status = payload.get("status")
    if status == "skipped":
        lines.append(f"- {label}: skipped: {payload.get('reason', 'n/a')}")
        return
    if status != "ok":
        lines.append(f"- {label}: not computed")
        return

    summary = payload.get("summary") or {}
    lo, hi = summary.get("abs_d_min"), summary.get("abs_d_max")
    if lo is None or hi is None:
        lines.append(f"- {label}: no finite pairwise d")
        return

    widest = summary.get("largest_contrast") or {}
    pair = f"{widest.get('group_a')} vs {widest.get('group_b')}"
    line = (
        f"- {label}: |d| {_fmt(lo)} to {_fmt(hi)} "
        f"across {summary.get('n_pairs')} contrasts; largest: {pair}"
    )
    excluded = (summary.get("n_sentinel_pairs") or 0) + (
        summary.get("n_undefined_pairs") or 0
    )
    if excluded:
        line += (
            f" (range covers {len(payload.get('pairs', [])) - excluded} of "
            f"{summary.get('n_pairs')}: "
            f"{summary.get('n_sentinel_pairs')} infinite, "
            f"{summary.get('n_undefined_pairs')} undefined)"
        )
    lines.append(line)


def _fmt(x: Any) -> str:
    """Format a stat value for the report: ``n/a`` for None, 4 sig-figs for floats."""
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def main(argv: list[str] | None = None) -> int:
    """
    Run the analysis pipeline end to end and write all outputs.

    Returns a process exit code: 0 on success, 1 if the database path does
    not exist. A missing/empty database is not an error: it produces
    empty-status outputs and still returns 0.
    """
    args = _parse_args(argv)

    if not args.db.exists():
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        return 1

    now_utc = datetime.now(timezone.utc)
    # _run_dir creates the (unique) directory itself.
    run_dir = _run_dir(args.out, now_utc)

    # Read-only DB access: analysis never writes experiment rows, and mode=ro
    # enforces that at the boundary instead of relying on a no-op commit.
    with get_readonly_connection(args.db) as conn:
        df = load_dataframe(conn)

    if df.empty:
        print(
            "WARN: no metriced runs found in the database; "
            "writing empty-status outputs.",
            file=sys.stderr,
        )

    results: dict[str, dict[str, Any]] = {}
    for filename, fn in _ANALYSES:
        payload = fn(df)
        results[filename] = payload
        _write_json(run_dir / filename, payload)

    # Convention-dependent analyses: one file per (analysis, convention).
    for stem, fn in _CONVENTION_ANALYSES:
        for convention in _CONVENTIONS:
            filename = f"{stem}__{convention}.json"
            payload = fn(df, convention)
            results[filename] = payload
            _write_json(run_dir / filename, payload)

    # report.md (interpretation layer).
    report = _build_report(
        results, db_path=args.db, now_utc=now_utc, display_tz=args.display_tz
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    # figures/ is intentionally left to the visualizer: figure styling must
    # follow the overall plotting/export design and is not produced by the
    # compute pipeline. Leave a documented placeholder so the directory
    # contract is visible without committing to a style here.
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "README.md").write_text(
        "# Figures (not produced here)\n\n"
        "Figure generation belongs to the visualizer, which consumes the JSON "
        "files in the parent directory. Styling (thesis-serif vs. slides-sans, "
        "export formats) is a visualization concern and is intentionally not "
        "produced by the compute pipeline.\n",
        encoding="utf-8",
    )

    print(f"Analysis written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
