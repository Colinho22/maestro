"""
MAESTRO — analysis CLI entry point.

    python -m maestro.analysis [--db PATH] [--out DIR] [--display-tz TZ]

Runs the statistical analysis pipeline against the experiment database
(read-only) and writes canonical JSON outputs plus an assembled
``report.md`` to ``<out>/<timestamp>/``. The visualizer (#19) consumes
these JSON files; it does not recompute the statistics.

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
    SCHEMA_VERSION,
    anova_strategy,
    anova_strategy_by_model,
    anova_strategy_by_tier,
    describe,
    effect_sizes,
    error_taxonomy_by_strategy,
    load_dataframe,
    posthoc_strategy,
    tradeoff_correctness_efficiency,
)
from maestro.analysis.timestamps import format_for_display
from maestro.db.client import get_connection
from maestro.experiment_config import DB_PATH

# Default output root, relative to project root (two parents up from this
# package: src/maestro/analysis -> src/maestro -> src -> <root>).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _PROJECT_ROOT / "output" / "analysis"

# (filename, callable) for each analysis. Order is the report order.
# Callables take the DataFrame and return a JSON-serializable dict.
_ANALYSES: list[tuple[str, Callable]] = [
    ("descriptive.json", describe),
    ("anova_strategy.json", anova_strategy),
    ("anova_strategy_by_tier.json", anova_strategy_by_tier),
    ("anova_strategy_by_model.json", anova_strategy_by_model),
    ("posthoc_strategy.json", posthoc_strategy),
    ("effect_sizes.json", effect_sizes),
    ("error_taxonomy_by_strategy.json", error_taxonomy_by_strategy),
    ("tradeoff_correctness_efficiency.json", tradeoff_correctness_efficiency),
]

# RQ → file mapping, surfaced in report.md. This is the *interpretation*
# layer: it lives in the human-facing report, never in the data files (so
# the JSON stays reusable if the RQs are reframed). Keep in sync with the
# thesis RQ definitions.
_RQ_MAP: list[tuple[str, str, str]] = [
    (
        "RQ1",
        "anova_strategy.json + posthoc_strategy.json",
        "Multi-agent strategies vs. single-agent baseline correctness.",
    ),
    (
        "RQ2",
        "anova_strategy_by_tier.json",
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
        "tradeoff_correctness_efficiency.json + effect_sizes.json",
        "Correctness vs. efficiency trade-off across strategies.",
    ),
    (
        "robustness",
        "anova_strategy_by_model.json",
        "Cross-cutting: does the strategy effect hold across models / "
        "providers (strategy×model interaction)?",
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
    numeric suffix (``-1``, ``-2``, …) is appended until an unused name is
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
    # JSON parsers (and the visualizer, #19) reject. The statistics layer
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
    lines.append("# MAESTRO — Statistical Analysis Report")
    lines.append("")
    lines.append(f"- Generated: {when}")
    lines.append(f"- Database: `{db_path}`")
    lines.append(f"- Output schema version: {SCHEMA_VERSION}")
    lines.append("")

    # RQ → file mapping (interpretation layer — lives here, not in the JSON).
    lines.append("## Research question → output mapping")
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

    _summarize_anova(lines, "RQ1 — strategy", results.get("anova_strategy.json", {}))
    _summarize_anova(
        lines, "RQ2 — strategy×tier", results.get("anova_strategy_by_tier.json", {})
    )
    _summarize_anova(
        lines,
        "robustness — strategy×model",
        results.get("anova_strategy_by_model.json", {}),
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Analyses reported as *skipped* lacked a factor with ≥2 observed "
        "levels in the current corpus (e.g. a single input tier). They "
        "recompute automatically once the corpus grows — no code change."
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
        lines.append(f"- {label}: skipped — {payload.get('reason', 'n/a')}")
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
    not exist. A missing/empty database is not an error — it produces
    empty-status outputs and still returns 0.
    """
    args = _parse_args(argv)

    if not args.db.exists():
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        return 1

    now_utc = datetime.now(timezone.utc)
    # _run_dir creates the (unique) directory itself.
    run_dir = _run_dir(args.out, now_utc)

    # Read-only DB access. get_connection commits on exit, but the analysis
    # issues no writes, so the commit is a no-op.
    with get_connection(args.db) as conn:
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

    # report.md (interpretation layer).
    report = _build_report(
        results, db_path=args.db, now_utc=now_utc, display_tz=args.display_tz
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    # figures/ is intentionally deferred to the visualizer (#19): figure
    # styling must follow the overall plotting/export design, which is not
    # settled. Leave a documented placeholder so the directory contract is
    # visible to #19 without committing to a style here.
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "README.md").write_text(
        "# Figures (deferred)\n\n"
        "Figure generation belongs to the visualizer (#19), which consumes "
        "the JSON files in the parent directory. Styling (thesis-serif vs. "
        "slides-sans, export formats) is a #19 concern and is intentionally "
        "not produced by the compute pipeline.\n",
        encoding="utf-8",
    )

    print(f"Analysis written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
