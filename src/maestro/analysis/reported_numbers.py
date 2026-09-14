"""
Canonical reported-numbers dump: the values docs quote and slide decks cite
are computed from the results database here and written to a
machine-readable JSON file. Downstream text (README, CHANGELOG, docs
markdown) is checked against this file by the consistency test, so a
transcribed number cannot silently disagree with the database.

Scope is deliberately narrow: only headline totals that already appear in
the docs today (total cell count, success / failure split, aggregate
cost). Statistical results have their own richer artefacts under
``maestro.analysis.__main__``; this module is the join point where prose
meets truth.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from maestro.db.client import get_readonly_connection
from maestro.experiment_config import DB_PATH

# Emitted schema version. Bump when a field is renamed or its meaning
# changes; consumers (docs, the consistency check) can then pin against a
# specific shape rather than a moving target.
SCHEMA_VERSION = "1.0"

# Fixed relative location under the analysis output tree. Kept stable so
# the consistency check does not need a discovery step.
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "output"
    / "analysis"
    / "reported_numbers.json"
)


def compute_reported_numbers(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Aggregate the docs-referenced totals from a results DB, read-only.

    Returns ``status="empty"`` when no runs have been recorded yet: every
    numeric field is still present but set to 0 so a downstream diff still
    works against a fresh (v2.0.0) checkout.
    """
    totals_row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_runs,
            COALESCE(SUM(CASE WHEN error IS NULL AND output_diagram_code IS NOT NULL
                              AND TRIM(output_diagram_code) <> ''
                         THEN 1 ELSE 0 END), 0) AS successes,
            COALESCE(SUM(CASE WHEN error IS NOT NULL OR output_diagram_code IS NULL
                              OR TRIM(COALESCE(output_diagram_code, '')) = ''
                         THEN 1 ELSE 0 END), 0) AS failures,
            COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd
        FROM run_results
        """
    ).fetchone()

    total_runs = int(totals_row["total_runs"])
    if total_runs == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "empty",
            "total_runs": 0,
            "successes": 0,
            "failures": 0,
            "total_cost_usd": 0.0,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "total_runs": total_runs,
        "successes": int(totals_row["successes"]),
        "failures": int(totals_row["failures"]),
        # Rounded to cents: docs quote e.g. USD 171.62, so writing extra
        # digits would create a mismatch on the last decimal that is
        # cosmetic, not real. Two-place rounding matches the reporting
        # convention and keeps the diff meaningful.
        "total_cost_usd": round(float(totals_row["total_cost_usd"]), 2),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (``--db``, ``--out``)."""
    parser = argparse.ArgumentParser(
        prog="python -m maestro.analysis.reported_numbers",
        description=(
            "Dump the docs-referenced headline totals to a machine-readable "
            "JSON file. Consumed by the model-registry consistency check."
        ),
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
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Destination file for the reported-numbers JSON "
            f"(default: {DEFAULT_OUTPUT_PATH})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Write the reported-numbers JSON file. Exit code 1 if the DB path does
    not exist; 0 otherwise (including an empty DB, which still emits a
    valid file so the docs pipeline can run against a fresh checkout).
    """
    args = _parse_args(argv)
    if not args.db.exists():
        print(f"ERROR: database not found: {args.db}")
        return 1

    with get_readonly_connection(args.db) as conn:
        payload = compute_reported_numbers(conn)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
