# Database schema

Field-level reference for `maestro.db`. Written for anyone querying the
database directly (SQL, pandas, a notebook). Assumes you have already read
`docs/running.md` for how a run gets written.

The database is a plain SQLite file. The runner is the only writer; the
analysis module and the dashboard open it read-only. The schema lives in
code (`src/maestro/db/client.py`); the `.db` file is a build output.

---

## 1. Foreign-key chain

```text
run_environments      one row per CLI invocation
    |
    | environment_id
    v
run_configs           one row per matrix cell (the "what was asked")
    |
    | run_id
    v
run_results           one row per matrix cell (the "what came back")
    |
    +--> sub_results     0..N rows per cell (multi-step strategy steps)
    +--> metric_results  0..1 row per successful cell (scored metrics)
```

`PRAGMA foreign_keys = ON` is set on every connection.

- `run_configs.environment_id` may be NULL for old rows that predate the
  provenance column. The additive migration in `init_db` adds the column
  without backfilling.
- `run_results.retry_count` is `INTEGER NOT NULL DEFAULT 0`, so old rows
  that predate the column read as 0 after the additive migration, not NULL.
- On `metric_results`, the container and attachment P/R/F1 columns are
  nullable REAL (NULL = metric not applicable for that diagram), while
  the corresponding count columns are `INTEGER NOT NULL DEFAULT 0`.
  Pre-Phase-3b rows therefore read as NULL P/R/F1 and 0 counts after the
  additive migration.
- **Application-level invariant on `metric_results`**: at most one row
  per `run_id`. The schema does not carry a `UNIQUE(run_id)` constraint;
  uniqueness is enforced by the runner, which writes at most one metric
  row per cell (`_execute_cell` returns a single `MetricResult` or None).
  A downstream tool that inserts extra rows would break this contract.

---

## 2. Tables

### 2.1 `run_environments`

One row per CLI invocation. Every `run_configs` row links here so a
result traces back to the exact software stack that produced it.

| Column | Type | Notes |
|---|---|---|
| `environment_id` | TEXT PK | UUID generated at capture time. |
| `os` | TEXT | `platform.platform()` output (e.g. `Linux-6.5.0-...`). |
| `arch` | TEXT | `platform.machine()` (`x86_64`, `arm64`, etc.). |
| `python` | TEXT | Full `sys.version`. |
| `hostname` | TEXT | `platform.node()`; NULL if unavailable. |
| `git_commit` | TEXT | `git rev-parse HEAD` at run time. NULL under Docker (no `.git` in image) and NULL if git is not on `$PATH`. |
| `git_dirty` | INTEGER | Tri-state: 1 = dirty, 0 = clean, NULL = probe failed. Never conflate NULL with clean. |
| `lib_versions` | TEXT | JSON blob: `{"anthropic": "0.34.2", "openai": "...", ...}`. Whitelisted runtime deps only (see `db/environment.py`). |
| `docker_image_digest` | TEXT | Value of `MAESTRO_IMAGE_DIGEST` at build time. NULL if the build did not pass it. |
| `captured_at` | TEXT NOT NULL | UTC ISO 8601. |

### 2.2 `run_configs`

One row per matrix cell. Keyed by a run-scoped UUID.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK | UUID. Foreign key target for `run_results`, `sub_results`, `metric_results`. |
| `strategy` | TEXT NOT NULL | `Strategy` enum value (`single_agent`, `sop_based`, `crew_ai`, `lang_graph`, `null_control`, `copy_control`, `ground_truth_control`). |
| `model` | TEXT NOT NULL | Model name from `MODELS` registry, or the literal `"control"` for control rows. |
| `example_id` | TEXT NOT NULL | Input identifier from `INPUTS` (e.g. `bpmn_1_03`). |
| `tier` | INTEGER NOT NULL | 1 = simple, 2 = complex, 3 = cross-layer. |
| `run_number` | INTEGER NOT NULL | Repeat index (1..N). Controls always use 1. |
| `timestamp` | TEXT NOT NULL | UTC ISO 8601 at cell start. |
| `environment_id` | TEXT | FK -> `run_environments.environment_id`. NULL on pre-provenance rows. |

### 2.3 `run_results`

One row per matrix cell (paired with `run_configs` by `run_id`).
A failed cell still gets a row; that is the "recorded, never silent"
guarantee.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK | FK -> `run_configs.run_id`. |
| `output_diagram_code` | TEXT | The extracted Mermaid diagram. NULL on failure. |
| `raw_response` | TEXT | Unprocessed model output as returned by the provider, kept even when the cell fails so the malformed text can be inspected. NULL if the provider returned no candidate at all (safety block, empty response). |
| `prompt_tokens` | INTEGER NOT NULL | Prompt/input token count. 0 for controls and for cells that failed before any call. |
| `completion_tokens` | INTEGER NOT NULL | Completion/output token count. |
| `duration_ms` | INTEGER NOT NULL | Wall-clock latency of the cell. |
| `cost_usd` | REAL NOT NULL | Computed at write time from token counts and the `ModelPricing` rate captured in `experiment_config.py`. Never recomputed at read time, so a later pricing change does not alter historical rows. |
| `error` | TEXT | Human-readable error string. NULL means success (the sole flag for `is_success`). |
| `retry_count` | INTEGER NOT NULL DEFAULT 0 | Number of retries the provider's retry policy consumed for this cell. |

A cell "succeeded" iff `error IS NULL AND output_diagram_code IS NOT NULL`.

### 2.4 `sub_results`

Intermediate step outputs for multi-step strategies (SOP, CrewAI,
LangGraph). One row per step (typically three: entity extraction,
relationship extraction, render). Absent for `single_agent` and controls.

| Column | Type | Notes |
|---|---|---|
| `sub_id` | TEXT PK | UUID. |
| `run_id` | TEXT NOT NULL | FK -> `run_configs.run_id`. |
| `step_number` | INTEGER NOT NULL | 1-indexed. |
| `step_name` | TEXT NOT NULL | Free-form (e.g. `extract_entities`). |
| `output_text` | TEXT | Extracted step output (JSON or Mermaid). NULL if the step failed validation. |
| `raw_response` | TEXT | Raw model output for this step, kept even when the step's parsed output is rejected. |
| `prompt_tokens` | INTEGER NOT NULL | |
| `completion_tokens` | INTEGER NOT NULL | |
| `duration_ms` | INTEGER NOT NULL | |
| `cost_usd` | REAL NOT NULL | Per-step cost; the parent `run_results.cost_usd` is the total. |
| `error` | TEXT | NULL on success. |
| `retry_count` | INTEGER NOT NULL DEFAULT 0 | |

### 2.5 `metric_results`

Scored metrics for successful cells. Not written for failed cells (they
have no diagram to score).

**Structural validity:**

| Column | Type | Notes |
|---|---|---|
| `metric_id` | TEXT PK | UUID. |
| `run_id` | TEXT NOT NULL | FK -> `run_configs.run_id`. |
| `parses_valid` | INTEGER | 1 if `mmdc` parsed the diagram, 0 if it rejected it, NULL if `mmdc` was unavailable (metric skipped). |
| `parse_error` | TEXT | `mmdc` stderr excerpt when `parses_valid = 0`. NULL otherwise. |

**Entity metrics** (three variants: id, name, lemma):

Each variant contributes `_precision`, `_recall`, `_f1` (REAL NOT NULL):

- `entity_id_*`: exact-id match. This is the strictest.
- `entity_name_*`: label match on `name + [Type]` (descriptor line
  intentionally out of contract). Uses fuzzy comparison against the
  ground truth.
- `entity_lemma_*`: label match after lemmatisation (plural stripping,
  separator normalisation). Catches "Tasks" -> "task".

**Relationship metrics** (two variants):

- `relationship_relaxed_*`: matches direction and endpoints, ignores
  edge label.
- `relationship_strict_*`: also requires the edge label to match.

**Container metrics** (subgraphs; pool / lane / boundary / expanded
sub-process). `_precision`, `_recall`, `_f1` are nullable REAL: NULL means
the ground truth had no containers, so the metric is not applicable for
that diagram.

- `container_id_*`
- `container_name_*`

**Attachment metrics** (`o--o` edges: BPMN boundary events, network
associations). Nullable for the same reason.

- `attachment_precision`, `attachment_recall`, `attachment_f1`

**Counts** (all `INTEGER NOT NULL`):

- `entities_in_output`, `entities_in_truth`
- `relationships_in_output`, `relationships_in_truth`
- `containers_in_output`, `containers_in_truth`
- `attachments_in_output`, `attachments_in_truth`

**Error taxonomy** (all `INTEGER NOT NULL`; counts, not F1):

- `missing_entities`, `extra_entities`, `false_entities`, `duplicate_entities`
- `missing_relationships`, `extra_relationships`, `false_relationships`, `duplicate_relationships`

`extra_*` counts predictions the ground truth does not have; `false_*` is
a further breakdown of those into ones the model hallucinated versus ones
that describe something not in the input. See
`src/maestro/analysis/metrics.py` for the exact predicates.

---

## 3. Accessing the database

### 3.1 Read-only, from Python

Use the runner's own helper to enforce the read-only contract:

```python
from maestro.db.client import get_readonly_connection
from maestro.experiment_config import DB_PATH

with get_readonly_connection(DB_PATH) as conn:
    row = conn.execute(
        "SELECT COUNT(*) FROM run_results WHERE error IS NULL"
    ).fetchone()
    print(row[0])
```

### 3.2 Read-only, from the shell

```bash
sqlite3 'file:out/maestro.db?mode=ro'
```

The `mode=ro` opens the file read-only even if a run is currently
writing, so an ad-hoc query never contends with the writer.

### 3.3 Read-only, with pandas

```python
import sqlite3
import pandas as pd

with sqlite3.connect("file:out/maestro.db?mode=ro", uri=True) as conn:
    df = pd.read_sql(
        """
        SELECT c.strategy, c.model, c.tier, m.entity_id_f1, m.relationship_relaxed_f1
        FROM run_configs c
        JOIN metric_results m ON m.run_id = c.run_id
        WHERE c.strategy NOT IN ('null_control', 'copy_control', 'ground_truth_control')
        """,
        conn,
    )
```

---

## 4. Common queries

### 4.1 Success rate per strategy

```sql
SELECT
  c.strategy,
  COUNT(*)                                  AS total,
  SUM(r.error IS NULL AND r.output_diagram_code IS NOT NULL) AS successes,
  1.0 * SUM(r.error IS NULL AND r.output_diagram_code IS NOT NULL) / COUNT(*) AS success_rate
FROM run_configs c
JOIN run_results r ON r.run_id = c.run_id
GROUP BY c.strategy
ORDER BY c.strategy;
```

### 4.2 Cost per strategy, excluding controls

```sql
SELECT
  c.strategy,
  ROUND(SUM(r.cost_usd), 4) AS total_usd,
  ROUND(AVG(r.cost_usd), 6) AS mean_cost_per_cell
FROM run_configs c
JOIN run_results r ON r.run_id = c.run_id
WHERE c.strategy IN ('single_agent', 'sop_based', 'crew_ai', 'lang_graph')
GROUP BY c.strategy;
```

### 4.3 Trace a specific result to its runtime environment

```sql
SELECT
  c.run_id, c.strategy, c.model, c.example_id, c.run_number,
  e.os, e.python, e.git_commit, e.docker_image_digest, e.lib_versions
FROM run_configs c
LEFT JOIN run_environments e ON e.environment_id = c.environment_id
WHERE c.run_id = ?;
```

### 4.4 Retrieve a failed cell's raw model output

```sql
SELECT c.strategy, c.model, r.error, r.raw_response
FROM run_configs c
JOIN run_results r ON r.run_id = c.run_id
WHERE r.error IS NOT NULL
ORDER BY c.timestamp DESC
LIMIT 10;
```

---

## 5. Schema evolution

Additive migrations only. `init_db` inspects `PRAGMA table_info` and issues
`ALTER TABLE ... ADD COLUMN` for any missing column, so an older database
opens without loss. There is no down-migration and no backfill; old rows
keep their NULLs.

A change that is not additive (a column removal, a type change, a
semantic redefinition) requires a fresh database. Delete `out/maestro.db`
and re-run the matrix; see `docs/running.md` for the resume semantics.