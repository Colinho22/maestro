# Analysis

Operational reference for `python -m maestro.analysis`: what it computes,
what it writes, and how to consume its output. Written for anyone who
wants scored numbers out of `maestro.db`. Assumes you have already read
`docs/schema.md` for the underlying tables.

The analysis module is compute-only. It reads `maestro.db` read-only,
computes descriptive and inferential statistics into a set of JSON files,
and writes an assembled markdown report. Figures are the dashboard's job
(see `docs/dashboard.md` once viz work lands).

---

## 1. Invocation

```bash
python -m maestro.analysis [--db PATH] [--out DIR] [--display-tz TZ]
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--db` | `out/maestro.db` (or `$MAESTRO_DB_PATH`) | Source database. Opened read-only. |
| `--out` | `output/analysis` | Root directory. A timestamped subdirectory is created inside it per invocation, so consecutive runs never overwrite each other. |
| `--display-tz` | system local (or `$MAESTRO_DISPLAY_TZ`) | IANA timezone for human-readable timestamps in `report.md`. Stored timestamps stay UTC regardless. |

Exit code:

- 0 on success. This includes the case where the database is empty; the
  module writes empty-status outputs and still exits 0 so a downstream
  pipeline can treat the file layout as guaranteed.
- 1 if the database file does not exist.

---

## 2. Output layout

Each invocation writes to `<out>/<timestamp>/`:

```text
output/analysis/20260621T111935Z/
    report.md
    descriptive.json
    anova_strategy.json
    anova_strategy_by_tier.json
    anova_strategy_by_model.json
    posthoc_strategy.json
    effect_sizes.json
    error_taxonomy_by_strategy.json
    tradeoff_correctness_efficiency.json
    failure_rates.json
    survivor_bias.json
    figures/README.md
```

The timestamp is UTC, filesystem-safe, second-precision. Two invocations
in the same second get a `-1`, `-2`, ... suffix.

`figures/` is a placeholder with a README noting that figure generation
is the visualizer's responsibility, not the compute pipeline's. The
directory exists so the layout contract is stable even before any figure
lands.

---

## 3. What each JSON file contains

Every file has a top-level `status` string:

- `"ok"`: the analysis ran and its results are present.
- `"skipped"`: the analysis needed at least two levels of some factor and
  the corpus does not currently provide them (e.g. a single input tier).
  A `reason` string explains which factor was underpopulated. Re-runs
  will populate the analysis automatically once the corpus grows; no
  code change is needed.
- `"empty"`: there were no experimental rows to analyse at all, so the
  analysis had nothing to run on rather than a factor too sparse to fit.
  The distinction from `"skipped"` is deliberate: `"empty"` means no data,
  `"skipped"` means data that will not support this particular test.

### 3.1 `descriptive.json`

Cell-level means (F1 metrics, cost, duration, retry count) grouped by
strategy, model, tier, and their combinations. Controls are included
here as sanity anchors: `null_control` and `copy_control` should sit near
0, `ground_truth_control` should sit near 1.

### 3.2 `anova_strategy.json`

One-way ANOVA on `entity_id_f1` with `strategy` as the sole factor.
`single_agent` is the reference (comparison baseline). Controls are
excluded (their F1 is 0 or 1 by construction and would break the
homoscedasticity assumption).

Payload shape (abridged):

```json
{
  "status": "ok",
  "n": 5612,
  "term_of_interest": "strategy",
  "terms": {
    "strategy": {
      "F": 12.34,
      "p": 0.0001,
      "df": 3,
      "partial_eta_sq": 0.007
    },
    "Residual": { "df": 5608 }
  }
}
```

### 3.3 `anova_strategy_by_tier.json`

Two-way ANOVA: strategy, tier, and their interaction. Answers whether
input complexity moderates the strategy effect.

### 3.4 `anova_strategy_by_model.json`

Two-way ANOVA: strategy, model, and their interaction. A significant
interaction means the strategy effect is model-specific.

### 3.5 `posthoc_strategy.json`

Tukey HSD pairwise comparisons across strategies. Consumed with
`anova_strategy.json` to answer which pairs actually differ.

### 3.6 `effect_sizes.json`

Cohen's d for every strategy pair. Small, medium, and large thresholds
follow the conventional cutoffs (0.2 / 0.5 / 0.8).

### 3.7 `error_taxonomy_by_strategy.json`

Per-strategy counts of hallucinated entities, missing entities, missing
relationships, and the other taxonomy columns. Descriptive only; no
inferential test. Useful for the "what did each strategy get wrong"
question.

### 3.8 `tradeoff_correctness_efficiency.json`

Per-strategy medians on both dimensions: `entity_id_f1` (correctness)
and `cost_usd` / `duration_ms` (efficiency). Consumed by the dashboard's
Pareto view.

### 3.9 `failure_rates.json`

Failure rate and cause breakdown per strategy, model, and tier, plus an
`overall` block. This is the reliability counterpart to the accuracy
tables: `error_taxonomy_by_strategy.json` scores the content of diagrams
that were produced, while this file covers the runs that produced nothing
scorable at all. The two populations are disjoint.

Rates are **pooled counts**: the denominator is every run attempted in the
group, not a mean of per-cell rates. A per-cell mean would weight a cell
with one run as heavily as a cell with five, which for a rate is the wrong
grain. This deliberately differs from the F1 path, which aggregates per
cell because it averages a score rather than counting events.

Every cause appears in every `causes` block, including the zeros: an
absent key would be ambiguous between "never happened" and "not measured".
Controls are excluded (they never call a model).

Payload shape (abridged):

```json
{
  "status": "ok",
  "statistic": "pooled_failure_rate",
  "overall": {
    "n_runs": 6000,
    "n_failed": 478,
    "failure_rate": 0.0797,
    "causes": { "schema_violation": 261, "truncation": 135, "...": 0 }
  },
  "by_strategy": [
    {
      "strategy": "sop_based",
      "n_runs": 1500,
      "n_failed": 151,
      "failure_rate": 0.1007,
      "causes": { "schema_violation": 85, "...": 0 }
    }
  ]
}
```

### 3.10 `survivor_bias.json`

Per strategy, the primary DV under `valid_only` (survivors only) against
`intent_to_treat` (every run, failures scored 0.0). The gap between them
is the survivor bias: a strategy that fails often looks better under
`valid_only` because its failures were dropped rather than scored, and
`survivor_bias` measures exactly how much that dropping flatters it.

Reported per strategy rather than as one pooled figure, because the bias
scales with each strategy's failure rate: a single number would hide the
comparison that matters. `n_cells_dropped` counts cells where every run
failed, so `valid_only` had nothing to average.

---

## 3a. Failure taxonomy

`failure_rates.json` classifies each invalid run into exactly one primary
cause. The classifier lives in `src/maestro/analysis/failures.py`.

### What counts as a failure

Two disjoint shapes, both rejected by `RunResult.success`:

- an **errored** run: `error` is set, and the string is the evidence.
- a **silent-empty** run: `error` is `None` but the diagram is missing or
  blank. A provider can return whitespace without raising, so this is a
  real category, not a data defect.

### Where the evidence lives

On a failed run, `run_results.raw_response` is always `NULL`: the error
result is built before any text exists. The failing text is retained one
level down, on the `sub_results` row that failed. Classification therefore
reads the run's `error` plus the first failed sub-result's `raw_response`.

This means historical runs can be classified retroactively with no model
re-invocation, which is why re-scoring was never needed for this analysis.

### The causes

| Cause | Meaning |
|---|---|
| `rate_limit` | Provider returned a rate-limit error. |
| `timeout` | Request exceeded the client deadline. |
| `api_error` | Generic provider API error (the SDKs' catch-all base class). |
| `safety_block` | Response blocked by a content or safety filter. |
| `empty_output` | Provider returned no text, or only whitespace. |
| `truncation` | Response stopped mid-structure, consistent with a token limit. |
| `parse_error` | Output was not parseable in the requested format. |
| `schema_violation` | Output parsed but broke the Mermaid output contract. |
| `orchestration_error` | The framework misbehaved, not the model output. |
| `unknown` | No rule matched. Visible by design rather than mis-filed. |

### Classification rules

Causes are not mutually exclusive in practice: a truncated response is
usually *also* a parse error, because the truncation is what broke the
parse. Rather than multi-label (which makes rates hard to sum and
compare), each failure gets one primary cause under a fixed precedence,
most specific first:

1. **Infrastructure** (`rate_limit`, `timeout`, `safety_block`). If the
   API never returned, nothing downstream is meaningful.
2. **`empty_output`**. A missing response cannot be a parse error.
3. **`schema_violation`**, before the generic parse rule: well-formed text
   that violates the output contract is a different failure from text that
   is not parseable at all.
4. **`parse_error`**, promoted to **`truncation`** when the parser message
   is truncation-shaped (an unterminated string, or a structure that simply
   stops) *and* the raw response is long enough that running out of tokens
   is plausible. Without the raw text the two are indistinguishable, so the
   conservative `parse_error` label stands: under-reporting truncation is
   safer than inventing it.
5. **`orchestration_error`**.
6. **`api_error`** last among the infrastructure family, since a more
   specific subclass above must win over the catch-all base class.
7. **`unknown`** as the explicit fallback.

Adding a provider means checking whether its error prefixes are covered.
An unmatched prefix surfaces as `unknown` rather than being mis-filed,
which is the failure mode this ordering exists to prevent.

---

## 4. `report.md`

Human-readable summary. Four sections:

1. Header (generation time, DB path, output schema version).
2. **RQ -> output-file mapping** table. This is the interpretation layer,
   deliberately outside the JSON so the numeric outputs stay reusable if
   the research questions are reframed.
3. **Summary**: headline numbers (descriptive cell count, three ANOVA
   one-liners).
4. **Notes**: caveats about skipped analyses and the control exclusion.

`report.md` is safe to include directly in a paper appendix or a defence
slide deck. The JSON files are the machine-readable source; `report.md`
is the reader-friendly view.

---

## 5. Metric definitions

Every scored dimension is documented at the field level in
`docs/schema.md`. This section documents how the numbers are computed;
the code lives in `src/maestro/analysis/metrics.py`.

### 5.1 Entity metrics (`entity_*`)

Three variants of the same P/R/F1 shape, differing only in how a
predicted entity is matched to a ground-truth entity:

- **id**: exact string match on the Mermaid node id. The strictest
  metric and the ANOVA target (`entity_id_f1`).
- **name**: fuzzy match on `name + [Type]` (the third descriptor line is
  intentionally out of contract; see `_label_core` in `metrics.py`).
- **lemma**: normalised match after lowercasing, separator collapsing,
  and basic plural stripping.

An input node with an empty `name` is scored by id only (see
`docs/data.md` section 3).

### 5.2 Relationship metrics (`relationship_*`)

- **relaxed**: matches direction and endpoints; ignores the edge label.
- **strict**: also requires the edge label to match.

Both are computed on the entity-id space, so a relationship that
references a wrongly-labelled entity still counts if the ids line up.

### 5.3 Container metrics (`container_*`)

Subgraphs (pools, lanes, boundaries, expanded sub-processes). Same P/R/F1
shape as entities. Nullable: a diagram whose ground truth has no
subgraphs contributes NULL, not 0, so downstream analysis can skip the
metric where it does not apply.

### 5.4 Attachment metrics (`attachment_*`)

`o--o` edges (BPMN boundary events, network associations). Undirected.
Same nullable behaviour as containers.

### 5.5 Structural validity (`parses_valid`)

Boolean-typed: 1 if `mmdc` parsed the diagram, 0 if it rejected it, NULL
if `mmdc` was unavailable. The metric is skip-friendly: an absent `mmdc`
does not invalidate a run, it just leaves this column blank. Docker
users always get the metric; local users without `mmdc` installed get
NULL regardless of operating system.

---

## 6. Handling sparse corpora

The corpus can under-populate a factor (only one input tier, only one
model, only two strategies). Analyses that need at least two levels of
that factor return `status="skipped"` with a `reason` string that names
the underpopulated factor. An analysis with no experimental rows at all
returns `status="empty"` instead, and carries no `reason`: nothing was
too sparse to fit, there was simply nothing to fit. Downstream code
should check the status before reading terms:

```python
import json

payload = json.loads(open("output/analysis/.../anova_strategy_by_tier.json").read())
if payload["status"] == "ok":
    interaction_p = payload["terms"]["strategy:tier"]["p"]
else:
    # "empty" carries no reason, so do not index it unconditionally.
    print(f"{payload['status']}: {payload.get('reason', 'no experimental rows')}")
```

Every consumer (the report builder, the dashboard) uses this pattern; a
mid-development run against a partial corpus never breaks the layout.

---

## 7. Reproducibility

The analysis module reads `maestro.db` and produces deterministic output.
Nothing about a scored number depends on the analysis-run timestamp, the
timezone, or the machine the analysis is invoked on. Two runs of
`python -m maestro.analysis` against the same database produce
byte-identical JSON (setting aside the timestamp string in `report.md`).

The exact `statsmodels` / `scipy` / `pandas` versions used are captured
in `run_environments.lib_versions` at experiment time and again at
analysis time (via the analysis-side script's own environment); pinning
them in `pyproject.toml` is what keeps historical numbers stable.

---

## 8. Reported-numbers dump

A companion entry point,

```bash
python -m maestro.analysis.reported_numbers
```

emits `output/analysis/reported_numbers.json`: the headline totals docs
and slide decks cite (total cell count, success / failure split,
aggregate cost). The file is the machine-readable source that
transcribed numbers in tracked prose must match, and the model-registry
consistency test in CI enforces that. Regenerate it after any run that
changes those totals, then update the docs from the file rather than
from a screenshot of the runner output.

Empty database produces a valid `status: "empty"` payload so a fresh
checkout can still write the file and pass the consistency test.

---

## 9. Related documentation

- `docs/schema.md`: full database schema reference.
- `docs/running.md`: how to produce the database in the first place.
- `docs/reproducibility.md`: provenance and integrity model.