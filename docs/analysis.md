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

```
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

---

## 4. `report.md`

Human-readable summary. Six sections:

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
users always get the metric; local users on Windows or macOS may not.

---

## 6. Handling sparse corpora

The corpus can under-populate a factor (only one input tier, only one
model, only two strategies). Analyses that need at least two levels of
that factor return `status="skipped"` with a `reason` string that names
the underpopulated factor. Downstream code should check the status
before reading terms:

```python
import json

payload = json.loads(open("output/analysis/.../anova_strategy_by_tier.json").read())
if payload["status"] == "ok":
    interaction_p = payload["terms"]["strategy:tier"]["p"]
else:
    print(f"skipped: {payload['reason']}")
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

## 8. Related documentation

- `docs/schema.md`: full database schema reference.
- `docs/running.md`: how to produce the database in the first place.
- `docs/reproducibility.md`: provenance and integrity model.