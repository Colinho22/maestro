# Changelog

All notable changes to MAESTRO are recorded here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/), and the project uses
[semantic versioning](https://semver.org/): a major bump marks a milestone (see
the release table in `.github/CONTRIBUTING.md`). Each tagged release is also a
GitHub Release. A published `maestro.db` is integrity-anchored by the SHA-256
committed alongside its data release, so a downloaded database can be verified
against the exact file the results came from.

## [1.0.2] - 2026-07-24

Polished visualization and analysis refinements on the `v1.0.1` dataset. The
result database is unchanged; this release changes how those results are
analysed and presented, so pre-change and post-change statistics must not be
mixed (see the scoring-contract note below).

### Added

- Two runnable notebooks under `src/maestro/viz/`, each reproducing a set of
  thesis exhibits from the frozen database. `core-visualization.ipynb` renders
  the figures (reliability funnel, correctness with confidence intervals,
  pairwise strategy contrasts, error profiles, correctness-vs-cost, the
  strategy x model heatmap) and writes an SVG and a PNG per figure plus a
  `figure_values.json` of the exact plotted values. `analysis_tables.ipynb`
  formats the statistical tables (per-strategy F1, the one- and two-way
  ANOVAs, per-tier outcomes, error rates, efficiency, mixed-effects estimates,
  per-model reliability) by reading the canonical analysis JSON, never
  recomputing a statistic.
- Reusable read-only queries in `viz/queries.py`, promoted from the notebooks
  so the notebooks and the dashboard share one tested source:
  `run_outcomes_by_strategy`, `mean_entity_id_f1_by_strategy_by_convention`,
  `run_rates_by_tier`, `valid_rate_by_model`, `taxonomy_rates_per_valid_diagram`,
  and `efficiency_by_strategy`. Each carries a test pinning its population and
  grain (per-cell means for correctness, pooled counts for rates).
- Outcome-neutral theme colors (`OUTCOME_ERROR_COLOR`, `OUTCOME_INVALID_COLOR`)
  for the reliability funnel, kept achromatic so the two failure modes never
  read as additional strategies.
- An `nbstripout` pre-commit hook that strips notebook outputs, execution
  counts, and volatile metadata before commit, so the notebooks are committed
  as runnable sources rather than result artifacts.
- `mixed_effects_robustness` output: a `MixedLM` fit of
  `entity_id_f1 ~ strategy * tier` on the un-aggregated runs, with crossed
  random intercepts for model and input, as a robustness check on the
  aggregated ANOVA. Degrades to a skip-stub if the mixed model does not
  converge.
- Output contract `schema_version` bumped to `1.1`; each inferential file
  records its `unit_of_analysis` and `scoring_convention`.
- `effect_sizes` output gains a `summary` block: the range of absolute
  Cohen's d across the pairwise strategy contrasts, plus the widest contrast.
  Reporting a null needs an effect size beside the non-significant p, and a
  results table cannot carry one value per pair. Pairs with no finite d are
  counted (`n_sentinel_pairs`, `n_undefined_pairs`) rather than dropped, so a
  partial range is never presented as a complete one. `report.md` prints the
  range for both conventions. Purely additive, so `schema_version` stays
  `1.1`: no previously reported number changes.

### Changed

- The `crew_ai` strategy displays as `CrewAI` (one word, as the vendor spells
  it) across figures, tables, and the dashboard.
- **Scoring contract: inferential tests now aggregate to per-cell means under
  an explicit scoring convention.** This changes what the reported statistics
  mean, so it is a versioned scoring change: pre-change and post-change numbers
  must not be mixed in one analysis.
  - The ANOVA / Tukey / Cohen's d path previously fit the raw per-run rows,
    treating the five repeats of a cell as independent (pseudoreplication).
    They now fit one observation per (strategy, model, input) after averaging
    repeats. Reported `n` is the cell count, not the run count.
  - Two scoring conventions are made explicit and are each emitted as their
    own file (`<analysis>__intent_to_treat.json`, `<analysis>__valid_only.json`).
    `intent_to_treat` (primary) scores a failed or unrenderable diagram as 0.0;
    `valid_only` keeps only renderable diagrams. The prior implicit behaviour
    (drop failures, keep an unrenderable diagram at its partial F1) matched
    neither convention and is gone.
  - `db.queries.fetch_analysis_rows` now LEFT JOINs `metric_results` so
    outright failures (no scored row) reach the analysis layer and can be
    scored 0.0 under `intent_to_treat`.

## [1.0.1] - 2026-06-21

Experiment data. The result database produced by the `v1.0.0` code, published
as a release asset alongside its SHA-256 (`maestro.db.sha256`), and archived on
Zenodo as [`10.5281/zenodo.20792757`](https://doi.org/10.5281/zenodo.20792757).

### Dataset

- 6,000 evaluated cells (30 inputs x 4 strategies x 10 models x 5 repeats) plus
  90 deterministic control rows. 5,612 cells scored successfully; 478 failures,
  all recorded with the raw model output (`raw_response`) for diagnosis.
- Produced by `v1.0.0` in Docker, run timestamp 2026-06-21T07:16 to 11:19 UTC,
  total API cost USD 171.62. Library versions are captured per invocation in
  `run_environments.lib_versions`.

### Provenance note

The run executed inside the Docker image, which does not contain the `.git`
directory, so the automatic `git_commit` / `git_dirty` / `docker_image_digest`
columns in `run_environments` are NULL (environment capture fails soft by
design rather than aborting the run). The data-to-code link is instead
established by: this database was produced by the `v1.0.0` tag (the only tagged
code at the run time, on a clean working tree, with nothing committed between
the tag and the run), and its integrity is anchored by the committed SHA-256.
A future run that captures `git_commit` in Docker (passing the commit and image
digest as build arguments) is tracked for `v1.0.2+`.

## [1.0.0] - 2026-06-21

Thesis experimental run. The frozen code state that produced the experimental
data reported in the MAESTRO thesis (FHGR FS26).

### Under test

- Four orchestration strategies: SingleAgent, SOP, CrewAI, LangGraph, holding
  prompts and the output contract identical so only orchestration differs.
- Three control conditions (no LLM, deterministic): NullControl and
  CopyInputControl (score floor), GroundTruthEchoControl (score ceiling).
- Ten models across five providers: Anthropic (claude-opus-4-8,
  claude-haiku-4-5-20251001), OpenAI (gpt-5.5-2026-04-23,
  gpt-5.4-mini-2026-03-17), Mistral (mistral-medium-3-5, mistral-small-2603),
  Google (gemini-3.5-flash, gemini-3.1-flash-lite), DeepSeek (deepseek-v4-pro,
  deepseek-v4-flash).

### Added

- Diagram-type-aware label rendering: C4 and network-topology diagrams use the
  `name\n[Type]\ntech` label, BPMN keeps bare names. The diagram type is read
  from input metadata and given to every strategy as task context.
- Concurrent matrix execution, capped per provider (`--provider-concurrency`,
  default 4), with the main thread as the sole DB writer.
- `raw_response` captured on every cell (including failures) for post-run
  diagnosis, alongside per-call retry counts and per-invocation environment
  capture (OS, arch, Python, git commit, library versions, image digest).

### Changed

- The entity-name metric scores the input-derivable label core (name and type);
  inconsistently-authored descriptor lines and labels for input-unnamed nodes are
  out of the scored contract by design.
- Step-3 output is structurally validated (empty-label brackets, concatenated
  nodes, unbalanced subgraphs) so a fixable malformation consumes the retry
  budget instead of scoring as a parse failure.
- CrewAI's delivered prompt is stripped to match SOP byte-for-byte, removing a
  prompt-content confound.

### Reproduce

```bash
git clone https://github.com/Colinho22/maestro.git
cd maestro && git checkout v1.0.0
cp .env.template .env            # add API keys
docker compose build
docker compose run --rm maestro python -m maestro.run --repeats 5
```

## [1.0.0-rc.1] - 2026-06-14

Release candidate cut to validate the toolchain (Docker build, matrix shape,
scoring pipeline, a small smoke run) before committing to the full paid run. Not
the thesis dataset; that is produced under `v1.0.0`.

### Added

- Pre-freeze code cleanup: ASCII-only sweep, modern typing throughout, a
  read-only analysis DB connection, `tenacity` in the provenance whitelist,
  structured empty-response handling across all providers, and a single shared
  step-output contract for the multi-step strategies. None of it changes
  successful scored output.
