# Running MAESTRO

Operational reference for running the benchmark. Assumes you have already
cloned the repository and installed it (see the top-level `README.md` for the
one-time setup). This document is the source of truth for CLI flags, the
default experiment matrix, resume semantics, cost and time expectations, and
common failure modes.

Two ways to run the benchmark are supported:

- **Docker (recommended)**: `docker compose run --rm maestro python -m maestro.run ...`
- **Local**: `python -m maestro.run ...`

Every command below is shown in Docker form first, with the local
equivalent underneath.

---

## 1. Prerequisites

- Python 3.11 (local) or Docker (containerised).
- API keys for every provider you intend to exercise. The pre-flight check
  aggregates all missing keys into one error, so you never discover a missing
  key partway through a paid run.
- `mmdc` (mermaid-cli) for the structural-validity metric. Bundled in the
  Docker image. Optional locally: if absent, `parses_valid` is recorded as
  NULL and every other metric still runs.

### 1.1 Configure API keys

Copy the template and fill in the keys you have:

```bash
cp .env.template .env
```

The five recognised variables:

| Variable | Provider |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (Claude models) |
| `OPENAI_API_KEY` | OpenAI (GPT models) |
| `MISTRAL_API_KEY` | Mistral |
| `GEMINI_API_KEY` | Google (Gemini models) |
| `DEEPSEEK_API_KEY` | DeepSeek |

Keys are read from the process environment at run time. `.env` is loaded
automatically by `python-dotenv` in both local and Docker runs.

You only need the keys for the providers you actually want to run. The
matrix builder honours `--model` and `--strategy` filters before the
pre-flight check, so a partial-key run works as long as your filters
exclude the providers whose keys are missing.

---

## 2. The default matrix

Without filters, `python -m maestro.run` executes the full experiment matrix:

- **5 repeats** per non-control cell (`DEFAULT_REPEATS = 5`)
- **4 real strategies**: `single_agent`, `sop_based`, `crew_ai`, `lang_graph`
- **3 control strategies**: `null_control`, `copy_control`, `ground_truth_control`
- **10 models** across 5 providers, in flagship / efficiency pairs:

  | Provider | Flagship | Efficiency |
  |---|---|---|
  | Anthropic | `claude-opus-4-8` | `claude-haiku-4-5-20251001` |
  | OpenAI | `gpt-5.5-2026-04-23` | `gpt-5.4-mini-2026-03-17` |
  | Mistral | `mistral-medium-3-5` | `mistral-small-2603` |
  | Google | `gemini-3.5-flash` | `gemini-3.1-flash-lite` |
  | DeepSeek | `deepseek-v4-pro` | `deepseek-v4-flash` |

- **30 inputs** across three tiers:

  | Tier | Value | Contents |
  |---|---|---|
  | `SIMPLE` | 1 | Fewer than 10 entities |
  | `COMPLEX` | 2 | 10 to 25 entities |
  | `CROSS_LAYER` | 3 | 25+ entities, multi-pool, or cross-layer flows |

Controls collapse both the model and repeat dimensions to a single row per
`(input, control_strategy)` cell, since neither dimension varies for a
deterministic strategy. So the total matrix is:

```text
30 inputs x 4 strategies x 10 models x 5 repeats  = 6000 real cells
30 inputs x 3 control strategies                  =   90 control cells
                                                   ------
                                                    6090 cells
```

The published `v1.0.1` dataset was produced by exactly this matrix.

---

## 3. CLI reference

```text
python -m maestro.run [FILTER FLAGS] [RESUME FLAG] [--dry-run]
```

### 3.1 Filter flags

Every filter accepts a comma-separated list. Filters are validated up front:
a misspelled value aborts with exit code 2 before any API call, so a typo
cannot silently shrink the matrix.

| Flag | Type | Effect |
|---|---|---|
| `--strategy <list>` | comma-separated | Run only these strategies. Values: `single_agent`, `sop_based`, `crew_ai`, `lang_graph`, `null_control`, `copy_control`, `ground_truth_control`. |
| `--tier <int>` | 1, 2, or 3 | Run only inputs of this complexity tier. |
| `--model <list>` | comma-separated | Run only these model names. See the table in section 2. |
| `--example <list>` | comma-separated | Run only these `example_id`s (e.g. `bpmn_1_03,it_1_07`). |
| `--repeats <int>` | integer, default 5 | Override the per-cell repeat count. Controls are unaffected. |

The `--model` filter applies only to real (LLM) strategies. Control rows do
not consume a model and are preserved by every `--model` value, so a
`--strategy null_control --model anything` combination stays a valid no-op.

### 3.2 Concurrency

```text
--provider-concurrency <int>   default: 4
```

Maximum in-flight requests per provider. The runner uses one semaphore per
provider (Anthropic, OpenAI, Mistral, Gemini, DeepSeek), so the total
in-flight ceiling is `providers x --provider-concurrency`. Concurrency does
not change results; it only changes how fast the matrix runs.

Recommended values:

- **1**: free-tier keys, or when a provider is rate-limiting.
- **4** (default): safe for typical paid accounts.
- **8+**: only when your account documents a high rate limit.

Set to 1 if you see repeated 429s from one provider; the retry path
(`providers/_retry.py`) will still handle transient bursts, but a sustained
cap violation is best solved at the source.

### 3.3 Resume flags

Resume behaviour is mutually exclusive:

| Flag | Effect |
|---|---|
| _(default)_ | Skip cells that already have a successful row in the database. Re-run cells whose prior row is a failure (transient errors deserve another attempt). |
| `--no-resume` | Ignore the database and execute every cell in the filtered matrix. Use after deleting or replacing the database, or when a code change means prior rows are no longer comparable. |
| `--rerun-failed` | Execute only cells that have a prior failed row. Skip both successful cells and cells with no row yet. |

Cells are keyed by `(example_id, strategy, model, run_number)`, so a resumed
run picks up exactly where the previous one left off.

### 3.4 Preview

```text
--dry-run
```

Prints the filtered matrix and exits. No API calls, no writes. Useful for
verifying a filter combination before spending money.

---

## 4. Typical invocations

### 4.1 Smoke test

Before any long run, execute one tier-1 cell to confirm the install, keys,
and scoring pipeline work end to end. `--example` and `--model` narrow
the matrix to exactly one cell so the cost is bounded and the check is
fast:

```bash
docker compose run --rm maestro python -m maestro.run \
  --strategy single_agent \
  --example bpmn_1_01 \
  --model claude-haiku-4-5-20251001 \
  --repeats 1
# Local:
python -m maestro.run \
  --strategy single_agent \
  --example bpmn_1_01 \
  --model claude-haiku-4-5-20251001 \
  --repeats 1
```

Expected: one row inserted into `out/maestro.db`, a printed cost around
USD 0.01, no errors.

### 4.2 Full matrix

```bash
docker compose run --rm maestro python -m maestro.run
# Local:
python -m maestro.run
```

Runs the entire 6,090-cell matrix. Resumable: interrupt with Ctrl+C, then
re-run the same command to pick up where it left off.

### 4.3 One provider only

```bash
python -m maestro.run --model claude-opus-4-8,claude-haiku-4-5-20251001
```

The pre-flight check will pass with only `ANTHROPIC_API_KEY` set. Includes
control rows (which do not use a model) automatically.

### 4.4 Tier-2 subset for a paper revision

```bash
python -m maestro.run --tier 2 --strategy sop_based,lang_graph --repeats 3
```

### 4.5 Re-run failures after a provider outage

```bash
python -m maestro.run --rerun-failed
```

### 4.6 Preview the matrix a filter combination would produce

```bash
python -m maestro.run --tier 3 --strategy crew_ai --dry-run
```

---

## 5. Where results go

- **Local**: `out/maestro.db` (relative to the project root).
- **Docker**: `out/maestro.db` on the host, mounted into the container at
  `/app/out/maestro.db`.

Override the path with the `MAESTRO_DB_PATH` environment variable if you
need a different location:

```bash
MAESTRO_DB_PATH=/tmp/experiment.db python -m maestro.run --tier 1 --repeats 1
```

The runner, the analysis module, and the dashboard all read `MAESTRO_DB_PATH`,
so setting it once puts every consumer on the same file.

The database contains:

- `run_environments`: one row per invocation (OS, Python version, library
  versions, git commit, image digest).
- `run_configs`: one row per cell, keyed by `run_id` and linked to an
  environment.
- `run_results`: one row per cell (token counts, duration, cost, raw model
  output, error field).
- `sub_results`: intermediate step outputs for multi-step strategies.
- `metric_results`: scored metrics for successful cells.

See `docs/schema.md` for the full field reference.

---

## 6. What "reproducible" means here

Every invocation snapshots its runtime environment once and links every row
it writes to that snapshot. To reproduce a specific number:

1. Query `run_results` for the row.
2. Follow the foreign key to `run_configs`.
3. Follow that to `run_environments` for the exact stack.

Under Docker, some columns (`git_commit`, `git_dirty`, `docker_image_digest`)
can be NULL because the image does not contain the `.git` directory and the
digest is optional at build time. See `docs/reproducibility.md` for the full
provenance model.

---

## 7. Cost and duration estimates

Ballpark figures from the `v1.0.1` production run (6,090 cells, 5 repeats,
full matrix, Docker, `--provider-concurrency 4`):

- **Wall clock**: about 4 hours.
- **Total API cost**: USD 171.62.

Actual numbers depend on provider pricing on the day, retry activity, and
network conditions. Use the smoke run (section 4.1) plus a tier-1 subset to
estimate before committing.

The runner prints running cost as cells complete, and a final total on
exit. Cost is also recomputed from persisted token counts and the
`ModelPricing` rate captured at write time, so a later change to pricing
does not alter historical rows.

---

## 8. Troubleshooting

### 8.1 "Missing API keys for the following providers"

The pre-flight check found one or more required keys absent. The error
message lists every missing key and which models needed it. Set the missing
keys in `.env` (or your shell environment) and re-run.

If you did not intend to run those providers, narrow the matrix with
`--strategy`, `--model`, or `--example` so the pre-flight check no longer
requires them.

### 8.2 "unknown --strategy value(s)" or similar

You passed a value the runner does not recognise. Exit code 2. The error
message lists the known values. Common causes:

- Typo: `sop` instead of `sop_based`, `crew` instead of `crew_ai`.
- Stale docs: `--strategy` accepts the enum's string value (see section 3.1),
  not the class name.

### 8.3 One provider is being rate-limited

Symptoms: sustained 429s from one provider in the log, cells finishing
slowly, retry counts climbing on `run_results.retry_count`. Options:

- Lower `--provider-concurrency` (try 1 or 2).
- Narrow `--model` to exclude the affected provider and run the rest first,
  then rerun the excluded provider on its own.

### 8.4 `parses_valid` is NULL on every row

The structural-validity metric is skipped when `mmdc` (mermaid-cli) is not
installed. Run under Docker to get the metric, or install `mermaid-cli` and
a Puppeteer Chrome build locally:

```bash
npm install -g @mermaid-js/mermaid-cli
npx puppeteer browsers install chrome
```

Every other metric is unaffected.

### 8.5 "Git working tree is dirty"

Warning, not an error. The runner captures the git commit hash for
provenance, and a dirty tree means the commit hash is an incomplete
description of the code that produced the numbers. Commit or stash before a
run you intend to reference later.

### 8.6 The run halted mid-batch

The design guarantees a single failing cell cannot halt the batch: every
strategy/provider error is caught and recorded as a failed row. A true halt
means either:

- Ctrl+C. Re-run with the same command; resume mode skips the completed
  cells.
- An unexpected exception in the runner harness itself. The stack trace is
  printed to stderr and the cell is logged as "worker crashed; cell lost".
  Other cells continue. Re-run with the same command to retry the lost
  cell.
- The Python process itself was killed (OOM, host restart). Re-run with
  the same command.

### 8.7 Crew AI prompts asking for tracing consent

The runner sets `CREWAI_TESTING=true` and related environment variables
before importing CrewAI to short-circuit the interactive tracing prompt.
If you see the prompt anyway, you are on an older CrewAI or these
variables have been unset. Do not remove them from `run.py`: they only
disable trace prompts and telemetry, not agent execution, and their
absence causes a 20-second stdin timeout per crew cell.

---

## 9. Related documentation

- `docs/schema.md`: full database schema reference.
- `docs/analysis.md`: how to invoke the analysis module and read its output.
- `docs/reproducibility.md`: environment capture, provenance, and DB
  integrity verification.
- `docs/extending.md`: adding a provider, a strategy, or a metric.