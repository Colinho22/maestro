# Contributing to MAESTRO

These are the conventions the codebase already follows, written down in one place so
that cleanup, future development, and replication all pull in the same direction. They
are descriptive first (this is how MAESTRO is built) and prescriptive second (build new
code the same way).

If a rule here ever conflicts with the code right next to you, match the code and open
an issue. Local consistency beats any rule below.

---

## 0. Guiding principle

MAESTRO began as a master's thesis artifact, but it is built as a benchmark meant to be
replicated, extended, and run by others. It is not a throwaway experiment script, and it
is not a general-purpose library. The thesis is its origin; the benchmark is its purpose.
The `v2.0.0` line, which ships with an empty DB so anyone can run the matrix from scratch,
is the explicit target: a self-service benchmark a stranger can clone and use without the
author in the loop.

Optimise for that, replication and independent use, ahead of API ergonomics or exhaustive
library-style polish.

Three things are non-negotiable; everything else is a preference:

- A run is **reproducible**. The exact stack that produced a number is recorded.
- A failure is **recorded, never silent**. A bad cell becomes a row, not a missing one.
- The experiment **never crashes on one bad cell**. One failure does not end a multi-hour run.

Every rule below exists to protect one of those three, so that a stranger with a clone and
API keys gets the same numbers you did.

---

## 1. Development setup

Two paths. Use whichever fits what you are doing. The README has the full experiment-run
guide; this section is just enough to get a working dev environment.

**Local** (fast iteration, tested on macOS and Linux):

```bash
pip install -e ".[dev]"     # package + ruff + pytest
pre-commit install          # wire the ruff git hook
pytest
ruff check .
ruff format --check .
```

The structural-validity metric shells out to `mmdc` (mermaid-cli). Locally it is optional:
if `mmdc` is absent the metric is recorded as NULL and everything else still runs.

**Docker** (cross-platform, reproducible, and the recommended path on Windows):

```bash
docker compose build
docker compose run --rm maestro python -m maestro.run --strategy single_agent --tier 1 --repeats 1
```

The image bundles Python, mermaid-cli, and Chromium, so the structural-validity metric
always works and the run is platform-independent.

Rule of thumb: **local** for day-to-day code changes and tests, **Docker** for a clean
reproducible run, for Windows, or to match the environment a replicator will use.

---

## 2. Project layout & module boundaries

`src/maestro/` is the only shipped code. `tests/` mirrors it. Everything else (`data/`,
lockfiles, build output) is input or generated, and is excluded from review tooling.

Each module has one job. Keep the boundaries clean:

- **`providers/`**: one adapter per LLM SDK. Translate an SDK call into a `RunResult`. No
  orchestration logic here.
- **`strategies/`**: orchestration only. Build prompts, sequence calls, assemble a
  `(RunResult, list[SubResult])`. No SDK specifics leak in.
- **`analysis/`**: scoring (`metrics.py`) and statistics (`statistics.py`). Reads the DB,
  never writes experiment rows.
- **`db/`**: schema, persistence, environment capture. The only place that writes
  experiment data.
- **`schemas.py`**: every persisted or cross-layer shape, as Pydantic models. This is the
  contract between layers.
- **`prompts.py`**: the canonical Mermaid output contract. Imports nothing from `maestro`
  (circular-import guard, keep it that way).
- **`run.py`**: CLI entry and matrix building. The only module allowed to read env or argv
  and set process-level state.

**Single-image monorepo:** the Streamlit dashboard ships in core dependencies, not an
extra, because one container serves either the experiment or the dashboard, never both at
once. One `pip install -e .` covers everything.

---

## 3. Naming

- LLM provider classes end in **`Provider`** (`AnthropicProvider`). This avoids a collision
  with the vendor client class (`anthropic.Anthropic`).
- Strategy classes end in **`Strategy`** (`SOPStrategy`).
- `_private` for module-internal helpers, `_PRIVATE_CONST` for internal class attributes
  (`_PROVIDER_NAME`, `_RETRYABLE_STATUS`), module-level `CAPS` for tunables (`MAX_ATTEMPTS`).
- An adapter that bridges a third-party contract to ours gets a descriptive name, not a
  suffix (`MaestroBackedLLM`).
- The persisted `Strategy` enum value and a strategy's runtime `.name` (its class name) are
  different things. Do not conflate them in logs or rows.

---

## 4. Typing & data contracts

- `from __future__ import annotations` at the top of every module.
- Modern builtin generics: `str | None`, `list[dict]`, `tuple[RunResult, list[SubResult]]`.
  No `typing.Optional`, `List`, or `Tuple` in new code.
- Type every public signature. A private one-liner may omit the return type when it is
  obvious from the body.
- Anything that crosses a layer boundary or is persisted is a Pydantic model in
  `schemas.py`, not a bare dict and not a tuple. Dicts are fine inside a function.

---

## 5. Docstrings & comments

The house style is **"why," not "what."** The signature already says what; the docstring
says why the thing exists, why it is shaped this way, and what would break if you changed
it. `providers/_retry.py` and `providers/deepseek.py` set the bar.

Scope (mirrors `coderabbit.yaml`). Write a substantive docstring on:

- classes,
- public functions,
- any function whose behaviour is not obvious from name plus type signature.

Do **not** add one to:

- `__init__` that only calls `super()` and assigns attributes,
- single-line private helpers under a parent that already documents the pattern,
- trivial `@property` accessors.

Style rules for all prose in code:

- **Concise and clear.** Every sentence earns its place. Depth is welcome when it records a
  real decision (see `_retry.py`); padding and restating the code are not.
- **No typographic dashes.** Use a plain ASCII hyphen (`-`) or rephrase. Do not use en
  dashes or em dashes in docstrings or comments: they are an authorship tell and can trip
  encoding. Write a range as `3 to 5` or `3-5`, never with an en dash.
- Comments record **decisions and tradeoffs, not mechanics.** A comment that explains a
  non-obvious choice (a version pin, a workaround, an intentional no-op) is required; one
  that restates the code is noise.
- Keep the `# ---- Section ----` dividers in longer modules. They are how the files stay
  navigable.

---

## 6. Error handling & resilience

- **Providers never raise.** `complete()` catches every SDK exception and returns a
  `RunResult` with `.error` set. The boundary should still fail loudly on a programmer error
  (calling a real strategy with `provider=None` gives an `AttributeError`), but a transient
  API error must become a recorded failed row, not a crash.
- **Observability code fails soft.** Environment capture records `None` for any probe that
  fails and keeps going. It must never abort the run it is describing. The nullable schema
  fields exist for exactly this.
- **One retry mechanism.** All retryable work goes through `providers/_retry.call_with_retry`:
  tenacity, `MAX_ATTEMPTS`, exponential-jitter backoff, and a `RetryStats` for `retry_count`.
  Each provider supplies its own `is_retryable` predicate (SDK exception hierarchies differ),
  but the policy lives in one place.
- **`retry_count` is caller-owned.** Pre-create a `RetryStats` and pass it in, so the count
  survives an exhausted-retries exception and still lands on the failed `RunResult`.

The retry path and the `_error_result` contract are not yet uniform across the codebase.
See the Cleanup backlog at the end of this doc.

---

## 7. Persistence & DB interaction

How the code interacts with the database, independent of which DB file is committed at
which release.

- **Schema is code.** The `CREATE TABLE` statements in `db/` are the single source of truth;
  the `.db` is a build output, not a hand-edited file. Recreating from schema must always
  produce the same shape.
- **One writer, many read-only readers.** `db/` is the only writer of experiment data.
  `analysis/` and `viz/` open the DB read-only. The dashboard never mutates results.
- **Writes are idempotent, runs are resumable.** Completed cells are skipped on restart, so
  an interrupted matrix resumes with the same command and never double-writes.
- **Provenance is linked, not loose.** `run_environments` to `run_configs` to `run_results`
  and `metric_results`, by foreign key. Every result traces back via its `environment_id` to
  the exact stack (git commit, library versions, image digest) that produced it.
- **Derived values are computed at write time.** `cost_usd` is computed from token counts and
  the `ModelPricing` rate at the moment of the run, not recomputed at read time. The stored
  number is the one that was true when the call happened.
- **Keep the library whitelist in sync.** `db/environment.py:_LIB_WHITELIST` records the
  runtime deps in `pyproject.toml` that affect a recorded number: anything that produces or
  scores a run. A dep that affects results but is not whitelisted silently stops being
  recorded, and you only notice when a replication diverges and the smoking gun is missing.
  The viz-only deps (streamlit, matplotlib) ship in core but never change a recorded number,
  so they are deliberately excluded; the list is provenance, not an installed-package
  inventory. A whitelisted-but-transitive dep (scipy) carries a one-line why, same as a pin.

**Accessing the results DB.** `maestro.db` is a standard SQLite file (`./out/maestro.db`
under Docker). To inspect it:

- `sqlite3 maestro.db` for ad-hoc SQL, or any SQLite GUI (DB Browser for SQLite, DataGrip).
- From Python, the stdlib `sqlite3`, or `pandas.read_sql(query, sqlite3.connect(...))`.
- `python -m maestro.analysis` runs the canonical analysis for you; the dashboard reads the
  same file read-only.

When poking at a DB that a run might be using, open it read-only so you do not lock the
writer: `sqlite3 'file:maestro.db?mode=ro'` on the CLI, or
`sqlite3.connect('file:maestro.db?mode=ro', uri=True)` in Python. Join along the foreign-key
chain (`run_environments` to `run_configs` to `run_results` and `metric_results`).

---

## 8. Reproducibility rules

- **`temperature = 0`**, set once on `LLMProvider` and inherited by all. Do not re-set it per
  provider.
- **Repeat every cell** (3 to 5 times) and report variance, not single scores. Determinism is
  best-effort even at temperature 0.
- **Versioned changes.** Any edit that alters scoring, such as a prompt-contract change or a
  metric redefinition, must be recorded as a version bump so pre-change and post-change runs
  are never mixed in one analysis. This is why the Mermaid-contract refactor had to land
  before scored runs.
- **Pin runtime deps with a reason.** Any non-obvious pin carries a one-line why (a
  supply-chain incident, a moved import path, a required new symbol). `pyproject.toml` is the
  pattern to follow.

---

## 9. Testing

- `tests/` mirrors `src/maestro/`. A module's tests live at the same relative path.
- **Fail fast on config errors.** `build_matrix` rejects an unknown `--strategy` or
  `--example` with exit code 2, so a typo shrinks the matrix loudly, not silently halfway
  through a run.
- **Pin contracts with tests.** Snapshot the canonical prompt text, and assert that every
  provider's `SYSTEM_PROMPT is` the shared constant, so re-inlining is caught.
- **Do not test vendor SDK internals.** Test the boundary: that the right kwargs reach
  `complete()`, and that errors become failed `RunResult`s.
- **`pre-commit` runs ruff only, not pytest.** The suite is slow enough (the LangGraph
  import) to make every commit annoying. CI is the test gate.

---

## 10. Tooling & CI

- **ruff** is both linter and formatter (`E/F/I/W`, line length 88, target py311).
  `ruff format` is authoritative, so do not hand-format.
- The `pre-commit` ruff `rev` tracks `pyproject`'s `ruff>=` in lockstep. Bump them together.
- **Document every per-file ignore inline**, with a why (`run.py` `E402`, `prompts.py` and
  `experiment_config.py` `E501`). No undocumented ignores.
- **CI gates every PR to `main`** (ruff check, ruff format check, pytest). Branch pushes do
  not run CI. GitHub Actions are pinned to commit SHAs, not tags, so a republished tag cannot
  change what runs.

---

## 11. Git, versioning & releases

- **Commit messages:** `type: short description`, where type is one of `feat`, `fix`,
  `refactor`, `docs`, `test`, `init`. Imperative, lower-case.
- **Incremental history is the scientific norm** for an artifact: small, meaningful commits
  that show the work, not one end-of-thesis dump.
- **Release line** (semantic; a major bump marks a milestone):

| Tag | Contents |
|---|---|
| `v1.0.0` | **Experiment state.** The code exactly as it produced the thesis results. |
| `v1.0.1` | **Experiment data.** The result DB published alongside the code. |
| `v1.0.2` | **Polished visualization** included. |
| `v2.0.0` | **Post-publication.** Ships with an empty DB so anyone can run the matrix from scratch. |

- **Tag against the chapter or experiment** each release backs. `CITATION.cff` is the
  canonical citation (GitHub's "Cite this repository" button).

---

## 12. Extending MAESTRO

The extension points are a deliverable, so this doubles as replicator documentation.

**Add a provider**

- Subclass `LLMProvider`, or `OpenAIProvider` if the API is OpenAI-compatible (like
  `DeepSeekProvider`).
- Implement `complete()` and `_is_retryable`; set `_PROVIDER_NAME` and `MAX_TOKENS`.
- Add the package to `_LIB_WHITELIST` and add a `ModelPricing` row.
- Inherit `TEMPERATURE` and `SYSTEM_PROMPT`; do not redefine them.

**Add a strategy**

- Subclass `BaseStrategy`; implement `run()` returning `(RunResult, list[SubResult])`.
- Use the shared retry path, and build failed results via the standard error helper.
- Register it in the strategy registry and the `Strategy` enum.

**Both**

- Add a snapshot or identity test, then run the tier-1 single-cell smoke command before the
  full matrix.