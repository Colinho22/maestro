# Reproducibility

How MAESTRO makes a scored number traceable to the exact software stack
that produced it, and how to verify a downloaded dataset matches the one
the results came from. Written for anyone auditing, replicating, or
extending a MAESTRO run.

Three properties are non-negotiable:

- **A run is reproducible.** The exact stack that produced any single
  number is recorded alongside it.
- **A failure is recorded, never silent.** A cell that does not produce a
  scorable diagram becomes a labelled row, not a gap.
- **One bad cell never crashes a multi-hour run.**

This document documents the first: the provenance model and how to use it.

---

## 1. What gets captured

Every non-`--dry-run` invocation of `python -m maestro.run` writes exactly
one row to `run_environments` and links every result it produces to that
row. `--dry-run` prints the filtered matrix and exits before any write. The
captured fields are:

| Field | What it records |
|---|---|
| `environment_id` | UUID generated at capture time. |
| `os` | `platform.platform()` output (kernel, distribution, or macOS version). |
| `arch` | CPU architecture (`x86_64`, `arm64`, ...). |
| `python` | Full `sys.version`. |
| `hostname` | `platform.node()`. |
| `git_commit` | `git rev-parse HEAD` at invocation time. NULL when git is unavailable. |
| `git_dirty` | Tri-state: 1 (dirty), 0 (clean), NULL (probe failed). |
| `lib_versions` | JSON blob of installed versions for every whitelisted runtime dependency. |
| `docker_image_digest` | Value of the `MAESTRO_IMAGE_DIGEST` build arg, or NULL. |
| `captured_at` | UTC ISO 8601. |

The full field-level reference lives in `docs/schema.md` section 2.1.

The capture code is intentionally soft: every probe (git, subprocess,
package resolution) that fails records `None` rather than aborting the
run. The whole point of the module is observability, so it must never
abort the run it is supposed to describe.

---

## 2. The library whitelist

`run_environments.lib_versions` records not every installed package, but
the ones whose version can materially change a recorded number. Two
categories qualify:

- **Producers**: SDKs the strategies call (anthropic, openai, mistralai,
  google-genai) and the frameworks they use (crewai, langgraph).
- **Scorers**: analysis dependencies whose implementation details affect
  the statistics output (statsmodels, pandas, scipy).

Runtime tools that never influence a scored number (streamlit,
matplotlib) are deliberately excluded. The whitelist is provenance, not
an installed-package inventory. The list lives in
`src/maestro/db/environment.py:_LIB_WHITELIST` and must be kept in sync
with the runtime dependencies in `pyproject.toml`; the docstring on
`_LIB_WHITELIST` explains why each entry is present.

---

## 3. Tracing a number back to its stack

Every row on `run_configs` carries an `environment_id`. Follow the
foreign key:

```sql
SELECT
  c.run_id, c.strategy, c.model, c.example_id, c.run_number,
  e.os, e.python, e.git_commit, e.git_dirty,
  e.docker_image_digest,
  e.lib_versions
FROM run_configs c
LEFT JOIN run_environments e ON e.environment_id = c.environment_id
WHERE c.run_id = ?;
```

`lib_versions` is a JSON string; parse it with your SQL client's JSON
support or in application code:

```python
import json
libs = json.loads(row["lib_versions"])
print(libs["anthropic"], libs["openai"])
```

`environment_id` may be NULL for rows produced before the provenance
column existed. The additive migration in `init_db` adds the column
without backfilling. In practice, the `v1.0.1` dataset and every later
run carry a linked environment.

---

## 4. Docker vs local capture

Docker and local runs record the same fields, but three fields behave
differently under Docker:

- **`git_commit`**: NULL under Docker. The image does not ship the
  `.git` directory, so `git rev-parse HEAD` returns nothing. The
  data-to-code link is instead established by the release tag the
  image was built from.
- **`git_dirty`**: NULL for the same reason.
- **`docker_image_digest`**: populated only if the build was passed the
  `MAESTRO_IMAGE_DIGEST` build argument. `docker-compose.yml` reads it
  from the shell environment:

  ```bash
  MAESTRO_IMAGE_DIGEST=$(git rev-parse HEAD) docker compose build
  ```

  A build without the argument records NULL. The published `v1.0.1`
  dataset was produced without this argument set; see section 6.

Under local (non-Docker) runs, all three fields are populated as long as
git is available. Capturing everything under Docker is a `v1.0.2+`
improvement tracked separately.

---

## 5. The published `v1.0.1` dataset

The result database produced by `v1.0.0` is published as a GitHub Release
asset alongside its SHA-256 (`maestro.db.sha256`) and archived on Zenodo
as [`10.5281/zenodo.20792757`](https://doi.org/10.5281/zenodo.20792757).
Its integrity is anchored by the committed SHA-256: a downloaded database
can be verified byte-for-byte against the exact file the results came
from.

### 5.1 Verify the download

```bash
sha256sum -c maestro.db.sha256
```

Expected output:

```text
maestro.db: OK
```

The file itself:

```text
2244cae2c6c24999fc9d8889637d4007f819c5658947c89d9a8a0f5a7fb89b0b  maestro.db
```

If the check fails, the database is not the one the results were
produced from. Re-download.

### 5.2 What is inside

- 6,000 evaluated cells: 30 inputs x 4 strategies x 10 models x 5
  repeats.
- 90 deterministic control rows: 30 inputs x 3 controls.
- 5,612 cells scored successfully; 478 failures, each recorded as a row
  in `run_results` with the raw model output (`run_results.raw_response`)
  retained where the provider returned one. `raw_response` is NULL only
  when the provider returned no candidate at all (safety block, empty
  response).
- Produced by `v1.0.0` in Docker, run window 2026-06-21T07:16 to 11:19
  UTC, total API cost USD 171.62.

Library versions are captured per invocation in
`run_environments.lib_versions`.

---

## 6. The `v1.0.1` provenance caveat

The `v1.0.1` run executed inside Docker without `MAESTRO_IMAGE_DIGEST`
passed at build time, so `run_environments.docker_image_digest` is NULL.
The image did not contain the `.git` directory either, so `git_commit`
and `git_dirty` are also NULL. Environment capture failed soft by
design; the run was not aborted.

The data-to-code link is instead established two ways:

- The database was produced by `v1.0.0`, which was the only tagged code
  at run time, on a clean working tree, with nothing committed between
  the tag and the run.
- The database's integrity is anchored by the committed SHA-256.

A future run that captures `git_commit` inside Docker (by passing the
commit and image digest as build arguments) is tracked for `v1.0.2+`.

---

## 7. Determinism at temperature 0

Every provider inherits `TEMPERATURE = 0` from `LLMProvider` and never
overrides it. In practice, temperature 0 is best-effort determinism, not
byte-for-byte determinism: minor variance across repeats is expected and
is why the experiment matrix ships with 5 repeats per cell. Analysis
reports variance rather than single-run scores.

---

## 8. What is not captured

The following are outside the provenance model and should be handled
externally when they matter:

- **Vendor-side model updates.** A provider that silently updates a
  pinned model snapshot changes what `claude-opus-4-8` refers to. Model
  ids are pinned to dated snapshots where the vendor offers them; where
  they do not (or when the vendor rolls a snapshot forward), replication
  after that date is not guaranteed. `captured_at` bounds the window a
  replication must land in.
- **Network conditions.** `duration_ms` includes network latency to the
  provider. A geographically distant replicator will see different
  numbers on the efficiency axis; the correctness axis is unaffected.
- **Rate-limit dynamics.** `retry_count` records how many retries the
  provider's policy consumed, so a run under sustained rate limiting is
  distinguishable from one that was not. This does not change the
  scored diagram, but it changes wall-clock cost expectations.

---

## 9. Related documentation

- `docs/schema.md`: full database schema reference, including every
  provenance column.
- `docs/running.md`: how to invoke the runner and what the
  `MAESTRO_IMAGE_DIGEST` build argument does.
- `.github/CONTRIBUTING.md`: the release-line conventions
  (`v1.0.0` / `v1.0.1` / `v1.0.2` / `v2.0.0`) and their contents.
- `CHANGELOG.md`: history, keyed to release tags.