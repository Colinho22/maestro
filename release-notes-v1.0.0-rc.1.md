## MAESTRO v1.0.0-rc.1 - Release candidate (pipeline validation)

A release candidate cut before the frozen `v1.0.0` thesis run. Its purpose is to
confirm the toolchain produces sane output end to end, Docker build, matrix
shape, scoring pipeline, and a small smoke run, before committing to the full
multi-hour experiment. It is NOT the code state that produced the thesis data;
that will be the `v1.0.0` tag.

### Why this candidate exists

The full matrix is a long, paid run. Tagging a candidate first lets the Docker
image, the experiment matrix, and the metric pipeline be validated on a small
subset (for example `--repeats 1` on tier 1) so a configuration or environment
problem is caught cheaply rather than halfway through the real run.

### What this release contains

- Four orchestration strategies under test: `single_agent`, `sop_based`,
  `crew_ai`, `lang_graph`, holding prompts and the output contract identical so
  only the orchestration differs.
- Three control conditions (no LLM, deterministic): `null_control` and
  `copy_control` (score floor), `ground_truth_control` (score ceiling).
- Five providers: Anthropic, OpenAI, Mistral, Gemini, DeepSeek, across a matrix
  of `inputs x strategies x models x repeats`, stratified by complexity tier.
- Evaluation pipeline: structural validity (via `mmdc`), entity F1 (id / name /
  lemma), relationship F1 (relaxed / strict), container and attachment metrics,
  and an error taxonomy.
- Reproducibility instrumentation: per-invocation environment capture (OS, arch,
  Python, git commit, library versions, Docker image digest), per-call retry
  counts, and control-condition sanity floors and ceiling.

### What changed going into the candidate

This candidate folds in the pre-freeze code cleanup: an ASCII-only sweep of code
and tests, modern typing throughout (`from __future__ import annotations`, no
`typing.Optional`), a read-only analysis DB connection, `tenacity` added to the
provenance whitelist, structured empty-response handling across all providers,
and a single shared step-output contract for the multi-step strategies. None of
it changes successful scored output; the empty-response work only affects how an
already-failing cell is labeled.

### How to validate the pipeline with this candidate

```bash
git clone https://github.com/Colinho22/maestro.git
cd maestro
git checkout v1.0.0-rc.1
cp .env.template .env            # add API keys for the providers you will run

# Matrix shape, no API calls:
python -m maestro.run --dry-run

# Small smoke run end to end:
python -m maestro.run --strategy single_agent --tier 1 --repeats 1
# Docker: docker compose run --rm maestro \
#   python -m maestro.run --strategy single_agent --tier 1 --repeats 1
```

If the smoke run scores and persists a row and `--dry-run` shows the expected
matrix, the toolchain is ready for the `v1.0.0` freeze and full run.

### Scope

In scope: validating Docker, matrix shape, and the scoring pipeline on a small
subset. Out of scope: the thesis dataset (produced under `v1.0.0`) and any
post-candidate cleanup (which lands on `main` toward `v1.0.0`). This candidate
tag stays frozen.
