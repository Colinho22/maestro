# Changelog

All notable changes to MAESTRO are recorded here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/), and the project uses
[semantic versioning](https://semver.org/): a major bump marks a milestone (see
the release table in `.github/CONTRIBUTING.md`). Each tagged release is also a
GitHub Release; every row in a published `maestro.db` carries the `git_commit`
of the tag that produced it, so data and code stay cross-verifiable.

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
