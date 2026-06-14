# CLAUDE.md

Project memory for Claude Code. MAESTRO is a benchmark comparing agentic orchestration
strategies (single-agent, SOP, CrewAI, LangGraph) that generate relational Mermaid
diagrams from structured inputs and score them against ground truth. It began as a
master's thesis artifact and is built to be replicated and run by others.

Full conventions live in `.github/CONTRIBUTING.md`, imported here so they are always in
context:

@.github/CONTRIBUTING.md

## Non-negotiables

Protect these three above convenience; every rule exists to serve one of them:

1. A run is reproducible. The exact stack that produced a number is recorded.
2. A failure is recorded, never silent. A bad cell becomes a row, not a missing one.
3. One bad cell never crashes a multi-hour run.

## Always

- No typographic dashes anywhere in code, docstrings, comments, or commit messages. Use a
  plain ASCII hyphen or rephrase. Never en dashes or em dashes.
- Docstrings explain why, not what, and stay concise. Skip them on trivial `__init__`,
  one-line private helpers, and trivial properties.
- One retry mechanism: route all retryable LLM work through
  `providers/_retry.call_with_retry`. Never hand-roll a retry loop.
- Providers never raise: `complete()` catches SDK errors and returns a `RunResult` with
  `.error` set. Environment capture fails soft (records `None`), never aborts.
- `db/` is the only writer. `analysis/` and `viz/` open the database read-only. Schema is
  code; the `.db` is a build output.
- Never change scoring silently: a prompt-contract or metric change is a version bump, so
  pre-change and post-change runs are never mixed. Do not touch experiment behaviour during
  a freeze without flagging it.
- Typing: `from __future__ import annotations` at the top, modern generics (`str | None`,
  `list[...]`), Pydantic models in `schemas.py` for anything persisted or crossing a layer.
- Naming: provider classes end in `Provider`, strategy classes end in `Strategy`.

## Module map

- `providers/` one adapter per LLM SDK (SDK call to `RunResult`)
- `strategies/` orchestration only (build prompts, sequence calls, assemble results)
- `analysis/` scoring and statistics, reads the DB
- `db/` schema, persistence, environment capture, the only writer
- `schemas.py` Pydantic contracts between layers
- `prompts.py` the Mermaid output contract, imports nothing from `maestro`
- `run.py` CLI entry and matrix building

## Commands

```bash
pip install -e ".[dev]"     # set up the dev environment
pytest                      # run tests (the CI gate; pre-commit does not run them)
ruff check .                # lint
ruff format --check .       # formatting check (ruff format is authoritative)
python -m maestro.run --strategy single_agent --tier 1 --repeats 1   # single-cell smoke run
```

## Before committing

- Run `ruff check .` and `pytest` locally. CI gates every PR to `main`.
- Commit messages: `type: short description` (`feat`, `fix`, `refactor`, `docs`, `test`),
  imperative and lower-case.
- Keep commits small and self-contained; do not batch unrelated changes.