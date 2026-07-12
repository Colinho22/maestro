# MAESTRO documentation

Operational documentation for running the benchmark, querying its
outputs, and extending it. For the project overview and quickstart, see
the top-level [`README.md`](../README.md).

## Index

- [**running.md**](running.md): Full CLI reference for
  `python -m maestro.run`. Prerequisites, the default matrix, filter and
  resume semantics, provider concurrency, cost and duration expectations,
  troubleshooting.
- [**data.md**](data.md): The benchmark corpus. Input JSON format,
  ground-truth Mermaid conventions (including the `"a"` placeholder for
  empty labels), how to add a new input.
- [**schema.md**](schema.md): Field-level database reference. The
  foreign-key chain, every column, common query recipes, read-only access
  patterns.
- [**analysis.md**](analysis.md): The `python -m maestro.analysis`
  pipeline. Invocation, output files, metric definitions, and how to
  handle sparse-corpus skip statuses.
- [**extending.md**](extending.md): How to add a provider, a strategy, a
  metric, or an input. Each recipe ends with a smoke test.
- [**reproducibility.md**](reproducibility.md): Provenance model, Docker
  vs local capture caveats, the `v1.0.1` dataset SHA-256 verification,
  the Zenodo archive.
- [**visualization_design_guide.md**](visualization_design_guide.md):
  Chart standards used by the thesis and the dashboard.

`dashboard.md` (Streamlit dashboard walk-through) will land alongside
the final visualization work.