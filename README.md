# MAESTRO

[![ci](https://github.com/Colinho22/maestro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Colinho22/maestro/actions/workflows/ci.yml) ![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Colinho22/maestro?utm_source=oss&utm_medium=github&utm_campaign=Colinho22%2Fmaestro&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.11-blue.svg) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20792756.svg)](https://doi.org/10.5281/zenodo.20792756)

**M**ulti-**A**gent **E**valuation for **S**tructured **R**elational **O**utput

Comparing agentic orchestration frameworks for automated relational diagram generation.

---

## What it evaluates

MAESTRO is a benchmark. It gives every configuration the same task, a structured
input dataset to turn into a relational [Mermaid](https://mermaid.js.org/)
diagram, then scores the output against a ground-truth diagram. The question it
answers is whether multi-agent orchestration produces better relational output
than a single agent, and at what cost.

**Four orchestration strategies** generate the diagram, holding prompts and the
output contract identical so only the orchestration differs:

- `single_agent`: one prompt, one LLM call (the baseline)
- `sop_based`: a hand-coded three-step procedure (extract entities, extract
  relationships, render Mermaid)
- `crew_ai`: the same three steps orchestrated with CrewAI
- `lang_graph`: the same three steps orchestrated with LangGraph

**Three control conditions** (no LLM, deterministic) bracket the score range so a
strategy's numbers are interpretable: `null_control` (empty diagram) and
`copy_control` (raw input) are the floor; `ground_truth_control` (the answer
verbatim) is the ceiling.

**Five providers** are under test: Anthropic, OpenAI, Mistral, Gemini, and
DeepSeek, across a matrix of `inputs x strategies x models x repeats`, stratified
by complexity tier.

**Scoring** covers structural validity (does it parse, via `mmdc`), entity F1
(id / name / lemma), relationship F1 (relaxed / strict), and an error taxonomy of
what each diagram got wrong. Every cell is repeated and variance is reported.

---

## Quickstart

The benchmark runs a matrix of `inputs x strategies x models x repeats`, scores
each generated Mermaid diagram against its ground truth, and records every
result (plus the runtime environment) in a SQLite database. The four steps
below are the minimum to get a scored row into the database from a clean
checkout. For the full CLI reference, resume semantics, cost expectations, and
troubleshooting, see [`docs/running.md`](docs/running.md).

**Docker is the recommended way to run MAESTRO.** Local install is
supported for day-to-day development.

### Prerequisites

- Docker (recommended), or Python 3.11 for the local path.
- API keys for the providers you intend to exercise:
  [Anthropic](https://docs.anthropic.com/en/api/overview),
  [OpenAI](https://platform.openai.com/docs/api-reference/authentication),
  [Mistral](https://docs.mistral.ai/getting-started/quickstarts/studio/activate-and-generate-api-key),
  [Gemini](https://ai.google.dev/gemini-api/docs/api-key),
  [DeepSeek](https://api-docs.deepseek.com/).
- [`mmdc`](https://github.com/mermaid-js/mermaid-cli) for the
  structural-validity metric. Bundled in the Docker image; optional locally
  (the metric is skipped if absent).

### 1. Clone and install

```bash
git clone https://github.com/Colinho22/maestro.git
cd maestro
```

Docker (recommended), which bundles Python, mermaid-cli, and Chromium:

```bash
docker compose build
```

Or install locally:

```bash
pip install -e .            # add ".[dev]" for the test/lint tools
```

### 2. Configure API keys

```bash
cp .env.template .env       # then edit .env with your keys
```

### 3. Smoke run

Docker:

```bash
docker compose run --rm maestro python -m maestro.run \
  --strategy single_agent --tier 1 --repeats 1
```

Local:

```bash
python -m maestro.run --strategy single_agent --tier 1 --repeats 1
```

### 4. Analyse or explore

Docker:

```bash
docker compose up                    # dashboard at http://localhost:8501
docker compose run --rm maestro python -m maestro.analysis
```

Local:

```bash
python -m maestro.analysis          # scored summary to stdout
streamlit run src/maestro/viz/app.py
```

The full matrix, filter combinations, and troubleshooting live in
[`docs/running.md`](docs/running.md). The provenance model and DB integrity
verification live in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Local development

Setup is tested on macOS. Install the dev extras and run the test suite and
linters from the project root:

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

[pre-commit](https://pre-commit.com/) hooks (ruff lint + format) are configured
in `.pre-commit-config.yaml`; enable them with `pre-commit install`.

---

## Citing

If you use MAESTRO in your work, please cite the archived release on Zenodo
([`10.5281/zenodo.20792756`](https://doi.org/10.5281/zenodo.20792756), which
always resolves to the latest version) or the [`CITATION.cff`](CITATION.cff)
file (GitHub's "Cite this repository" button).
