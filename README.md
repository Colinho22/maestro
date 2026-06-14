# MAESTRO

[![ci](https://github.com/Colinho22/maestro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Colinho22/maestro/actions/workflows/ci.yml) ![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Colinho22/maestro?utm_source=oss&utm_medium=github&utm_campaign=Colinho22%2Fmaestro&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.11-blue.svg) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**M**ulti-**A**gent **E**valuation for **S**tructured **R**elational **O**utput

Comparing agentic orchestration frameworks for automated relational diagram generation.

---

## Running the experiment

The benchmark runs a matrix of `inputs × strategies × models × repeats`, scores
each generated Mermaid diagram against its ground truth, and records every
result — plus the runtime environment — in a SQLite database. The steps below
run the experiment from a clean checkout.

> This is a high-level walkthrough. A detailed guide (troubleshooting, full CLI
> reference) will follow as the code stabilises.

### Prerequisites

- Python 3.11
- API keys for the providers you intend to run —
  [Anthropic](https://docs.anthropic.com/en/api/overview),
  [OpenAI](https://platform.openai.com/docs/api-reference/authentication),
  [Mistral](https://docs.mistral.ai/getting-started/quickstarts/studio/activate-and-generate-api-key),
  [Gemini](https://ai.google.dev/gemini-api/docs/api-key),
  [DeepSeek](https://api-docs.deepseek.com/) (see each provider's docs for
  obtaining a key)
- [`mmdc`](https://github.com/mermaid-js/mermaid-cli) (mermaid-cli) for the
  structural-validity metric — optional locally (the metric is skipped if it is
  absent), bundled in the Docker image
- Docker (optional) — only if you prefer the container path over a local install

The local install path is tested on macOS and works on Windows. The Docker path
runs Linux inside the container, so it is platform-independent and is the
recommended route on Windows — it bundles a headless Chromium, which the
`parses_valid` structural-validity metric needs. A native Windows install
computes that metric only if mermaid-cli and a Puppeteer Chrome build
(`npx puppeteer browsers install chrome`) are present; without them the metric
is skipped, and the rest of the pipeline is unaffected.

### 1. Clone and install

```bash
git clone https://github.com/Colinho22/maestro.git
cd maestro
pip install -e .            # or: pip install -e ".[dev]" for the test/lint tools
```

Or build the container, which bundles Python, mermaid-cli, and Chromium:

```bash
docker compose build
```

### 2. Configure API keys

Copy the template and fill in the keys for the providers you will use:

```bash
cp .env.template .env
# edit .env — keys are read from the environment at run time
```

### 3. Validate the setup with a small run

A single tier-1 cell confirms the install, keys, and scoring pipeline work
before committing to the full matrix:

```bash
python -m maestro.run --strategy single_agent --tier 1 --repeats 1
# Docker: docker compose run --rm maestro python -m maestro.run --strategy single_agent --tier 1 --repeats 1
```

### 4. Run the full matrix

```bash
python -m maestro.run
# Docker: docker compose run --rm maestro python -m maestro.run
```

Runs are resumable by default: already-completed cells are skipped, so an
interrupted run can be restarted with the same command. Results are written to
`maestro.db` (or `./out/maestro.db` under Docker).

### 5. Analyse the results

```bash
python -m maestro.analysis
```

### 6. Explore the results in the dashboard

```bash
docker compose up          # → http://localhost:8501
# Local (without Docker): streamlit run src/maestro/viz/app.py
```

### Reproducibility audit trail

Every invocation snapshots its runtime environment — OS, architecture, Python
version, library versions, git commit, and (under Docker) the image digest —
into the `run_environments` table, linked to each run. This lets a later
replication attempt diagnose diverging numbers against the exact stack that
produced the original data.

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

If you use MAESTRO in your work, please cite it via the
[`CITATION.cff`](CITATION.cff) file (GitHub's "Cite this repository" button), or
see that file for the reference details.
