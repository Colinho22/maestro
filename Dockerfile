# MAESTRO — Multi-Agent Evaluation for Structured Relational Output
# Dockerfile for cross-platform reproducibility.
#
# The image carries both halves of the pipeline:
#   * Python 3.11 — runs the experiment (models, strategies, scoring, DB).
#   * mermaid-cli (mmdc) + Chromium — backs the structural-validity metric
#     (analysis/metrics.py shells out to `mmdc` to compute parses_valid; without
#     it that metric is recorded as NULL for every run).

FROM python:3.11-slim

WORKDIR /app

# System dependencies:
#   * git              — environment.py records the commit hash per run
#   * nodejs / npm     — runtime for mermaid-cli
#   * chromium         — mmdc renders via Puppeteer, which needs a browser
#   * the lib*/fonts*  — shared libraries Chromium needs to start headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nodejs \
    npm \
    chromium \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# mermaid-cli, pinned for reproducibility. Puppeteer must use the system
# Chromium (installed above) rather than downloading its own — and Chromium
# refuses to run as root without --no-sandbox, which is the norm in CI/Docker.
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
RUN npm install -g @mermaid-js/mermaid-cli@11.4.2

# A Puppeteer launch config so mmdc starts Chromium headless without a sandbox.
# Chromium refuses to run as root without --no-sandbox; --disable-dev-shm-usage
# avoids crashes from the small /dev/shm Docker allocates by default. mmdc only
# honours these via a config file passed with `-p`, so metrics.py reads this
# path from MERMAID_PUPPETEER_CONFIG and forwards it as `-p`.
RUN printf '{"args":["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"]}' \
    > /app/puppeteer.json
ENV MERMAID_PUPPETEER_CONFIG=/app/puppeteer.json

# Python project. Copy metadata first so the dependency layer caches across
# source-only changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Sanity: fail the build if mmdc can't actually render, so a broken
# Chromium/Puppeteer setup is caught here, not 80% into a real run. Uses a temp
# file (not /dev/stdin) so this checks the browser, not the input path; -p makes
# the launch config explicit rather than relying on env discovery.
RUN printf 'flowchart LR\n  a["A"] --> b["B"]\n' > /tmp/smoke.mmd \
    && mmdc -p /app/puppeteer.json -i /tmp/smoke.mmd -o /tmp/smoke.png -e png \
    && rm -f /tmp/smoke.mmd /tmp/smoke.png

# Default: print the version. Override with the experiment runner, e.g.
#   docker compose run --rm maestro python -m maestro.run --tier 1
CMD ["python", "-c", "import maestro; print(f'MAESTRO v{maestro.__version__}')"]
