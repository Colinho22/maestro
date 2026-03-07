# MAESTRO - Multi-Agent Evaluation for Structured Relational Output
# Dockerfile for cross-platform reproducibility

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python", "-c", "import maestro; print(f'MAESTRO v{maestro.__version__}')"]
