# Extending MAESTRO

How to add a new provider, a new strategy, a new metric, or a new input.
Written for anyone using MAESTRO as a benchmark to run their own
comparisons. Each recipe ends with a smoke test that verifies the
addition without spending real API money.

Two invariants apply to every extension:

- **Providers never raise.** `complete()` must catch every SDK error and
  return a `RunResult` with `.error` set. A raise inside a provider
  crashes the whole batch.
- **The output contract is shared.** Every strategy uses the Mermaid
  rules in `src/maestro/prompts.py`. Do not fork or paraphrase them.

---

## 1. Add a provider

Use case: a new vendor SDK, or an OpenAI-compatible endpoint hosted
elsewhere.

### 1.1 Subclass `LLMProvider`

Create `src/maestro/providers/<vendor>.py`:

```python
from __future__ import annotations

import time

import somevendor
from maestro.providers._retry import RetryStats, call_with_retry
from maestro.providers.base import LLMProvider
from maestro.schemas import ModelPricing, RunConfig, RunResult, compute_cost


class SomeVendorProvider(LLMProvider):
    """Provider for SomeVendor's chat completion API."""

    _PROVIDER_NAME = "somevendor"
    MAX_TOKENS = 4096

    def __init__(self, api_key: str, pricing: ModelPricing) -> None:
        super().__init__(api_key, pricing)
        self._client = somevendor.Client(api_key=api_key)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Which SDK exceptions warrant a retry. Vendor-specific."""
        return isinstance(exc, somevendor.RateLimitError)

    def complete(
        self,
        prompt: str,
        config: RunConfig,
        system_prompt: str | None = None,
    ) -> RunResult:
        retry_stats = RetryStats()
        start = time.monotonic()
        try:
            response = call_with_retry(
                lambda: self._client.chat(
                    model=self.pricing.model,
                    system=system_prompt or self.SYSTEM_PROMPT,
                    prompt=prompt,
                    max_tokens=self.MAX_TOKENS,
                    temperature=self.TEMPERATURE if self.pricing.supports_temperature else None,
                ),
                is_retryable=self._is_retryable,
                stats=retry_stats,
                provider_name=self._PROVIDER_NAME,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return RunResult(
                run_id=config.run_id,
                output_diagram_code=None,
                raw_response=None,
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=duration_ms,
                cost_usd=0.0,
                error=f"{type(exc).__name__}: {exc}",
                retry_count=retry_stats.count,
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        text = response.output_text or ""
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        return RunResult(
            run_id=config.run_id,
            output_diagram_code=text or None,
            raw_response=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            cost_usd=compute_cost(prompt_tokens, completion_tokens, self.pricing),
            error=None if text else "EmptyResponse: no text content",
            retry_count=retry_stats.count,
        )
```

Rules:

- **Never raise.** Every failure mode returns a `RunResult` with `.error`
  set. The `call_with_retry` wrapper handles the retry policy; a
  post-retry failure lands in the outer `except` and becomes a failed
  row.
- **Inherit `TEMPERATURE = 0` and `SYSTEM_PROMPT`.** Do not redefine
  them in the subclass. If a model does not accept a temperature
  parameter, set `supports_temperature=False` on its `ModelPricing`
  entry and skip the parameter in the call.
- **Route retries through `call_with_retry`.** Do not hand-roll a retry
  loop. `_is_retryable` is the only vendor-specific decision the retry
  path needs.

If the vendor exposes an OpenAI-compatible endpoint, subclass
`OpenAIProvider` instead (see `providers/deepseek.py` for the reference).

### 1.2 Register the dispatch

In `src/maestro/run.py`, add an entry to `_PROVIDER_DISPATCH`:

```python
_PROVIDER_DISPATCH = (
    ("claude", AnthropicProvider, "ANTHROPIC_API_KEY"),
    ("gpt", OpenAIProvider, "OPENAI_API_KEY"),
    ...,
    ("somevendor", SomeVendorProvider, "SOMEVENDOR_API_KEY"),
)
```

The needle (`"somevendor"`) must appear in every model name for the new
provider, and must not collide with any other vendor's needles. Dispatch
picks the first matching substring.

### 1.3 Add the environment variable

Add the key to `.env.template`:

```
SOMEVENDOR_API_KEY=<API_KEY>
```

### 1.4 Whitelist the library

In `src/maestro/db/environment.py`, add the SDK package name to
`_LIB_WHITELIST`:

```python
_LIB_WHITELIST: tuple[str, ...] = (
    "anthropic",
    ...,
    "somevendor",
)
```

A missing entry means the SDK version silently stops being recorded in
`run_environments.lib_versions`, and the omission surfaces only when a
future replication attempt diverges and the smoking gun is missing.

### 1.5 Declare the dependency

Add the SDK to `pyproject.toml`'s runtime dependencies with a version
pin. Include a one-line comment on the pin if it is non-obvious (a
supply-chain incident, a required new symbol, a moved import path).

### 1.6 Register a model

Add an entry to `MODELS` in `src/maestro/experiment_config.py`:

```python
ModelPricing(
    model="somevendor-flagship-1",
    input_price_per_1m=3.00,
    output_price_per_1m=15.00,
),
```

Pricing is USD per 1M tokens, sourced from the provider's public pricing
page on the date of the run. `cost_usd` is computed at write time from
this rate, so a later pricing change does not alter historical rows.

### 1.7 Smoke test

```bash
python -m maestro.run \
  --model somevendor-flagship-1 \
  --strategy single_agent \
  --tier 1 \
  --example bpmn_1_01 \
  --repeats 1
```

Expected: one row in `run_results`, `error IS NULL`, `raw_response`
populated, `cost_usd > 0`, `duration_ms > 0`.

---

## 2. Add a strategy

Use case: a new orchestration pattern (a different tool loop, a
different message hierarchy, a different agent framework).

### 2.1 Subclass `BaseStrategy`

Create `src/maestro/strategies/<name>.py`:

```python
from __future__ import annotations

import time

from maestro.prompts import render_rules
from maestro.schemas import InputFile, RunConfig, RunResult, SubResult
from maestro.strategies.base import BaseStrategy


class MyStrategy(BaseStrategy):
    """One-line rationale for what this orchestration adds."""

    def run(
        self,
        input_file: InputFile,
        config: RunConfig,
    ) -> tuple[RunResult, list[SubResult]]:
        start = time.monotonic()
        try:
            input_text = input_file.file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return self._error_result(config, f"input read failed: {exc}", start=start)

        prompt = f"{render_rules()}\n\nInput:\n{input_text}"
        result = self.provider.complete(prompt=prompt, config=config)
        return result, []
```

Rules:

- **Class name ends with `Strategy`.**
- **Return `(RunResult, list[SubResult])`.** `SubResult` rows describe
  intermediate steps of a multi-step orchestration; a single-call
  strategy returns an empty list.
- **Use `render_rules()` for the output contract.** Do not paraphrase or
  fork the Mermaid rules. If the strategy needs a variant, extend
  `render_rules()`'s `skill` parameter (see `prompts.py`).
- **Structural errors use `self._error_result`.** A file-read failure or
  a step-validation rejection becomes a failed `RunResult`, not a raise.

### 2.2 Register the strategy

Two edits:

- Add a value to the `Strategy` enum in `src/maestro/schemas.py`:

  ```python
  MY_STRATEGY = "my_strategy"
  ```

  The string value is persisted to `run_configs.strategy` and exposed as
  a `--strategy` choice. Once data is collected, never rename it.

- Wire the strategy into `STRATEGY_MAP` in `src/maestro/run.py`:

  ```python
  STRATEGY_MAP = {
      Strategy.SINGLE_AGENT: SingleAgentStrategy,
      ...,
      Strategy.MY_STRATEGY: MyStrategy,
  }
  ```

- Add the enum value to `STRATEGIES` in
  `src/maestro/experiment_config.py` so it is part of the default
  matrix.

### 2.3 Smoke test

```bash
python -m maestro.run \
  --strategy my_strategy \
  --model claude-haiku-4-5-20251001 \
  --tier 1 \
  --example bpmn_1_01 \
  --repeats 1
```

Expected: one successful row, plus rows in `sub_results` if the strategy
has intermediate steps.

---

## 3. Add a metric

Use case: a new scoring dimension the entity / relationship / container
axes do not cover.

### 3.1 Author the metric

Add a computation to `src/maestro/analysis/metrics.py`. Follow the P/R/F1
shape used by existing metrics; nullable if the metric can be "not
applicable" for some diagrams. Wire it into `evaluate_run` so it lands on
the `MetricResult` returned to the runner.

### 3.2 Extend the schema

Add the columns to the `metric_results` `CREATE TABLE` in
`src/maestro/db/client.py`. Then add an additive migration alongside the
existing ones so old databases open without loss:

```python
def _migrate_add_my_metric_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(metric_results)")}
    if "my_metric_f1" not in cols:
        conn.execute("ALTER TABLE metric_results ADD COLUMN my_metric_f1 REAL")
        conn.execute("ALTER TABLE metric_results ADD COLUMN my_metric_precision REAL")
        conn.execute("ALTER TABLE metric_results ADD COLUMN my_metric_recall REAL")
```

Call the new migration from `init_db`.

### 3.3 Extend the persistence layer

Add the fields to the `MetricResult` schema in `src/maestro/schemas.py`
and to `insert_metric_result` in `src/maestro/db/queries.py`.

### 3.4 Smoke test

Re-run a single scored cell against a small database:

```bash
python -m maestro.run \
  --strategy single_agent \
  --model claude-haiku-4-5-20251001 \
  --tier 1 \
  --example bpmn_1_01 \
  --repeats 1
```

Then query the new columns:

```bash
sqlite3 'file:out/maestro.db?mode=ro' \
  "SELECT my_metric_f1, my_metric_precision FROM metric_results ORDER BY ROWID DESC LIMIT 1;"
```

---

## 4. Add an input

Covered in full in `docs/data.md` section 5.

---

## 5. Version and release discipline

Any change that alters what a number means requires a version bump so
pre-change and post-change runs are never mixed. In particular:

- A metric definition change (a new scoring rule, a nullable becoming
  non-nullable, a new taxonomy category) is a bump.
- A prompt contract change (an edit to `prompts.py`) is a bump.
- A new provider or strategy is additive and does not require a bump on
  its own; the new rows are recognisably from the new configuration.

A pricing change on an existing model is not a bump: `cost_usd` is
computed at write time and stored, so historical rows retain their
original cost even after a repricing.

See `CHANGELOG.md` and `.github/CONTRIBUTING.md` for the release-line
conventions.