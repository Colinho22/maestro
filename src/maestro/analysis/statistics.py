"""
MAESTRO — Statistical analysis pipeline (compute only; no visualization).

Reads the experiment database (read-only) and produces canonical,
paper-ready statistics as JSON plus an assembled ``report.md``. The
visualizer is a separate consumer of these JSON outputs — it does
not duplicate the math here.

## What this module computes

- **Descriptives** — per-cell (strategy × model × tier) mean / median / std
  for the primary correctness DV and the efficiency DVs. Controls are
  *included* here (they are the sanity floor/ceiling anchors).
- **ANOVA** — factorial models on the primary correctness DV. Controls are
  *excluded* (their F1 is 0 or 1 by construction and would distort the
  F-statistic). ``single_agent`` is the reference level: it is the
  comparison *baseline*, distinct from the control conditions.
- **Effect sizes** — partial η² per ANOVA term, Cohen's d for pairwise
  strategy contrasts.
- **Error taxonomy** — descriptive characterization of the eight taxonomy
  counts per strategy (exploratory; no inferential test).
- **Correctness/efficiency trade-off** — per-strategy correctness against
  cost and latency, plus a correctness-to-cost ratio.

## Naming & traceability

Output filenames are content-based (test + factors), not RQ-numbered: an
``anova_strategy_by_tier.json`` stays truthful even if the research
questions are reframed. Each JSON carries *method* metadata
(``analysis``, ``dependent_variable``, ``factors``, ``term_of_interest``,
``schema_version``) but no ``research_question`` field — the RQ→file
mapping is an interpretation concern that lives in ``report.md`` and the
visualizer, not in the data. See the project memory note for the full
RQ→file table.

## Graceful degradation on sparse data

A factor with fewer than two observed levels (today: ``tier``, with a
single COMPLEX input in the corpus) makes its ANOVA uncomputable. Rather
than crash, the affected analysis emits a ``status: "skipped"`` stub
naming the under-leveled factor. The same code path produces real
statistics unchanged once the corpus grows.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from maestro.db.queries import fetch_analysis_rows
from maestro.experiment_config import CONTROL_STRATEGIES
from maestro.schemas import Strategy

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Version of the JSON output contract. Bump on any breaking change to the
# shape of the emitted files so the visualizer can detect it.
SCHEMA_VERSION = "1.0"

# Primary correctness dependent variable. The metric_results table carries
# a family of F1 scores; entity_id_f1 (exact ID match) is the strictest
# entity measure and the agreed primary correctness signal. The other F1s
# ride along in descriptives for context.
PRIMARY_DV = "entity_id_f1"

# Efficiency dependent variables, summarized descriptively alongside
# correctness.
EFFICIENCY_DVS = ("cost_usd", "duration_ms", "retry_count")

# The comparison baseline (NOT a control). Used as the ANOVA reference
# level so coefficients read as "agent strategy vs. single-agent baseline".
BASELINE_STRATEGY = Strategy.SINGLE_AGENT.value

# Error-taxonomy columns characterized descriptively for the exploratory
# error-pattern analysis.
TAXONOMY_COLUMNS = (
    "missing_entities",
    "extra_entities",
    "false_entities",
    "duplicate_entities",
    "missing_relationships",
    "extra_relationships",
    "false_relationships",
    "duplicate_relationships",
)

# String values of the control strategies, for DataFrame filtering.
_CONTROL_VALUES = frozenset(s.value for s in CONTROL_STRATEGIES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataframe(conn: sqlite3.Connection) -> "pd.DataFrame":
    """
    Load the analysis rows (run_configs ⋈ run_results ⋈ metric_results)
    into a pandas DataFrame. Read-only.

    Returns an empty DataFrame (no rows) when the database has no metriced
    runs yet — callers handle the empty case rather than this raising.
    """
    import pandas as pd

    rows = fetch_analysis_rows(conn)
    # sqlite3.Row objects are mapping-like; dict() gives clean column keys.
    return pd.DataFrame([dict(r) for r in rows])


def _experimental(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Subset to experimental (non-control) rows for inferential tests.

    Controls have F1 fixed at 0 or 1 by construction; including them in an
    ANOVA would inflate between-group variance and corrupt the F-statistic.
    They remain in descriptive outputs (see ``describe``).
    """
    if df.empty:
        return df
    return df[~df["strategy"].isin(_CONTROL_VALUES)]


# ---------------------------------------------------------------------------
# Factor guard — graceful degradation
# ---------------------------------------------------------------------------


def _factor_levels(df: "pd.DataFrame", factor: str) -> list:
    """
    Distinct observed values of ``factor`` (order-stable, NaNs dropped).

    Values are coerced to native Python types so the list is JSON-safe when
    it lands in a skip-stub: ``.unique()`` on an integer column (e.g. tier)
    yields numpy int64, which ``json.dumps`` cannot serialize.
    """
    if df.empty or factor not in df.columns:
        return []
    return [_to_native(v) for v in df[factor].dropna().unique()]


def _guard_factors(df: "pd.DataFrame", factors: list[str]) -> dict | None:
    """
    Check every factor has at least two observed levels. Returns ``None``
    when all factors are usable (the caller proceeds), or a skip-stub dict
    describing the first under-leveled factor when not.

    The stub is the canonical "we could not compute this, and here's
    exactly why" record — honest output on a sparse corpus, never a crash.
    """
    if df.empty:
        return {"status": "skipped", "reason": "no experimental rows in database"}
    for factor in factors:
        levels = _factor_levels(df, factor)
        if len(levels) < 2:
            return {
                "status": "skipped",
                "reason": (
                    f"factor {factor!r} has {len(levels)} level(s) "
                    f"(need >= 2): {levels}"
                ),
                "factor": factor,
                "levels": levels,
            }
    return None


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def describe(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Per-cell descriptive statistics (mean / median / std) for the primary
    correctness DV and the efficiency DVs, grouped by
    (strategy, model, tier). Controls are INCLUDED — their cells are the
    sanity floor/ceiling anchors for interpreting absolute F1.
    """
    metrics = [PRIMARY_DV, *EFFICIENCY_DVS]
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "descriptive",
        "grouping": ["strategy", "model", "tier"],
        "metrics": list(metrics),
        "includes_controls": True,
        "cells": [],
    }
    if df.empty:
        out["status"] = "empty"
        return out

    grouped = df.groupby(["strategy", "model", "tier"], dropna=False)
    for (strategy, model, tier), cell in grouped:
        entry: dict[str, Any] = {
            "strategy": strategy,
            "model": model,
            "tier": _to_native(tier),
            "n": int(len(cell)),
            "is_control": strategy in _CONTROL_VALUES,
        }
        for metric in metrics:
            series = cell[metric].dropna()
            entry[metric] = {
                "mean": _to_native(series.mean()) if len(series) else None,
                "median": _to_native(series.median()) if len(series) else None,
                "std": _to_native(series.std(ddof=1)) if len(series) > 1 else None,
            }
        out["cells"].append(entry)
    return out


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


def _anova(
    df: "pd.DataFrame",
    factors: list[str],
    *,
    dv: str = PRIMARY_DV,
    term_of_interest: str | None = None,
) -> dict[str, Any]:
    """
    Fit an OLS model ``dv ~ C(f1) [* C(f2) ...]`` on the experimental
    (non-control) rows and run a Type-II ANOVA, returning F / p / df and
    partial η² per term.

    ``factors`` of length 2 are crossed with ``*`` so the interaction term
    is included — that interaction is the quantity of interest for the
    "does complexity moderate the effect" and "does it hold across models"
    questions. ``term_of_interest`` records which term answers the driving
    question (e.g. ``"C(strategy):C(tier)"``); it is metadata only.

    Returns a skip-stub (never raises) when any factor is under-leveled.
    """
    exp = _experimental(df)
    guard = _guard_factors(exp, factors)

    base_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "anova",
        "dependent_variable": dv,
        "factors": list(factors),
        "term_of_interest": term_of_interest,
        "reference_level": {"strategy": BASELINE_STRATEGY}
        if "strategy" in factors
        else None,
        "excludes_controls": True,
    }
    if guard is not None:
        return {**base_meta, **guard}

    import pandas as pd  # noqa: F401  (ensures pandas present for statsmodels)
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Build a patsy formula. Treat every factor as categorical via C().
    # ``Treatment`` coding with single_agent as the explicit reference makes
    # strategy coefficients read against the baseline.
    terms = []
    for f in factors:
        if f == "strategy":
            terms.append(f"C(strategy, Treatment(reference={BASELINE_STRATEGY!r}))")
        else:
            terms.append(f"C({f})")
    rhs = " * ".join(terms)
    formula = f"{dv} ~ {rhs}"

    model = ols(formula, data=exp).fit()
    # Type II is the conventional choice for unbalanced factorial designs
    # without a strong a-priori ordering of effects.
    table = sm.stats.anova_lm(model, typ=2)

    # Partial η² = SS_effect / (SS_effect + SS_residual).
    ss_resid = float(table.loc["Residual", "sum_sq"])
    results: dict[str, Any] = {}
    for term in table.index:
        if term == "Residual":
            continue
        ss_effect = float(table.loc[term, "sum_sq"])
        denom = ss_effect + ss_resid
        partial_eta_sq = ss_effect / denom if denom > 0 else None
        f_val = table.loc[term, "F"]
        p_val = table.loc[term, "PR(>F)"]
        results[term] = {
            "sum_sq": ss_effect,
            "df": _to_native(table.loc[term, "df"]),
            "F": _to_native(f_val),
            "p": _to_native(p_val),
            "partial_eta_sq": partial_eta_sq,
        }

    return {
        **base_meta,
        "status": "ok",
        "n": int(len(exp)),
        "residual_df": _to_native(table.loc["Residual", "df"]),
        "terms": results,
    }


def anova_strategy(df: "pd.DataFrame") -> dict[str, Any]:
    """One-way ANOVA: correctness ~ strategy (baseline = single_agent)."""
    return _anova(
        df,
        ["strategy"],
        term_of_interest="C(strategy, Treatment(reference='single_agent'))",
    )


def anova_strategy_by_tier(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Two-way ANOVA: correctness ~ strategy × tier. The interaction term is
    the quantity of interest (does input complexity moderate the effect).
    Self-skips while the corpus has a single tier.
    """
    return _anova(
        df,
        ["strategy", "tier"],
        term_of_interest=("C(strategy, Treatment(reference='single_agent')):C(tier)"),
    )


def anova_strategy_by_model(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Two-way ANOVA: correctness ~ strategy × model. The interaction probes
    whether the strategy effect holds across models / providers — a
    cross-cutting robustness check, not tied to a single RQ.
    """
    return _anova(
        df,
        ["strategy", "model"],
        term_of_interest=("C(strategy, Treatment(reference='single_agent')):C(model)"),
    )


# ---------------------------------------------------------------------------
# Post-hoc
# ---------------------------------------------------------------------------


def posthoc_strategy(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Tukey HSD pairwise comparison of strategies on the primary DV. Run
    regardless of ANOVA significance for completeness; the consumer decides
    whether to report it (convention: only when the omnibus ANOVA is
    significant). Self-skips when fewer than two strategies are present.
    """
    exp = _experimental(df)
    base_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "posthoc_tukey_hsd",
        "dependent_variable": PRIMARY_DV,
        "factor": "strategy",
    }
    guard = _guard_factors(exp, ["strategy"])
    if guard is not None:
        return {**base_meta, **guard}

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey = pairwise_tukeyhsd(
        endog=exp[PRIMARY_DV].astype(float),
        groups=exp["strategy"].astype(str),
        alpha=0.05,
    )
    # Build the comparison rows from the *public* TukeyHSDResults attributes
    # rather than the private ``_results_table``: groupsunique holds the
    # group labels, and meandiffs / confint / pvalues / reject are aligned
    # arrays over the upper-triangular (i < j) group pairs in the same order
    # statsmodels generates them. Reproducing that ordering here keeps the
    # output identical to the summary table without depending on a private
    # attribute that can move between statsmodels releases.
    group_names = [str(g) for g in tukey.groupsunique]
    pairs = [
        (i, j) for i in range(len(group_names)) for j in range(i + 1, len(group_names))
    ]
    comparisons = []
    for k, (i, j) in enumerate(pairs):
        lower, upper = tukey.confint[k]
        comparisons.append(
            {
                "group1": group_names[i],
                "group2": group_names[j],
                "meandiff": _to_native(tukey.meandiffs[k]),
                "p_adj": _to_native(tukey.pvalues[k]),
                "lower": _to_native(lower),
                "upper": _to_native(upper),
                "reject": bool(tukey.reject[k]),
            }
        )

    return {
        **base_meta,
        "status": "ok",
        "n": int(len(exp)),
        "alpha": 0.05,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def effect_sizes(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Cohen's d for every pairwise strategy contrast on the primary DV, using
    a pooled standard deviation. Partial η² for the overall effects is
    reported within each ANOVA file; here we provide the pairwise d that
    the strategy comparison narrative needs.
    """
    exp = _experimental(df)
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "effect_sizes",
        "dependent_variable": PRIMARY_DV,
        "measure": "cohens_d_pairwise",
        "pairs": [],
    }
    guard = _guard_factors(exp, ["strategy"])
    if guard is not None:
        return {**out, **guard}

    groups = {
        str(name): grp[PRIMARY_DV].dropna().astype(float)
        for name, grp in exp.groupby("strategy")
    }
    names = sorted(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = groups[names[i]], groups[names[j]]
            d = _cohens_d(a, b)
            out["pairs"].append(
                {
                    "group_a": names[i],
                    "group_b": names[j],
                    "mean_a": _to_native(a.mean()) if len(a) else None,
                    "mean_b": _to_native(b.mean()) if len(b) else None,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "cohens_d": d,
                }
            )
    out["status"] = "ok"
    return out


def _cohens_d(a, b) -> float | str | None:
    """
    Cohen's d with pooled SD.

    Returns ``None`` when either group has < 2 observations, ``0.0`` when
    the groups are identical, the signed string ``"inf"`` / ``"-inf"`` when
    pooled variance is zero but the means differ, and the float d otherwise.

    Zero pooled variance is a real case here: every run in a cell can score
    an identical F1 (e.g. a deterministic strategy on a single input). When
    that happens, the effect size is *not* zero unless the means also match
    — two perfectly-consistent strategies at different score levels are
    maximally separated, i.e. an infinite standardized difference. Returning
    a string sentinel (instead of 0.0 or null) preserves that signal in
    JSON, which cannot represent infinity.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = (((na - 1) * va) + ((nb - 1) * vb)) / (na + nb - 2)
    mean_diff = a.mean() - b.mean()
    # Treat negligible pooled variance as zero. Identical scores stored as
    # floats (e.g. three 0.1s) leave a ~1e-34 residual variance rather than
    # an exact 0, which would otherwise divide a real mean difference by a
    # near-zero SD and yield an absurd ~1e16 "finite" d instead of the
    # correct infinity. The threshold is far below any real F1 spread.
    if pooled <= 1e-12:
        if abs(mean_diff) <= 1e-12:
            return 0.0
        # Infinite standardized difference. JSON has no infinity and the
        # writer enforces allow_nan=False, so emit a signed string sentinel
        # that survives serialization and still carries the sign — rather
        # than letting _to_native collapse inf to null (indistinguishable
        # from "not computed").
        return "inf" if mean_diff > 0 else "-inf"
    return _to_native(mean_diff / (pooled**0.5))


# ---------------------------------------------------------------------------
# Error taxonomy (exploratory, descriptive)
# ---------------------------------------------------------------------------


def error_taxonomy_by_strategy(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Descriptive characterization of the eight error-taxonomy counts per
    strategy (mean count per run). Exploratory: no inferential test, per
    the exploratory framing of the error-pattern research question.

    Controls are included — their taxonomy profile is itself informative
    (e.g. the copy control's "extra" counts reveal input/diagram overlap).
    """
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "error_taxonomy_descriptive",
        "grouping": ["strategy"],
        "taxonomy_columns": list(TAXONOMY_COLUMNS),
        "statistic": "mean_count_per_run",
        "includes_controls": True,
        "by_strategy": [],
    }
    if df.empty:
        out["status"] = "empty"
        return out

    for strategy, grp in df.groupby("strategy"):
        entry: dict[str, Any] = {
            "strategy": strategy,
            "n": int(len(grp)),
            "is_control": strategy in _CONTROL_VALUES,
            "counts": {col: _to_native(grp[col].mean()) for col in TAXONOMY_COLUMNS},
        }
        out["by_strategy"].append(entry)
    return out


# ---------------------------------------------------------------------------
# Correctness / efficiency trade-off
# ---------------------------------------------------------------------------


def tradeoff_correctness_efficiency(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Per-strategy mean correctness against mean cost and latency, plus a
    correctness-to-cost ratio — the numeric backing for the Pareto
    trade-off figure (the figure itself is rendered by the visualizer).

    Experimental rows only: the trade-off question compares the orchestration
    strategies and the baseline, not the zero-cost controls.
    """
    exp = _experimental(df)
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "tradeoff_correctness_efficiency",
        "correctness_metric": PRIMARY_DV,
        "efficiency_metrics": ["cost_usd", "duration_ms"],
        "excludes_controls": True,
        "by_strategy": [],
    }
    if exp.empty:
        out["status"] = "empty"
        return out

    for strategy, grp in exp.groupby("strategy"):
        mean_corr = _to_native(grp[PRIMARY_DV].mean())
        mean_cost = _to_native(grp["cost_usd"].mean())
        mean_lat = _to_native(grp["duration_ms"].mean())
        # Correctness per US-dollar; None when cost is zero/undefined to
        # avoid a divide-by-zero or a meaningless infinity.
        ratio = (
            mean_corr / mean_cost
            if mean_cost not in (None, 0) and mean_corr is not None
            else None
        )
        out["by_strategy"].append(
            {
                "strategy": strategy,
                "n": int(len(grp)),
                "mean_correctness": mean_corr,
                "mean_cost_usd": mean_cost,
                "mean_duration_ms": mean_lat,
                "correctness_per_usd": ratio,
            }
        )
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# JSON-safety helper
# ---------------------------------------------------------------------------


def _to_native(value: Any) -> Any:
    """
    Coerce numpy/pandas scalars to plain Python types so ``json.dumps``
    succeeds, and NaN/inf to ``None``. pandas means/vars come back as
    numpy float64; json can't serialize those, and NaN is not valid JSON.
    """
    if value is None:
        return None
    # numpy scalars expose .item(); pandas NA / NaN handled via != self.
    try:
        import math

        # numpy bool_/int_/float_ all carry .item()
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    except (ValueError, TypeError):
        return None
