"""
MAESTRO Statistical analysis pipeline (compute only; no visualization).

Reads the experiment database (read-only) and produces canonical,
paper-ready statistics as JSON plus an assembled ``report.md``. The
visualizer is a separate consumer of these JSON outputs; it does
not duplicate the math here.

## Unit of analysis and scoring convention

Two choices shape every inferential number here, and both are recorded in
each emitted file's metadata so a reader never has to guess.

*Unit of analysis.* The five repeats of a cell are technical replicates,
not independent observations. Fitting the raw per-run rows would treat
correlated repeats as independent and overstate significance
(pseudoreplication). All inferential tests therefore run on **per-cell
means**: one observation per (strategy, model, input) after averaging the
repeats. Descriptives keep the per-run grain (they report spread, not a
test).

*Scoring convention.* A run can succeed with a good diagram, succeed with
a diagram that does not render, or fail outright with no scored row at all.
The two conventions below make explicit which of those count and how; the
old implicit behaviour (drop failures, keep an unrenderable diagram at its
partial F1) was neither, so it is gone.

- ``intent_to_treat`` (the primary convention): every experimental run
  counts. A run that failed, or whose diagram does not parse, scores 0.0
  on the primary DV. This measures what a user actually receives, and it
  denies a strategy the chance to look good by failing and being dropped.
  Because failures are informative here (they concentrate in the more
  complex orchestrations), excluding them would bias the comparison.
- ``valid_only`` (a robustness convention): keep only runs whose diagram
  parsed. This measures quality *given* the model produced a renderable
  diagram, and is reported alongside intent-to-treat as a sensitivity
  check, not as the headline.

## What this module computes

- **Descriptives**: per-cell (strategy × model × tier) mean / median / std
  for the primary correctness DV and the efficiency DVs, on the raw per-run
  grain. Controls are *included* here (they are the sanity floor/ceiling
  anchors).
- **ANOVA**: factorial models on the primary correctness DV, fit on the
  per-cell aggregated frame under a chosen scoring convention. Controls are
  *excluded* (their F1 is 0 or 1 by construction and would distort the
  F-statistic). ``single_agent`` is the reference level: it is the
  comparison *baseline*, distinct from the control conditions.
- **Effect sizes**: partial η² per ANOVA term, Cohen's d for pairwise
  strategy contrasts (also on the aggregated frame).
- **Mixed-effects robustness**: a ``MixedLM`` on the un-aggregated rows with
  random intercepts for model and input, as a check that the aggregated
  ANOVA conclusion survives a model that accounts for the crossed grouping
  structure directly instead of by pre-averaging.
- **Error taxonomy**: descriptive characterization of the eight taxonomy
  counts per strategy (exploratory; no inferential test).
- **Correctness/efficiency trade-off**: per-strategy correctness against
  cost and latency, plus a correctness-to-cost ratio.

## Naming & traceability

Output filenames are content-based (test + factors), not RQ-numbered: an
``anova_strategy_by_tier.json`` stays truthful even if the research
questions are reframed. Each JSON carries *method* metadata
(``analysis``, ``dependent_variable``, ``factors``, ``term_of_interest``,
``schema_version``) but no ``research_question`` field: the RQ->file
mapping is an interpretation concern that lives in ``report.md`` and the
visualizer, not in the data. See the project memory note for the full
RQ->file table.

## Graceful degradation on sparse data

A factor with fewer than two observed levels (today: ``tier``, with a
single COMPLEX input in the corpus) makes its ANOVA uncomputable. Rather
than crash, the affected analysis emits a ``status: "skipped"`` stub
naming the under-leveled factor. The same code path produces real
statistics unchanged once the corpus grows.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Literal

from maestro.db.queries import fetch_analysis_rows
from maestro.experiment_config import CONTROL_STRATEGIES
from maestro.schemas import Strategy

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Version of the JSON output contract. Bumped to 1.1 when the inferential
# path moved to per-cell aggregation under an explicit scoring convention:
# the numbers changed meaning, so the contract version had to move with them.
SCHEMA_VERSION = "1.1"

# Scoring conventions for the primary DV. See the module docstring for the
# full rationale; in short, ``intent_to_treat`` scores failures and
# unrenderable diagrams as 0.0 (what the user receives), ``valid_only`` keeps
# only renderable diagrams (quality given a parse).
ScoringConvention = Literal["intent_to_treat", "valid_only"]
INTENT_TO_TREAT: ScoringConvention = "intent_to_treat"
VALID_ONLY: ScoringConvention = "valid_only"

# The primary convention: the headline inferential numbers use it. valid_only
# rides along as a robustness/sensitivity report.
DEFAULT_CONVENTION: ScoringConvention = INTENT_TO_TREAT

# The columns that identify one experimental cell. Repeats (run_number) are
# averaged away within a cell; tier is carried along because it is a function
# of example_id, not an independent grouping key.
_CELL_KEYS = ("strategy", "model", "example_id")

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
    runs yet; callers handle the empty case rather than this raising.
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
# Per-cell aggregation under a scoring convention
# ---------------------------------------------------------------------------


def _apply_convention(
    df: "pd.DataFrame", convention: ScoringConvention
) -> "pd.DataFrame":
    """
    Resolve the primary DV per the scoring convention, on the raw per-run
    frame, before aggregation.

    ``intent_to_treat`` keeps every experimental run and rewrites the primary
    DV to 0.0 for any run that failed (no metric row, so ``entity_id_f1`` is
    NaN) or whose diagram did not parse (``parses_valid`` is 0). Note the two
    are distinct: a run can parse-fail yet still carry a partial ``entity_id_f1``
    from the scorer, so a coalesce of NaN alone would leave those unrenderable
    diagrams at their partial score. Both are zeroed here.

    ``valid_only`` drops every run that did not parse (``parses_valid`` != 1),
    which also drops outright failures (their ``parses_valid`` is NaN).
    """
    out = df.copy()
    parses = out["parses_valid"]
    if convention == INTENT_TO_TREAT:
        # A run counts as scorable only if it parsed. Anything else (failed
        # run with NaN parses_valid, or a parsed-invalid diagram) is 0.0.
        scored = out[PRIMARY_DV].where(parses == 1, other=0.0)
        # A parsed run with a NaN DV should not exist, but guard anyway so a
        # stray NaN never survives into the aggregate as a silent drop.
        out[PRIMARY_DV] = scored.fillna(0.0)
        return out
    # valid_only: keep parsed runs only. NaN parses_valid (failures) compare
    # false and are dropped, which is the intended exclusion.
    return out[parses == 1]


def aggregate_experimental(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> "pd.DataFrame":
    """
    Collapse the experimental rows to one observation per cell for the
    inferential tests: filter to experimental, apply the scoring convention,
    then average the DVs across the repeats of each
    (strategy, model, example_id) cell.

    ``tier`` is carried along unchanged (it is constant within a cell, being a
    function of example_id). Under ``intent_to_treat`` every experimental cell
    survives, because failures contribute a 0.0 rather than dropping out;
    under ``valid_only`` a cell with no parsed run disappears entirely, which
    is the honest outcome (there is nothing to average).

    Returns an empty frame unchanged (callers guard on emptiness).
    """
    import pandas as pd  # noqa: F401

    exp = _experimental(df)
    if exp.empty:
        return exp
    scored = _apply_convention(exp, convention)
    if scored.empty:
        return scored

    # Average the numeric DVs the inferential and effect-size code consume.
    # tier rides along via a stable per-cell first value (constant in a cell).
    dv_cols = [PRIMARY_DV, *EFFICIENCY_DVS]
    present = [c for c in dv_cols if c in scored.columns]
    agg = (
        scored.groupby(list(_CELL_KEYS), dropna=False)
        .agg({**{c: "mean" for c in present}, "tier": "first"})
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Factor guard: graceful degradation
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
    exactly why" record: honest output on a sparse corpus, never a crash.
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
    (strategy, model, tier). Controls are INCLUDED: their cells are the
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
    convention: ScoringConvention = DEFAULT_CONVENTION,
) -> dict[str, Any]:
    """
    Fit an OLS model ``dv ~ C(f1) [* C(f2) ...]`` on the per-cell aggregated
    frame (one mean per strategy×model×input under ``convention``) and run a
    Type-II ANOVA, returning F / p / df and partial η² per term.

    Aggregating before the fit is deliberate: the repeats are technical
    replicates, so fitting the raw rows would be pseudoreplication and
    overstate significance. ``n`` in the result is therefore the number of
    cells, not the number of runs.

    ``factors`` of length 2 are crossed with ``*`` so the interaction term
    is included: that interaction is the quantity of interest for the
    "does complexity moderate the effect" and "does it hold across models"
    questions. ``term_of_interest`` records which term answers the driving
    question (e.g. ``"C(strategy):C(tier)"``); it is metadata only.

    Returns a skip-stub (never raises) when any factor is under-leveled.
    """
    agg = aggregate_experimental(df, convention)
    guard = _guard_factors(agg, factors)

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
        "unit_of_analysis": "mean over repeats per (strategy, model, input)",
        "scoring_convention": convention,
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

    model = ols(formula, data=agg).fit()
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
        "n": int(len(agg)),
        "residual_df": _to_native(table.loc["Residual", "df"]),
        "terms": results,
    }


def anova_strategy(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> dict[str, Any]:
    """One-way ANOVA: correctness ~ strategy (baseline = single_agent)."""
    return _anova(
        df,
        ["strategy"],
        term_of_interest="C(strategy, Treatment(reference='single_agent'))",
        convention=convention,
    )


def anova_strategy_by_tier(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> dict[str, Any]:
    """
    Two-way ANOVA: correctness ~ strategy × tier. The interaction term is
    the quantity of interest (does input complexity moderate the effect).
    Self-skips while the corpus has a single tier.
    """
    return _anova(
        df,
        ["strategy", "tier"],
        term_of_interest=("C(strategy, Treatment(reference='single_agent')):C(tier)"),
        convention=convention,
    )


def anova_strategy_by_model(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> dict[str, Any]:
    """
    Two-way ANOVA: correctness ~ strategy × model. The interaction probes
    whether the strategy effect holds across models / providers: a
    cross-cutting robustness check, not tied to a single RQ.
    """
    return _anova(
        df,
        ["strategy", "model"],
        term_of_interest=("C(strategy, Treatment(reference='single_agent')):C(model)"),
        convention=convention,
    )


# ---------------------------------------------------------------------------
# Post-hoc
# ---------------------------------------------------------------------------


def posthoc_strategy(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> dict[str, Any]:
    """
    Tukey HSD pairwise comparison of strategies on the primary DV, on the
    per-cell aggregated frame. Run regardless of ANOVA significance for
    completeness; the consumer decides whether to report it (convention: only
    when the omnibus ANOVA is significant). Self-skips when fewer than two
    strategies are present.
    """
    agg = aggregate_experimental(df, convention)
    base_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "posthoc_tukey_hsd",
        "dependent_variable": PRIMARY_DV,
        "factor": "strategy",
        "unit_of_analysis": "mean over repeats per (strategy, model, input)",
        "scoring_convention": convention,
    }
    guard = _guard_factors(agg, ["strategy"])
    if guard is not None:
        return {**base_meta, **guard}

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey = pairwise_tukeyhsd(
        endog=agg[PRIMARY_DV].astype(float),
        groups=agg["strategy"].astype(str),
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
        "n": int(len(agg)),
        "alpha": 0.05,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def effect_sizes(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> dict[str, Any]:
    """
    Cohen's d for every pairwise strategy contrast on the primary DV, using
    a pooled standard deviation, on the per-cell aggregated frame. Partial η²
    for the overall effects is reported within each ANOVA file; here we
    provide the pairwise d that the strategy comparison narrative needs.
    """
    agg = aggregate_experimental(df, convention)
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "effect_sizes",
        "dependent_variable": PRIMARY_DV,
        "measure": "cohens_d_pairwise",
        "unit_of_analysis": "mean over repeats per (strategy, model, input)",
        "scoring_convention": convention,
        "pairs": [],
    }
    guard = _guard_factors(agg, ["strategy"])
    if guard is not None:
        return {**out, **guard}

    groups = {
        str(name): grp[PRIMARY_DV].dropna().astype(float)
        for name, grp in agg.groupby("strategy")
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
    two perfectly-consistent strategies at different score levels are
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
        # that survives serialization and still carries the sign, rather
        # than letting _to_native collapse inf to null (indistinguishable
        # from "not computed").
        return "inf" if mean_diff > 0 else "-inf"
    return _to_native(mean_diff / (pooled**0.5))


# ---------------------------------------------------------------------------
# Mixed-effects robustness model
# ---------------------------------------------------------------------------


def mixed_effects_robustness(
    df: "pd.DataFrame", convention: ScoringConvention = DEFAULT_CONVENTION
) -> dict[str, Any]:
    """
    Robustness check on the aggregated ANOVA: a linear mixed model
    ``entity_id_f1 ~ strategy * tier`` fit on the un-aggregated experimental
    runs, with crossed random intercepts for model and input.

    The aggregated ANOVA removes the repeat correlation by pre-averaging, but
    the resulting cells still share structure: every cell on a strong model
    is lifted, every cell on a hard input is dragged down. This model accounts
    for that crossed grouping directly (model and example_id as random
    effects) instead of by averaging, so agreement between the two is evidence
    the strategy conclusion is not an artifact of the simpler error model.

    Crossed (non-nested) random effects in statsmodels go through the
    ``vc_formula`` variance-components mechanism on a single dummy grouping
    column, and can fail to converge. This is a secondary check, so on any
    failure it degrades to a skip-stub (never raises), matching the
    ``_guard_factors`` contract used elsewhere.
    """
    base_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "mixed_effects_robustness",
        "dependent_variable": PRIMARY_DV,
        "fixed_effects": "strategy * tier",
        "random_effects": ["model", "example_id"],
        "unit_of_analysis": "per-run (un-aggregated); repeats modeled, not averaged",
        "scoring_convention": convention,
        "role": "robustness_check",
    }

    # Apply the convention on the raw experimental rows, but do NOT aggregate:
    # the mixed model wants one row per run so the random effects have within
    # cell replication to estimate from.
    exp = _experimental(df)
    if exp.empty:
        return {**base_meta, "status": "skipped", "reason": "no experimental rows"}
    scored = _apply_convention(exp, convention)

    # tier must vary for the strategy*tier fixed effect to be estimable, and
    # both grouping factors must have spread for the random effects.
    guard = _guard_factors(scored, ["strategy", "tier"])
    if guard is not None:
        return {**base_meta, **guard}

    try:
        import warnings

        import statsmodels.formula.api as smf
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        data = scored.copy()
        # A single constant grouping column lets both model and example_id
        # enter as variance components, giving crossed (not nested) effects.
        data["_grp"] = 1
        vc = {"model": "0 + C(model)", "example_id": "0 + C(example_id)"}
        md = smf.mixedlm(
            f"{PRIMARY_DV} ~ C(strategy, Treatment(reference={BASELINE_STRATEGY!r})) "
            "* C(tier)",
            data=data,
            groups=data["_grp"],
            vc_formula=vc,
        )
        # Non-convergence is handled below via fit.converged and reported as a
        # skip-stub; the warnings it emits along the way are noise here, so
        # silence them rather than letting them clutter the run log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = md.fit()
    except Exception as exc:  # noqa: BLE001 - robustness check, never fatal
        # Non-convergence, singular design, or a statsmodels internal error all
        # land here: report the reason honestly rather than aborting the run.
        return {
            **base_meta,
            "status": "skipped",
            "reason": f"mixed model did not fit: {type(exc).__name__}: {exc}",
        }

    if not getattr(fit, "converged", True):
        return {**base_meta, "status": "skipped", "reason": "did not converge"}

    # Report the fixed-effect coefficients (the strategy/tier terms) with their
    # standard errors and p-values; the random-effect variances go alongside.
    params = fit.params
    fixed = {}
    for name in params.index:
        # Skip the variance-component parameters (their names carry "Var").
        if "Var" in name or name == "Group Var":
            continue
        fixed[name] = {
            "coef": _to_native(params[name]),
            "std_err": _to_native(fit.bse.get(name)),
            "p": _to_native(fit.pvalues.get(name)),
        }

    return {
        **base_meta,
        "status": "ok",
        "n": int(len(scored)),
        "n_groups": {
            "model": int(scored["model"].nunique()),
            "example_id": int(scored["example_id"].nunique()),
        },
        "fixed_effects_estimates": fixed,
        "converged": bool(getattr(fit, "converged", True)),
    }


# ---------------------------------------------------------------------------
# Error taxonomy (exploratory, descriptive)
# ---------------------------------------------------------------------------


def error_taxonomy_by_strategy(df: "pd.DataFrame") -> dict[str, Any]:
    """
    Descriptive characterization of the eight error-taxonomy counts per
    strategy (mean count per run). Exploratory: no inferential test, per
    the exploratory framing of the error-pattern research question.

    Controls are included: their taxonomy profile is itself informative
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
    correctness-to-cost ratio: the numeric backing for the Pareto
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
