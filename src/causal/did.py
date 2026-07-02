"""Difference-in-Differences (DiD) for the observational case.

When there is no clean randomised experiment — a campaign is rolled out to some
regions/customers at a point in time, not randomly — DiD recovers the causal
effect by comparing the *change* in a treated group to the *change* in a control
group, differencing out any fixed group gap and any common time trend.

The identifying assumption is **parallel trends**: absent treatment, the two
groups would have moved in parallel. That is testable in the pre-period, so this
module ships the test alongside the estimator (a DiD number without a parallel-
trends check is not evidence).

    diff_in_diff     : ATT via OLS with a treatment×post interaction, HC3 SEs
    parallel_trends  : pre-period interaction test (should be non-significant)
    event_study      : per-period dynamic effects for the classic event plot
"""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def diff_in_diff(df: pd.DataFrame, outcome: str, treatment_col: str, post_col: str,
                 controls: list[str] | None = None) -> dict:
    """Estimate the ATT via OLS DiD.

    df must contain: outcome, treatment_col (0/1 group), post_col (0/1 period).
    The ATT is the coefficient on the treatment×post interaction. HC3 robust
    standard errors handle heteroskedasticity.
    """
    covariates = " + ".join(controls) if controls else ""
    formula = f"{outcome} ~ {treatment_col} * {post_col}" + (f" + {covariates}" if controls else "")
    result = smf.ols(formula, data=df).fit(cov_type="HC3")
    key = f"{treatment_col}:{post_col}"
    return {
        "att": float(result.params[key]),
        "p_value": float(result.pvalues[key]),
        "ci_95": [float(v) for v in result.conf_int().loc[key].tolist()],
        "n": int(result.nobs),
    }


def parallel_trends_test(df: pd.DataFrame, outcome: str, treatment_col: str,
                         period_col: str, treat_period: int) -> dict:
    """Test the parallel-trends assumption on pre-treatment periods only.

    Regresses the outcome on treatment × period-dummies using data *before*
    treatment starts. If any interaction is significant, the groups were already
    diverging and a plain DiD is not credible.
    """
    pre = df[df[period_col] < treat_period].copy()
    formula = f"{outcome} ~ {treatment_col} * C({period_col})"
    result = smf.ols(formula, data=pre).fit(cov_type="HC3")
    inter = {k: (float(result.params[k]), float(result.pvalues[k]))
             for k in result.params.index if f"{treatment_col}:" in k}
    min_p = min((p for _, p in inter.values()), default=1.0)
    return {
        "interaction_terms": inter,
        "min_p_value": min_p,
        # parallel trends holds if no pre-period interaction is significant
        "parallel_trends_ok": bool(min_p >= 0.05),
    }


def event_study(df: pd.DataFrame, outcome: str, treatment_col: str,
                period_col: str, reference_period: int) -> pd.DataFrame:
    """Dynamic DiD: treatment effect in every period relative to a reference
    (usually the last pre-treatment period). Pre-treatment coefficients should
    hover around zero; a jump at treatment onset is the effect. Returns a table
    ready to plot with confidence intervals.
    """
    d = df.copy()
    d[period_col] = d[period_col].astype(int)
    ref = f"C({period_col}, Treatment(reference={reference_period}))"
    formula = f"{outcome} ~ {treatment_col} * {ref}"
    result = smf.ols(formula, data=d).fit(cov_type="HC3")

    rows = []
    for k in result.params.index:
        if f"{treatment_col}:" in k:
            # parse the period out of the term name
            period = int("".join(ch for ch in k.split("[T.")[-1] if ch.isdigit()))
            ci = result.conf_int().loc[k]
            rows.append({"period": period, "effect": float(result.params[k]),
                         "ci_low": float(ci[0]), "ci_high": float(ci[1])})
    out = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    return out
