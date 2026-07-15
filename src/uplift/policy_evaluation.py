"""Bootstrap confidence intervals for a targeting policy's net profit.

Notebook 04 reports a single point estimate: uplift-targeting the top-k%
by predicted uplift beats treating everyone by some £ amount. A point
estimate alone can't distinguish "the policy is genuinely better" from
"this particular test-set split happened to favour it" -- exactly the gap
this module closes, via the nonparametric (percentile) bootstrap: resample
the test set with replacement many times, recompute both policies' net
profit on each resample (using each customer's already-fitted uplift
score, never refit), and report the percentile interval of the
*difference*. A difference whose 95% CI excludes zero is a real effect at
this sample size, not sampling noise on one split.
"""

from __future__ import annotations

import numpy as np


def _net_profit(
    treatment: np.ndarray,
    conversion: np.ndarray,
    idx: np.ndarray,
    cost_per_email: float,
    value_per_conversion: float,
) -> float:
    """Net profit of e-mailing exactly the customers at `idx` (a subset of
    row positions into treatment/conversion), using the same
    difference-in-conditional-means incremental-conversion estimator as
    notebook 04's policy_value() -- valid because treatment is randomised
    within the evaluation set."""
    t_sub, c_sub = treatment[idx], conversion[idx]
    treated_mask, control_mask = t_sub == 1, t_sub == 0
    treated_rate = c_sub[treated_mask].mean() if treated_mask.any() else 0.0
    control_rate = c_sub[control_mask].mean() if control_mask.any() else 0.0
    incremental_rate = treated_rate - control_rate

    n_targeted = len(idx)
    incremental_conversions = incremental_rate * n_targeted
    revenue = incremental_conversions * value_per_conversion
    cost = n_targeted * cost_per_email
    return revenue - cost


def bootstrap_policy_comparison(
    treatment: np.ndarray,
    conversion: np.ndarray,
    uplift_scores: np.ndarray,
    k_frac: float,
    cost_per_email: float,
    value_per_conversion: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for (uplift-targeted top-k% net profit) minus
    (treat-everyone net profit), on the same resampled test set each time.

    Args:
        treatment, conversion, uplift_scores: arrays over the evaluation
            set, same length and row order.
        k_frac: fraction of customers the uplift policy targets (top-k by
            uplift_scores).
        n_boot: number of bootstrap resamples.

    Returns:
        dict with the point estimate, percentile CI, and whether it
        excludes zero.
    """
    treatment = np.asarray(treatment)
    conversion = np.asarray(conversion)
    uplift_scores = np.asarray(uplift_scores)
    n = len(treatment)
    rng = np.random.default_rng(seed)

    point_order = np.argsort(-uplift_scores)
    cutoff = int(np.ceil(k_frac * n))
    point_targeted_idx = point_order[:cutoff]
    point_targeted_profit = _net_profit(
        treatment, conversion, point_targeted_idx, cost_per_email, value_per_conversion
    )
    point_all_idx = np.arange(n)
    point_all_profit = _net_profit(
        treatment, conversion, point_all_idx, cost_per_email, value_per_conversion
    )
    point_diff = point_targeted_profit - point_all_profit

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        resample_idx = rng.integers(0, n, size=n)
        # re-rank the resample by each customer's own (already-fitted) uplift score
        resample_order = resample_idx[np.argsort(-uplift_scores[resample_idx])]
        targeted_idx = resample_order[:cutoff]

        targeted_profit = _net_profit(
            treatment, conversion, targeted_idx, cost_per_email, value_per_conversion
        )
        all_profit = _net_profit(
            treatment, conversion, resample_idx, cost_per_email, value_per_conversion
        )
        diffs[b] = targeted_profit - all_profit

    lower, upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_targeted_profit": point_targeted_profit,
        "point_all_profit": point_all_profit,
        "point_diff": point_diff,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "significant": bool(lower > 0 or upper < 0),
        "n_boot": n_boot,
    }
