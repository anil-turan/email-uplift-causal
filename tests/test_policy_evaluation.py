"""Tests for the bootstrap policy-value comparison (targeting policy vs
treat-everyone)."""

import numpy as np

from src.uplift.policy_evaluation import bootstrap_policy_comparison


def _synthetic_policy_data(n=6000, seed=0):
    """Half the population is persuadable (treatment helps them a lot); the
    other half are "sleeping dogs" (treatment genuinely hurts them). Uplift
    scores rank persuadables first, so targeting the top-k should beat
    treating everyone by a large, unambiguous margin -- not just the small
    wasted-email-cost margin a "some customers are simply neutral" setup
    would produce, which is too subtle to reliably clear bootstrap noise
    at this sample size."""
    rng = np.random.default_rng(seed)
    persuadable = rng.integers(0, 2, size=n)
    treatment = rng.integers(0, 2, size=n)
    base = 0.05
    p = base + 0.5 * persuadable * treatment - 0.04 * (1 - persuadable) * treatment
    p = np.clip(p, 0.0, 1.0)
    conversion = (rng.random(n) < p).astype(int)
    # uplift score correlates with true persuadability, plus noise
    uplift_scores = persuadable + rng.normal(0, 0.1, size=n)
    return treatment, conversion, uplift_scores


def test_targeting_beats_treat_everyone_when_scores_are_informative():
    treatment, conversion, uplift_scores = _synthetic_policy_data()
    result = bootstrap_policy_comparison(
        treatment, conversion, uplift_scores,
        k_frac=0.5, cost_per_email=0.10, value_per_conversion=45.0,
        n_boot=500, seed=1,
    )
    assert result["point_diff"] > 0
    assert result["significant"] is True
    assert result["ci_low"] > 0


def test_no_difference_when_scores_are_pure_noise():
    rng = np.random.default_rng(2)
    n = 3000
    treatment = rng.integers(0, 2, size=n)
    conversion = (rng.random(n) < 0.05).astype(int)  # no true heterogeneity
    uplift_scores = rng.normal(size=n)  # uninformative

    result = bootstrap_policy_comparison(
        treatment, conversion, uplift_scores,
        k_frac=0.5, cost_per_email=0.10, value_per_conversion=45.0,
        n_boot=500, seed=3,
    )
    assert result["significant"] is False
    assert result["ci_low"] < 0 < result["ci_high"]


def test_targeting_100_percent_equals_treat_everyone():
    treatment, conversion, uplift_scores = _synthetic_policy_data()
    result = bootstrap_policy_comparison(
        treatment, conversion, uplift_scores,
        k_frac=1.0, cost_per_email=0.10, value_per_conversion=45.0,
        n_boot=200, seed=4,
    )
    assert result["point_diff"] == 0.0


def test_ci_widens_with_fewer_bootstrap_replicates_is_not_required_but_ci_is_finite():
    treatment, conversion, uplift_scores = _synthetic_policy_data()
    result = bootstrap_policy_comparison(
        treatment, conversion, uplift_scores,
        k_frac=0.3, cost_per_email=0.10, value_per_conversion=45.0,
        n_boot=300, seed=5,
    )
    assert np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_high"])
    assert result["ci_low"] <= result["point_diff"] <= result["ci_high"]
