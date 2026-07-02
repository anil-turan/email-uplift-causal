"""A/B experiment analysis: power/sample-size, two-proportion z-test,
sample-ratio-mismatch (SRM), and CUPED variance reduction.

These functions treat the experiment the way a data scientist should:
compute the required sample size *before* looking at results, validate
that randomisation actually held, and report effect size with a
confidence interval rather than a bare p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def calculate_sample_size(
    baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Required sample size *per variant* for a two-proportion test.

    baseline_rate : current conversion rate, e.g. 0.10
    mde           : minimum detectable effect, relative, e.g. 0.05 == 5% lift
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    effect_size = abs(p2 - p1) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))


@dataclass
class ABResult:
    control_rate: float
    treatment_rate: float
    abs_diff: float
    lift: float
    z: float
    p_value: float
    ci_95: tuple[float, float]
    significant: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        sig = "significant" if self.significant else "not significant"
        return (
            f"control={self.control_rate:.4f} treatment={self.treatment_rate:.4f} "
            f"lift={self.lift:+.1%} p={self.p_value:.4f} "
            f"CI95=[{self.ci_95[0]:+.4f}, {self.ci_95[1]:+.4f}] ({sig})"
        )


def two_proportion_ztest(
    control: dict, treatment: dict, alpha: float = 0.05
) -> ABResult:
    """Two-proportion z-test. control/treatment are dicts with
    'conversions' and 'visitors'. Returns lift, CI and significance."""
    p_c = control["conversions"] / control["visitors"]
    p_t = treatment["conversions"] / treatment["visitors"]
    pooled = (control["conversions"] + treatment["conversions"]) / (
        control["visitors"] + treatment["visitors"]
    )
    se = np.sqrt(
        pooled * (1 - pooled) * (1 / control["visitors"] + 1 / treatment["visitors"])
    )
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    # CI uses the unpooled SE, which is the correct SE for the difference itself.
    se_diff = np.sqrt(
        p_c * (1 - p_c) / control["visitors"] + p_t * (1 - p_t) / treatment["visitors"]
    )
    margin = stats.norm.ppf(1 - alpha / 2) * se_diff
    return ABResult(
        control_rate=p_c,
        treatment_rate=p_t,
        abs_diff=p_t - p_c,
        lift=(p_t - p_c) / p_c if p_c > 0 else np.nan,
        z=z,
        p_value=p_value,
        ci_95=((p_t - p_c) - margin, (p_t - p_c) + margin),
        significant=p_value < alpha,
    )


def sample_ratio_mismatch(n_control: int, n_treatment: int, expected: float = 0.5) -> dict:
    """Chi-square test that the observed allocation matches the intended
    split. A significant result means the randomisation or logging is
    broken and the experiment should NOT be trusted.
    """
    total = n_control + n_treatment
    exp_control = total * expected
    exp_treatment = total * (1 - expected)
    chi2 = (n_control - exp_control) ** 2 / exp_control + (
        n_treatment - exp_treatment
    ) ** 2 / exp_treatment
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {
        "observed_ratio": n_control / total,
        "chi2": chi2,
        "p_value": p_value,
        "srm_detected": p_value < 0.001,  # conventional strict SRM threshold
    }


def cuped_adjust(
    y: np.ndarray, covariate: np.ndarray
) -> tuple[np.ndarray, float]:
    """CUPED (Controlled-experiment Using Pre-Existing Data).

    Uses a pre-experiment covariate correlated with the outcome to remove
    variance from the metric without introducing bias. Returns the adjusted
    outcome and the fraction of variance removed.

        y_cuped = y - theta * (covariate - mean(covariate))
        theta   = cov(y, covariate) / var(covariate)
    """
    cov = np.cov(y, covariate, ddof=1)[0, 1]
    theta = cov / np.var(covariate, ddof=1)
    y_adj = y - theta * (covariate - covariate.mean())
    var_reduction = 1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1)
    return y_adj, var_reduction


def bonferroni(alpha: float, n_tests: int) -> float:
    """Family-wise corrected alpha for multiple metric testing."""
    return alpha / n_tests
