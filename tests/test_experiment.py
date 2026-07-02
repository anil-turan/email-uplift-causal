"""Tests for the A/B experiment analysis module."""

import numpy as np
import pytest

from src.data.load import add_treatment_flag, load_raw
from src.experiment.ab import (
    calculate_sample_size,
    cuped_adjust,
    sample_ratio_mismatch,
    two_proportion_ztest,
)


def test_dataset_loads_and_is_randomised():
    """The raw data should load and pass its integrity checks, and the
    control visit rate should be well below the treated rate."""
    df = load_raw()
    assert df.shape[0] == 64000
    two = add_treatment_flag(df)
    assert two.loc[two.treatment == 1, "visit"].mean() > two.loc[
        two.treatment == 0, "visit"
    ].mean()


def test_ztest_matches_manual_computation():
    """Two-proportion z-test on a textbook example."""
    res = two_proportion_ztest(
        {"conversions": 100, "visitors": 1000},
        {"conversions": 130, "visitors": 1000},
    )
    assert res.control_rate == pytest.approx(0.10)
    assert res.treatment_rate == pytest.approx(0.13)
    assert res.lift == pytest.approx(0.30)
    assert res.significant  # 10% vs 13% at n=1000 is significant
    # CI for the difference should bracket the observed 0.03 difference
    assert res.ci_95[0] < 0.03 < res.ci_95[1]


def test_no_effect_is_not_significant():
    res = two_proportion_ztest(
        {"conversions": 100, "visitors": 1000},
        {"conversions": 101, "visitors": 1000},
    )
    assert not res.significant
    assert res.p_value > 0.05


def test_sample_size_shrinks_with_larger_effect():
    """Detecting a bigger effect needs fewer samples."""
    small_mde = calculate_sample_size(0.10, 0.05)
    large_mde = calculate_sample_size(0.10, 0.20)
    assert large_mde < small_mde


def test_srm_flags_broken_split():
    balanced = sample_ratio_mismatch(5000, 5000)
    broken = sample_ratio_mismatch(5000, 6000)
    assert not balanced["srm_detected"]
    assert broken["srm_detected"]


def test_cuped_is_unbiased_and_reduces_variance():
    """CUPED must not change the mean (unbiased) and should reduce variance
    when the covariate is correlated with the outcome."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 2 * x + rng.normal(size=5000)  # strongly correlated with x
    y_adj, var_reduction = cuped_adjust(y, x)
    assert y_adj.mean() == pytest.approx(y.mean(), abs=1e-9)
    assert var_reduction > 0.5  # ~80% of variance explained by x
