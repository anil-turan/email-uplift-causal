"""Tests for the Difference-in-Differences module.

The headline guarantee: on a synthetic panel with a known effect, DiD recovers
it, the parallel-trends test passes when trends are parallel, and it *fails*
when they are not.
"""

import numpy as np
import pandas as pd

from src.causal.did import diff_in_diff, event_study, parallel_trends_test


def _panel(true_att=5.0, pre_divergence=0.0, seed=42, n=1500, t=8, treat_period=4):
    """Customer-month panel. `pre_divergence` adds a treated-only trend even in
    the pre-period, which should break parallel trends when non-zero."""
    rng = np.random.default_rng(seed)
    fe = rng.normal(20, 5, n)
    trend = np.arange(t) * 0.8
    rows = []
    for c in range(n):
        treated = int(c >= n // 2)
        for p in range(t):
            post = int(p >= treat_period)
            spend = (fe[c] + trend[p]
                     + true_att * treated * post
                     + pre_divergence * treated * p       # treated-specific trend
                     + rng.normal(0, 3))
            rows.append((c, treated, p, post, spend))
    return pd.DataFrame(rows, columns=["customer", "treated", "period", "post", "spend"])


def test_did_recovers_known_effect():
    df = _panel(true_att=5.0)
    res = diff_in_diff(df, "spend", "treated", "post")
    assert res["ci_95"][0] <= 5.0 <= res["ci_95"][1]      # CI covers the truth
    assert abs(res["att"] - 5.0) < 0.5
    assert res["p_value"] < 1e-6


def test_did_detects_no_effect():
    df = _panel(true_att=0.0)
    res = diff_in_diff(df, "spend", "treated", "post")
    assert res["ci_95"][0] <= 0.0 <= res["ci_95"][1]


def test_parallel_trends_passes_when_parallel():
    df = _panel(true_att=5.0, pre_divergence=0.0)
    assert parallel_trends_test(df, "spend", "treated", "period", 4)["parallel_trends_ok"]


def test_parallel_trends_fails_when_violated():
    # a strong treated-only pre-trend must be flagged
    df = _panel(true_att=5.0, pre_divergence=2.0)
    assert not parallel_trends_test(df, "spend", "treated", "period", 4)["parallel_trends_ok"]


def test_event_study_shape_and_reference_excluded():
    df = _panel()
    es = event_study(df, "spend", "treated", "period", reference_period=3)
    assert list(es.columns) == ["period", "effect", "ci_low", "ci_high"]
    assert 3 not in es["period"].tolist()                 # reference period is omitted
    # post-treatment effects should be clearly positive
    assert es[es["period"] >= 4]["effect"].mean() > 3.0
