"""Tests for the uplift meta-learners and Qini evaluation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.uplift.learners import (
    DRLearner,
    SLearner,
    TLearner,
    XLearner,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)


def _synthetic_uplift_data(n=4000, seed=0):
    """Half the population is 'persuadable' (treatment raises P(y)), half is
    inert. A good uplift model should rank the persuadables first."""
    rng = np.random.default_rng(seed)
    persuadable = rng.integers(0, 2, size=n)
    treatment = rng.integers(0, 2, size=n)
    base = 0.2
    p = base + 0.5 * persuadable * treatment  # effect only for persuadables
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"persuadable": persuadable, "noise": rng.normal(size=n)})
    return X, treatment, y


def test_qini_curve_starts_at_origin_and_is_ordered():
    X, t, y = _synthetic_uplift_data()
    frac, qini = qini_curve(np.arange(len(y))[::-1], t, y)
    assert frac[0] == 0.0 and qini[0] == 0.0
    assert frac[-1] == 1.0
    assert len(frac) == len(y) + 1


def test_perfect_ranking_beats_random_qini():
    """A model that perfectly ranks persuadables first should have a much
    higher Qini coefficient than random scoring."""
    X, t, y = _synthetic_uplift_data()
    perfect = X["persuadable"].values + 1e-6 * np.random.default_rng(1).normal(size=len(y))
    random_scores = np.random.default_rng(2).normal(size=len(y))
    assert qini_coefficient(perfect, t, y) > qini_coefficient(random_scores, t, y)


def test_learners_recover_persuadable_signal():
    """Both meta-learners should assign higher uplift to persuadables than to
    inert customers on data where only persuadables respond."""
    X, t, y = _synthetic_uplift_data(n=8000)
    for Learner in (SLearner, TLearner):
        model = Learner(LogisticRegression(max_iter=1000)).fit(X, t, y)
        up = model.predict_uplift(X)
        mask = X["persuadable"] == 1
        assert up[mask].mean() > up[~mask].mean()


def test_x_and_dr_learners_recover_persuadable_signal():
    """The two-stage learners (X, DR) should also rank persuadables above inert
    customers. They take a classifier for outcomes and a regressor for effects."""
    X, t, y = _synthetic_uplift_data(n=8000)
    mask = (X["persuadable"] == 1).to_numpy()
    for Learner in (XLearner, DRLearner):
        model = Learner(LogisticRegression(max_iter=1000), LinearRegression()).fit(X, t, y)
        up = model.predict_uplift(X)
        assert up[mask].mean() > up[~mask].mean()


def test_uplift_at_k_returns_expected_keys():
    X, t, y = _synthetic_uplift_data()
    scores = X["persuadable"].values.astype(float)
    r = uplift_at_k(scores, t, y, k=0.3)
    assert set(r) == {"k", "n_targeted", "response_treated", "response_control", "incremental_rate"}
    assert r["n_targeted"] == int(np.ceil(0.3 * len(y)))
