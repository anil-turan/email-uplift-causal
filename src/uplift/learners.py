"""Meta-learners for uplift (heterogeneous treatment effect) modeling,
implemented from scratch on top of any scikit-learn style classifier.

Uplift = P(outcome | treated) - P(outcome | not treated), estimated per
individual. Unlike a response model ("who will convert?"), an uplift model
answers the causal question "who converts *because* we treated them?" —
the persuadables — so budget is not wasted on sure-things or sleeping-dogs.

    S-learner : one model on [X, treatment]; uplift = f(X, 1) - f(X, 0)
    T-learner : two models, one per arm; uplift = f_t(X) - f_c(X)

Evaluation uses the Qini curve and Qini coefficient, the uplift analogue
of the ROC curve / AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


class SLearner:
    """Single model with treatment as a feature."""

    def __init__(self, model):
        self.model = clone(model)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> SLearner:
        Xt = X.copy()
        Xt["treatment"] = np.asarray(treatment)
        self.model.fit(Xt, y)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X1 = X.copy()
        X1["treatment"] = 1
        X0 = X.copy()
        X0["treatment"] = 0
        p1 = self.model.predict_proba(X1)[:, 1]
        p0 = self.model.predict_proba(X0)[:, 1]
        return p1 - p0


class TLearner:
    """Separate model per treatment arm."""

    def __init__(self, model):
        self.model_t = clone(model)
        self.model_c = clone(model)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> TLearner:
        treatment = np.asarray(treatment)
        self.model_t.fit(X[treatment == 1], y[treatment == 1])
        self.model_c.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.model_t.predict_proba(X)[:, 1]
        p0 = self.model_c.predict_proba(X)[:, 1]
        return p1 - p0


class XLearner:
    """X-learner (Künzel et al. 2019).

    Fixes the T-learner's weakness on imbalanced arms by a second stage. First
    fit outcome models per arm (like the T-learner). Then *impute* the treatment
    effect for every unit against the other arm's model and fit a dedicated
    effect (CATE) regressor on each arm's imputed effects. Finally blend the two
    effect models by the propensity, which lets the arm with more data dominate
    where it is more reliable.

        1. mu0, mu1  : outcome models on control / treated  (classifiers)
        2. D1 = Y1 - mu0(X1)   (imputed effect for treated units)
           D0 = mu1(X0) - Y0   (imputed effect for control units)
        3. tau1, tau0 : regressors fit on D1 / D0
        4. tau(x) = e(x)*tau0(x) + (1 - e(x))*tau1(x)

    ``outcome_model`` is a classifier (uses predict_proba); ``effect_model`` is a
    regressor for the imputed continuous effects.
    """

    def __init__(self, outcome_model, effect_model, propensity: float | None = None):
        self.mu0 = clone(outcome_model)
        self.mu1 = clone(outcome_model)
        self.tau0 = clone(effect_model)
        self.tau1 = clone(effect_model)
        self.propensity = propensity  # constant e; if None, use treated share

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> XLearner:
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        Xt, Xc = X[treatment == 1], X[treatment == 0]
        yt, yc = y[treatment == 1], y[treatment == 0]

        self.mu1.fit(Xt, yt)
        self.mu0.fit(Xc, yc)

        # impute individual treatment effects against the opposite arm
        d1 = yt - self.mu0.predict_proba(Xt)[:, 1]
        d0 = self.mu1.predict_proba(Xc)[:, 1] - yc
        self.tau1.fit(Xt, d1)
        self.tau0.fit(Xc, d0)

        self._e = treatment.mean() if self.propensity is None else self.propensity
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._e * self.tau0.predict(X) + (1 - self._e) * self.tau1.predict(X)


class DRLearner:
    """Doubly-Robust learner (single-split).

    Builds a pseudo-outcome that is unbiased if *either* the outcome model or the
    propensity is right — the same doubly-robust idea used for off-policy
    evaluation in the sibling recsys project, here applied to individual
    treatment effects:

        psi = (mu1 - mu0)
              + W*(Y - mu1)/e
              - (1-W)*(Y - mu0)/(1-e)

    A regressor is then fit on ``psi`` to give a smooth CATE estimate. This
    single-split version is fine for a demonstration; production DR-learners
    cross-fit the nuisance models to remove own-observation bias.
    """

    def __init__(self, outcome_model, effect_model, propensity: float | None = None):
        self.mu0 = clone(outcome_model)
        self.mu1 = clone(outcome_model)
        self.tau = clone(effect_model)
        self.propensity = propensity

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> DRLearner:
        w = np.asarray(treatment).astype(float)
        y = np.asarray(y)
        e = w.mean() if self.propensity is None else self.propensity
        e = np.clip(e, 1e-3, 1 - 1e-3)

        self.mu1.fit(X[w == 1], y[w == 1])
        self.mu0.fit(X[w == 0], y[w == 0])
        m1 = self.mu1.predict_proba(X)[:, 1]
        m0 = self.mu0.predict_proba(X)[:, 1]

        psi = (m1 - m0) + w * (y - m1) / e - (1 - w) * (y - m0) / (1 - e)
        self.tau.fit(X, psi)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.tau.predict(X)


def qini_curve(
    uplift: np.ndarray, treatment: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Qini curve.

    Rank the population by predicted uplift (descending). At each depth we
    count the *incremental* outcomes gained by treating that top fraction,
    corrected for the different arm sizes:

        qini(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))

    Returns (fraction_targeted, incremental_outcomes).
    """
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    y = np.asarray(y)

    order = np.argsort(-uplift)
    t, r = treatment[order], y[order]

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(r * t)
    y_c = np.cumsum(r * (1 - t))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c > 0, n_t / n_c, 0.0)
    qini = y_t - y_c * ratio

    n = len(uplift)
    fraction = np.arange(1, n + 1) / n
    # prepend the origin so the curve starts at (0, 0)
    return np.insert(fraction, 0, 0.0), np.insert(qini, 0, 0.0)


def qini_coefficient(uplift: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal,
    normalised. Higher is better; 0 means no better than random."""
    frac, qini = qini_curve(uplift, treatment, y)
    # random line: straight from origin to the final incremental value
    random_line = frac * qini[-1]
    area_model = _trapezoid(qini, frac)
    area_random = _trapezoid(random_line, frac)
    return area_model - area_random


def uplift_at_k(
    uplift: np.ndarray, treatment: np.ndarray, y: np.ndarray, k: float = 0.3
) -> dict:
    """Observed incremental response rate among the top-k fraction ranked by
    predicted uplift. This is the number a marketer actually acts on."""
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    y = np.asarray(y)

    order = np.argsort(-uplift)
    cutoff = int(np.ceil(k * len(uplift)))
    idx = order[:cutoff]

    t, r = treatment[idx], y[idx]
    rate_t = r[t == 1].mean() if (t == 1).any() else np.nan
    rate_c = r[t == 0].mean() if (t == 0).any() else np.nan
    return {
        "k": k,
        "n_targeted": cutoff,
        "response_treated": rate_t,
        "response_control": rate_c,
        "incremental_rate": rate_t - rate_c,
    }
