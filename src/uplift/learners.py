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
