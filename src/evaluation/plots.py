"""Reusable plotting helpers for the experiment and uplift notebooks.
All functions return a matplotlib Axes so notebooks can save the figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_qini(frac: np.ndarray, qini: np.ndarray, label: str, ax=None):
    """Qini curve vs the random-targeting diagonal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frac, qini, label=label, linewidth=2)
    ax.plot(
        [0, 1], [0, qini[-1]], linestyle="--", color="grey", label="Random targeting"
    )
    ax.set_xlabel("Fraction of population targeted (ranked by predicted uplift)")
    ax.set_ylabel("Incremental outcomes")
    ax.set_title("Qini curve")
    ax.legend()
    return ax


def plot_uplift_by_bin(uplift: np.ndarray, n_bins: int = 10, ax=None):
    """Distribution of predicted uplift across deciles — shows how much of the
    population is 'persuadable' vs neutral vs negative (sleeping dogs)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(-uplift)
    binned = np.array_split(uplift[order], n_bins)
    means = [b.mean() for b in binned]
    colors = ["#2a9d8f" if m > 0 else "#e76f51" for m in means]
    ax.bar(range(1, n_bins + 1), means, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Uplift decile (1 = highest predicted uplift)")
    ax.set_ylabel("Mean predicted uplift")
    ax.set_title("Predicted uplift by decile")
    return ax


def plot_balance(balance_df, ax=None):
    """Standardised mean differences between arms — a randomisation check.
    |SMD| < 0.1 is the conventional 'well balanced' threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.barh(balance_df.index, balance_df["smd"], color="#264653")
    ax.axvline(0.1, color="red", linestyle="--", linewidth=1)
    ax.axvline(-0.1, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Standardised mean difference (treatment - control)")
    ax.set_title("Covariate balance check")
    return ax
