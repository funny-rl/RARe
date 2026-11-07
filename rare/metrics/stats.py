"""Seed statistics of the paper tables: IQM and percentile bootstrap intervals."""

import numpy as np
from scipy.stats import trim_mean


def iqm(values, axis=None):
    """Interquartile mean: mean of the middle 50% (25% cut per side, as in rliable)."""
    return trim_mean(values, .25, axis=0 if axis is None else axis)


def bootstrap_ci(values, n_bootstrap=10000, seed=0):
    """95% percentile bootstrap CI of the IQM over seeds (the first axis)."""
    values = np.asarray(values)
    indices = np.random.default_rng(seed).integers(0, len(values), (n_bootstrap, len(values)))
    low, high = np.percentile(iqm(values[indices], axis=1), [2.5, 97.5], axis=0)
    return low, high


def paired(left, right, n_bootstrap=10000, seed=1):
    """IQM difference with seeds resampled jointly, its 95% CI, and the seed win rate."""
    left, right = np.asarray(left), np.asarray(right)
    indices = np.random.default_rng(seed).integers(0, len(left), (n_bootstrap, len(left)))
    samples = iqm(left[indices], axis=1) - iqm(right[indices], axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"difference": float(iqm(left) - iqm(right)), "ci": (float(low), float(high)),
            "win_rate": float(np.mean((left > right) + .5 * (left == right)))}
