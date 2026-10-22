"""Uncertainty and significance for model comparisons.

Test sequences are *not* independent: sequences of the same homology
component (or clone, gene, species) share errors. Confidence intervals are
therefore computed with a cluster (block) bootstrap that resamples whole
groups, and paired comparisons resample the same groups for both systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float


def _group_index(groups: Sequence[Hashable]) -> list[np.ndarray]:
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]


def cluster_bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[Hashable] | None = None,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI; resamples groups with replacement when ``groups`` is given.

    ``values`` are per-sequence scores (e.g. 1/0 exact numbering, or per-sequence
    position accuracy). With ``groups=None`` this is the ordinary i.i.d. bootstrap.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx_groups = _group_index(groups) if groups is not None else [np.array([i]) for i in range(len(v))]
    G = len(idx_groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([idx_groups[g] for g in pick])
        boots[b] = statistic(v[idx])
    a = (1 - level) / 2
    return Interval(float(statistic(v)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), level)


def paired_cluster_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    groups: Sequence[Hashable],
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[Interval, float]:
    """CI of mean(a) - mean(b) resampling the same groups for both systems.

    Returns (interval, two-sided bootstrap p-value approximated by the
    fraction of resampled differences on the opposite side of 0, times 2).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx_groups = _group_index(groups)
    G = len(idx_groups)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([idx_groups[g] for g in pick])
        boots[k] = d[idx].mean()
    alpha = (1 - level) / 2
    est = float(d.mean())
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return Interval(est, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha)), level), float(min(1.0, p))


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes.

    b = A right & B wrong, c = A wrong & B right; under H0, b ~ Binomial(b + c, 1/2).
    Returns (b, c, two-sided p-value). Assumes independent pairs: use per-group
    aggregation or the paired cluster bootstrap when sequences are clustered.
    """
    a = np.asarray(correct_a, dtype=bool)
    bb = np.asarray(correct_b, dtype=bool)
    b = int((a & ~bb).sum())
    c = int((~a & bb).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sps.binomtest(b, n, 0.5, alternative="two-sided").pvalue
    return b, c, float(p)


def holm_bonferroni(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
