"""Reference (NumPy) formulas for multi-task loss weighting.

These mirror the PyTorch modules in ``protannotml.models.mtl`` and are used in
the tests and the tutorial to check closed-form properties.

Notation: ``s_t = log(sigma_t ** 2)`` is the learnable log-variance of task t.

Kendall, Gal & Cipolla (CVPR 2018), homoscedastic uncertainty:
    regression     : 0.5 * exp(-s) * L + 0.5 * s
    classification : exp(-s) * L + 0.5 * s          (their approximation)
Liebel & Koerner (2018), positive regulariser variant:
    0.5 * exp(-s) * L + log(1 + exp(s))
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def weighted_sum(losses: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(weights[t] * losses[t] for t in losses))


def kendall_term(loss: float, s: float, kind: str = "classification") -> float:
    if kind == "classification":
        return float(np.exp(-s) * loss + 0.5 * s)
    if kind == "regression":
        return float(0.5 * np.exp(-s) * loss + 0.5 * s)
    raise ValueError(kind)


def kendall_optimal_s(loss: float, kind: str = "classification") -> float:
    """Closed-form minimiser of the Kendall term in s for a fixed task loss.

    classification: d/ds [e^{-s} L + s/2] = 0  ->  s* = log(2L), weight e^{-s*} = 1/(2L)
    regression:     d/ds [e^{-s} L/2 + s/2] = 0 ->  s* = log(L),  weight e^{-s*}/2 = 1/(2L)
    In both cases each task's weighted loss contribution at the optimum is 1/2.
    """
    if loss <= 0:
        raise ValueError("loss must be positive")
    return float(np.log(2 * loss)) if kind == "classification" else float(np.log(loss))


def liebel_term(loss: float, s: float) -> float:
    return float(0.5 * np.exp(-s) * loss + np.log1p(np.exp(s)))


def liebel_optimal_s(loss: float) -> float:
    """Minimiser of 0.5 e^{-s} L + log(1 + e^{s}).

    Setting the derivative to zero: -0.5 L e^{-s} + e^{s}/(1 + e^{s}) = 0.
    With u = e^{s}: u^2 - 0.5 L u - 0.5 L = 0, so u = (0.5L + sqrt(0.25L^2 + 2L)) / 2.
    The weight 0.5/u stays bounded (<= ~ 1/sqrt(2L) for small L) instead of 1/(2L).
    """
    a = 0.5 * loss
    u = (a + np.sqrt(a * a + 4 * a)) / 2
    return float(np.log(u))


def simulate_uncertainty_weighting(
    loss_curves: np.ndarray,
    lr: float = 0.05,
    s0: float = 0.0,
    kind: str = "classification",
    s_min: float | None = None,
) -> np.ndarray:
    """Gradient descent on the log-variances s_t while task losses follow given curves.

    loss_curves: array (steps, tasks). Returns the trajectory of s (steps, tasks).
    ``s_min`` optionally clamps s from below (caps the weight of nearly solved tasks).
    """
    steps, tasks = loss_curves.shape
    s = np.full(tasks, s0, dtype=float)
    traj = np.empty((steps, tasks))
    for k in range(steps):
        L = loss_curves[k]
        if kind == "classification":
            grad = -np.exp(-s) * L + 0.5
        else:
            grad = -0.5 * np.exp(-s) * L + 0.5
        s = s - lr * grad
        if s_min is not None:
            s = np.maximum(s, s_min)
        traj[k] = s
    return traj


def masked_task_mean(per_sample_losses: Sequence[float], has_label: Sequence[bool]) -> float:
    """Average a task loss over the samples that carry a label for it (0 if none)."""
    v = np.asarray(per_sample_losses, dtype=float)
    m = np.asarray(has_label, dtype=bool)
    return float(v[m].mean()) if m.any() else 0.0


def marginal_nll(probs: np.ndarray, target_set: Sequence[int]) -> float:
    """-log sum_{a in target_set} p(a): loss for set-valued targets (equivalent alleles)
    and for coarse labels obtained by marginalising a fine-grained softmax."""
    return float(-np.log(np.asarray(probs)[list(target_set)].sum()))
