"""Masking strategies for masked language modelling (DAPT / TAPT).

Framework-agnostic (NumPy). Works on token ids produced by the base model's
own tokenizer; the caller supplies the ids of the mask token, the special
tokens, and the ids that are valid amino-acid replacements.

Conventions follow BERT/ESM: a selected position is replaced by the mask
token with probability 0.8, by a random amino acid with probability 0.1, and
left unchanged with probability 0.1. Unselected positions get label -100
(ignored by the loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MaskingConfig:
    rate: float = 0.15  # expected fraction of maskable tokens selected
    p_mask: float = 0.8
    p_random: float = 0.1  # the remaining 1 - p_mask - p_random stay unchanged

    def __post_init__(self) -> None:
        if not 0 < self.rate < 1:
            raise ValueError("rate must be in (0, 1)")
        if self.p_mask + self.p_random > 1:
            raise ValueError("p_mask + p_random must be <= 1")


def uniform_probs(n: int, rate: float) -> np.ndarray:
    return np.full(n, rate, dtype=float)


def region_fixed_rate_probs(
    regions: Sequence[str],
    fixed_rates: Mapping[str, float],
    overall_rate: float,
) -> np.ndarray:
    """Per-token selection probabilities with fixed rates for some regions.

    The rate of the remaining tokens is set so that the expected overall
    fraction equals ``overall_rate`` (preferential masking, Ng & Briney 2025):

        r_rest = (overall_rate * N - sum_r n_r * rate_r) / n_rest

    Raises ValueError if no valid ``r_rest`` in [0, 1] exists.
    """
    regions = list(regions)
    n = len(regions)
    probs = np.empty(n)
    fixed_mass = 0.0
    rest_idx = []
    for i, r in enumerate(regions):
        if r in fixed_rates:
            probs[i] = fixed_rates[r]
            fixed_mass += fixed_rates[r]
        else:
            rest_idx.append(i)
    if rest_idx:
        r_rest = (overall_rate * n - fixed_mass) / len(rest_idx)
        if not 0.0 <= r_rest <= 1.0:
            raise ValueError(f"infeasible rates: remaining rate would be {r_rest:.3f}")
        probs[rest_idx] = r_rest
    return probs


def region_weighted_probs(regions: Sequence[str], weights: Mapping[str, float], overall_rate: float) -> np.ndarray:
    """Probabilities proportional to region weights, rescaled to ``overall_rate`` on average."""
    w = np.array([weights.get(r, 1.0) for r in regions], dtype=float)
    probs = overall_rate * w * len(w) / w.sum()
    if probs.max() > 1:
        raise ValueError("weights too extreme for the requested overall rate")
    return probs


def span_selection(n: int, rate: float, mean_span: float, rng: np.random.Generator) -> np.ndarray:
    """Select contiguous spans (geometric lengths) until ~rate * n tokens are selected."""
    target = int(round(rate * n))
    selected = np.zeros(n, dtype=bool)
    p = 1.0 / mean_span
    guard = 0
    while selected.sum() < target and guard < 10 * n:
        length = int(rng.geometric(p))
        start = int(rng.integers(0, n))
        selected[start:min(n, start + length)] = True
        guard += 1
    # trim overshoot at random to hit the target exactly
    idx = np.flatnonzero(selected)
    if len(idx) > target:
        drop = rng.choice(idx, size=len(idx) - target, replace=False)
        selected[drop] = False
    return selected


def apply_mask(
    input_ids: np.ndarray,
    selected: np.ndarray,
    mask_token_id: int,
    replacement_ids: Sequence[int],
    config: MaskingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the 80/10/10 corruption to selected positions; return (inputs, labels)."""
    inputs = input_ids.copy()
    labels = np.full_like(input_ids, IGNORE_INDEX)
    labels[selected] = input_ids[selected]
    u = rng.random(input_ids.shape)
    to_mask = selected & (u < config.p_mask)
    to_random = selected & (u >= config.p_mask) & (u < config.p_mask + config.p_random)
    inputs[to_mask] = mask_token_id
    replacement_ids = np.asarray(replacement_ids)
    inputs[to_random] = rng.choice(replacement_ids, size=int(to_random.sum()))
    return inputs, labels


def mask_sequence(
    input_ids: np.ndarray,
    special_mask: np.ndarray,
    probs: np.ndarray,
    mask_token_id: int,
    replacement_ids: Sequence[int],
    config: MaskingConfig = MaskingConfig(),
    rng: np.random.Generator | None = None,
    min_masked: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select positions with per-token ``probs`` (special tokens never), then corrupt.

    ``probs`` is aligned with the maskable (non-special) tokens, in order.
    At least ``min_masked`` tokens are selected so every sequence contributes.
    """
    rng = rng or np.random.default_rng()
    maskable = np.flatnonzero(~special_mask)
    if len(probs) != len(maskable):
        raise ValueError("probs must have one entry per non-special token")
    sel_local = rng.random(len(maskable)) < probs
    if sel_local.sum() < min_masked:
        sel_local[rng.choice(len(maskable), size=min_masked, replace=False, p=probs / probs.sum())] = True
    selected = np.zeros_like(special_mask, dtype=bool)
    selected[maskable[sel_local]] = True
    return apply_mask(input_ids, selected, mask_token_id, replacement_ids, config, rng)


def temperature_sampling_weights(source_sizes: Mapping[str, int], alpha: float) -> dict[str, float]:
    """Sampling probability per corpus source, p_s proportional to n_s ** alpha.

    alpha = 1 reproduces the natural proportions; alpha -> 0 tends to uniform
    over sources. Used to keep rare species/loci visible during DAPT.
    """
    z = {s: n ** alpha for s, n in source_sizes.items()}
    tot = sum(z.values())
    return {s: v / tot for s, v in z.items()}


def mlm_loss_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy over positions whose label != -100 (reference implementation)."""
    keep = labels != IGNORE_INDEX
    if not keep.any():
        return 0.0
    z = logits[keep]
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(z)), labels[keep]].mean())
