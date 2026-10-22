"""Constrained decoding of IMGT numbering.

Given per-residue log-probabilities over a position vocabulary sorted in
sequence order (``protannotml.imgt.vocab``), find the labelling with maximal
total log-probability such that numbered labels are *strictly increasing*
along the sequence. Label 0 (``UNNUMBERED``) is allowed anywhere and does not
constrain its neighbours.

Dynamic programme, O(L * K) time with a running prefix maximum:

    best[i, k] = logp[i, k] + max_{k' < k} B[i-1, k']     for k >= 1
    B[i, k]    = best value of residues 0..i whose last numbered label is k
                 (k = 0: nothing numbered yet)

An optional ``allowed`` mask (L x K booleans) restricts labels per residue,
e.g. to positions compatible with the predicted domain type.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1e30


def monotonic_decode(logp: np.ndarray, allowed: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Return (labels, score) maximising sum_i logp[i, labels[i]] under the constraint."""
    logp = np.asarray(logp, dtype=float)
    L, K = logp.shape
    if allowed is not None:
        logp = np.where(allowed, logp, NEG_INF)
    # B[k]: best score so far with last numbered label k (k=0 means none yet)
    B = np.full(K, NEG_INF)
    B[0] = 0.0
    back_choice = np.zeros((L, K), dtype=np.int64)  # label chosen at i for state k
    back_prev = np.zeros((L, K), dtype=np.int64)  # state before i
    for i in range(L):
        newB = np.full(K, NEG_INF)
        # Option 1: residue i is UNNUMBERED, state unchanged.
        stay = B + logp[i, 0]
        newB = np.maximum(newB, stay)
        back_choice[i] = 0
        back_prev[i] = np.arange(K)
        # Option 2: residue i takes label k >= 1, previous state k' < k.
        prefix_val = np.maximum.accumulate(B)  # best over states <= k
        prefix_arg = np.zeros(K, dtype=np.int64)
        arg = 0
        for k in range(K):
            if B[k] >= B[arg]:
                arg = k
            prefix_arg[k] = arg
        cand = np.full(K, NEG_INF)
        cand[1:] = prefix_val[:-1] + logp[i, 1:]
        better = cand > newB
        newB = np.where(better, cand, newB)
        back_choice[i] = np.where(better, np.arange(K), back_choice[i])
        prev_for_k = np.concatenate([[0], prefix_arg[:-1]])
        back_prev[i] = np.where(better, prev_for_k, back_prev[i])
        B = newB
    state = int(np.argmax(B))
    score = float(B[state])
    labels = np.zeros(L, dtype=np.int64)
    for i in range(L - 1, -1, -1):
        labels[i] = back_choice[i, state]
        state = int(back_prev[i, state])
    return labels, score


def is_valid_numbering(labels: np.ndarray) -> bool:
    numbered = [int(x) for x in labels if x != 0]
    return all(a < b for a, b in zip(numbered, numbered[1:]))


def brute_force_decode(logp: np.ndarray) -> tuple[np.ndarray, float]:
    """Exhaustive search for tests (tiny L and K only)."""
    import itertools

    L, K = logp.shape
    best, best_lab = NEG_INF, None
    for lab in itertools.product(range(K), repeat=L):
        lab = np.array(lab)
        if is_valid_numbering(lab):
            sc = logp[np.arange(L), lab].sum()
            if sc > best:
                best, best_lab = sc, lab
    return best_lab, float(best)
