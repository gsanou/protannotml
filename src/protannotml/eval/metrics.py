"""Task metrics for numbering, segmentation and assignment.

All functions take plain Python/NumPy inputs so that baselines (ANARCI,
AbNumber, IgBLAST, DomainGapAlign outputs) and models are scored identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------

def position_accuracy(pred: Sequence[Sequence[str]], true: Sequence[Sequence[str]]) -> float:
    """Fraction of residues whose predicted IMGT label equals the reference (micro average)."""
    hits = total = 0
    for p, t in zip(pred, true):
        if len(p) != len(t):
            raise ValueError("prediction and reference must cover the same residues")
        hits += sum(a == b for a, b in zip(p, t))
        total += len(t)
    return hits / total if total else float("nan")


def sequence_exact_match(pred: Sequence[Sequence[str]], true: Sequence[Sequence[str]]) -> np.ndarray:
    """Per-sequence boolean: every residue correctly numbered."""
    return np.array([list(p) == list(t) for p, t in zip(pred, true)], dtype=bool)


def per_label_accuracy(pred: Sequence[Sequence[str]], true: Sequence[Sequence[str]]) -> dict[str, float]:
    """Accuracy for each reference label (e.g. each IMGT position), to locate errors."""
    hit: dict[str, int] = {}
    tot: dict[str, int] = {}
    for p, t in zip(pred, true):
        for a, b in zip(p, t):
            tot[b] = tot.get(b, 0) + 1
            hit[b] = hit.get(b, 0) + (a == b)
    return {k: hit[k] / tot[k] for k in tot}


# ---------------------------------------------------------------------------
# Segments and boundaries
# ---------------------------------------------------------------------------

def spans(labels: Sequence[Hashable], ignore: Iterable[Hashable] = ("OUT",)) -> list[tuple[Hashable, int, int]]:
    """Contiguous runs (label, start, end_exclusive), skipping ``ignore`` labels."""
    ignore = set(ignore)
    out = []
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] not in ignore:
            out.append((labels[i], i, j))
        i = j
    return out


def boundaries(labels: Sequence[Hashable], ignore: Iterable[Hashable] = ("OUT",)) -> set[tuple[Hashable, str, int]]:
    """Set of (label, 'start'|'end', index) for every span."""
    b = set()
    for lab, s, e in spans(labels, ignore):
        b.add((lab, "start", s))
        b.add((lab, "end", e))
    return b


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(p, r, f, tp, fp, fn)


def boundary_f1(pred_labels: Sequence[Sequence[Hashable]], true_labels: Sequence[Sequence[Hashable]],
                tolerance: int = 0, ignore: Iterable[Hashable] = ("OUT",)) -> PRF:
    """Boundary precision/recall/F1 with a +/- ``tolerance`` residue window.

    A predicted boundary matches a reference boundary of the same label and
    side if their indices differ by at most ``tolerance``; matching is
    one-to-one (greedy by distance).
    """
    tp = fp = fn = 0
    for pl, tl in zip(pred_labels, true_labels):
        pb, tb = boundaries(pl, ignore), boundaries(tl, ignore)
        used = set()
        for key in sorted(pb, key=lambda x: (str(x[0]), x[1], x[2])):
            cands = [t for t in tb if t[:2] == key[:2] and t not in used and abs(t[2] - key[2]) <= tolerance]
            if cands:
                best = min(cands, key=lambda t: abs(t[2] - key[2]))
                used.add(best)
                tp += 1
            else:
                fp += 1
        fn += len(tb) - len(used)
    return _prf(tp, fp, fn)


def span_exact_f1(pred_labels: Sequence[Sequence[Hashable]], true_labels: Sequence[Sequence[Hashable]],
                  ignore: Iterable[Hashable] = ("OUT",)) -> PRF:
    """Span-level F1 where a span counts only if label, start and end all match."""
    tp = fp = fn = 0
    for pl, tl in zip(pred_labels, true_labels):
        ps, ts = set(spans(pl, ignore)), set(spans(tl, ignore))
        tp += len(ps & ts)
        fp += len(ps - ts)
        fn += len(ts - ps)
    return _prf(tp, fp, fn)


# ---------------------------------------------------------------------------
# Gene / allele assignment
# ---------------------------------------------------------------------------

def set_accuracy(pred: Sequence[Hashable], true_sets: Sequence[Iterable[Hashable]]) -> float:
    """Fraction of predictions that fall in the set of equally correct answers."""
    return float(np.mean([p in set(s) for p, s in zip(pred, true_sets)]))


def top_k_set_accuracy(ranked: Sequence[Sequence[Hashable]], true_sets: Sequence[Iterable[Hashable]], k: int) -> float:
    return float(np.mean([bool(set(r[:k]) & set(s)) for r, s in zip(ranked, true_sets)]))


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """ECE = sum_b |B_b|/n * |acc(B_b) - conf(B_b)| with equal-width bins on [0, 1]."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def coverage_accuracy_curve(confidence: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Selective prediction: accuracy among the c most confident predictions, for every coverage c."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(corr[order])
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), cum / n
