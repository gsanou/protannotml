"""Leakage-resistant partitioning of sequence datasets.

Design
------
1. Compute an all-vs-all similarity graph with MMseqs2 (``easy-search``) and
   keep every pair at or above a threshold ``t``. Do *not* rely on the default
   ``--max-seqs 300``: dense IG/TR families have thousands of neighbours per
   sequence and truncated hit lists silently drop edges.
2. Split by *connected components* of that graph (single linkage). Unlike
   representative ("set-cover") clustering, this guarantees that no pair of
   sequences in different splits exceeds ``t`` - up to the sensitivity of the
   search.
3. When a giant component appears (typical for IG V domains at low ``t``),
   either raise ``t``, hold out whole biological groups (gene clusters,
   species, loci), or remove a minimal set of bridging sequences
   (GraphPart-style), and report what was removed.
4. Audit the final split: maximum cross-split identity and number of
   violating pairs, computed on an independent, exhaustive search.

All functions below are pure Python/NumPy except :func:`mmseqs_all_vs_all`,
which shells out to the ``mmseqs`` binary (tested with release 18).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

Edge = tuple[str, str, float]  # (id_a, id_b, identity in [0, 1])


# ---------------------------------------------------------------------------
# MMseqs2 wrapper
# ---------------------------------------------------------------------------

def mmseqs_all_vs_all(
    fasta: str | Path,
    min_seq_id: float,
    coverage: float = 0.8,
    cov_mode: int = 0,
    sensitivity: float = 7.5,
    max_seqs: int = 100_000,
    threads: int = 8,
    workdir: str | Path | None = None,
    mmseqs: str = "mmseqs",
) -> list[Edge]:
    """All-vs-all search; returns undirected edges with identity >= ``min_seq_id``.

    Identity is MMseqs2 ``fident`` (identical positions / alignment length,
    ``--seq-id-mode 0``). ``cov_mode=0`` requires ``coverage`` on both query
    and target, which prevents a short fragment from linking two full domains.
    The edge keeps the maximum identity over the two search directions.
    """
    if shutil.which(mmseqs) is None:
        raise FileNotFoundError("mmseqs binary not found on PATH")
    tmp = Path(workdir or tempfile.mkdtemp(prefix="mmseqs_"))
    out = tmp / "hits.tsv"
    cmd = [
        mmseqs, "easy-search", str(fasta), str(fasta), str(out), str(tmp / "tmp"),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode),
        "--seq-id-mode", "0", "-s", str(sensitivity), "--max-seqs", str(max_seqs),
        "--format-output", "query,target,fident,alnlen,qcov,tcov",
        "--threads", str(threads),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return read_hits(out, min_seq_id)


def read_hits(path: str | Path, min_seq_id: float) -> list[Edge]:
    best: dict[tuple[str, str], float] = {}
    with open(path) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            q, t, fident = row[0], row[1], float(row[2])
            if q == t or fident < min_seq_id:
                continue
            key = (q, t) if q < t else (t, q)
            best[key] = max(best.get(key, 0.0), fident)
    return [(a, b, s) for (a, b), s in best.items()]


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items: Iterable[Hashable]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in self.parent}

    def find(self, x: Hashable) -> Hashable:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_components(ids: Sequence[str], edges: Iterable[Edge], threshold: float) -> list[list[str]]:
    """Single-linkage components of the graph restricted to edges >= threshold."""
    uf = UnionFind(ids)
    for a, b, s in edges:
        if s >= threshold:
            uf.union(a, b)
    groups: dict[Hashable, list[str]] = defaultdict(list)
    for x in ids:
        groups[uf.find(x)].append(x)
    return sorted(groups.values(), key=len, reverse=True)


def greedy_set_cover_clusters(ids: Sequence[str], edges: Iterable[Edge], threshold: float) -> list[list[str]]:
    """Representative clustering (MMseqs2 ``--cluster-mode 0``-like), for comparison.

    Repeatedly picks the unassigned sequence with most unassigned neighbours as
    representative and assigns it its neighbours. Members are within
    ``threshold`` of their representative, *not* of each other, and pairs across
    clusters can exceed ``threshold``.
    """
    nbrs: dict[str, set[str]] = {x: set() for x in ids}
    for a, b, s in edges:
        if s >= threshold:
            nbrs[a].add(b)
            nbrs[b].add(a)
    unassigned = set(ids)
    clusters = []
    while unassigned:
        rep = max(sorted(unassigned), key=lambda x: len(nbrs[x] & unassigned))
        members = [rep] + sorted(nbrs[rep] & unassigned)
        unassigned -= set(members)
        clusters.append(members)
    return clusters


# ---------------------------------------------------------------------------
# Assigning groups to splits
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    assignment: dict[str, str]  # id -> split name
    removed: list[str]  # ids dropped to break bridges (never silently)

    def ids(self, split: str) -> list[str]:
        return [k for k, v in self.assignment.items() if v == split]


def assign_groups_to_splits(
    groups: Sequence[Sequence[str]],
    ratios: Mapping[str, float],
    labels: Mapping[str, Hashable] | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Greedy, label-aware bin packing of whole groups into splits.

    Groups are processed from largest to smallest (ties broken by a seeded
    shuffle). Each group goes to the split with the largest remaining deficit,
    measured on total size and, if ``labels`` is given, on the label
    distribution (sum of absolute deficits). Groups are never broken.
    """
    rng = np.random.default_rng(seed)
    names = list(ratios)
    total = sum(len(g) for g in groups)
    target = {s: ratios[s] / sum(ratios.values()) * total for s in names}
    size = {s: 0 for s in names}
    label_target: dict[str, Counter] = {}
    label_size: dict[str, Counter] = {s: Counter() for s in names}
    if labels is not None:
        overall = Counter(labels[x] for g in groups for x in g)
        label_target = {s: Counter({k: v * ratios[s] / sum(ratios.values()) for k, v in overall.items()}) for s in names}

    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda i: -len(groups[i]))
    assignment: dict[str, str] = {}
    for i in order:
        g = groups[i]
        g_labels = Counter(labels[x] for x in g) if labels is not None else Counter()

        def cost(s: str) -> float:
            # deficit after adding the group (lower is better)
            c = abs(target[s] - (size[s] + len(g))) - abs(target[s] - size[s])
            if labels is not None:
                for k, v in g_labels.items():
                    c += 0.5 * (abs(label_target[s][k] - (label_size[s][k] + v)) - abs(label_target[s][k] - label_size[s][k]))
            return c

        best = min(names, key=lambda s: (cost(s), size[s] / max(target[s], 1e-9)))
        size[best] += len(g)
        label_size[best].update(g_labels)
        for x in g:
            assignment[x] = best
    return assignment


def split_by_components(
    ids: Sequence[str],
    edges: Sequence[Edge],
    threshold: float,
    ratios: Mapping[str, float],
    labels: Mapping[str, Hashable] | None = None,
    max_component_fraction: float = 0.2,
    seed: int = 0,
) -> SplitResult:
    """Component-based split with GraphPart-style bridge removal.

    If the largest component exceeds ``max_component_fraction`` of the data,
    the highest-degree nodes of that component are removed one by one until it
    falls apart below the limit. Removed ids are returned, not hidden.
    """
    kept = list(ids)
    removed: list[str] = []
    live_edges = [e for e in edges if e[2] >= threshold]
    while True:
        comps = connected_components(kept, live_edges, threshold)
        if len(comps[0]) <= max_component_fraction * len(ids) or len(comps[0]) <= 1:
            break
        giant = set(comps[0])
        degree = Counter()
        for a, b, _ in live_edges:
            if a in giant and b in giant:
                degree[a] += 1
                degree[b] += 1
        victim = max(sorted(giant), key=lambda x: degree[x])
        removed.append(victim)
        kept.remove(victim)
        live_edges = [e for e in live_edges if victim not in (e[0], e[1])]
    assignment = assign_groups_to_splits(comps, ratios, labels, seed)
    return SplitResult(assignment, removed)


# ---------------------------------------------------------------------------
# Biological hold-outs
# ---------------------------------------------------------------------------

def merge_near_identical_groups(
    group_of: Mapping[str, str],
    group_edges: Iterable[tuple[str, str, float]],
    threshold: float,
) -> dict[str, str]:
    """Merge biological groups (e.g. germline genes) whose references are near-identical.

    Example: IGHV3-23 and its duplicate IGHV3-23D. Holding out one but not the
    other leaks the held-out gene. ``group_edges`` are identities between the
    groups' reference sequences. Returns id -> merged group label.
    """
    groups = sorted(set(group_of.values()))
    uf = UnionFind(groups)
    for a, b, s in group_edges:
        if s >= threshold and a in uf.parent and b in uf.parent:
            uf.union(a, b)
    members: dict[Hashable, list[str]] = defaultdict(list)
    for g in groups:
        members[uf.find(g)].append(g)
    label = {g: "+".join(sorted(members[uf.find(g)])) for g in groups}
    return {x: label[g] for x, g in group_of.items()}


def holdout_groups(group_of: Mapping[str, str], held_out: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (kept_ids, held_out_ids) for a set of held-out group labels."""
    held = set(held_out)
    kept = [x for x, g in group_of.items() if g not in held]
    out = [x for x, g in group_of.items() if g in held]
    return kept, out


# ---------------------------------------------------------------------------
# Audit and novelty tiers
# ---------------------------------------------------------------------------

@dataclass
class LeakageAudit:
    n_violations: int
    max_cross_identity: float
    worst_pairs: list[Edge]


def audit_split(assignment: Mapping[str, str], edges: Iterable[Edge], threshold: float,
                splits: tuple[str, str] = ("train", "test"), top: int = 10) -> LeakageAudit:
    """Count cross-split pairs >= threshold between two splits."""
    a_name, b_name = splits
    cross = []
    for a, b, s in edges:
        sa, sb = assignment.get(a), assignment.get(b)
        if {sa, sb} == {a_name, b_name}:
            cross.append((a, b, s))
    cross.sort(key=lambda e: -e[2])
    viol = [e for e in cross if e[2] >= threshold]
    return LeakageAudit(len(viol), cross[0][2] if cross else 0.0, viol[:top])


NOVELTY_BINS: tuple[tuple[str, float], ...] = (
    (">=95%", 0.95), ("90-95%", 0.90), ("80-90%", 0.80), ("70-80%", 0.70), ("<70%", 0.0),
)


def novelty_tier(max_identity_to_train: float) -> str:
    """Bin a test sequence by its maximum identity to any training sequence."""
    for name, lo in NOVELTY_BINS:
        if max_identity_to_train >= lo:
            return name
    return NOVELTY_BINS[-1][0]


def max_identity_to_split(test_ids: Iterable[str], train_ids: Iterable[str], edges: Iterable[Edge]) -> dict[str, float]:
    test, train = set(test_ids), set(train_ids)
    best = {x: 0.0 for x in test}
    for a, b, s in edges:
        if a in test and b in train:
            best[a] = max(best[a], s)
        elif b in test and a in train:
            best[b] = max(best[b], s)
    return best


def filter_pretraining_corpus(corpus_ids: Iterable[str], protected_ids: Iterable[str],
                              edges: Iterable[Edge], threshold: float) -> list[str]:
    """Drop corpus sequences too similar to validation/test sequences.

    Applies to the DAPT and TAPT corpora: masked-language-model pretraining on
    a test sequence is leakage, even without its labels.
    """
    protected = set(protected_ids)
    banned = set()
    for a, b, s in edges:
        if s >= threshold:
            if a in protected:
                banned.add(b)
            if b in protected:
                banned.add(a)
    return [x for x in corpus_ids if x not in banned and x not in protected]
