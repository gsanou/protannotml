"""Label vocabularies for the per-residue heads.

The IMGT position vocabulary is sorted in *sequence order*. Hence a valid
numbering is exactly a strictly increasing sequence of label indices, which is
what the monotonic decoder (``protannotml.decoding.monotonic``) enforces.
"""

from __future__ import annotations

from dataclasses import dataclass

from .numbering import DomainType, ImgtPosition, V_REGIONS

UNNUMBERED = "UNNUMBERED"  # residue outside any domain, or an unmappable FR insertion


@dataclass(frozen=True)
class PositionVocabulary:
    labels: tuple[str, ...]
    positions: tuple[ImgtPosition | None, ...]

    @property
    def size(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        return self._index[label]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {lab: i for i, lab in enumerate(self.labels)})

    @property
    def unnumbered_index(self) -> int:
        return self._index[UNNUMBERED]


def v_domain_vocabulary(
    max_cdr3_insertions_per_side: int = 40,
    max_cdr12_insertions_per_side: int = 3,
) -> PositionVocabulary:
    """Vocabulary for V-domain numbering.

    Index 0 is ``UNNUMBERED``; indices 1.. follow sequence order.

    ``max_cdr3_insertions_per_side`` bounds CDR3-IMGT length to
    13 + 2 * K residues (K=40 -> 93, enough for bovine ultralong CDR3; check
    the empirical maximum in the training data and document the choice).
    """
    pos: list[ImgtPosition] = [ImgtPosition(n) for n in range(1, 129)]
    for k in range(1, max_cdr3_insertions_per_side + 1):
        pos += [ImgtPosition(111, k), ImgtPosition(112, k)]
    # Not documented by IMGT: centre insertions for CDR1 (32/33) and CDR2 (60/61).
    for k in range(1, max_cdr12_insertions_per_side + 1):
        pos += [ImgtPosition(32, k), ImgtPosition(33, k), ImgtPosition(60, k), ImgtPosition(61, k)]
    pos = sorted(pos)
    labels = (UNNUMBERED,) + tuple(str(p) for p in pos)
    return PositionVocabulary(labels=labels, positions=(None,) + tuple(pos))


def region_vocabulary(domain: DomainType = DomainType.V) -> tuple[str, ...]:
    if domain is DomainType.V:
        return ("OUT",) + tuple(name for name, _, _ in V_REGIONS)
    raise NotImplementedError("region labels for C/G domains: use strand names")
