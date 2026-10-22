"""IMGT unique numbering primitives.

This module encodes the deterministic parts of the IMGT unique numbering that a
learned model must reproduce:

* positions with insertion codes (111.1, 112.1, 45.3, 61A, ...) and their
  *sequence order*, which differs from their numeric order for "right-side"
  insertion sites (112.2 comes before 112.1, which comes before 112);
* the gap / insertion rule for loops (CDR1-, CDR2-, CDR3-IMGT of V domains and
  the FG loop of C domains);
* region membership (FR/CDR for V domains, strands and turns for C and G domains);
* assembly of a numbered V domain from its regions and a germline framework
  template (framework gaps are gene-specific and must come from IMGT references).

References
----------
Lefranc et al. (2003) Dev Comp Immunol 27:55-77 (V domain).
Lefranc et al. (2005) Dev Comp Immunol 29:185-203 (C domain).
Lefranc et al. (2005) Dev Comp Immunol 29:917-938 (G domain).
IMGT Scientific chart (FR-IMGT / CDR-IMGT definitions), consulted 2026-10-20.

Caveat
------
IMGT documents the CDR3-IMGT / FG-loop insertion order (112.1, 111.1, 112.2,
111.2, ...). It does not, in the pages consulted, define insertion positions for
CDR1-IMGT > 12 or CDR2-IMGT > 10. For those cases this module follows the same
symmetric convention as ANARCI (insertions at the loop centre, right side first)
and flags them through ``LoopNumbering.extrapolated``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Iterable, Sequence


class DomainType(str, Enum):
    """IMGT domain families with distinct numbering grids."""

    V = "V"  # V-DOMAIN and V-LIKE-DOMAIN
    C = "C"  # C-DOMAIN and C-LIKE-DOMAIN
    G = "G"  # G-DOMAIN and G-LIKE-DOMAIN


# Base numbers whose insertions are placed *before* the base number in sequence
# order, in decreasing insertion index (e.g. ... 112.2, 112.1, 112).
_PREFIX_INSERTION_BASES: dict[DomainType, frozenset[int]] = {
    # 33 and 61 are the right-centre insertion sites of CDR1/CDR2 (extrapolated,
    # not documented by IMGT); 112 is documented for CDR3-IMGT.
    DomainType.V: frozenset({112, 61, 33, 1}),
    DomainType.C: frozenset({112, 85, 1}),
    DomainType.G: frozenset({1}),
}

# Large offset used to sort prefix insertions after the previous base's own
# suffix insertions. Assumes fewer than 10_000 insertions at a single site.
_PREFIX_OFFSET = 10_000


@total_ordering
@dataclass(frozen=True)
class ImgtPosition:
    """One IMGT position, e.g. ``111``, ``111.1``, ``112.3`` or ``61A``.

    Attributes
    ----------
    number:
        Base IMGT number.
    insertion:
        0 for a base position, k >= 1 for the k-th insertion at this base.
    letter:
        If True, display the insertion with a capital letter (G-domain style:
        61A, 61B); otherwise with a dotted index (V/C style: 111.1).
    domain:
        Domain family, needed to know the sequence order of insertions.
    """

    number: int
    insertion: int = 0
    letter: bool = False
    domain: DomainType = DomainType.V

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"IMGT numbers start at 1, got {self.number}")
        if self.insertion < 0:
            raise ValueError("insertion index must be >= 0")

    # ---- ordering -------------------------------------------------------
    def sort_key(self) -> tuple[int, int]:
        """Key reproducing the order of residues along the sequence."""
        if self.insertion == 0:
            return (self.number, 0)
        if self.number in _PREFIX_INSERTION_BASES[self.domain]:
            # e.g. 112.k sits between 111.x and 112, in decreasing k.
            return (self.number - 1, _PREFIX_OFFSET - self.insertion)
        return (self.number, self.insertion)

    def __lt__(self, other: "ImgtPosition") -> bool:
        if not isinstance(other, ImgtPosition):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    # ---- formatting -----------------------------------------------------
    def __str__(self) -> str:
        if self.insertion == 0:
            return str(self.number)
        if self.letter:
            if self.insertion > 26:
                raise ValueError("letter-style insertion beyond Z")
            return f"{self.number}{chr(64 + self.insertion)}"
        return f"{self.number}.{self.insertion}"

    @classmethod
    def parse(cls, text: str, domain: DomainType = DomainType.V) -> "ImgtPosition":
        """Parse ``"111"``, ``"111.2"`` or ``"61A"``."""
        text = text.strip()
        if "." in text:
            base, ins = text.split(".", 1)
            return cls(int(base), int(ins), letter=False, domain=domain)
        if text and text[-1].isalpha():
            return cls(int(text[:-1]), ord(text[-1].upper()) - 64, letter=True, domain=domain)
        return cls(int(text), 0, domain=domain)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

V_REGIONS: tuple[tuple[str, int, int], ...] = (
    ("FR1", 1, 26),
    ("CDR1", 27, 38),
    ("FR2", 39, 55),
    ("CDR2", 56, 65),
    ("FR3", 66, 104),
    ("CDR3", 105, 117),
    ("FR4", 118, 128),
)

# Strand / loop structure of the V domain (IMGT Scientific chart).
V_STRANDS: tuple[tuple[str, int, int], ...] = (
    ("A", 1, 15), ("B", 16, 26), ("BC", 27, 38), ("C", 39, 46), ("C'", 47, 55),
    ("C'C''", 56, 65), ("C''", 66, 74), ("D", 75, 84), ("E", 85, 96),
    ("F", 97, 104), ("FG", 105, 117), ("G", 118, 128),
)

# C domain (Lefranc et al. 2005). Positions 37-38 and 46-76 do not exist.
C_STRANDS: tuple[tuple[str, int, int], ...] = (
    ("A", 1, 15), ("B", 16, 26), ("BC", 27, 36), ("C", 39, 45), ("D", 77, 84),
    ("E", 85, 96), ("F", 97, 104), ("FG", 105, 117), ("G", 118, 128),
)
# Insertion sites of the C domain -> region they belong to.
C_INSERTION_SITES: dict[int, str] = {1: "A", 15: "AB", 45: "CD", 84: "DE", 85: "DE", 96: "EF", 111: "FG", 112: "FG"}

# G domain (IMGT Scientific chart, G superfamily).
G_STRANDS: tuple[tuple[str, int, int], ...] = (
    ("A", 1, 14), ("AB", 15, 17), ("B", 18, 28), ("BC", 29, 30), ("C", 31, 38),
    ("CD", 39, 41), ("D", 42, 49), ("HELIX", 50, 92),
)

V_ANCHORS: dict[int, tuple[str, frozenset[str]]] = {
    23: ("1st-CYS", frozenset("C")),
    41: ("CONSERVED-TRP", frozenset("W")),
    89: ("hydrophobic", frozenset("ACFILMVWY")),
    104: ("2nd-CYS", frozenset("C")),
    118: ("J-TRP/J-PHE", frozenset("WF")),
}


def region_of(position: ImgtPosition) -> str:
    """Return the IMGT region (V: FR/CDR, C/G: strand or turn) of a position."""
    n = position.number
    if position.domain is DomainType.V:
        # Insertions take the region of their base (111.x/112.x are CDR3).
        for name, lo, hi in V_REGIONS:
            if lo <= n <= hi:
                return name
        raise ValueError(f"position {position} outside the V-domain grid (1-128)")
    if position.domain is DomainType.C:
        if position.insertion and n in C_INSERTION_SITES:
            return C_INSERTION_SITES[n]
        for name, lo, hi in C_STRANDS:
            if lo <= n <= hi:
                return name
        raise ValueError(f"position {position} does not exist in a C domain")
    for name, lo, hi in G_STRANDS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"position {position} outside the G-domain grid (1-92)")


# ---------------------------------------------------------------------------
# Loop gap / insertion rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopSpec:
    """A loop whose occupancy follows the IMGT 'top of the loop' rule."""

    name: str
    start: int
    end: int
    insertions_documented: bool

    @property
    def max_length(self) -> int:
        return self.end - self.start + 1


CDR1_IMGT = LoopSpec("CDR1", 27, 38, insertions_documented=False)
CDR2_IMGT = LoopSpec("CDR2", 56, 65, insertions_documented=False)
CDR3_IMGT = LoopSpec("CDR3", 105, 117, insertions_documented=True)
FG_LOOP_C = LoopSpec("FG", 105, 117, insertions_documented=True)


@dataclass(frozen=True)
class LoopNumbering:
    positions: tuple[ImgtPosition, ...]
    gaps: tuple[int, ...]
    extrapolated: bool = False  # True when the insertion rule is not IMGT-documented


def number_loop(length: int, loop: LoopSpec, domain: DomainType = DomainType.V) -> LoopNumbering:
    """IMGT positions occupied by a loop of ``length`` residues, in sequence order.

    Shorter loops are filled alternately from both anchors (left first), so the
    gaps open at the top of the loop. Longer loops receive insertions at the two
    central positions, right-centre first: for CDR3-IMGT this yields
    112.1, 111.1, 112.2, 111.2, ...

    Example
    -------
    >>> [str(p) for p in number_loop(12, CDR3_IMGT).positions][5:7]
    ['110', '112']
    """
    if length < 0:
        raise ValueError("loop length must be >= 0")
    m = loop.max_length
    n_core = min(length, m)
    n_left = (n_core + 1) // 2
    n_right = n_core // 2
    left = list(range(loop.start, loop.start + n_left))
    right = list(range(loop.end - n_right + 1, loop.end + 1))
    occupied = set(left) | set(right)
    gaps = tuple(p for p in range(loop.start, loop.end + 1) if p not in occupied)

    left_pos = [ImgtPosition(p, domain=domain) for p in left]
    right_pos = [ImgtPosition(p, domain=domain) for p in right]
    extra = max(0, length - m)
    left_ins: list[ImgtPosition] = []
    right_ins: list[ImgtPosition] = []
    if extra:
        # Centre positions of a full loop: for an odd max (13) the left half is
        # one longer, so the centre pair is (111, 112) for CDR3-IMGT.
        centre_left = loop.start + (m + 1) // 2 - 1
        centre_right = centre_left + 1
        for k in range(1, extra + 1):
            if k % 2 == 1:  # right-centre first
                right_ins.append(ImgtPosition(centre_right, (k + 1) // 2, domain=domain))
            else:
                left_ins.append(ImgtPosition(centre_left, k // 2, domain=domain))
    # Sequence order: left anchors, left insertions (increasing), right
    # insertions (decreasing index), right anchors.
    ordered = left_pos + left_ins + sorted(right_ins) + right_pos
    ordered = sorted(ordered)
    assert len(ordered) == length
    return LoopNumbering(tuple(ordered), gaps, extrapolated=bool(extra) and not loop.insertions_documented)


# ---------------------------------------------------------------------------
# V-domain assembly
# ---------------------------------------------------------------------------

FULL_FR_POSITIONS: dict[str, tuple[int, ...]] = {
    name: tuple(range(lo, hi + 1)) for name, lo, hi in V_REGIONS if name.startswith("FR")
}


@dataclass(frozen=True)
class FrameworkTemplate:
    """Occupied FR positions of a germline gene (framework gaps are gene-specific).

    Build it from an IMGT-gapped V-REGION amino-acid sequence (dots = gaps),
    available from IMGT/GENE-DB, with :meth:`from_gapped_v_region`.
    """

    missing: frozenset[int] = field(default_factory=frozenset)

    def positions(self, region: str) -> tuple[int, ...]:
        return tuple(p for p in FULL_FR_POSITIONS[region] if p not in self.missing)

    @classmethod
    def from_gapped_v_region(cls, gapped: str) -> "FrameworkTemplate":
        """``gapped``: IMGT-gapped V-REGION (positions 1..104+), '.' marks a gap.

        Only framework gaps are kept; CDR occupancy is recomputed from lengths.
        """
        missing = set()
        for i, ch in enumerate(gapped, start=1):
            if i > 104:
                break
            if ch in ".-":
                region = region_of(ImgtPosition(i))
                if region.startswith("FR"):
                    missing.add(i)
        return cls(frozenset(missing))


@dataclass(frozen=True)
class NumberedResidue:
    position: ImgtPosition
    residue: str
    region: str


def number_v_domain(
    regions: dict[str, str],
    template: FrameworkTemplate | None = None,
) -> list[NumberedResidue]:
    """Number a V domain given its FR/CDR segmentation.

    Parameters
    ----------
    regions:
        Mapping with keys FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4 (FR4 may be
        shorter than 11 if the sequence is truncated at its C-terminus).
    template:
        Germline framework template. Required whenever a framework region is
        shorter than its full IMGT length.

    Raises
    ------
    ValueError
        If a framework length is incompatible with the template.
    """
    template = template or FrameworkTemplate()
    out: list[NumberedResidue] = []
    loops = {"CDR1": CDR1_IMGT, "CDR2": CDR2_IMGT, "CDR3": CDR3_IMGT}
    for name, _, _ in V_REGIONS:
        seq = regions.get(name, "")
        if name in loops:
            positions = number_loop(len(seq), loops[name]).positions
        else:
            available = template.positions(name)
            if name == "FR4":
                if len(seq) > len(available):
                    raise ValueError(f"FR4 longer than the {len(available)} template positions")
                positions = tuple(ImgtPosition(p) for p in available[: len(seq)])
            elif name == "FR1" and len(seq) < len(available):
                # N-terminal truncation: drop leading positions only.
                positions = tuple(ImgtPosition(p) for p in available[len(available) - len(seq):])
            else:
                if len(seq) != len(available):
                    raise ValueError(
                        f"{name} has {len(seq)} residues but the template has {len(available)} "
                        f"positions; framework insertions/deletions need explicit handling"
                    )
                positions = tuple(ImgtPosition(p) for p in available)
        out.extend(NumberedResidue(p, aa, name) for p, aa in zip(positions, seq))
    return out


def check_anchors(numbered: Sequence[NumberedResidue]) -> dict[int, bool]:
    """Return, for each V-domain anchor present, whether its residue is canonical."""
    by_pos = {r.position.number: r.residue for r in numbered if r.position.insertion == 0}
    return {p: by_pos[p] in allowed for p, (_, allowed) in V_ANCHORS.items() if p in by_pos}


def gapped_string(numbered: Iterable[NumberedResidue], start: int = 1, end: int = 128) -> str:
    """Render base positions start..end with '.' for gaps (insertions omitted)."""
    by_pos = {r.position.number: r.residue for r in numbered if r.position.insertion == 0}
    return "".join(by_pos.get(p, ".") for p in range(start, end + 1))
