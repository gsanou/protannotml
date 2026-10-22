"""Parse IMGT gene and allele names into a label hierarchy.

Covers the common human forms (IGHV3-23*01, IGHV3-23D*01, IGHV1/OR15-1*01,
TRAV29/DV5*01, IGHD1-1*01, IGHG1*01, IGKC*01). Other species and exotic forms
must be validated against IMGT/GENE-DB before use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LOCUS = re.compile(r"^(IG[HKL]|TR[ABGD])(.*)$")
_VDJ = re.compile(r"^([VDJ])(\d+)?(?:/(OR\d+|DV\d+))?(?:-(\d+))?(D)?$")
_IGH_CONST = re.compile(r"^([MDGAE])(\d*)$")
_CONST = re.compile(r"^C(\d*)$")


@dataclass(frozen=True)
class GeneName:
    locus: str
    gene_type: str  # V, D, J or C
    subgroup: str | None
    gene: str  # gene name without allele, e.g. IGHV3-23
    allele: str | None  # e.g. "01"
    orphon: bool = False
    duplicated: bool = False

    @property
    def subgroup_name(self) -> str | None:
        return f"{self.locus}{self.gene_type}{self.subgroup}" if self.subgroup else None

    def hierarchy(self) -> list[str]:
        """Coarse-to-fine labels: locus, gene type, subgroup, gene, allele."""
        levels = [self.locus, f"{self.locus}{self.gene_type}"]
        if self.subgroup_name:
            levels.append(self.subgroup_name)
        levels.append(self.gene)
        if self.allele:
            levels.append(f"{self.gene}*{self.allele}")
        return levels


def parse_gene_name(name: str) -> GeneName:
    raw = name.strip()
    allele = None
    if "*" in raw:
        raw, allele = raw.split("*", 1)
    m = _LOCUS.match(raw)
    if not m:
        raise ValueError(f"unknown locus prefix in {name!r}")
    locus, rest = m.groups()
    if (locus == "IGH" and _IGH_CONST.match(rest)) or _CONST.match(rest):
        # IGHD alone is the constant delta gene, not a diversity gene.
        return GeneName(locus, "C", None, raw, allele)
    g = _VDJ.match(rest)
    if not g:
        raise ValueError(f"unsupported gene name form {name!r}")
    gtype, first_num, extra, gene_num, dup = g.groups()
    subgroup = first_num if (gtype == "V" or gene_num) else None
    return GeneName(
        locus=locus,
        gene_type=gtype,
        subgroup=subgroup,
        gene=raw,
        allele=allele,
        orphon=bool(extra and extra.startswith("OR")),
        duplicated=bool(dup),
    )
