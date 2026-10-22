import pytest

from protannotml.imgt import (
    CDR1_IMGT,
    CDR2_IMGT,
    CDR3_IMGT,
    DomainType,
    FrameworkTemplate,
    ImgtPosition,
    number_loop,
    number_v_domain,
    parse_gene_name,
    region_of,
    v_domain_vocabulary,
)
from protannotml.imgt.numbering import check_anchors, gapped_string

TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSL"
    "RAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)


def labels(loop_numbering):
    return [str(p) for p in loop_numbering.positions]


def test_cdr3_gap_order_matches_imgt():
    # IMGT: gaps open in the order 111, 112, 110, 113, 109, 114 ...
    expected_gap_order = [111, 112, 110, 113, 109, 114]
    for missing in range(1, 7):
        gaps = set(number_loop(13 - missing, CDR3_IMGT).gaps)
        assert gaps == set(expected_gap_order[:missing])


def test_cdr1_cdr2_gap_order_matches_imgt_tables():
    # Inferred from IMGT length tables: CDR1 33, 32, 34, 31, 35, 30, 36; CDR2 61, 60, 62, 59 ...
    cdr1_order = [33, 32, 34, 31, 35, 30, 36]
    for missing in range(1, 8):
        assert set(number_loop(12 - missing, CDR1_IMGT).gaps) == set(cdr1_order[:missing])
    cdr2_order = [61, 60, 62, 59, 63, 58]
    for missing in range(1, 7):
        assert set(number_loop(10 - missing, CDR2_IMGT).gaps) == set(cdr2_order[:missing])


def test_cdr3_insertions_order():
    assert labels(number_loop(14, CDR3_IMGT))[6:9] == ["111", "112.1", "112"]
    assert labels(number_loop(15, CDR3_IMGT))[6:10] == ["111", "111.1", "112.1", "112"]
    assert labels(number_loop(16, CDR3_IMGT))[6:11] == ["111", "111.1", "112.2", "112.1", "112"]
    ln = number_loop(16, CDR3_IMGT)
    assert not ln.extrapolated
    cdr1 = number_loop(14, CDR1_IMGT)
    assert cdr1.extrapolated
    assert labels(cdr1)[5:9] == ["32", "32.1", "33.1", "33"]


def test_position_order_and_parse():
    seq = [ImgtPosition.parse(s) for s in ["112", "111", "112.1", "111.1", "112.2", "110"]]
    assert [str(p) for p in sorted(seq)] == ["110", "111", "111.1", "112.2", "112.1", "112"]
    c = DomainType.C
    cpos = [ImgtPosition.parse(s, c) for s in ["85", "84.1", "85.1", "85.7", "84", "1", "1.2", "1.1"]]
    assert [str(p) for p in sorted(cpos)] == ["1.2", "1.1", "1", "84", "84.1", "85.7", "85.1", "85"]
    assert str(ImgtPosition.parse("61A", DomainType.G)) == "61A"


def test_regions():
    assert region_of(ImgtPosition(112, 3)) == "CDR3"
    assert region_of(ImgtPosition(104)) == "FR3"
    assert region_of(ImgtPosition(45, 4, domain=DomainType.C)) == "CD"
    with pytest.raises(ValueError):
        region_of(ImgtPosition(60, domain=DomainType.C))
    assert region_of(ImgtPosition(61, 1, letter=True, domain=DomainType.G)) == "HELIX"


def test_trastuzumab_vh_numbering():
    regions = {
        "FR1": "EVQLVESGGGLVQPGGSLRLSCAAS", "CDR1": "GFNIKDTY", "FR2": "IHWVRQAPGKGLEWVAR",
        "CDR2": "IYPTNGYT", "FR3": "RYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC",
        "CDR3": "SRWGGDGFYAMDY", "FR4": "WGQGTLVTVSS",
    }
    assert "".join(regions[k] for k in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]) == TRASTUZUMAB_VH
    with pytest.raises(ValueError):
        number_v_domain(regions)  # FR3 has 38 residues: a template is required
    template = FrameworkTemplate(frozenset({10, 73}))
    numbered = number_v_domain(regions, template)
    assert len(numbered) == 120
    occupied = {r.position.number for r in numbered}
    assert set(range(1, 129)) - occupied == {10, 31, 32, 33, 34, 60, 61, 73}
    assert all(check_anchors(numbered).values())
    assert gapped_string(numbered)[22] == "C" and gapped_string(numbered)[103] == "C"


def test_template_from_gapped_germline():
    # Synthetic gapped V-REGION with dots at FR positions 10 and 73 and CDR gaps.
    gapped = ["X"] * 104
    for p in (10, 73, 31, 32, 33, 34):
        gapped[p - 1] = "."
    t = FrameworkTemplate.from_gapped_v_region("".join(gapped))
    assert t.missing == frozenset({10, 73})


def test_vocabulary_is_sequence_ordered():
    vocab = v_domain_vocabulary(max_cdr3_insertions_per_side=5)
    pos = vocab.positions[1:]
    assert all(a < b for a, b in zip(pos, pos[1:]))
    assert vocab.index("111.1") < vocab.index("112.5") < vocab.index("112.1") < vocab.index("112")
    assert vocab.labels[0] == "UNNUMBERED"


@pytest.mark.parametrize(
    "name, gene_type, subgroup, hierarchy_tail",
    [
        ("IGHV3-23*01", "V", "3", ["IGHV3", "IGHV3-23", "IGHV3-23*01"]),
        ("IGHV3-23D*01", "V", "3", ["IGHV3", "IGHV3-23D", "IGHV3-23D*01"]),
        ("IGHV1/OR15-1*01", "V", "1", ["IGHV1", "IGHV1/OR15-1", "IGHV1/OR15-1*01"]),
        ("TRAV29/DV5*01", "V", "29", ["TRAV29", "TRAV29/DV5", "TRAV29/DV5*01"]),
        ("IGHD1-1*01", "D", "1", ["IGHD1", "IGHD1-1", "IGHD1-1*01"]),
        ("IGHJ6*02", "J", None, ["IGHJ", "IGHJ6", "IGHJ6*02"]),
        ("IGHD*01", "C", None, ["IGHC", "IGHD", "IGHD*01"]),
        ("IGKC*01", "C", None, ["IGKC", "IGKC", "IGKC*01"]),
    ],
)
def test_gene_names(name, gene_type, subgroup, hierarchy_tail):
    g = parse_gene_name(name)
    assert g.gene_type == gene_type
    assert g.subgroup == subgroup
    assert g.hierarchy()[-len(hierarchy_tail):] == hierarchy_tail
