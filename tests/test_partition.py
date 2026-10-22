from protannotml.data.partition import (
    assign_groups_to_splits,
    audit_split,
    connected_components,
    filter_pretraining_corpus,
    greedy_set_cover_clusters,
    holdout_groups,
    max_identity_to_split,
    merge_near_identical_groups,
    novelty_tier,
    split_by_components,
)

# A chain a-b-c-d where consecutive pairs are similar but a and d are not.
IDS = ["a", "b", "c", "d", "e", "f"]
EDGES = [("a", "b", 0.95), ("b", "c", 0.93), ("c", "d", 0.92), ("e", "f", 0.97), ("a", "c", 0.88)]


def test_components_are_single_linkage():
    comps = connected_components(IDS, EDGES, 0.9)
    assert sorted(map(sorted, comps)) == [["a", "b", "c", "d"], ["e", "f"]]


def test_set_cover_can_leak_but_components_cannot():
    clusters = greedy_set_cover_clusters(IDS, EDGES, 0.9)
    # b is picked first (neighbours a, c); d ends up alone although d-c = 0.92.
    assert any(set(c) == {"d"} for c in clusters)
    assignment = {x: "train" for x in IDS}
    assignment["d"] = "test"
    assert audit_split(assignment, EDGES, 0.9).n_violations == 1
    comps = connected_components(IDS, EDGES, 0.9)
    comp_assignment = {x: ("test" if x in comps[1] else "train") for x in IDS}
    assert audit_split(comp_assignment, EDGES, 0.9).n_violations == 0


def test_bin_packing_respects_groups_and_ratios():
    groups = [[f"g{i}_{j}" for j in range(size)] for i, size in enumerate([5, 4, 3, 3, 2, 2, 1, 1, 1])]
    assignment = assign_groups_to_splits(groups, {"train": 0.7, "test": 0.3}, seed=1)
    for g in groups:
        assert len({assignment[x] for x in g}) == 1
    n_test = sum(v == "test" for v in assignment.values())
    assert 5 <= n_test <= 9  # 30% of 22 = 6.6


def test_bridge_removal_breaks_giant_component():
    ids = [f"s{i}" for i in range(10)]
    hub_edges = [("s0", f"s{i}", 0.95) for i in range(1, 10)]
    res = split_by_components(ids, hub_edges, 0.9, {"train": 0.8, "test": 0.2}, max_component_fraction=0.3)
    assert res.removed == ["s0"]
    assert audit_split(res.assignment, hub_edges, 0.9).n_violations == 0


def test_duplicate_genes_are_merged_before_holdout():
    group_of = {"x1": "IGHV3-23", "x2": "IGHV3-23D", "x3": "IGHV1-2"}
    merged = merge_near_identical_groups(group_of, [("IGHV3-23", "IGHV3-23D", 1.0)], 0.98)
    assert merged["x1"] == merged["x2"] == "IGHV3-23+IGHV3-23D"
    kept, out = holdout_groups(merged, ["IGHV3-23+IGHV3-23D"])
    assert sorted(out) == ["x1", "x2"] and kept == ["x3"]


def test_novelty_and_corpus_filter():
    best = max_identity_to_split(["d"], ["a", "b", "c"], EDGES)
    assert best["d"] == 0.92 and novelty_tier(best["d"]) == "90-95%"
    corpus = filter_pretraining_corpus(["a", "b", "c", "e"], ["d"], EDGES, 0.9)
    assert corpus == ["a", "b", "e"]
