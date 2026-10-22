import numpy as np
import pytest

from protannotml.eval.metrics import (
    boundary_f1,
    coverage_accuracy_curve,
    expected_calibration_error,
    per_label_accuracy,
    position_accuracy,
    sequence_exact_match,
    set_accuracy,
    span_exact_f1,
    spans,
    top_k_set_accuracy,
)
from protannotml.eval.stats import (
    cluster_bootstrap_ci,
    holm_bonferroni,
    mcnemar_exact,
    paired_cluster_bootstrap_diff,
)


def test_position_metrics():
    true = [["1", "2", "3", "111", "112.1", "112"], ["1", "2"]]
    pred = [["1", "2", "3", "111", "112", "112.1"], ["1", "2"]]
    assert position_accuracy(pred, true) == pytest.approx(6 / 8)
    assert list(sequence_exact_match(pred, true)) == [False, True]
    acc = per_label_accuracy(pred, true)
    assert acc["112.1"] == 0.0 and acc["1"] == 1.0


def test_boundaries_with_tolerance():
    true = [["FR1"] * 4 + ["CDR1"] * 3 + ["FR2"] * 3]
    pred = [["FR1"] * 5 + ["CDR1"] * 2 + ["FR2"] * 3]  # CDR1 start shifted by +1
    assert spans(true[0]) == [("FR1", 0, 4), ("CDR1", 4, 7), ("FR2", 7, 10)]
    strict = boundary_f1(pred, true, tolerance=0)
    assert (strict.tp, strict.fp, strict.fn) == (4, 2, 2)
    loose = boundary_f1(pred, true, tolerance=1)
    assert loose.f1 == pytest.approx(1.0)
    assert span_exact_f1(pred, true).tp == 1  # only FR2 is exact


def test_assignment_metrics():
    assert set_accuracy(["a1", "b", "c"], [{"a1", "a2"}, {"x"}, {"c"}]) == pytest.approx(2 / 3)
    assert top_k_set_accuracy([["x", "a2"], ["y", "z"]], [{"a1", "a2"}, {"q"}], k=2) == 0.5


def test_calibration():
    conf = [0.9] * 10
    correct = [True] * 9 + [False]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-12)
    correct = [True] * 5 + [False] * 5
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4)
    cov, acc = coverage_accuracy_curve([0.9, 0.2, 0.8], [True, False, True])
    assert list(acc) == pytest.approx([1.0, 1.0, 2 / 3])


def test_cluster_bootstrap_is_wider_for_correlated_data():
    rng = np.random.default_rng(0)
    groups, values = [], []
    for g in range(30):
        p = rng.uniform(0.6, 1.0)  # cluster-level accuracy varies
        n = int(rng.integers(5, 40))
        values += list(rng.random(n) < p)
        groups += [g] * n
    iid = cluster_bootstrap_ci(values, None, n_boot=2000, seed=1)
    clu = cluster_bootstrap_ci(values, groups, n_boot=2000, seed=1)
    assert (clu.high - clu.low) > 1.3 * (iid.high - iid.low)
    assert clu.low <= clu.estimate <= clu.high


def test_paired_tests():
    a = [1] * 40 + [0] * 10
    b = [1] * 30 + [0] * 20
    b_, c_, p = mcnemar_exact(np.array(a, bool), np.array(b, bool))
    assert (b_, c_) == (10, 0)
    assert p == pytest.approx(2 * 0.5 ** 10)
    ci, pb = paired_cluster_bootstrap_diff(a, b, groups=list(range(50)), n_boot=2000)
    assert ci.estimate == pytest.approx(0.2) and ci.low > 0 and pb < 0.05


def test_holm():
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert list(adj) == pytest.approx([0.03, 0.06, 0.06])
