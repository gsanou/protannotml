import math

import numpy as np
import pytest

from protannotml.decoding.monotonic import brute_force_decode, is_valid_numbering, monotonic_decode
from protannotml.training.weighting_ref import (
    kendall_optimal_s,
    kendall_term,
    liebel_optimal_s,
    liebel_term,
    marginal_nll,
    masked_task_mean,
    simulate_uncertainty_weighting,
)


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_kendall_optimum_is_minimum(kind):
    for L in (0.02, 0.15, 1.2):
        s_star = kendall_optimal_s(L, kind)
        grid = np.linspace(s_star - 3, s_star + 3, 2001)
        vals = [kendall_term(L, s, kind) for s in grid]
        assert abs(grid[int(np.argmin(vals))] - s_star) < 5e-3
        # weighted contribution at the optimum is 1/2 for both forms
        w = math.exp(-s_star) if kind == "classification" else 0.5 * math.exp(-s_star)
        assert w * L == pytest.approx(0.5)


def test_liebel_optimum_and_bounded_weight():
    for L in (0.001, 0.02, 1.2):
        s_star = liebel_optimal_s(L)
        grid = np.linspace(s_star - 3, s_star + 3, 4001)
        vals = [liebel_term(L, s) for s in grid]
        assert abs(grid[int(np.argmin(vals))] - s_star) < 5e-3
    # nearly solved task: Kendall weight explodes, Liebel weight stays moderate
    L = 0.001
    w_kendall = math.exp(-kendall_optimal_s(L))
    w_liebel = 0.5 * math.exp(-liebel_optimal_s(L))
    assert w_kendall == pytest.approx(500.0)
    assert w_liebel < 25


def test_gradient_descent_tracks_closed_form():
    steps = 4000
    curves = np.tile(np.array([0.05, 0.4, 1.5]), (steps, 1))
    traj = simulate_uncertainty_weighting(curves, lr=0.05)
    expected = [math.log(2 * L) for L in (0.05, 0.4, 1.5)]
    assert traj[-1] == pytest.approx(expected, rel=1e-3, abs=1e-3)
    clamped = simulate_uncertainty_weighting(curves, lr=0.05, s_min=-1.0)
    assert clamped[-1][0] == pytest.approx(-1.0)


def test_masked_mean_and_marginal():
    assert masked_task_mean([1.0, 3.0, 5.0], [True, False, True]) == pytest.approx(3.0)
    assert masked_task_mean([1.0], [False]) == 0.0
    p = np.array([0.45, 0.40, 0.15])
    assert marginal_nll(p, [0, 1]) == pytest.approx(-math.log(0.85))


def test_monotonic_decoder_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(40):
        L, K = rng.integers(2, 6), rng.integers(2, 6)
        logits = rng.normal(size=(L, K)) * 2
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        lab, score = monotonic_decode(logp)
        lab_bf, score_bf = brute_force_decode(logp)
        assert is_valid_numbering(lab)
        assert score == pytest.approx(score_bf)
        assert logp[np.arange(L), lab].sum() == pytest.approx(score)


def test_decoder_repairs_out_of_order_argmax():
    # labels: 0=UNNUMBERED, 1..5 = positions in sequence order
    logp = np.log(np.array([
        [0.01, 0.90, 0.03, 0.02, 0.02, 0.02],
        [0.01, 0.02, 0.50, 0.45, 0.01, 0.01],
        [0.01, 0.02, 0.60, 0.30, 0.05, 0.02],   # argmax repeats label 2
        [0.01, 0.01, 0.02, 0.03, 0.03, 0.90],
    ]))
    argmax = logp.argmax(1)
    assert not is_valid_numbering(argmax)
    lab, _ = monotonic_decode(logp)
    assert list(lab) == [1, 2, 3, 5]


def test_allowed_mask():
    logp = np.log(np.full((3, 4), 0.25))
    allowed = np.ones((3, 4), dtype=bool)
    allowed[:, 2] = False  # e.g. a position that does not exist in this domain type
    lab, _ = monotonic_decode(logp, allowed)
    assert 2 not in lab
