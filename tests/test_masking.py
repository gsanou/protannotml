import math

import numpy as np
import pytest

from protannotml.data.masking import (
    IGNORE_INDEX,
    MaskingConfig,
    mask_sequence,
    mlm_loss_from_logits,
    region_fixed_rate_probs,
    region_weighted_probs,
    span_selection,
    temperature_sampling_weights,
)

# Trastuzumab VH region layout: FR1 25, CDR1 8, FR2 17, CDR2 8, FR3 38, CDR3 13, FR4 11
REGIONS = ["FR1"] * 25 + ["CDR1"] * 8 + ["FR2"] * 17 + ["CDR2"] * 8 + ["FR3"] * 38 + ["CDR3"] * 13 + ["FR4"] * 11


def test_preferential_rates_keep_overall_mean():
    probs = region_fixed_rate_probs(REGIONS, {"CDR3": 0.25}, overall_rate=0.15)
    assert len(probs) == 120
    assert probs.mean() == pytest.approx(0.15)
    rest = (0.15 * 120 - 13 * 0.25) / 107
    assert probs[0] == pytest.approx(rest)  # ~0.1379
    with pytest.raises(ValueError):
        region_fixed_rate_probs(REGIONS, {"CDR3": 0.25, "FR3": 0.9}, overall_rate=0.15)


def test_weighted_probs_mean():
    probs = region_weighted_probs(REGIONS, {"CDR1": 2, "CDR2": 2, "CDR3": 2}, 0.15)
    assert probs.mean() == pytest.approx(0.15)
    assert probs[30] == pytest.approx(2 * probs[0])


def test_mask_statistics_and_specials():
    rng = np.random.default_rng(0)
    n = 120
    ids = np.concatenate([[0], rng.integers(4, 24, n), [2]])  # <cls> ... <eos>
    special = np.zeros(n + 2, dtype=bool)
    special[[0, -1]] = True
    probs = region_fixed_rate_probs(REGIONS, {"CDR3": 0.25}, 0.15)
    sel_counts = np.zeros(n + 2)
    mask_hits = 0
    total_sel = 0
    for _ in range(4000):
        inp, lab = mask_sequence(ids, special, probs, mask_token_id=32, replacement_ids=range(4, 24),
                                 config=MaskingConfig(), rng=rng)
        sel = lab != IGNORE_INDEX
        assert not sel[0] and not sel[-1]
        assert (lab[sel] == ids[sel]).all()
        sel_counts += sel
        mask_hits += (inp[sel] == 32).sum()
        total_sel += sel.sum()
    freq = sel_counts[1:-1] / 4000
    cdr3 = slice(25 + 8 + 17 + 8 + 38, 25 + 8 + 17 + 8 + 38 + 13)
    assert abs(freq[cdr3].mean() - 0.25) < 0.02
    assert abs(freq.mean() - 0.15) < 0.01
    assert abs(mask_hits / total_sel - 0.8) < 0.02


def test_span_selection_hits_target():
    rng = np.random.default_rng(1)
    sel = span_selection(200, 0.15, mean_span=3.0, rng=rng)
    assert sel.sum() == 30


def test_temperature_sampling():
    w = temperature_sampling_weights({"human": 900_000, "mouse": 90_000, "llama": 1_000}, alpha=0.5)
    assert w["llama"] > 1_000 / 991_000  # rare source up-weighted
    assert sum(w.values()) == pytest.approx(1.0)
    w1 = temperature_sampling_weights({"a": 3, "b": 1}, alpha=1.0)
    assert w1["a"] == pytest.approx(0.75)


def test_mlm_loss_reference():
    logits = np.zeros((3, 20))
    labels = np.array([5, IGNORE_INDEX, 7])
    assert mlm_loss_from_logits(logits, labels) == pytest.approx(math.log(20))
