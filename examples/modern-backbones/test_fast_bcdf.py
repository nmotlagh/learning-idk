"""fast_bcdf must pick the same thresholds as the package's reference implementation.

delta = 0.5 is left out on purpose: with an odd-sized reject region the binomial CDF is
exactly 0.5, and SciPy's rounding decides the reference's answer either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from learning_idk.data import load_data
from learning_idk.threshold import learn_thresholds as reference

import fast_bcdf

SYNTH = Path(__file__).resolve().parents[2] / "synth_logits"
DELTAS = (0.05, 0.25, 0.49, 0.51, 0.75, 0.95)


def _compare(logits: torch.Tensor, targets: torch.Tensor, delta: float) -> None:
    ref = reference(logits, targets, delta=delta, thresh_func="b_cdf").numpy()
    mx, pr = torch.softmax(logits, 1).max(1)
    ours = fast_bcdf.learn_thresholds(mx.numpy(), pr.numpy(), (pr == targets).numpy(),
                                      logits.shape[1], delta)
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-7)


@pytest.mark.parametrize("version", range(1, 9))
@pytest.mark.parametrize("split", ["val", "test"])
def test_matches_reference_on_synthetic_logits(version: int, split: str) -> None:
    logits, targets = load_data(SYNTH / f"{split}_logits_v{version}.pt", synth=True)
    for delta in DELTAS:
        _compare(logits, targets, delta)


@pytest.mark.parametrize("trial", range(20))
def test_matches_reference_on_random_logits(trial: int) -> None:
    # Many classes, quantized scores (lots of ties), some near-empty classes.
    g = torch.Generator().manual_seed(trial)
    c = int(torch.randint(3, 60, (1,), generator=g))
    n = int(torch.randint(50, 3000, (1,), generator=g))
    targets = torch.randint(0, c, (n,), generator=g)
    logits = torch.randn(n, c, generator=g) * float(torch.rand(1, generator=g) * 3)
    logits[torch.arange(n), targets] += float(torch.rand(1, generator=g) * 4)
    if trial % 2:
        logits = (logits * 4).round() / 4
    for delta in (0.05, 0.3, 0.7, 0.9):
        _compare(logits, targets, delta)


def test_global_threshold_accepts_negative_scores() -> None:
    scores = np.array([-3.0, -2.0, -1.0, 0.5, 2.0, 3.0])
    correct = np.array([False, False, True, True, True, True])
    lf = fast_bcdf.log_factorials(scores.size)
    assert fast_bcdf.learn_threshold(scores, correct, 0.05, lf, floor=-np.inf) == -2.0
    assert fast_bcdf.learn_threshold(scores[2:], correct[2:], 0.05, lf, floor=-np.inf) == -np.inf
