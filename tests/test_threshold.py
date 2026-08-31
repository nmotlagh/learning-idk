"""Tests for threshold learning and reject checks."""

from __future__ import annotations

import math

import pytest
import torch

from learning_idk.threshold import (
    SUPPORTED_THRESH_FUNCS,
    check_reject,
    evaluate,
    learn_thresholds,
    wilson_cc_bound,
)


def test_wilson_cc_bound_rejects_zero_trials() -> None:
    with pytest.raises(ValueError, match="n must be > 0"):
        wilson_cc_bound(k=0, n=0, delta=0.05)


def test_wilson_cc_bound_increases_with_successes() -> None:
    low = wilson_cc_bound(k=3, n=20, delta=0.05)
    high = wilson_cc_bound(k=12, n=20, delta=0.05)
    assert low < high


def test_check_reject_returns_false_for_empty_region() -> None:
    preds = torch.tensor([], dtype=torch.int64)
    targets = torch.tensor([], dtype=torch.int64)
    assert check_reject(preds, targets, delta=0.05, thresh_func="b_cdf") is False


@pytest.mark.parametrize("thresh_func", sorted(SUPPORTED_THRESH_FUNCS))
def test_check_reject_returns_bool_for_all_supported_methods(thresh_func: str) -> None:
    preds = torch.tensor([0, 1, 2, 2], dtype=torch.int64)
    targets = torch.tensor([0, 2, 2, 1], dtype=torch.int64)
    out = check_reject(preds, targets, delta=0.05, thresh_func=thresh_func)
    assert isinstance(out, bool)


def test_learn_thresholds_uses_logits_dimension_for_class_count() -> None:
    # Targets do not include all classes, but threshold output must.
    logits = torch.tensor(
        [
            [4.0, 0.1, 0.0, 0.0],
            [3.0, 0.3, 0.0, 0.0],
            [0.2, 3.5, 0.0, 0.0],
            [0.1, 4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    thresholds = learn_thresholds(logits, targets, delta=0.05, thresh_func="b_cdf")
    assert thresholds.shape == (4,)
    assert all(math.isfinite(v) for v in thresholds.tolist())


def test_learn_thresholds_rejects_unsupported_function() -> None:
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="Unsupported thresh_func"):
        learn_thresholds(logits, targets, delta=0.05, thresh_func="not_real")


def test_evaluate_raises_for_short_threshold_vector() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.int64)
    thresholds = torch.tensor([0.5, 0.5], dtype=torch.float32)
    with pytest.raises(ValueError, match="fewer entries than model classes"):
        evaluate(logits, targets, thresholds)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_learn_thresholds_keeps_coverage_when_accuracy_ties(dtype, capsys) -> None:
    # Both confidence groups are 75% accurate; rejecting one cannot improve accuracy.
    logits = torch.tensor([[0.6, 0.4]] * 4 + [[0.9, 0.1]] * 4, dtype=dtype).log()
    targets = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1])

    thresholds = learn_thresholds(logits, targets)

    torch.testing.assert_close(thresholds, torch.zeros(2, dtype=dtype))
    evaluate(logits, targets, thresholds)
    output = capsys.readouterr().out
    assert "Select Accuracy: 75.0" in output
    assert "Coverage: 100.0" in output
