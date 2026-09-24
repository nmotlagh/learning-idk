"""Fast NumPy B-CDF thresholds, with optional grouping of classes.

Same decisions as ``learning_idk.threshold.learn_class_threshold`` (checked in
test_fast_bcdf.py), but sorted + prefix sums instead of an O(n^2) scan, and exact
integer comparisons for ties. ``learn_grouped_thresholds`` shares one threshold
across a group of predicted classes: one group per class is the paper's per-class
method, a single group is one global threshold.
"""

from __future__ import annotations

import math

import numpy as np

# Log-space sums land ~1e-12 off exact boundaries (e.g. CDF = 0.5 when delta = 0.5
# and n is odd); SciPy returns those exactly, so compare with a small tolerance.
CDF_EPS = 1e-9


def log_factorials(n: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])


def binom_half_cdf(k: int, n: int, log_fact: np.ndarray) -> float:
    """P[X <= k] for X ~ Binomial(n, 0.5), summed in log space."""
    if k >= n:
        return 1.0
    i = np.arange(k + 1)
    log_terms = log_fact[n] - log_fact[i] - log_fact[n - i] - n * math.log(2.0)
    m = log_terms.max()
    return float(math.exp(m) * np.exp(log_terms - m).sum())


def learn_threshold(scores: np.ndarray, correct: np.ndarray, delta: float,
                    log_fact: np.ndarray, floor: float = 0.0) -> float:
    """Highest-select-accuracy threshold whose reject region (score <= t) is not
    significantly better than chance: BinomCDF(k_correct; n_rejected, 0.5) <= 1 - delta.

    Returns ``floor`` when nothing should be rejected (0.0 for softmax scores; pass
    -inf for scores that can be negative, such as log-odds).
    """
    n = scores.size
    if n == 0 or correct.all():
        return floor
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    cum_correct = np.cumsum(correct[order].astype(np.int64))
    total_correct = int(cum_correct[-1])

    candidates = np.unique(scores[~correct])  # ascending
    rejected = np.searchsorted(s, candidates, side="right")  # count with score <= t

    # Best so far as an exact fraction; starts at the base accuracy with coverage -1.
    best_t, best_k, best_n, best_cov = floor, total_correct, n, -1
    for t, rej in zip(candidates, rejected):
        k_rej = int(cum_correct[rej - 1])
        if binom_half_cdf(k_rej, int(rej), log_fact) > 1.0 - delta + CDF_EPS:
            continue
        sel = n - int(rej)
        k_sel = total_correct - k_rej
        if sel == 0:
            # The reference scores an empty select region as 1.1 (beats anything).
            better, tie = best_n != 0, False
        elif best_n == 0:  # current best is that 1.1 sentinel
            better, tie = False, False
        else:
            lhs, rhs = k_sel * best_n, best_k * sel
            better, tie = lhs > rhs, lhs == rhs
        if better or (tie and sel > best_cov):
            best_t, best_k, best_n, best_cov = float(t), k_sel, sel, sel
    return best_t


def learn_grouped_thresholds(max_sm: np.ndarray, preds: np.ndarray, correct: np.ndarray,
                             groups: np.ndarray, delta: float) -> np.ndarray:
    """One B-CDF threshold per group of predicted classes (groups[c] = group of class c)."""
    log_fact = log_factorials(max_sm.size)
    out = np.zeros(groups.size)
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        m = np.isin(preds, members)
        out[members] = learn_threshold(max_sm[m], correct[m], delta, log_fact)
    return out


def learn_thresholds(max_sm: np.ndarray, preds: np.ndarray, correct: np.ndarray,
                     num_classes: int, delta: float) -> np.ndarray:
    """Per-class thresholds, as in the paper."""
    return learn_grouped_thresholds(max_sm, preds, correct, np.arange(num_classes), delta)


def evaluate(max_sm: np.ndarray, preds: np.ndarray, correct: np.ndarray,
             thresholds: np.ndarray) -> dict[str, float]:
    select = max_sm > thresholds[preds]
    return {
        "coverage": float(select.mean()),
        "select_acc": float(correct[select].mean()) if select.any() else float("nan"),
        "reject_acc": float(correct[~select].mean()) if (~select).any() else float("nan"),
    }


def global_at_coverage(max_sm: np.ndarray, coverage: float, num_classes: int) -> np.ndarray:
    """A single threshold that selects the given fraction of these scores."""
    return np.full(num_classes, np.quantile(max_sm, 1.0 - coverage))
