"""Core thresholding algorithm for Learning When to Say "I Don't Know".

Paper: https://arxiv.org/abs/2209.04944
Authors: Nicholas Kashani Motlagh*, Jim Davis*,
         Tim Anderson+, and Jeremy Gwinnup+
Affiliation: *Department of Computer Science & Engineering, Ohio State University
             +Air Force Research Laboratory, Wright-Patterson AFB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from .calibration import ModelWithTemperature
from .data import get_logits_loader, load_data, load_decisions

# Defaults
DEFAULT_THRESHOLD_PATH = "thresholds.pt"
N_BINS = 15
THRESH_FUNC = "b_cdf"
DELTA = 0.05
SUPPORTED_THRESH_FUNCS = {
    "b_cdf",
    "wilson",
    "wilson_cc",
    "clopper_pearson",
    "agresti_coull",
}


def _parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="learning-idk threshold learner")
    parser.add_argument(
        "--threshold_path",
        type=str,
        default=DEFAULT_THRESHOLD_PATH,
        help="Path to save computed thresholds",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to validation data (logits,targets)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Path to test data (logits,targets)",
    )
    parser.add_argument(
        "--synth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean flag indicating data is synthetic",
    )
    parser.add_argument(
        "--skip_ts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean flag indicating whether to skip temperature scaling.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DELTA,
        help="User-provided significance level",
    )
    parser.add_argument(
        "--thresh_func",
        type=str,
        default=THRESH_FUNC,
        choices=sorted(SUPPORTED_THRESH_FUNCS),
        help="Method to compute thresholds (b_cdf, wilson, wilson_cc, clopper_pearson, agresti_coull)",
    )
    return parser.parse_args()


def _validate_inputs(delta: float, thresh_func: str) -> None:
    """Validate user-facing algorithm parameters."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    if thresh_func not in SUPPORTED_THRESH_FUNCS:
        raise ValueError(
            f"Unsupported thresh_func '{thresh_func}'. "
            f"Expected one of: {sorted(SUPPORTED_THRESH_FUNCS)}."
        )


def learn_temp(dataloader: torch.utils.data.DataLoader) -> ModelWithTemperature:
    """Learns per-class temperatures on logits.

    Args:
        dataloader: DataLoader for logits.

    Returns:
        A ModelWithTemperature that temperature scales inputs (per-class).
    """
    model_ts = ModelWithTemperature(n_bins=N_BINS, strategy="grid", per_class=True)
    model_ts.set_temperature(dataloader, t_vals=list(torch.linspace(0.25, 4.0, 100)))
    model_ts.eval()
    return model_ts


def get_ts_data(
    model_ts: ModelWithTemperature, dataloader: torch.utils.data.DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts temperature scaled logits from a dataloader.

    Args:
        model_ts: A temperature scaled model.
        dataloader: A dataloader of logits.

    Returns:
        Tuple of (ts_logits, targets).
    """
    all_ts_logits = []
    all_targets = []
    for inputs, targets in dataloader:
        ts_logits = model_ts(inputs)
        all_ts_logits.append(ts_logits)
        all_targets.append(targets)
    all_ts_logits = torch.cat(all_ts_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_ts_logits, all_targets


def wilson_cc_bound(k: int, n: int, delta: float = DELTA) -> float:
    """Generates a (1-delta) upper bound using the Wilson interval with continuity correction.

    This strategy is approximately equivalent to the Binomial CDF with delta area in the tail.

    Args:
        k: Number of successes.
        n: Number of trials.
        delta: User defined significance level.

    Returns:
        The upper bound.
    """
    if n <= 0:
        raise ValueError("n must be > 0 for Wilson bound computation.")
    p = k / n
    q = 1.0 - p
    z = stats.norm.isf(delta)
    z2 = z**2
    denom = 2 * (n + z2)
    num = 2.0 * n * p + z2 + 1.0 + z * np.sqrt(z2 + 2 - 1.0 / n + 4 * p * (n * q - 1))
    bound = num / denom
    if p == 0:
        bound = 0.0
    elif p == 1:
        bound = 1.0
    return bound


def learn_thresholds(
    logits: torch.Tensor,
    targets: torch.Tensor,
    delta: float = DELTA,
    thresh_func: str = THRESH_FUNC,
) -> torch.Tensor:
    """Learns per-class thresholds on logits using the proposed approach.

    The method validates the reject region using thresh_func at a user-provided
    significance level.

    Args:
        logits: Tensor of logits.
        targets: Tensor of targets.
        delta: User defined significance level.
        thresh_func: Implementation used to validate reject region. Options
            are (b_cdf, wilson, wilson_cc, clopper_pearson, agresti_coull).

    Returns:
        A tensor of per-class thresholds.
    """
    _validate_inputs(delta, thresh_func)
    num_classes = logits.shape[1]
    thresholds = torch.zeros(num_classes, dtype=logits.dtype, device=logits.device)
    sm_scores = torch.softmax(logits, dim=1)
    max_sms, preds = torch.max(sm_scores, dim=1)
    for c in range(num_classes):
        class_idx = torch.where(preds == c)[0]
        thresholds[c] = learn_class_threshold(
            preds[class_idx], max_sms[class_idx], targets[class_idx], delta, thresh_func
        )
    return thresholds


def accuracy(is_correct: torch.Tensor) -> float:
    """Computes accuracy from a binary tensor.

    Args:
        is_correct: Binary tensor of successes and failures.

    Returns:
        Accuracy of the trials.
    """
    if is_correct.numel() == 0:
        return float("nan")
    return float(is_correct.float().mean().item())


def check_reject(
    preds: torch.Tensor,
    targets: torch.Tensor,
    delta: float,
    thresh_func: str,
) -> bool:
    """Validate whether the reject region is viable.

    Args:
        preds: Tensor of predictions.
        targets: Tensor of targets.
        delta: User defined significance level.
        thresh_func: Implementation used to validate reject region.

    Returns:
        Whether the reject region is viable at the given significance level.
    """
    if preds.numel() == 0:
        return False
    is_correct = preds == targets
    k = int(torch.sum(is_correct).item())
    n = is_correct.numel()
    if thresh_func == "b_cdf":
        return bool(stats.binom.cdf(k, n, 0.5) <= 1 - delta)
    elif thresh_func == "wilson_cc":
        return bool(wilson_cc_bound(k, n, delta=delta) <= 0.5)
    else:
        if thresh_func == "clopper_pearson":
            thresh_func = "beta"
        # Convert a one-sided upper bound at level (1-delta) into two-sided alpha.
        _, ci_u = proportion_confint(k, n, alpha=2 * delta, method=thresh_func)
        return bool(ci_u <= 0.5)


def learn_class_threshold(
    preds: torch.Tensor,
    max_sms: torch.Tensor,
    targets: torch.Tensor,
    delta: float,
    thresh_func: str,
) -> float:
    """Learns a threshold for a single class.

    Args:
        preds: Tensor of predictions.
        max_sms: Tensor of softmax scores corresponding to predictions.
        targets: Tensor of targets.
        delta: User defined significance level.
        thresh_func: Implementation used to validate reject region.

    Returns:
        Threshold that optimizes select accuracy while adhering to constraint.
    """
    if preds.numel() == 0:
        return 0.0

    # Only need to check thresholds that optimize select accuracy
    incorrect_idx = torch.where(preds != targets)[0]
    if incorrect_idx.numel() == 0:
        return 0.0
    possible_thresholds = torch.unique(max_sms[incorrect_idx])
    best_thresh = torch.tensor(0.0, dtype=max_sms.dtype, device=max_sms.device)
    best_cov = -1.0
    best_sacc = accuracy(preds == targets)
    for thresh in possible_thresholds:
        select_idx = torch.where(max_sms > thresh)[0]
        reject_idx = torch.where(max_sms <= thresh)[0]
        if select_idx.numel() > 0:
            sacc = accuracy(preds[select_idx] == targets[select_idx])
        else:
            sacc = 1.1
        cov = float(select_idx.numel() / (select_idx.numel() + reject_idx.numel()))
        if check_reject(preds[reject_idx], targets[reject_idx], delta, thresh_func):
            if sacc > best_sacc or (sacc == best_sacc and cov > best_cov):
                best_thresh = thresh
                best_sacc = sacc
                best_cov = cov

    return float(best_thresh.item())


def sanity_check(logits: torch.Tensor, targets: torch.Tensor) -> None:
    """Prints the accuracy of logits against targets as a sanity check.

    Args:
        logits: Tensor of logits.
        targets: Tensor of targets.
    """
    print(f"Base accuracy: {accuracy(torch.argmax(logits, dim=1) == targets)}")


def evaluate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
    decisions: torch.Tensor | None = None,
) -> None:
    """Computes select accuracy, reject accuracy, and coverage.

    If decisions are provided (for synthetic equal-density data), IDA is also computed.

    Args:
        logits: Tensor of logits.
        targets: Tensor of targets.
        thresholds: Tensor of per-class thresholds.
        decisions: Tensor of ideal decisions (for synthetic data).
    """
    if thresholds.numel() < logits.shape[1]:
        raise ValueError(
            "thresholds tensor has fewer entries than model classes. "
            f"Expected at least {logits.shape[1]}, got {thresholds.numel()}."
        )

    sm_scores = torch.softmax(logits, dim=1)
    max_sms, preds = torch.max(sm_scores, dim=1)
    class_thresholds = thresholds.to(logits.device)[preds]
    select_idx = torch.where(max_sms > class_thresholds)[0]
    reject_idx = torch.where(max_sms <= class_thresholds)[0]
    if select_idx.numel() > 0:
        sacc = accuracy(preds[select_idx] == targets[select_idx])
    else:
        sacc = float("nan")
    if reject_idx.numel() > 0:
        racc = accuracy(preds[reject_idx] == targets[reject_idx])
    else:
        racc = float("nan")
    cov = float(select_idx.numel() / (select_idx.numel() + reject_idx.numel()))
    print(f"Select Accuracy: {sacc * 100 :.1f}")
    print(f"Reject Accuracy: {racc * 100 :.1f}")
    print(f"Coverage: {cov * 100 :.1f}")
    if decisions is not None:
        selected = max_sms > class_thresholds
        ida = accuracy(selected == decisions)
        print(f"IDA: {ida * 100 :.1f}")


def main(
    data_path: str | Path,
    threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    synth: bool = False,
    delta: float = DELTA,
    skip_ts: bool = False,
    thresh_func: str = THRESH_FUNC,
    test_data_path: str | Path | None = None,
) -> None:
    """Run the full thresholding pipeline.

    Args:
        data_path: Path to validation data.
        threshold_path: Path to save computed thresholds.
        synth: Flag indicating synthetic data.
        delta: Significance level.
        skip_ts: Whether to skip temperature scaling.
        thresh_func: Method to compute thresholds.
        test_data_path: Path to test data (optional).
    """
    if data_path is None:
        raise ValueError("data_path is required.")
    _validate_inputs(delta, thresh_func)

    # Load data to learn thresholds
    print("Loading Data")
    logits, targets = load_data(data_path, synth=synth)
    decisions = None

    # Load data to evaluate thresholds
    test_logits, test_targets, test_decisions = None, None, None
    if test_data_path:
        test_logits, test_targets = load_data(test_data_path, synth=synth)

    # Get decisions if synthetic
    if synth:
        decisions = load_decisions(data_path)
        if test_data_path:
            test_decisions = load_decisions(test_data_path)

    # Print accuracy
    sanity_check(logits, targets)

    # Temperature scale data
    model_ts = None
    if not skip_ts:
        print("Temperature Scaling")
        data_loader = get_logits_loader(logits, targets)
        model_ts = learn_temp(data_loader)
        logits, targets = get_ts_data(model_ts, data_loader)

        # Temp scale test data
        if test_data_path:
            test_data_loader = get_logits_loader(test_logits, test_targets)
            test_logits, test_targets = get_ts_data(model_ts, test_data_loader)

    # Learn thresholds
    print("Learning Thresholds")
    thresholds = learn_thresholds(logits, targets, delta=delta, thresh_func=thresh_func)
    torch.save(thresholds, threshold_path)

    # Evaluate thresholds on validation data
    print(f"Evaluating {data_path}")
    evaluate(logits, targets, thresholds, decisions=decisions)

    # Evaluate on test data
    if test_data_path:
        print(f"Evaluating {test_data_path}")
        evaluate(test_logits, test_targets, thresholds, test_decisions)


def _cli() -> None:
    """CLI entrypoint for ``learning-idk`` console script."""
    args = _parse_args()
    main(
        args.data_path,
        threshold_path=args.threshold_path,
        synth=args.synth,
        delta=args.delta,
        skip_ts=args.skip_ts,
        thresh_func=args.thresh_func,
        test_data_path=args.test_data_path,
    )


if __name__ == "__main__":
    _cli()
