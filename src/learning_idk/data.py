"""Data loading utilities for logit datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Batch size for data loaders
NUM_IN_BATCH: int = 32


def _safe_torch_load(data_path: Path) -> torch.Tensor:
    """Load tensor files across PyTorch versions.

    ``weights_only`` is only available in newer PyTorch releases.
    """
    try:
        return torch.load(data_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(data_path, map_location="cpu")


class LogitDataset(Dataset):
    """Simple torch Dataset with examples as logits."""

    def __init__(self, samples: torch.Tensor, targets: torch.Tensor) -> None:
        """Instantiates the Logit Dataset.

        Args:
            samples: Tensor of logits (samples x classes).
            targets: Tensor of targets (samples).
        """
        self.samples = samples
        self.targets = targets
        self.num_classes = torch.unique(targets).numel()

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx, :], self.targets[idx]


def load_data(
    data_path: str | Path, synth: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads the data from data_path and returns tensors for logits and targets.

    Args:
        data_path: Path to data (logits, targets).
        synth: Flag indicating whether data is synthetic.

    Returns:
        Tuple of (logits, targets).
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    data = _safe_torch_load(data_path)
    # Synthetic data has ground truth decision in last position
    if synth:
        return data[:, :-2].to(torch.float32), data[:, -2].to(torch.int64)
    else:
        return data[:, :-1].to(torch.float32), data[:, -1].to(torch.int64)


def load_decisions(data_path: str | Path) -> torch.Tensor:
    """Loads ground-truth decisions from synthetic data.

    Args:
        data_path: Path to data (logits, targets, decisions).

    Returns:
        Tensor of ideal decisions.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    data = _safe_torch_load(data_path)
    return data[:, -1].to(torch.int64)


def get_logits_loader(
    logits: torch.Tensor, targets: torch.Tensor
) -> DataLoader:
    """Generates a DataLoader from logits and targets.

    Args:
        logits: Tensor of logits.
        targets: Tensor of targets.

    Returns:
        DataLoader wrapping the logits and targets.
    """
    dataset = LogitDataset(logits, targets)
    return DataLoader(dataset, batch_size=NUM_IN_BATCH, shuffle=False)
