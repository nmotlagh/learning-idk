"""Tests for temperature scaling behavior."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from learning_idk.calibration import ModelWithTemperature


def test_temperature_scale_per_class_applies_predicted_class_temperature() -> None:
    model = ModelWithTemperature(per_class=True, strategy="grid")
    model.temperature = torch.tensor([1.0, 2.0, 4.0])
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.0, 4.0, 1.0]], dtype=torch.float32)

    scaled = model.temperature_scale(logits)

    assert torch.allclose(scaled[0], logits[0])
    assert torch.allclose(scaled[1], logits[1] / 2.0)


def test_temperature_scale_raises_for_mismatched_per_class_temperature_length() -> None:
    model = ModelWithTemperature(per_class=True, strategy="grid")
    model.temperature = torch.tensor([1.0])
    logits = torch.tensor([[0.0, 3.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="smaller than the number of predicted classes"):
        model.temperature_scale(logits)


def test_set_temperature_rejects_empty_loader() -> None:
    logits = torch.empty((0, 3), dtype=torch.float32)
    labels = torch.empty((0,), dtype=torch.int64)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    model = ModelWithTemperature(strategy="grid")

    with pytest.raises(ValueError, match="valid_loader is empty"):
        model.set_temperature(loader)


def test_set_temperature_grid_per_class_returns_temperature_per_logit_class() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.2, 0.1],
            [3.8, 0.3, 0.1],
            [0.2, 3.5, 0.3],
            [0.4, 3.7, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    model = ModelWithTemperature(strategy="grid", per_class=True)
    temps = model.set_temperature(loader, t_vals=[0.5, 1.0, 2.0])

    assert temps.shape == (3,)
    scaled = model(logits)
    assert scaled.shape == logits.shape
