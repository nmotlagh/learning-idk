"""Tests for data loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch

from learning_idk import data as data_mod
from learning_idk.data import load_data, load_decisions


def test_load_data_non_synth_splits_logits_and_targets(tmp_path: Path) -> None:
    payload = torch.tensor(
        [
            [1.2, 0.8, 0.0],
            [0.3, 1.9, 1.0],
        ],
        dtype=torch.float32,
    )
    data_path = tmp_path / "non_synth.pt"
    torch.save(payload, data_path)

    logits, targets = load_data(data_path, synth=False)

    assert logits.shape == (2, 2)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert targets.tolist() == [0, 1]


def test_load_data_and_decisions_for_synth_format(tmp_path: Path) -> None:
    # Each row is [logits..., target, decision].
    payload = torch.tensor(
        [
            [2.0, 0.2, 0.0, 1.0],
            [0.1, 3.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    data_path = tmp_path / "synth.pt"
    torch.save(payload, data_path)

    logits, targets = load_data(data_path, synth=True)
    decisions = load_decisions(data_path)

    assert logits.shape == (2, 2)
    assert targets.dtype == torch.int64
    assert decisions.dtype == torch.int64
    assert targets.tolist() == [0, 1]
    assert decisions.tolist() == [1, 0]


def test_safe_torch_load_falls_back_when_weights_only_is_unsupported(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    def fake_load(path: Path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("weights_only") is True:
            raise TypeError("weights_only unsupported")
        return expected

    monkeypatch.setattr(data_mod.torch, "load", fake_load)

    loaded = data_mod._safe_torch_load(Path("dummy.pt"))

    assert torch.equal(loaded, expected)
    assert len(calls) == 2
    assert calls[0]["map_location"] == "cpu"
    assert calls[1]["map_location"] == "cpu"
