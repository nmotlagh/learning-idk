"""CLI smoke test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


def test_cli_smoke_on_synthetic_data(tmp_path: Path) -> None:
    synth_path = tmp_path / "tiny_synth.pt"
    thresholds_path = tmp_path / "out_thresholds.pt"
    rows = torch.tensor(
        [
            [5.0, 1.0, 0.5, 0.0, 1.0],
            [4.5, 0.7, 0.6, 0.0, 1.0],
            [0.5, 4.2, 1.0, 1.0, 1.0],
            [0.7, 4.4, 0.8, 1.0, 1.0],
            [0.6, 1.1, 4.8, 2.0, 1.0],
            [0.9, 0.8, 4.5, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    torch.save(rows, synth_path)

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing

    cmd = [
        sys.executable,
        "-m",
        "learning_idk",
        "--data_path",
        str(synth_path),
        "--synth",
        "--skip_ts",
        "--threshold_path",
        str(thresholds_path),
    ]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert thresholds_path.exists()
    assert "Learning Thresholds" in proc.stdout
