"""Export the data behind the interactive demo on nmotlagh.github.io.

The demo learns one global B-CDF threshold in the browser, so it only needs each
prediction's confidence and whether it was right. Confidence is the log-odds of the
temperature-scaled max softmax, log(p / (1 - p)), stored as integers (x SCALE) so the
browser and this script see identical ties. It also ships a stratified sample of test
images for the dot plot, as one WebP sprite.

    uv run python export_demo.py --site ../../../nmotlagh.github.io

writes <site>/src/data/abstain-demo.json and <site>/public/demo/cifar100-sample.webp,
plus out/demo-reference.json (thresholds from this script, to check the JS port).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from learning_idk.calibration import ModelWithTemperature
from learning_idk.data import get_logits_loader
from PIL import Image

import fast_bcdf

HERE = Path(__file__).parent
MODEL = "vit_small_patch14_dinov2.lvd142m"
SCALE = 1000
SAMPLE_BLOCK = 10  # one sampled test image per 10 consecutive confidences
SPRITE_COLS = 40
DELTAS = (0.05, 0.25, 0.5, 0.75, 0.95)


def log_odds(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """log(p / (1 - p)) of the max softmax, computed without forming 1 - p."""
    z = logits.double()
    top, pred = z.max(1)
    rest = z.scatter(1, pred[:, None], float("-inf")).logsumexp(1)
    return (top - rest).numpy(), pred.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True, help="path to the site repo")
    args = ap.parse_args()

    d = np.load(HERE / "out" / f"{MODEL}.npz")
    val_lg, val_y = torch.from_numpy(d["val_logits"]), torch.from_numpy(d["val_targets"]).long()
    test_lg, test_y = torch.from_numpy(d["test_logits"]), torch.from_numpy(d["test_targets"]).long()

    ts = ModelWithTemperature(n_bins=15, strategy="grid", per_class=False)
    temperature = float(ts.set_temperature(get_logits_loader(val_lg, val_y),
                                           t_vals=list(torch.linspace(0.25, 4.0, 100))))

    def split(logits: torch.Tensor, targets: torch.Tensor):
        lo, pred = log_odds(logits / temperature)
        return np.round(lo * SCALE).astype(np.int64), pred, pred == targets.numpy()

    val_q, _, val_ok = split(val_lg, val_y)
    test_q, test_pred, test_ok = split(test_lg, test_y)

    # Stratified sample: sort test by confidence, take one random image per block.
    order = np.argsort(test_q, kind="stable")
    rng = np.random.default_rng(0)
    blocks = order[: len(order) // SAMPLE_BLOCK * SAMPLE_BLOCK].reshape(-1, SAMPLE_BLOCK)
    sample = blocks[np.arange(len(blocks)), rng.integers(0, SAMPLE_BLOCK, len(blocks))]

    cifar = np.load(HERE / "data" / "cifar100.npz")
    assert (cifar["test_y"][sample] == test_y.numpy()[sample]).all()
    rows = -(-len(sample) // SPRITE_COLS)
    sprite = np.zeros((rows * 32, SPRITE_COLS * 32, 3), dtype=np.uint8)
    for i, idx in enumerate(sample):
        r, c = divmod(i, SPRITE_COLS)
        sprite[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = cifar["test_x"][idx]
    names = [str(n).replace("_", " ") for n in cifar["fine_names"]]

    def sorted_ints(a: np.ndarray) -> list[int]:
        return np.sort(a).tolist()

    payload = {
        "model": "DINOv2 ViT-S/14 (frozen) + linear probe",
        "dataset": "CIFAR-100",
        "scale": SCALE,
        "temperature": round(temperature, 4),
        "classes": names,
        "val": {"right": sorted_ints(val_q[val_ok]), "wrong": sorted_ints(val_q[~val_ok])},
        "test": {"right": sorted_ints(test_q[test_ok]), "wrong": sorted_ints(test_q[~test_ok])},
        "sample": {
            "score": test_q[sample].tolist(),
            "pred": test_pred[sample].tolist(),
            "label": test_y.numpy()[sample].tolist(),
            "cols": SPRITE_COLS,
            "size": 32,
        },
    }
    (args.site / "src" / "data" / "abstain-demo.json").write_text(
        json.dumps(payload, separators=(",", ":")))
    (args.site / "public" / "demo").mkdir(parents=True, exist_ok=True)
    Image.fromarray(sprite).save(args.site / "public" / "demo" / "cifar100-sample.webp",
                                 quality=82, method=6)

    lf = fast_bcdf.log_factorials(val_q.size)
    reference = []
    print(f"{MODEL}: T={temperature:.3f}  test acc {test_ok.mean():.4f}  "
          f"sample acc {test_ok[sample].mean():.4f}")
    for delta in DELTAS:
        t = fast_bcdf.learn_threshold(val_q, val_ok, delta, lf, floor=float("-inf"))
        sel = test_q > t
        r = {"delta": delta, "threshold": t, "coverage": float(sel.mean()),
             "select_acc": float(test_ok[sel].mean()), "reject_acc": float(test_ok[~sel].mean())}
        reference.append(r)
        print(f"  delta={delta:<4} t={t / SCALE:+.3f} (p={1 / (1 + np.exp(-t / SCALE)):.3f})  "
              f"cov {r['coverage']:.3f}  sel {r['select_acc']:.4f}  rej {r['reject_acc']:.3f}")
    (HERE / "out" / "demo-reference.json").write_text(json.dumps(reference, indent=1))


if __name__ == "__main__":
    main()
