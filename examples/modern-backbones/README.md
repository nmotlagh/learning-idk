# B-CDF on modern backbones (CIFAR-100)

This example runs the method on CIFAR-100 using logits from current frozen vision backbones, each with a linear probe. It contains:

- a fast NumPy reimplementation of the B-CDF threshold search, checked against this package by parity tests;
- a script that generates the logits;
- the data export for the interactive demo on [nmotlagh.github.io](https://nmotlagh.github.io).

## Setup

- **Backbones** (from `timm`, kept frozen):
  - DINOv2 ViT-S/14 and ViT-B/14;
  - DINOv3 ViT-S/16;
  - SigLIP 2 ViT-B/16.
- **Features:** images are upsampled from 32 to 224 px (bicubic).
- **Linear probe:** AdamW, cosine schedule, 100 epochs. Weight decay is chosen on a split inside the probe's own training data.
- **Splits:** 40k train images fit the probe. The other 10k train images are the validation set used for temperatures and thresholds. The 10k CIFAR-100 test images are used for reporting only.

## Reproduce

```bash
uv sync
uv run pytest                      # fast_bcdf matches learning_idk's thresholds
uv run python make_logits.py       # needs a CUDA GPU; ~10 min on an RTX 4090
uv run python export_demo.py --site ../../../nmotlagh.github.io
```

- CIFAR-100 is read from the [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) Parquet mirror on Hugging Face and cached in `data/`.
- Logits go to `out/`. Neither directory is committed.
- GPU nondeterminism in probe training moves the last digit of some numbers between runs.

| File | Purpose |
|---|---|
| `fast_bcdf.py` | NumPy B-CDF: sorted prefix sums, exact ties, thresholds per class or per group of classes |
| `test_fast_bcdf.py` | Parity with `learning_idk.threshold.learn_thresholds` on the bundled synthetic logits and random stress cases |
| `make_logits.py` | Backbone features, linear probes, and val/test logits |
| `export_demo.py` | Data for the website demo: integer log-odds confidences and a sprite of 1,000 test images |

**Note on δ = 0.5:** with an odd-sized reject region, the binomial CDF is exactly 0.5, which equals 1 − δ. Because of floating-point rounding in SciPy, the reference implementation's answer can land on either side of that boundary. `fast_bcdf` uses a 1e-9 tolerance, and the parity tests skip δ = 0.5.

**Note on ties:** `fast_bcdf` compares select accuracies as exact fractions. The reference compares float32 accuracies. Two thresholds whose accuracies differ by less than float32 rounding can therefore break the tie differently. The parity tests on the bundled synthetic logits hit no such case.
