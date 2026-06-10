# Learning When to Say "I Don't Know"

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2209.04944-b31b1b.svg)](https://arxiv.org/abs/2209.04944)

Official code for ["Learning When to Say 'I Don't Know'"](https://arxiv.org/abs/2209.04944) by Nicholas Kashani Motlagh, [Jim Davis](http://web.cse.ohio-state.edu/~davis.1719/), Tim Anderson, and Jeremy Gwinnup (ISVC 2022).

> The exact code used in the paper is preserved at the [`v1.0.0`](https://github.com/osu-cvl/learning-idk/tree/v1.0.0) tag.

We propose a Reject Option Classification technique to identify and remove regions of uncertainty in the decision space for a given neural classifier and dataset. Rather than learning a rejection/selection function with a known rejection cost or strong accuracy/coverage constraints, we analyze the complementary reject region and employ a validation set to learn per-class softmax thresholds. The goal is to maximize the accuracy of the selected examples subject to a natural randomness allowance on the rejected examples (rejecting more incorrect than correct predictions).

## What's New in 3.0.0

`3.0.0` is a cleanup and correctness release focused on making the package easier to use safely.

### Key changes

- Strict CLI/input validation for `--data_path`, `--delta`, and `--thresh_func`.
- Per-class threshold count now follows the logits dimension (`logits.shape[1]`) instead of observed labels.
- Per-class temperature scaling now uses a deterministic 1D temperature vector by predicted class.
- Safer tensor loading across PyTorch versions (`weights_only` fallback + CPU map location).
- Added automated quality gates: `ruff` linting, `pytest` tests, and CI.

### Migration notes from 2.x

- If your labels omit classes that still exist in the logits head, output thresholds now include those classes.
- Invalid CLI values that were previously accepted may now fail fast with explicit error messages.
- Internal calibration temperature tensor shapes are now normalized (relevant if you use internal attributes directly).

## Installation

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

For development (tests + lint):

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

### Requirements

- Python >= 3.10
- PyTorch, NumPy, SciPy, statsmodels, Matplotlib

All dependencies are declared in `pyproject.toml` and installed automatically.

## Usage

### Learning thresholds

```bash
python -m learning_idk \
    --data_path <path to validation logits file> \
    --test_data_path <path to test logits (optional)> \
    --delta .05 \
    --thresh_func b_cdf
```

Or using the installed console script:

```bash
learning-idk \
    --data_path <path to validation logits file> \
    --delta .05 \
    --thresh_func b_cdf
```

#### Arguments

| Argument | Description | Default |
|---|---|---|
| `--data_path` | Path to validation logits (required) | — |
| `--test_data_path` | Path to test logits | `None` |
| `--delta` | Significance level for the BinomialCDF algorithm | `0.05` |
| `--thresh_func` | Method to check reject region viability (`b_cdf`, `wilson`, `wilson_cc`, `clopper_pearson`, `agresti_coull`) | `b_cdf` |
| `--threshold_path` | Path to save the threshold tensor | `thresholds.pt` |
| `--synth` | Flag indicating synthetic data format | `False` |
| `--skip_ts` | Skip temperature scaling | `False` |

### Synthetic data

```bash
python -m learning_idk \
    --data_path synth_logits/val_logits_v1.pt \
    --test_data_path synth_logits/test_logits_v1.pt \
    --delta .05 \
    --thresh_func b_cdf \
    --synth
```

Synthetic logit files (`v1`–`v8`) are included in `synth_logits/`.

### As a library

```python
from learning_idk import learn_thresholds, evaluate, load_data, ModelWithTemperature

logits, targets = load_data("synth_logits/val_logits_v1.pt", synth=True)
thresholds = learn_thresholds(logits, targets, delta=0.05, thresh_func="b_cdf")
evaluate(logits, targets, thresholds)
```

### Calibration

The `calibration` module provides temperature scaling and ECE computation. For a more comprehensive calibration library, see [`netcal`](https://github.com/fabiankueppers/calibration-framework).

## Project structure

```
learning-idk/
├── .github/workflows/ci.yml
├── src/learning_idk/
│   ├── __init__.py
│   ├── __main__.py        # python -m learning_idk entrypoint
│   ├── threshold.py       # core thresholding algorithm
│   ├── calibration.py     # temperature scaling + ECE
│   └── data.py            # LogitDataset, data loaders
├── tests/                 # pytest suite
├── synth_logits/          # example synthetic data
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

```bibtex
@inproceedings{KashaniMotlagh2022,
    title     = {Learning When to Say ``{I} Don't Know''},
    author    = {Kashani Motlagh, Nicholas and Davis, Jim and Anderson, Tim and Gwinnup, Jeremy},
    booktitle = {International Symposium on Visual Computing (ISVC)},
    year      = {2022},
    url       = {https://arxiv.org/abs/2209.04944}
}
```
