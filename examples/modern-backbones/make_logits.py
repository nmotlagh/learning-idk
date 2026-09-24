"""CIFAR-100 logits from modern frozen backbones + linear probes.

Splits: the probe trains on 40k train images, 10k held-out train images are the
calibration/threshold set ("val"), and the 10k test set is for reporting. Writes
out/<model>.npz with val/test logits and targets. Needs a CUDA GPU (~10 min on a 4090
for all four backbones).

    uv run python make_logits.py [timm model name ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

OUT = Path(__file__).parent / "out"
DATA = Path(__file__).parent / "data"
MODELS = sys.argv[1:] or [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_small_patch16_dinov3.lvd1689m",
    "vit_base_patch16_siglip_224.v2_webli",
]
dev = "cuda"
torch.manual_seed(0)


def load_cifar() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CIFAR-100 from the Hugging Face parquet mirror (the Toronto host is slow)."""
    cache = DATA / "cifar100.npz"
    if not cache.exists():
        import io
        import json

        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from PIL import Image

        DATA.mkdir(exist_ok=True)
        arrays = {}
        for split in ("train", "test"):
            path = hf_hub_download("uoft-cs/cifar100", f"cifar100/{split}-00000-of-00001.parquet",
                                   repo_type="dataset")
            table = pq.read_table(path)
            names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
            arrays[f"{split}_x"] = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))
                                             for r in table.column("img").to_pylist()])
            arrays[f"{split}_y"] = np.asarray(table.column("fine_label").to_pylist())
        arrays["fine_names"] = np.asarray(names["fine_label"]["names"])
        np.savez(cache, **arrays)
    d = np.load(cache)

    def as_t(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y)

    return (*as_t(d["train_x"], d["train_y"]), *as_t(d["test_x"], d["test_y"]))


@torch.no_grad()
def extract(name: str, images: torch.Tensor) -> torch.Tensor:
    kwargs = {"img_size": 224} if "dino" in name else {}
    model = timm.create_model(name, pretrained=True, num_classes=0, **kwargs).to(dev).eval()
    cfg = timm.data.resolve_model_data_config(model)
    size = model.patch_embed.img_size[0]  # cfg reports the pretrain size, not img_size
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(images), 500):
        x = images[i:i + 500].to(dev).float() / 255
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(model((x - mean) / std).float().cpu())
    return torch.cat(feats)


def train_probe(x: torch.Tensor, y: torch.Tensor, wd: float, epochs: int = 100) -> torch.nn.Linear:
    x, y = x.to(dev), y.to(dev)
    probe = torch.nn.Linear(x.shape[1], 100).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=dev).split(1024):
            loss = F.cross_entropy(probe(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    return probe.eval()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    xtr_img, ytr, xte_img, yte = load_cifar()
    perm = torch.randperm(len(ytr))
    fit_idx, val_idx = perm[:40_000], perm[40_000:]
    for name in MODELS:
        t0 = time.time()
        try:
            ftr, fte = extract(name, xtr_img), extract(name, xte_img)
        except Exception as e:  # gated or missing weights: skip, keep going
            print(f"SKIP {name}: {type(e).__name__}: {str(e)[:160]}")
            continue
        mu, sd = ftr[fit_idx].mean(0), ftr[fit_idx].std(0) + 1e-6
        ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
        # Pick weight decay on a split inside the 40k fit set (val stays untouched).
        inner, hold = fit_idx[:35_000], fit_idx[35_000:]
        best = max(
            (0.0, 1e-3, 1e-2, 1e-1),
            key=lambda wd: (train_probe(ftr[inner], ytr[inner], wd, 30)(ftr[hold].to(dev))
                            .argmax(1).cpu() == ytr[hold]).float().mean().item(),
        )
        probe = train_probe(ftr[fit_idx], ytr[fit_idx], best)
        with torch.no_grad():
            val_logits = probe(ftr[val_idx].to(dev)).cpu()
            test_logits = probe(fte.to(dev)).cpu()

        def acc(lg: torch.Tensor, y: torch.Tensor) -> float:
            return (lg.argmax(1) == y).float().mean().item()

        print(f"{name}: feat_dim={ftr.shape[1]} wd={best} "
              f"val_acc={acc(val_logits, ytr[val_idx]):.4f} test_acc={acc(test_logits, yte):.4f} "
              f"({time.time() - t0:.0f}s)")
        np.savez_compressed(
            OUT / f"{name}.npz",
            val_logits=val_logits.numpy(), val_targets=ytr[val_idx].numpy(),
            val_index=val_idx.numpy(),
            test_logits=test_logits.numpy(), test_targets=yte.numpy(),
        )


if __name__ == "__main__":
    main()
