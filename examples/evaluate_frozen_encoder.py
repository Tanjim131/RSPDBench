#!/usr/bin/env python3
"""RSPDBench protocol for your own frozen encoder (classification example on GEO-Bench m-EuroSAT).

1. Encode clean train/val images, fit a linear probe, pick its strength on clean validation only.
2. For every condition, degrade each test image in native units, then apply YOUR preprocessing
   and encoder, classify with the fixed clean-trained probe, and score.
3. Report relative drops from the clean score.

Plug in a model by editing `Encoder` (bands it consumes, preprocessing, feature extraction).
The default `--encoder random` is a smoke test that needs no weights.

    pip install geo-benchmark scikit-learn
    GEO_BENCH_DIR=/path/to/geobench python examples/evaluate_frozen_encoder.py --max-train 500 --max-test 300
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

import rspdbench as rb


class Encoder:
    """Replace with your model. It sees images AFTER degradation, in native units."""

    bands = ["B04", "B03", "B02", "B08", "B11", "B12"]  # physical band ids the model consumes

    def __init__(self, kind: str):
        g = torch.Generator().manual_seed(0)
        self.proj = torch.randn(len(self.bands) * 16, 256, generator=g) / 8.0 if kind == "random" else None

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x / 10000.0  # e.g. reflectance scaling, normalization, resizing

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> np.ndarray:
        z = F.adaptive_avg_pool2d(self.preprocess(batch), 4).flatten(1)  # toy features
        return torch.tanh(z @ self.proj).numpy()


def load_split(task, split: str, limit: int):
    """Images as (C, H, W) tensors in GEO-Bench native units, physical m-EuroSAT band order."""
    ds = task.get_dataset(split=split)
    n = min(len(ds), limit) if limit else len(ds)
    xs, ys, ids = [], [], []
    for i in range(n):
        s = ds[i]
        img, _ = s.pack_to_3d(s.band_names)  # stored order is rb.MEUROSAT_STORAGE_ORDER
        xs.append(torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1))
        ys.append(int(s.label))
        ids.append(str(getattr(s, "sample_name", i)))
    return xs, np.array(ys), ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="random")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    args = ap.parse_args()
    if "GEO_BENCH_DIR" not in os.environ:
        raise SystemExit("Set GEO_BENCH_DIR to your GEO-Bench download.")
    import geobench  # type: ignore

    task = next(t for t in geobench.task_iterator(benchmark_name="classification_v1.0") if t.dataset_name == "m-eurosat")
    contract = rb.get_contract("m-eurosat")
    all_ids = list(rb.MEUROSAT_STORAGE_ORDER)
    pick = [all_ids.index(b) for b in Encoder.bands]
    enc = Encoder(args.encoder)

    xtr, ytr, _ = load_split(task, "train", args.max_train)
    xva, yva, _ = load_split(task, "valid", args.max_train)
    xte, yte, ids = load_split(task, "test", args.max_test)
    band_mean = torch.stack(xtr).mean(dim=(0, 2, 3))  # clean training mean per band (P4/C1)

    feats = lambda xs: np.concatenate([enc(torch.stack([x[pick] for x in xs[i:i + 64]])) for i in range(0, len(xs), 64)])
    ftr, fva = feats(xtr), feats(xva)
    probe = max((LogisticRegression(C=c, max_iter=2000).fit(ftr, ytr) for c in (0.01, 0.1, 1.0, 10.0)), key=lambda m: m.score(fva, yva))

    scores = {}
    for cond in rb.conditions_for(contract, Encoder.bands):
        xs = [rb.degrade(x, cond, band_ids=all_ids, contract=contract, sample_id=sid, seed=42, band_mean=band_mean) for x, sid in zip(xte, ids)]
        scores[cond.name] = probe.score(feats(xs), yte)
    clean = scores["clean"]
    print(f"clean accuracy {clean:.4f}")
    for name, acc in scores.items():
        if name != "clean":
            print(f"{name:24s} acc {acc:.4f}  drop {rb.relative_drop(clean, acc):6.2f} %")


if __name__ == "__main__":
    main()
