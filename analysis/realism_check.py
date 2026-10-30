#!/usr/bin/env python3
"""Check P1 and P3 against Sentinel-2's own co-acquired 10 m and 20 m bands (paper Sec. 3.4).

Sentinel-2 acquires broad NIR (B08) at 10 m and narrow NIR (B8A) at 20 m over nearly
the same spectral range. For each test patch we pass B08 through P1's blur and x2 area
reduction for a grid of PSF widths and correlate it with the real B8A on the 20 m grid,
then estimate the residual B08-B8A displacement with sub-pixel cross-correlation.

Needs GEO-Bench (pip install geobench; set GEO_BENCH_DIR) and scikit-image.
Usage: python analysis/realism_check.py --dataset m-bigearthnet --out realism_m-bigearthnet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from skimage.registration import phase_cross_correlation

from rspdbench.operators import _gaussian_kernel, resolution_loss

# (subfolder, HDF5 key of B08, HDF5 key of B8A). GEO-Bench stores m-EuroSAT's B8A under "12 - SWIR".
DATASETS = {
    "m-eurosat": ("classification_v1.0/m-eurosat", "08 - NIR", "12 - SWIR"),
    "m-bigearthnet": ("classification_v1.0/m-bigearthnet", "08 - NIR", "08A - Vegetation Red Edge"),
    "m-SA-crop-type": ("segmentation_v1.0/m-SA-crop-type", "08 - NIR", "08A - Vegetation Red Edge"),
}
SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def blur_area(x: np.ndarray, sigma: float, n: int = 2) -> np.ndarray:
    t = torch.from_numpy(x[None, None].astype(np.float32))
    if sigma > 0:
        r = max(1, int(math.ceil(3.0 * sigma)))
        k = _gaussian_kernel(r, sigma, t.device, t.dtype).view(1, 1, 2 * r + 1, 2 * r + 1)
        t = F.conv2d(F.pad(t, (r, r, r, r), mode="reflect"), k)
    return F.interpolate(t, size=(x.shape[0] // n, x.shape[1] // n), mode="area")[0, 0].numpy()


def corr(a: np.ndarray, b: np.ndarray, border: int = 2) -> float:
    a, b = a[border:-border, border:-border].ravel(), b[border:-border, border:-border].ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--geobench-dir", default=os.environ.get("GEO_BENCH_DIR"), required=os.environ.get("GEO_BENCH_DIR") is None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rel, k08, k8a = DATASETS[args.dataset]
    root = Path(args.geobench_dir) / rel
    ids = json.load(open(root / "default_partition.json"))[args.split]
    rows = []
    for sid in ids:
        with h5py.File(root / f"{sid}.hdf5", "r") as h:
            b08, b8a = h[k08][()].astype(np.float64), h[k8a][()].astype(np.float64)
        if b08.std() < 1e-6 or b8a.std() < 1e-6:
            continue
        real20 = F.interpolate(torch.from_numpy(b8a[None, None].astype(np.float32)), scale_factor=0.5, mode="area")[0, 0].numpy()
        row = {"sample_id": sid, **{f"r_sigma{s:.2f}": corr(blur_area(b08, s), real20) for s in SIGMAS}}
        ref = resolution_loss(torch.from_numpy(b08[None].astype(np.float32)), 1.0, 2)[0].numpy()
        win = np.outer(np.hanning(b08.shape[0]), np.hanning(b08.shape[1]))
        shift, _, _ = phase_cross_correlation((ref - ref.mean()) * win, (b8a - b8a.mean()) * win, upsample_factor=50, normalization=None)
        g = np.hypot(*np.gradient(ref))
        row.update(shift_px=float(np.hypot(*shift)), texture=float(g.mean() / max(abs(ref.mean()), 1e-6)))
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = np.array([[row[f"r_sigma{s:.2f}"] for s in SIGMAS] for row in rows])
    best = np.array(SIGMAS)[np.nanargmax(r, axis=1)]
    tex = np.array([row["texture"] for row in rows])
    shift = np.array([row["shift_px"] for row in rows])[tex >= np.median(tex)]
    print(f"{args.dataset}: n={len(rows)}  median r by sigma: " + ", ".join(f"{s}:{np.nanmedian(r[:, i]):.4f}" for i, s in enumerate(SIGMAS)))
    print(f"  median best-fit sigma = {np.median(best):.2f};  B08-B8A shift on textured patches: median {np.median(shift):.3f} px, p95 {np.percentile(shift, 95):.3f} px")


if __name__ == "__main__":
    main()
