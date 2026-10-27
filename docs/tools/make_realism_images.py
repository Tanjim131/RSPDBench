#!/usr/bin/env python3
"""Mismatch maps for the project page's realism figure (paper Sec. 3.4).

For one m-BigEarthNet test patch, compares the real 20 m B8A band with B08 brought to the
20 m grid in two ways: area reduction alone (sigma = 0) and P1-S2 (sigma = 1, x2 area
reduction). Each version is matched to B8A in brightness with a least-squares gain and
offset, and the map shows the absolute difference that remains, in % of the B8A mean.

    GEO_BENCH_DIR=/path/to/geobench python docs/tools/make_realism_images.py

Writes docs/assets/realism/mismatch_raw.png, mismatch_p1s2.png and mismatch.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from realism_check import DATASETS, blur_area, corr  # noqa: E402

OUT = ROOT / "docs" / "assets" / "realism"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue, 100-700
SCALE_MAX = 20.0  # % of the B8A mean at the dark end of the color scale


def ramp_lut(steps: list[str], n: int = 256) -> np.ndarray:
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in steps], dtype=float)
    pos = np.linspace(0, 1, len(steps))
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, pos, rgb[:, c]) for c in range(3)], 1).round().astype(np.uint8)


def mismatch(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a, b = np.polyfit(x.ravel(), ref.ravel(), 1)
    return np.abs(ref - (a * x + b)) / ref.mean() * 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geobench-dir", default=os.environ.get("GEO_BENCH_DIR"), required=os.environ.get("GEO_BENCH_DIR") is None)
    ap.add_argument("--sample", default="id_433688")
    ap.add_argument("--size", type=int, default=360, help="output size in pixels")
    args = ap.parse_args()

    rel, k08, k8a = DATASETS["m-bigearthnet"]
    with h5py.File(Path(args.geobench_dir) / rel / f"{args.sample}.hdf5", "r") as h:
        b08, b8a = h[k08][()].astype(np.float64), h[k8a][()].astype(np.float64)
    real20 = F.interpolate(torch.from_numpy(b8a[None, None].astype(np.float32)), scale_factor=0.5, mode="area")[0, 0].numpy()
    lut = ramp_lut(RAMP)
    stats = {}
    for name, sigma in (("raw", 0.0), ("p1s2", 1.0)):
        x = blur_area(b08, sigma)
        m = mismatch(x.astype(np.float64), real20.astype(np.float64))
        idx = np.clip(m / SCALE_MAX, 0, 1) * 255
        img = Image.fromarray(lut[idx.round().astype(int)]).resize((args.size, args.size), Image.NEAREST)
        img.save(OUT / f"mismatch_{name}.png", optimize=True)
        stats[name] = {"sigma": sigma, "r": round(corr(x, real20), 4), "mean_mismatch_pct": round(float(m.mean()), 2)}
    stats.update(sample=args.sample, scale_max_pct=SCALE_MAX, ramp=RAMP)
    (OUT / "mismatch.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
