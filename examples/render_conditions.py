#!/usr/bin/env python3
"""Render one scene under every RSPDBench condition as false-color JPEGs.

Example (HLS Burn Scars tile from PANGAEA, bands Blue, Green, Red, NIR, SWIR1, SWIR2):
    python examples/render_conditions.py --tif subsetted_512x512_HLS.S30.T11SNS.2020276.v1.4_merged.tif \
        --dataset hls-burn-scars --composite SWIR_2 NIR_NARROW RED --out renders/

The display stretch is fixed from the clean image, so gain shifts stay visible as
brightness changes and misregistration as colored edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import rspdbench as rb


def stretch(img: np.ndarray, lo: np.ndarray, hi: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    return (np.clip((img - lo) / (hi - lo), 0, 1) ** gamma * 255).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", required=True, help="multiband GeoTIFF in the dataset's band order")
    ap.add_argument("--dataset", default="hls-burn-scars")
    ap.add_argument("--composite", nargs=3, default=["SWIR_2", "NIR_NARROW", "RED"], help="bands shown as R, G, B")
    ap.add_argument("--band-mean", nargs="*", type=float, help="clean training mean per band (P4/C1); defaults to this scene's means")
    ap.add_argument("--directions", nargs="+", default=["down"], help="gain directions to render for P2/C1/C2")
    ap.add_argument("--out", type=Path, default=Path("renders"))
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    import rasterio  # only needed for this example

    contract = rb.get_contract(args.dataset)
    with rasterio.open(args.tif) as src:
        x = torch.from_numpy(src.read().astype(np.float32))
    band_ids = list(contract.bands)
    mean = torch.tensor(args.band_mean) if args.band_mean else x.flatten(1).mean(1)
    pick = [band_ids.index(b) for b in args.composite]
    clean = x[pick].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(clean, 2, axis=(0, 1)), np.percentile(clean, 98, axis=(0, 1))

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for cond in rb.conditions_for(contract, band_ids):
        if cond.direction and cond.direction not in args.directions:
            continue
        y = rb.degrade(x, cond, band_ids=band_ids, contract=contract, sample_id=Path(args.tif).stem, seed=42, band_mean=mean)
        rgb = stretch(y[pick].permute(1, 2, 0).numpy(), lo, hi)
        Image.fromarray(rgb).save(args.out / f"{cond.name}.jpg", quality=args.quality, optimize=True, progressive=True)
        index.append({"name": cond.name, "family": cond.family, "severity": cond.severity, "direction": cond.direction})
    (args.out / "conditions.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} renders to {args.out}")


if __name__ == "__main__":
    main()
