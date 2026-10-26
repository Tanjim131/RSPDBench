#!/usr/bin/env python3
"""Draw the README figures, each in a light and a dark version.

defects_{light,dark}.png  one HLS Burn Scars crop under every defect at S3, degraded with the
                          released operators, with a map of how much each pixel changed
ranking_{light,dark}.png  m-EuroSAT foundation models ranked by their S3 drop under resolution
                          loss (P1) and under gain error (P2, worse direction)

    python analysis/figures/plot_readme_figures.py \
        --hls-tif /path/to/subsetted_512x512_HLS.S30.T11SNS.2020276.v1.4_merged.tif
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import rspdbench as rb  # noqa: E402

OUT = ROOT / "assets" / "readme"
DPI = 200

# Clean training-set mean per band (reflectance), PANGAEA HLS Burn Scars training split,
# the same values the project page uses for P4 and C1.
HLS_MEAN = [0.05189, 0.07617, 0.09353, 0.20407, 0.23159, 0.17121]
COMPOSITE = ["SWIR_2", "NIR_NARROW", "RED"]
CROP = (288, 336, 64)  # left, top, size in pixels: a reservoir arm inside the burn scar
CHANGE_MAX = 30.0      # change-map color scale, % of band mean

COLUMNS = [  # condition, code, name, setting
    ("clean", "Clean", "HLS scene", "30 m pixels"),
    ("p1_s3", "P1", "resolution loss", "30 m → 120 m"),
    ("p2_gain_down_s3", "P2", "gain error", "all bands × 0.85"),
    ("p3_s3", "P3", "misregistration", "NIR, SWIR by 2 px"),
    ("p4_s3", "P4", "sensor noise", "SNR 20"),
    ("c1_gain_down_s3", "C1", "acquisition", "P2 → P1 → P4"),
    ("c2_gain_down_s3", "C2", "harmonization", "NIR, SWIR only"),
]

MODEL_LABELS = {
    "dinov3_vit7b_web": "DINOv3-Web", "dinov3_vit7b_sat": "DINOv3-Sat", "dofa_v1_vit_large_e100": "DOFA",
    "galileo_base": "Galileo", "panopticon_vitb14": "Panopticon", "prithvi_eo_v2_600": "Prithvi 2.0",
    "ramen_base": "RAMEN",
}
HIGHLIGHT = ("ramen_base", "dinov3_vit7b_sat")

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # steps 100-700
THEMES = {
    "light": {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "surface": "#ffffff",
              "accent": ("#2a78d6", "#eb6834"), "ramp": BLUE_RAMP},
    "dark": {"ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "surface": "#0d1117",
             "accent": ("#3987e5", "#d95926"), "ramp": BLUE_RAMP[::-1]},
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def stretch(img: np.ndarray, lo: np.ndarray, hi: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    return np.clip((img - lo) / (hi - lo), 0, 1) ** gamma


def degraded_crops(hls_tif: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """False-color crops and per-pixel change (% of band mean) for every column."""
    import rasterio

    with rasterio.open(hls_tif) as src:
        x = torch.from_numpy(src.read().astype(np.float32))
    contract = rb.get_contract("hls-burn-scars")
    bands = list(contract.bands)
    mean = torch.tensor(HLS_MEAN)
    pick = [bands.index(b) for b in COMPOSITE]
    clean = x[pick].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(clean, 2, axis=(0, 1)), np.percentile(clean, 98, axis=(0, 1))
    left, top, size = CROP
    window = (slice(top, top + size), slice(left, left + size))
    images, changes = [], []
    for cond, *_ in COLUMNS:
        y = rb.degrade(x, cond, band_ids=bands, contract=contract, sample_id=hls_tif.stem, seed=42, band_mean=mean)
        images.append(stretch(y[pick].permute(1, 2, 0).numpy(), lo, hi)[window])
        rel = ((y - x).abs() / mean[:, None, None]).mean(0).numpy() * 100
        changes.append(rel[window])
    return images, changes


def plot_defects(images: list[np.ndarray], changes: list[np.ndarray], theme: dict, out: Path) -> None:
    n = len(COLUMNS)
    fig = plt.figure(figsize=(8.4, 3.2), dpi=DPI)
    grid = fig.add_gridspec(2, n, left=0.005, right=0.995, top=0.775, bottom=0.01, wspace=0.05, hspace=0.05)
    cmap = LinearSegmentedColormap.from_list("change", theme["ramp"])
    for j, (_, code, name, setting) in enumerate(COLUMNS):
        ax = fig.add_subplot(grid[0, j])
        ax.imshow(images[j], interpolation="nearest")
        ax.set_axis_off()
        for text, dy, color, weight in ((code, 30, theme["ink"], "bold"), (name, 17, theme["ink"], "normal"),
                                         (setting, 4, theme["ink2"], "normal")):
            ax.annotate(text, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                        ha="center", va="bottom", color=color, fontweight=weight, fontsize=8.5 if dy == 30 else 8)
        ax = fig.add_subplot(grid[1, j])
        ax.set_axis_off()
        if j == 0:
            ax.text(0.5, 0.86, "Where it changed", ha="center", va="center", color=theme["ink"], fontweight="bold",
                    fontsize=8.5, transform=ax.transAxes)
            ax.text(0.5, 0.71, "% of band mean", ha="center", va="center", color=theme["ink2"], transform=ax.transAxes)
            cax = ax.inset_axes([0.1, 0.42, 0.8, 0.1])
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, CHANGE_MAX), cmap=cmap), cax=cax,
                               orientation="horizontal", ticks=[0, 15, 30])
            bar.ax.set_xticklabels(["0", "15", "30+"], color=theme["ink2"])
            bar.ax.tick_params(length=0, pad=3)
            bar.outline.set_visible(False)
            continue
        ax.imshow(changes[j], cmap=cmap, vmin=0, vmax=CHANGE_MAX, interpolation="nearest")
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)


def s3_drops(csv_path: Path, dataset: str = "m-eurosat") -> dict[str, tuple[float, float]]:
    """(P1 drop, worse-direction P2 drop) at S3 per foundation model, in %."""
    drop: dict[tuple[str, str], float] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["model"] in MODEL_LABELS:
                drop[(r["model"], r["condition"])] = float(r["relative_drop_pct_mean"])
    return {m: (drop[(m, "p1_s3")], max(drop[(m, "p2_gain_down_s3")], drop[(m, "p2_gain_up_s3")]))
            for m in MODEL_LABELS}


def plot_ranking(drops: dict[str, tuple[float, float]], theme: dict, out: Path) -> None:
    models = list(drops)
    rank_p1 = {m: i + 1 for i, m in enumerate(sorted(models, key=lambda m: drops[m][0]))}
    rank_p2 = {m: i + 1 for i, m in enumerate(sorted(models, key=lambda m: drops[m][1]))}
    fig = plt.figure(figsize=(6.0, 3.4), dpi=DPI)
    ax = fig.add_axes([0.33, 0.04, 0.34, 0.66])
    ax.set_xlim(0, 1)
    ax.set_ylim(len(models) + 0.5, 0.5)
    ax.set_axis_off()
    for m in sorted(models, key=lambda m: m in HIGHLIGHT):
        color = theme["accent"][HIGHLIGHT.index(m)] if m in HIGHLIGHT else theme["muted"]
        ink = theme["ink"] if m in HIGHLIGHT else theme["ink2"]
        y1, y2 = rank_p1[m], rank_p2[m]
        ax.plot([0, 1], [y1, y2], color=color, lw=1.5, solid_capstyle="round", zorder=3 if m in HIGHLIGHT else 1,
                clip_on=False)
        ax.plot([0, 1], [y1, y2], ls="none", marker="o", ms=6.5, mfc=color, mec=theme["surface"], mew=1.5,
                zorder=4 if m in HIGHLIGHT else 2, clip_on=False)
        weight = "bold" if m in HIGHLIGHT else "normal"
        ax.text(-0.05, y1, f"{drops[m][0]:.0f}%", ha="right", va="center", color=theme["ink2"], fontsize=8)
        ax.text(-0.25, y1, MODEL_LABELS[m], ha="right", va="center", color=ink, fontweight=weight, fontsize=8.5)
        p2 = drops[m][1]
        ax.text(1.05, y2, f"{p2:.2f}%" if p2 < 1 else f"{p2:.1f}%", ha="left", va="center", color=theme["ink2"],
                fontsize=8)
        ax.text(1.32, y2, MODEL_LABELS[m], ha="left", va="center", color=ink, fontweight=weight, fontsize=8.5)
    for x, ha, head in ((-0.05, "right", "Resolution loss (P1)"), (1.05, "left", "Gain error (P2)")):
        ax.text(x, -0.15, head, ha=ha, va="bottom", color=theme["ink"], fontweight="bold", fontsize=8.5)
    ax.text(0.5, 0.75, "smallest drop", ha="center", va="center", color=theme["muted"], fontsize=7.5)
    ax.text(0.5, len(models) + 0.25, "largest drop", ha="center", va="center", color=theme["muted"], fontsize=7.5)
    fig.text(0.02, 0.96, "The model that loses least to resolution loss loses most to gain error",
             color=theme["ink"], fontweight="bold", fontsize=10, va="top")
    fig.text(0.02, 0.895, "m-EuroSAT, severity S3, ranked by each model's drop from its clean accuracy",
             color=theme["ink2"], fontsize=8.5, va="top")
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hls-tif", required=True, type=Path)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "rspdbench_aggregate.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    images, changes = degraded_crops(args.hls_tif)
    drops = s3_drops(args.results)
    for mode, theme in THEMES.items():
        plot_defects(images, changes, theme, OUT / f"defects_{mode}.png")
        plot_ranking(drops, theme, OUT / f"ranking_{mode}.png")
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
