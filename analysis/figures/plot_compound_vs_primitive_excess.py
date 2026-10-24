#!/usr/bin/env python3
"""Plot compound-vs-primitive interaction excess from canonical aggregates.

Two modes are supported:
- s3: one severity (S3), matching the main fingerprint heatmap.
- max: maximum excess across S1/S2/S3 and signed directions per cell.

Excess is defined as full compound relative drop minus the relative drop of the
worst primitive/component used by the aggregate table.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY_ALIASES = {
    "c1_acquisition_quality_chain": "C1",
    "C1": "C1",
    "c2_harmonization_chain": "C2",
    "C2": "C2",
}
DATASET_LABELS = {
    "m-eurosat": "m-EuroSAT",
    "m-bigearthnet": "m-BigEarthNet",
    "m-SA-crop-type": "m-SA-crop",
    "hls-burn-scars": "HLS-Burn-Scar",
    "m-chesapeake": "m-Chesapeake",
}
MODEL_LABELS = {
    "dinov3_vit7b_web": "DINOv3-Web",
    "dinov3_vit7b_sat": "DINOv3-Sat",
    "dofa_v1_vit_large_e100": "DOFA",
    "galileo_base": "Galileo",
    "panopticon_vitb14": "Panopticon",
    "prithvi_eo_v2_600": "Prithvi 2.0",
    "ramen_base": "RAMEN",
    "vit_b16_scratch": "ViT-B/16",
    "unet_scratch": "U-Net",
}
FAMILY_COLOR = {"C1": "#d62728", "C2": "#1f77b4"}
NEG_COLOR = "#7f7f7f"
MECHANISM_LABELS = {
    "dinov3_vit7b_web": "RGB unconditioned",
    "dinov3_vit7b_sat": "RGB unconditioned",
    "dofa_v1_vit_large_e100": "Wavelength-conditioned",
    "panopticon_vitb14": "Wavelength-conditioned",
    "galileo_base": "Channel-grouped",
    "ramen_base": "Resolution-conditioned",
    "prithvi_eo_v2_600": "Multispectral unconditioned",
    "vit_b16_scratch": "Supervised baseline",
    "unet_scratch": "Supervised baseline",
}
MECHANISM_COLORS = {
    "RGB unconditioned": "#4B1D83",
    "Wavelength-conditioned": "#006B4F",
    "Channel-grouped": "#B54500",
    "Resolution-conditioned": "#4E4A9E",
    "Multispectral unconditioned": "#B0005A",
    "Supervised baseline": "#3F3F3F",
    "Other": "#6F6F6F",
}


def parse_float(x: str) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def label_cell(dataset: str, model: str, variant: str) -> str:
    m = MODEL_LABELS.get(model, model)
    d = DATASET_LABELS.get(dataset, dataset)
    return f"{m} / {d}"


def mechanism_for_model(model: str) -> str:
    return MECHANISM_LABELS.get(model, "Other")


def direction_tag(direction: str) -> str:
    direction = (direction or "").lower()
    if direction == "down":
        return "↓"
    if direction == "up":
        return "↑"
    return "·"


def read_candidates(path: Path, mode: str) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("complete_seed_coverage", "").lower() != "true":
                continue
            if r.get("is_clean", "").lower() == "true":
                continue
            fam = FAMILY_ALIASES.get(r.get("condition_family", ""))
            if fam not in {"C1", "C2"}:
                continue
            excess = parse_float(r.get("compound_excess_relative_drop_pct", ""))
            compound = parse_float(r.get("relative_drop_pct_mean", ""))
            comp_abs = parse_float(r.get("component_max_absolute_drop_mean", ""))
            clean = parse_float(r.get("clean_metric_mean", ""))
            severity = parse_float(r.get("severity", ""))
            if excess is None or compound is None or severity is None:
                continue
            if mode == "s3" and int(severity) != 3:
                continue
            component_rel = ""
            if comp_abs is not None and clean not in (None, 0):
                component_rel = comp_abs / abs(clean) * 100.0
            rows.append({
                "dataset": r.get("dataset", ""),
                "task_type": r.get("task_type", ""),
                "model": r.get("model", ""),
                "variant": r.get("variant", ""),
                "family": fam,
                "condition": r.get("condition", ""),
                "severity": int(severity),
                "direction": r.get("direction", ""),
                "compound_relative_drop_pct": compound,
                "component_max_condition": r.get("component_max_condition", ""),
                "component_max_relative_drop_pct": component_rel,
                "excess_pp": excess,
                "excess_ratio": parse_float(r.get("compound_excess_ratio_vs_component_max", "")),
                "label": label_cell(r.get("dataset", ""), r.get("model", ""), r.get("variant", "")),
                "mechanism": mechanism_for_model(r.get("model", "")),
            })
    return rows


def select_rows(candidates: list[dict], mode: str) -> list[dict]:
    # One row per dataset/model/variant/family. Pick the direction/severity with largest excess.
    by_key: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in candidates:
        key = (r["dataset"], r["model"], r["variant"], r["family"])
        by_key.setdefault(key, []).append(r)
    selected = []
    for key, vals in by_key.items():
        vals = sorted(vals, key=lambda x: (x["excess_pp"], x["compound_relative_drop_pct"]), reverse=True)
        out = dict(vals[0])
        out["selection_mode"] = mode
        selected.append(out)
    return selected


def write_values(path: Path, selected: list[dict]) -> None:
    cols = [
        "selection_mode", "family", "dataset", "task_type", "model", "variant", "label",
        "condition", "severity", "direction", "compound_relative_drop_pct",
        "component_max_condition", "component_max_relative_drop_pct", "excess_pp", "excess_ratio",
        "mechanism",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(selected, key=lambda x: (x["family"], -x["excess_pp"], x["label"])):
            row = dict(r)
            for k in ("compound_relative_drop_pct", "component_max_relative_drop_pct", "excess_pp", "excess_ratio"):
                if isinstance(row.get(k), float):
                    row[k] = f"{row[k]:.5f}"
            w.writerow({c: row.get(c, "") for c in cols})


def plot(
    path: Path,
    selected: list[dict],
    mode: str,
    max_rows_per_family: Optional[int],
    dpi: int,
    plot_style: str,
    color_by: str,
) -> None:
    families = ["C1", "C2"]
    data = {}
    tails = {}
    for fam in families:
        vals = [r for r in selected if r["family"] == fam]
        vals.sort(key=lambda x: x["excess_pp"], reverse=True)
        if max_rows_per_family and len(vals) > max_rows_per_family:
            tails[fam] = vals[max_rows_per_family:]
            vals = vals[:max_rows_per_family]
        else:
            tails[fam] = []
        data[fam] = vals

    max_n = max((len(v) for v in data.values()), default=1)
    fig_h = max(4.2, 0.43 * max_n + 1.55)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, fig_h), sharex=False)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        vals = data[fam]
        labels = [r["label"] for r in vals]
        y = np.arange(len(vals))
        excess = [r["excess_pp"] for r in vals]
        if color_by == "mechanism":
            colors = [MECHANISM_COLORS.get(r["mechanism"], MECHANISM_COLORS["Other"]) if x >= 0 else NEG_COLOR for r, x in zip(vals, excess)]
        else:
            colors = [FAMILY_COLOR[fam] if x >= 0 else NEG_COLOR for x in excess]
        ax.barh(y, excess, color=colors, alpha=0.96, edgecolor="white", linewidth=0.7)
        ax.axvline(0, color="#222222", linewidth=1.0)
        ax.set_yticks(y, labels, fontsize=8.4 if plot_style == "main" else 7.2)
        ax.tick_params(axis="y", colors="#111111")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_color("#111111")
        ax.tick_params(axis="x", colors="#111111", labelsize=8.8 if plot_style == "main" else 7.8)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.invert_yaxis()
        if fam == "C1":
            baseline = "worst primitive (P1/P2/P4)"
        else:
            baseline = "worst C2 component"
        if plot_style == "main":
            panel_prefix = "A." if fam == "C1" else "B."
            ax.set_title(f"{panel_prefix} Top {len(vals)} {fam}: excess over {baseline}", fontsize=11.5, fontweight="bold")
        else:
            ax.set_title(f"{fam}: excess over {baseline}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Excess relative drop (percentage points)", fontsize=9.5, fontweight="bold")
        ax.grid(True, axis="x", color="#D0D0D0", linewidth=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Systematically annotate every displayed bar with selected severity/direction.
        for i, r in enumerate(vals):
            x = r["excess_pp"]
            ha = "left" if x >= 0 else "right"
            offset = 0.4 if x >= 0 else -0.4
            tag = f"S{r['severity']}{direction_tag(r['direction'])}"
            ax.text(x + offset, i, f"{tag}", va="center", ha=ha, fontsize=7.8 if plot_style == "main" else 6.8, color="#111111", fontweight="bold")
        if plot_style == "main" and tails[fam]:
            nonzero_tail = [r for r in tails[fam] if abs(r["excess_pp"]) > 1e-9]
            if nonzero_tail:
                min_tail = min(r["excess_pp"] for r in nonzero_tail)
                max_tail = max(r["excess_pp"] for r in nonzero_tail)
                tail_msg = f"remaining {len(tails[fam])}: {min_tail:.1f} to {max_tail:.1f} pp"
            else:
                tail_msg = f"remaining {len(tails[fam])}: ~0 pp"
            ax.text(
                0.99, 0.02, tail_msg,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.2,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
            )

    subtitle = "S3 only" if mode == "s3" else "maximum over S1/S2/S3 and signed directions"
    if plot_style == "main":
        suptitle = ""
    else:
        suptitle = f"Compound-vs-primitive interaction excess ({subtitle})"
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    if color_by == "mechanism":
        from matplotlib.patches import Patch

        mechanisms = []
        for fam_vals in data.values():
            for r in fam_vals:
                if r["mechanism"] not in mechanisms:
                    mechanisms.append(r["mechanism"])
        handles = [Patch(facecolor=MECHANISM_COLORS[m], edgecolor="#111111", linewidth=0.35, label=m) for m in mechanisms]
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(4, len(handles)),
            frameon=False,
            fontsize=8.8,
            bbox_to_anchor=(0.5, -0.01),
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_color("#111111")
        bottom = 0.14
    else:
        bottom = 0.04
    top = 0.98 if plot_style == "main" else 0.95
    fig.tight_layout(rect=(0, bottom, 1, top), w_pad=1.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("results/rspdbench_aggregate.csv"))
    ap.add_argument("--mode", choices=["s3", "max"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--values-output", type=Path, required=True)
    ap.add_argument("--max-rows-per-family", type=int, default=35, help="Limit plotted rows per family; CSV always contains all selected rows.")
    ap.add_argument("--plot-style", choices=["appendix", "main"], default="appendix")
    ap.add_argument("--color-by", choices=["family", "mechanism"], default="family")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    candidates = read_candidates(args.input, args.mode)
    selected = select_rows(candidates, args.mode)
    write_values(args.values_output, selected)
    plot(args.output, selected, args.mode, args.max_rows_per_family, args.dpi, args.plot_style, args.color_by)
    print(f"selected rows: {len(selected)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.values_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
