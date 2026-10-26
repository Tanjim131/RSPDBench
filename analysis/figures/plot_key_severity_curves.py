#!/usr/bin/env python3
"""Plot focused C1 severity-law trajectories from canonical 3-seed aggregates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_ALIASES = {
    "p1_spatial_resolution": "P1", "P1": "P1",
    "p2_uniform_gain": "P2", "P2": "P2",
    "p3_band_misregistration": "P3", "P3": "P3",
    "p4_sensor_noise": "P4", "P4": "P4",
    "c1_acquisition_quality_chain": "C1", "C1": "C1",
    "c2_harmonization_chain": "C2", "C2": "C2",
    "c2_component_gain_only": "C2_gain_only", "C2_solo": "C2_gain_only",
    "c2_component_resolution_only": "C2_resolution_only",
}

MODEL_LABELS = {
    "dinov3_vit7b_web": "DINOv3-Web",
    "dinov3_vit7b_sat": "DINOv3-Sat",
    "dofa_v1_vit_large_e100": "DOFA",
    "galileo_base": "Galileo",
    "panopticon_vitb14": "Panopticon",
    "prithvi_eo_v2_600": "Prithvi 2.0",
    "ramen_base": "RAMEN",
}
DATASET_LABELS = {
    "m-eurosat": "m-EuroSAT",
    "m-bigearthnet": "m-BigEarthNet",
    "m-SA-crop-type": "m-SA-crop",
    "hls-burn-scars": "HLS-Burn-Scar",
    "m-chesapeake": "m-Chesapeake",
}

COLOR = {
    "P1": "#1f77b4",
    "P2": "#ff7f00",
    "P3": "#7570b3",
    "P4": "#2ca02c",
    "C1": "#d62728",
    "C2": "#1f78b4",
    "C2 gain-only": "#6baed6",
    "C2 resolution-only": "#08519c",
    "HLS-Burn-Scar": "#9467bd",
    "m-SA-crop": "#ff7f0e",
}

MARKER = {
    "P1": "s",
    "P2 down": "^",
    "P4": "o",
    "C1 down": "*",
    "HLS-Burn-Scar": "D",
    "m-SA-crop": "X",
}


def ffloat(v: str) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    return out


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("complete_seed_coverage", "").lower() != "true":
                continue
            if r.get("is_clean", "").lower() == "true":
                continue
            rel = ffloat(r.get("relative_drop_pct_mean") or r.get("relative_drop_pct_from_mean") or "")
            sev = ffloat(r.get("severity", ""))
            if rel is None or sev is None or int(sev) not in (1, 2, 3):
                continue
            fam = FAMILY_ALIASES.get(r.get("condition_family", ""))
            if fam is None:
                continue
            rr = dict(r)
            rr["rel"] = rel
            rr["sev_int"] = int(sev)
            rr["family_norm"] = fam
            rows.append(rr)
    return rows


def curve(rows: List[dict], dataset: str, model: str, family: str, direction: Optional[str] = None) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for sev in (1, 2, 3):
        vals = [
            r["rel"] for r in rows
            if r["dataset"] == dataset
            and r["model"] == model
            and r["family_norm"] == family
            and r["sev_int"] == sev
            and (direction is None or r.get("direction", "") == direction)
        ]
        out.append(max(vals) if vals else None)
    return out


def excess_curve(
    rows: List[dict],
    dataset: str,
    model: str,
    compound_family: str,
    primitive_family: str,
    compound_direction: Optional[str] = None,
) -> List[Optional[float]]:
    compound = curve(rows, dataset, model, compound_family, compound_direction)
    primitive = curve(rows, dataset, model, primitive_family, None)
    out: List[Optional[float]] = []
    for c, p in zip(compound, primitive):
        out.append(None if c is None or p is None else c - p)
    return out


def write_values_csv(path: Path, plotted: List[Tuple[str, str, str, str, List[Optional[float]]]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "dataset", "model", "line", "S1", "S2", "S3"])
        for panel, dataset, model, label, vals in plotted:
            w.writerow([panel, dataset, model, label] + ["" if v is None else f"{v:.5f}" for v in vals])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("results/rspdbench_aggregate.csv"))
    ap.add_argument("--output", type=Path, default=Path("figures/key_severity_curves.png"))
    ap.add_argument("--values-output", type=Path, default=Path("figures/key_severity_curves_values.csv"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows = load_rows(args.input)
    panels = [
        {
            "id": "A",
            "dataset": "m-SA-crop-type",
            "model": "dinov3_vit7b_sat",
            "title": "C1 amplifies then saturates",
            "lines": [("P1", "P1", None), ("P2 down", "P2", "down"), ("P4", "P4", None), ("C1 down", "C1", "down")],
            "note": "DINOv3-Sat / m-SA-crop",
            "ylabel": "Relative drop (%)",
        },
        {
            "id": "B",
            "dataset": "hls-burn-scars",
            "model": "dinov3_vit7b_web",
            "title": "C1 follows P1",
            "lines": [("P1", "P1", None), ("P2 down", "P2", "down"), ("P4", "P4", None), ("C1 down", "C1", "down")],
            "note": "DINOv3-Web / HLS-Burn-Scar",
            "ylabel": "Relative drop (%)",
        },
        {
            "id": "C",
            "title": "Same model, dataset flip",
            "note": "DINOv3-Sat / HLS-Burn-Scar vs m-SA-crop",
            "ylabel": "C1 excess over P1 (pp)",
            "excess_lines": [
                ("HLS-Burn-Scar", "hls-burn-scars", "dinov3_vit7b_sat", "C1", "P1", "down"),
                ("m-SA-crop", "m-SA-crop-type", "dinov3_vit7b_sat", "C1", "P1", "down"),
            ],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharex=True)
    plotted = []
    x = [1, 2, 3]
    for ax, panel in zip(axes, panels):
        ymin, ymax = 0.0, 0.0
        if "excess_lines" in panel:
            line_specs = []
            for label, dataset, model, compound_family, primitive_family, direction in panel["excess_lines"]:
                vals = excess_curve(rows, dataset, model, compound_family, primitive_family, direction)
                line_specs.append((label, dataset, model, vals, COLOR.get(label, "#333333")))
        else:
            line_specs = []
            for label, fam, direction in panel["lines"]:
                vals = curve(rows, panel["dataset"], panel["model"], fam, direction)
                color = COLOR.get(label, COLOR.get(fam.replace("_", " "), "#333333"))
                if "C1" in label:
                    color = COLOR["C1"]
                elif label.startswith("P"):
                    color = COLOR.get(label.split()[0], color)
                line_specs.append((label, panel["dataset"], panel["model"], vals, color))

        for label, dataset, model, vals, color in line_specs:
            plotted.append((panel["id"], dataset, model, label, vals))
            yy = [float("nan") if v is None else v for v in vals]
            if "C1" in label:
                color = COLOR["C1"]
            elif label.startswith("P"):
                color = COLOR.get(label.split()[0], color)
            ax.plot(
                x,
                yy,
                marker=MARKER.get(label, "o"),
                linewidth=2.2,
                markersize=6.4,
                label=label,
                color=color,
            )
            present = [v for v in vals if v is not None]
            if present:
                ymin = min(ymin, min(present))
                ymax = max(ymax, max(present))
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, ["S1", "S2", "S3"], fontsize=10, fontweight="bold")
        ax.set_title(f"{panel['id']}. {panel['title']}", fontsize=10.5, fontweight="bold")
        ax.text(
            0.5,
            -0.24,
            panel["note"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#444444",
        )
        ax.set_ylabel(panel["ylabel"], fontsize=9.5, fontweight="bold")
        pad = max(2.0, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, 0.88))

    fig.tight_layout(w_pad=1.0, h_pad=1.25)
    fig.subplots_adjust(bottom=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    write_values_csv(args.values_output, plotted)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.values_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
