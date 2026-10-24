#!/usr/bin/env python3
"""Plot WACV degradation fingerprint heatmaps from canonical 3-seed aggregates.

The main figure uses one shared linear color scale across all datasets/families so
color represents absolute relative-drop magnitude. A companion normalized figure
can be emitted where each dataset-family column is independently scaled; that
figure is for within-axis contrast only.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath


FAMILIES = ["P1", "P2", "P3", "P4", "C1", "C2"]
DATASET_ORDER = ["m-eurosat", "m-bigearthnet", "m-SA-crop-type", "m-chesapeake", "hls-burn-scars"]
DATASET_LABELS = {
    "m-eurosat": "m-EuroSAT",
    "m-bigearthnet": "m-BigEarthNet",
    "m-SA-crop-type": "m-SA-crop",
    "hls-burn-scars": "HLS-Burn-Scar",
    "m-chesapeake": "m-Chesapeake",
}
MODEL_ORDER = [
    "dinov3_vit7b_web",
    "dinov3_vit7b_sat",
    "dofa_v1_vit_large_e100",
    "galileo_base",
    "panopticon_vitb14",
    "prithvi_eo_v2_600",
    "ramen_base",
    "vit_b16_scratch",
    "unet_scratch",
]
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

DATASET_TASK_TYPE = {
    "m-eurosat": "classification",
    "m-bigearthnet": "classification",
    "m-SA-crop-type": "segmentation",
    "hls-burn-scars": "segmentation",
    "m-chesapeake": "segmentation",
}

MODEL_TASK_APPLICABILITY = {
    "vit_b16_scratch": {"classification"},
    "unet_scratch": {"segmentation"},
}

FROZEN_ENCODER_MODELS = {
    "dinov3_vit7b_web",
    "dinov3_vit7b_sat",
    "dofa_v1_vit_large_e100",
    "galileo_base",
    "panopticon_vitb14",
    "prithvi_eo_v2_600",
    "ramen_base",
}

SUPERVISED_BASELINE_MODELS = {
    "vit_b16_scratch",
    "unet_scratch",
}

FIRE_FLAME_CURVED_VERTICES = [
    (281.6, 93.9),
    (297.6, 72.6),
    (301.6, 67.2), (308.0, 64.0), (314.7, 64.0),
    (326.4, 64.0), (336.0, 73.6), (336.0, 85.3),
    (336.0, 107.4),
    (336.0, 120.5), (341.4, 133.1), (350.9, 142.1),
    (435.6, 223.0),
    (484.4, 269.6), (512.0, 334.2), (512.0, 401.7),
    (512.0, 498.0), (434.0, 576.0), (337.7, 576.0),
    (320.0, 576.0),
    (214.0, 576.0), (128.0, 490.0), (128.0, 384.0),
    (128.0, 380.2),
    (128.0, 331.4), (147.4, 284.6), (181.9, 250.1),
    (185.4, 246.6),
    (189.6, 242.4), (195.4, 240.0), (201.4, 240.0),
    (213.9, 240.0), (224.0, 250.1), (224.0, 262.6),
    (224.0, 352.0),
    (224.0, 387.3), (252.7, 416.0), (288.0, 416.0),
    (323.3, 416.0), (352.0, 387.3), (352.0, 352.0),
    (352.0, 348.1),
    (352.0, 330.1), (344.8, 312.8), (332.1, 300.1),
    (293.5, 261.5),
    (269.5, 237.5), (256.0, 204.8), (256.0, 170.8),
    (256.0, 143.1), (265.0, 116.0), (281.6, 93.9),
    (281.6, 93.9),
]
FIRE_FLAME_CURVED_CODES = [
    MplPath.MOVETO,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.LINETO,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    MplPath.CLOSEPOLY,
]


def display_model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def add_flame_marker(fig, xy: Tuple[float, float], *, size: int = 10, zorder: int = 8) -> None:
    """Draw Font Awesome's fa-fire-flame-curved SVG path in figure coordinates."""
    da = DrawingArea(size, size, 0, 0)
    scale = size / 640.0
    verts = [(x * scale, (640.0 - y) * scale) for x, y in FIRE_FLAME_CURVED_VERTICES]
    path = MplPath(verts, FIRE_FLAME_CURVED_CODES)
    patch = PathPatch(path, facecolor="#D55E00", edgecolor="none")
    da.add_artist(patch)
    ab = AnnotationBbox(da, xy, xycoords=fig.transFigure, frameon=False, box_alignment=(0.0, 0.45), zorder=zorder)
    ab.set_clip_on(False)
    fig.add_artist(ab)

FAMILY_ALIASES = {
    "p1_spatial_resolution": "P1",
    "P1": "P1",
    "p2_uniform_gain": "P2",
    "P2": "P2",
    "p3_band_misregistration": "P3",
    "P3": "P3",
    "p4_sensor_noise": "P4",
    "P4": "P4",
    "c1_acquisition_quality_chain": "C1",
    "C1": "C1",
    "c2_harmonization_chain": "C2",
    "C2": "C2",
}

# Component-only rows are excluded from the headline C2 family cell.
EXCLUDED_FAMILIES = {"c2_component_gain_only", "c2_component_resolution_only", "C2_solo", "none", "clean"}


@dataclass(frozen=True)
class Key:
    dataset: str
    model: str
    family: str


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "na", "n/a"}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def model_key(row: dict) -> str:
    model = row.get("model", "")
    variant = row.get("variant", "")
    if model == "dinov3_vit7b" and variant in {"web", "sat"}:
        return f"dinov3_vit7b_{variant}"
    return model


def family_key(row: dict) -> Optional[str]:
    fam = row.get("condition_family", "")
    if fam in EXCLUDED_FAMILIES:
        return None
    return FAMILY_ALIASES.get(fam)


def read_aggregate(path: Path, severity: Optional[int] = 3) -> Tuple[Dict[Key, float], set[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    """Return max relative drop per dataset/model/family from complete 3-seed rows.

    If severity is set, only rows at that severity are used. Signed families still
    take the worse direction at that severity.
    """
    values: Dict[Key, float] = {}
    completed_runs: set[Tuple[str, str]] = set()
    condition_counts: Dict[Tuple[str, str], int] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("complete_seed_coverage", "").lower() != "true":
                continue
            dataset = row.get("dataset", "")
            model = model_key(row)
            if not dataset or not model:
                continue
            if row.get("is_clean", "").lower() == "true":
                completed_runs.add((dataset, model))
                cc = parse_float(row.get("condition_count", ""))
                if cc is not None:
                    condition_counts[(dataset, model)] = int(cc)
                continue
            fam = family_key(row)
            if fam is None:
                continue
            if severity is not None:
                row_severity = parse_float(row.get("severity", ""))
                if row_severity is None or int(row_severity) != int(severity):
                    continue
            rel = parse_float(row.get("relative_drop_pct_mean", ""))
            if rel is None:
                rel = parse_float(row.get("relative_drop_pct_from_mean", ""))
            if rel is None:
                continue
            key = Key(dataset, model, fam)
            # Family fingerprint = worst 3-seed-average drop over directions at the selected severity.
            values[key] = max(values.get(key, -float("inf")), rel)

    return values, completed_runs, condition_counts


def resolve_order(requested: Optional[str], default: List[str], available: Iterable[str]) -> List[str]:
    available_set = set(available)
    if requested:
        raw = [x.strip() for x in requested.split(",") if x.strip()]
        return raw
    out = [x for x in default if x in available_set]
    out.extend(sorted(available_set - set(out)))
    return out


def cell_status(
    dataset: str,
    model: str,
    family: str,
    completed_runs: set[Tuple[str, str]],
    condition_counts: Dict[Tuple[str, str], int],
) -> str:
    task_type = DATASET_TASK_TYPE.get(dataset)
    allowed_tasks = MODEL_TASK_APPLICABILITY.get(model)
    if allowed_tasks is not None and task_type not in allowed_tasks:
        return "na"
    if (dataset, model) not in completed_runs:
        return "missing"
    # Only mark invalid-by-contract cells as n/a. In this benchmark those are P3/C2
    # for completed 19-condition RGB/RGB-NIR runs. Other absent cells are treated as
    # not-yet-computed/blank rather than invalid.
    if family in {"P3", "C2"} and condition_counts.get((dataset, model)) == 19:
        return "na"
    return "missing"


def format_value(x: float) -> str:
    if abs(x) >= 10:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.1f}"
    return f"{x:.1f}"


def draw_heatmap(
    values: Dict[Key, float],
    completed_runs: set[Tuple[str, str]],
    condition_counts: Dict[Tuple[str, str], int],
    datasets: List[str],
    models: List[str],
    output: Path,
    normalized: bool,
    vmin: float,
    vmax: float,
    title: str,
    dpi: int,
    severity: Optional[int],
    group_compounds: bool,
) -> None:
    nrows = len(models)
    ncols = len(FAMILIES)
    width_per_dataset = 0.34 if group_compounds else 0.28
    fig_w = max(12.0, 1.35 * len(datasets) + width_per_dataset * len(datasets) * ncols)
    fig_h = max(4.2, 0.42 * nrows + 1.45)
    fig, axes = plt.subplots(1, len(datasets), figsize=(fig_w, fig_h), sharey=True, constrained_layout=False)
    if len(datasets) == 1:
        axes = [axes]

    if normalized:
        cmap = plt.get_cmap("OrRd").copy()
    else:
        # Avoid the darkest OrRd values; they obscure numeric annotations.
        base = plt.get_cmap("OrRd")
        cmap = LinearSegmentedColormap.from_list("OrRd_soft", base(np.linspace(0.02, 0.78, 256)))
    norm = Normalize(vmin=0, vmax=(1.0 if normalized else vmax))
    cmap.set_bad(color="#f7f7f7")
    negative_cmap = plt.get_cmap("Greens")

    # Negative relative drops are improvements. They are drawn with a separate
    # green scale computed from the plotted negative values, but no separate
    # colorbar is shown because the paper's primary colorbar is the 0--100% drop
    # scale.
    negative_values = [
        abs(values[Key(ds, model, fam)])
        for ds in datasets
        for model in models
        for fam in FAMILIES
        if Key(ds, model, fam) in values and values[Key(ds, model, fam)] < 0
    ]
    negative_abs_max = max(negative_values) if negative_values else 1.0

    # Column maxima for marker outlines and normalized scaling.
    maxima: Dict[Tuple[str, str], float] = {}
    for ds in datasets:
        for fam in FAMILIES:
            vals = [values[Key(ds, m, fam)] for m in models if Key(ds, m, fam) in values]
            if vals:
                maxima[(ds, fam)] = max(vals)

    im = None
    for ax_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        ax.set_facecolor("#FFFFFF")
        mat = np.full((nrows, ncols), np.nan, dtype=float)
        display_x = np.array([0, 1, 2, 3, 4.5, 5.5], dtype=float) if group_compounds else np.arange(ncols, dtype=float)
        cell_w = 1.0
        x_min = float(display_x[0] - cell_w / 2)
        x_max = float(display_x[-1] + cell_w / 2)
        text_vals: Dict[Tuple[int, int], float] = {}
        status: Dict[Tuple[int, int], str] = {}
        for i, model in enumerate(models):
            for j, fam in enumerate(FAMILIES):
                key = Key(ds, model, fam)
                if key in values:
                    raw = values[key]
                    text_vals[(i, j)] = raw
                    if normalized:
                        denom = maxima.get((ds, fam), 0.0)
                        mat[i, j] = 0.0 if denom <= 0 else max(raw, 0.0) / denom
                    else:
                        mat[i, j] = 0.0 if raw < 0 else min(max(raw, 0.0), vmax)
                    status[(i, j)] = "value"
                else:
                    status[(i, j)] = cell_status(ds, model, fam, completed_runs, condition_counts)

        if group_compounds:
            # Insert a real blank spacer column after P4, so the break reads as
            # primitives <space>|<space> compounds rather than as a cell border.
            x_edges = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.0, 5.0, 6.0], dtype=float)
            y_edges = np.arange(-0.5, nrows + 0.5, 1.0)
            mat_plot = np.full((nrows, ncols + 1), np.nan, dtype=float)
            mat_plot[:, :4] = mat[:, :4]
            mat_plot[:, 5:] = mat[:, 4:]
            im = ax.pcolormesh(x_edges, y_edges, mat_plot, cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=1.0)
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(nrows - 0.5, -0.5)
        else:
            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xlim(x_min, x_max)
        title_pad = 42 if group_compounds else 7
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold", pad=title_pad)
        ax.set_xticks(display_x, FAMILIES, fontsize=12, fontweight="bold")
        ax.set_yticks(range(nrows), [display_model_label(m) for m in models], fontsize=12, fontweight="bold")
        if ax_idx != 0:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.tick_params(axis="y", pad=14)

        if group_compounds:
            ax.set_xticks(np.arange(-0.5, ncols + 1, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)
        else:
            ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if group_compounds:
            # P1-P4 are primitives; C1-C2 are physically structured compound chains.
            # A true spacer plus separator helps readers parse super-additivity.
            ax.add_patch(Rectangle((3.5, -0.5), 0.5, nrows, facecolor="white", edgecolor="none", zorder=2))
            ax.axvline(3.75, color="#333333", linewidth=1.1, zorder=7)
            trans = ax.get_xaxis_transform()
            ax.text(
                1.5,
                1.015,
                "Primitives",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                transform=trans,
                clip_on=False,
            )
            ax.text(
                5.0,
                1.015,
                "Compounds",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                transform=trans,
                clip_on=False,
            )

        for i in range(nrows):
            for j, fam in enumerate(FAMILIES):
                x = float(display_x[j])
                st = status[(i, j)]
                if st == "value":
                    raw = text_vals[(i, j)]
                    if raw < 0 and not normalized:
                        frac = min(abs(raw) / negative_abs_max, 1.0) if negative_abs_max > 0 else 0.0
                        # Keep negative/improvement cells light-to-medium green so text remains readable.
                        face = negative_cmap(0.12 + 0.38 * frac)
                        ax.add_patch(Rectangle((x - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", linewidth=0.5, zorder=3))
                    ax.text(x, i, format_value(raw), ha="center", va="center", fontsize=10, color="#222222", zorder=5)
                    maxv = maxima.get((ds, fam))
                    if maxv is not None and abs(raw - maxv) < 1e-9 and maxv > 0:
                        # Inset and thicken the outline so it remains visible over the cell grid.
                        ax.add_patch(Rectangle((x - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor="#111111", linewidth=1.8, zorder=6))
                else:
                    if st == "na":
                        # Invalid-by-contract/task-inapplicable cells: gray fill with X marker.
                        ax.add_patch(Rectangle((x - 0.5, i - 0.5), 1, 1, facecolor="#bdbdbd", edgecolor="white", linewidth=0.5))
                        ax.text(x, i, "X", ha="center", va="center", fontsize=10.5, color="#3a3a3a", fontweight="bold", zorder=5)
                    else:
                        # Valid but not yet computed: blank.
                        ax.add_patch(Rectangle((x - 0.5, i - 0.5), 1, 1, facecolor="#f7f7f7", edgecolor="white", linewidth=0.5))

    top = 0.82 if group_compounds else 0.90
    wspace = 0.11 if group_compounds else 0.08
    fig.subplots_adjust(left=0.085, right=0.925, bottom=0.10, top=top, wspace=wspace)

    if len(axes) > 1:
        # Each dataset is a separate subplot. The visual center of the dataset
        # break is left of the raw axes-gap midpoint because the compounds block
        # and subplot title whitespace make the gap read asymmetrically.
        y0 = min(ax.get_position().y0 for ax in axes)
        y1 = max(ax.get_position().y1 for ax in axes)
        for left_ax, right_ax in zip(axes[:-1], axes[1:]):
            gap_x0 = left_ax.get_position().x1
            gap_x1 = right_ax.get_position().x0
            gap_w = gap_x1 - gap_x0
            gutter_center = gap_x0 + gap_w * 0.18
            gutter_w = min(0.009, max(0.0045, gap_w * 0.30))
            gutter_x = gutter_center - gutter_w / 2.0
            fig.add_artist(
                Rectangle(
                    (gutter_x, y0),
                    gutter_w,
                    y1 - y0,
                    transform=fig.transFigure,
                    facecolor="#E7F1F8",
                    edgecolor="#7EA1B9",
                    hatch="///",
                    linewidth=0.35,
                    alpha=0.88,
                    zorder=-5,
                    clip_on=False,
                )
            )

    # Align colorbar height with the heatmap row extent.
    first_pos = axes[0].get_position()
    cax = fig.add_axes([0.94, first_pos.y0, 0.012, first_pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("within-family normalized" if normalized else "relative drop (%)", fontsize=8)

    # Draw frozen-encoder markers as small blue superscripts anchored to the
    # rendered model labels. This keeps them close to the text like the dagger
    # marker, instead of tying their position to the heatmap/tick coordinate
    # system.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_ax = axes[0]
    for tick_label, model in zip(label_ax.get_yticklabels(), models):
        bbox = tick_label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        if model in FROZEN_ENCODER_MODELS:
            fig.text(
                bbox.x1 - 0.0005,
                bbox.y0 + bbox.height * 0.68,
                "\u2744",
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#159BD3",
            )
        elif model in SUPERVISED_BASELINE_MODELS:
            x = bbox.x1 - 0.0004
            y = bbox.y0 + bbox.height * 0.72
            add_flame_marker(fig, (x, y), size=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/rspdbench_aggregate.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/degradation_fingerprint_linear.png"))
    parser.add_argument("--normalized-output", type=Path, default=Path("figures/degradation_fingerprint_normalized.png"))
    parser.add_argument("--also-normalized", action="store_true", help="Also write the per-column normalized companion figure.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset order. Defaults to WACV order.")
    parser.add_argument("--models", default=None, help="Comma-separated model keys. Defaults to known WACV model order present in the input.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Deprecated; retained for CLI compatibility. Negative cells are green-scaled from observed negative values.")
    parser.add_argument("--vmax", type=float, default=100.0, help="Maximum for the positive-drop color scale in percent. Default: 100.")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3], help="Severity level to plot. Default: 3 (S3).")
    parser.add_argument("--title", default="Degradation fingerprints: relative drop by model x family")
    parser.add_argument("--group-compounds", action="store_true", help="Add a P1-P4 primitive / C1-C2 compound visual split.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    values, completed_runs, condition_counts = read_aggregate(args.input, severity=args.severity)
    available_datasets = {k.dataset for k in values} | {ds for ds, _ in completed_runs}
    available_models = {k.model for k in values} | {m for _, m in completed_runs}
    datasets = resolve_order(args.datasets, DATASET_ORDER, available_datasets)
    models = resolve_order(args.models, MODEL_ORDER, available_models)

    if not datasets:
        raise SystemExit("No datasets selected/found.")
    if not models:
        raise SystemExit("No models selected/found.")

    draw_heatmap(values, completed_runs, condition_counts, datasets, models, args.output, normalized=False, vmin=args.vmin, vmax=args.vmax, title=args.title, dpi=args.dpi, severity=args.severity, group_compounds=args.group_compounds)
    print(f"Wrote {args.output}")
    if args.also_normalized:
        draw_heatmap(values, completed_runs, condition_counts, datasets, models, args.normalized_output, normalized=True, vmin=args.vmin, vmax=args.vmax, title=args.title, dpi=args.dpi, severity=args.severity, group_compounds=args.group_compounds)
        print(f"Wrote {args.normalized_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
