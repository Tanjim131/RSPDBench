#!/usr/bin/env python3
"""Render the project-page explorer: one test scene per dataset under every condition.

Each scene is degraded with the released rspdbench operators in native product units,
using the clean training-set mean of each band for P4/C1, then shown as a fixed
false-color stretch taken from the clean image (so gain errors stay visible as
brightness changes and misregistration as colored edges).

    GEO_BENCH_DIR=/path/to/geobench python docs/tools/build_explorer_assets.py \
        --hls-tif /path/to/subsetted_512x512_HLS.S30.T11SNS.2020276.v1.4_merged.tif

Writes docs/assets/explorer/<dataset>/<condition>.(png|jpg) and docs/assets/explorer/scenes.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import rspdbench as rb  # noqa: E402

OUT = ROOT / "docs" / "assets" / "explorer"

WAVELENGTH_NM = {
    "B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740, "B07": 783, "B08": 842,
    "B8A": 865, "B09": 945, "B10": 1375, "B11": 1610, "B12": 2190,
    "BLUE": 490, "GREEN": 560, "RED": 665, "NIR_NARROW": 865, "SWIR_1": 1610, "SWIR_2": 2190, "NIR": 842,
}
S2_COMPOSITE = (["B12", "B8A", "B04"], "SWIR 2 (B12), narrow NIR (B8A) and red (B04) as R, G, B")

# One test scene per dataset. Place names are read off the scene-center coordinates.
SCENES = [
    {"key": "m-eurosat", "bench": "classification_v1.0", "index": 781, "label": "m-EuroSAT",
     "place": "Flevoland, Netherlands", "source": "Sentinel-2, 13 bands resampled to 10 m",
     "task": "land-cover classification", "composite": S2_COMPOSITE},
    {"key": "m-bigearthnet", "bench": "classification_v1.0", "index": 477, "label": "m-BigEarthNet",
     "place": "Algarve, Portugal", "source": "Sentinel-2A Level-2A, 12 bands",
     "task": "multi-label land-cover classification", "composite": S2_COMPOSITE,
     # sensing date from the BigEarthNet patch behind this GEO-Bench sample: S2A_MSIL2A_20171002T112112_73_81
     "date": "2017-10-02"},
    {"key": "m-SA-crop-type", "bench": "segmentation_v1.0", "index": 304, "label": "m-SA-crop",
     "place": "Western Cape, South Africa", "source": "Sentinel-2, 12 bands",
     "task": "crop-type segmentation", "composite": S2_COMPOSITE},
    {"key": "m-chesapeake", "bench": "segmentation_v1.0", "index": 390, "label": "m-Chesapeake",
     "place": "Chesapeake Bay watershed, USA", "source": "NAIP aerial imagery, red, green, blue and NIR",
     "task": "land-cover segmentation",
     "composite": (["NIR", "RED", "GREEN"], "NIR, red and green as R, G, B (color infrared)")},
]
HLS = {"key": "hls-burn-scars", "label": "HLS Burn Scar", "place": "San Diego County, California, USA",
       "source": "Harmonized Landsat Sentinel-2 (HLS S30), 6 bands", "task": "burn-scar segmentation",
       "composite": (["SWIR_2", "NIR_NARROW", "RED"], "SWIR 2, NIR and red as R, G, B"),
       # clean training-set mean per band (reflectance), PANGAEA HLS Burn Scars training split
       "band_mean": [0.05189, 0.07617, 0.09353, 0.20407, 0.23159, 0.17121]}

CHESAPEAKE_IDS = {"RED": "RED", "GREEN": "GREEN", "BLUE": "BLUE", "NEARINFRARED": "NIR"}
CARD_BOX = (272, 320, 96)  # left, top, size of the HLS crop shown on the defect cards


def stretch(img: np.ndarray, lo: np.ndarray, hi: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    return (np.clip((img - lo) / (hi - lo), 0, 1) ** gamma * 255).round().astype(np.uint8)


def band_ids_for(key: str, names: list[str]) -> list[str]:
    if key == "m-eurosat":
        return list(rb.MEUROSAT_STORAGE_ORDER)  # physical storage order, B8A last
    if key == "m-chesapeake":
        return [CHESAPEAKE_IDS[n.upper()] for n in names]
    return [rb.canonical_band_id(n) for n in names]


def geobench_scene(task, index: int, key: str):
    s = task.get_dataset(split="test")[index]
    img, _ = s.pack_to_3d(s.band_names)
    ids = band_ids_for(key, list(s.band_names))
    keep = [i for i, b in enumerate(ids) if b in WAVELENGTH_NM]  # drops m-SA-crop's cloud-probability layer
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)[:, :, keep]).permute(2, 0, 1).contiguous()
    band = s.bands[0]
    lat = lon = None
    if band.transform is not None and band.crs is not None:
        from rasterio.warp import transform as warp

        h, w = x.shape[1:]
        cx, cy = band.transform * (w / 2, h / 2)
        lons, lats = warp(band.crs, "EPSG:4326", [cx], [cy])
        lat, lon = round(lats[0], 4), round(lons[0], 4)
    return x, [ids[i] for i in keep], keep, str(s.sample_name), float(band.spatial_resolution), lat, lon


def train_mean(task, keep: list[int]) -> torch.Tensor:
    """Clean training-set mean per band, as in the paper's evaluation code."""
    ds = task.get_dataset(split="train")
    total, count = np.zeros(len(keep)), 0
    for i in range(len(ds)):
        s = ds[i]
        arr = np.asarray(s.pack_to_3d(s.band_names)[0], dtype=np.float64)[:, :, keep]
        total += arr.reshape(-1, len(keep)).sum(0)
        count += arr.shape[0] * arr.shape[1]
    return torch.tensor(total / count, dtype=torch.float32)


def render(meta: dict, x: torch.Tensor, band_ids: list[str], mean: torch.Tensor, sample_id: str) -> None:
    contract = rb.get_contract(meta["key"])
    out = OUT / meta["key"]
    out.mkdir(parents=True, exist_ok=True)
    pick = [band_ids.index(b) for b in meta["composite"][0]]
    clean = x[pick].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(clean, 2, axis=(0, 1)), np.percentile(clean, 98, axis=(0, 1))
    ext = "png" if max(x.shape[1:]) <= 128 else "jpg"
    names = []
    for cond in rb.conditions_for(contract, band_ids):
        if cond.family not in ("clean", "P1", "P2", "P3", "P4", "C1", "C2"):
            continue  # C2 component arms are analysis controls, not shown
        y = rb.degrade(x, cond, band_ids=band_ids, contract=contract, sample_id=sample_id, seed=42, band_mean=mean)
        im = Image.fromarray(stretch(y[pick].permute(1, 2, 0).numpy(), lo, hi))
        if ext == "png":
            im.save(out / f"{cond.name}.png", optimize=True)
        else:
            im.save(out / f"{cond.name}.jpg", quality=90, optimize=True, progressive=True)
        names.append(cond.name)
    h, w = x.shape[1:]
    gsd = meta["gsd_m"]
    meta.update({
        "ext": ext, "size": [int(h), int(w)], "extent_km": round(w * gsd / 1000, 2), "conditions": names,
        "bands": [{"id": b, "nm": WAVELENGTH_NM[b]} for b in band_ids],
        "composite": {"bands": meta["composite"][0], "text": meta["composite"][1]},
        "groups": {"p3": list(contract.p3_shifted), "c2_geo": list(contract.c2_geometric), "c2_rad": list(contract.c2_radiometric)},
        "multispectral": contract.multispectral(band_ids), "band_mean": [round(float(v), 5) for v in mean],
    })
    print(f"{meta['key']}: {len(names)} renders ({ext}), {h}x{w} px at {gsd} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hls-tif", required=True, type=Path)
    args = ap.parse_args()
    import geobench  # type: ignore
    import rasterio
    from rasterio.warp import transform as warp

    scenes = []
    for sc in SCENES:
        task = next(t for t in geobench.task_iterator(benchmark_name=sc["bench"]) if t.dataset_name == sc["key"])
        x, band_ids, keep, sid, gsd, lat, lon = geobench_scene(task, sc["index"], sc["key"])
        meta = {"date": None, **{k: v for k, v in sc.items() if k not in ("bench", "index")}}
        meta.update({"sample": sid, "gsd_m": gsd, "lat": lat, "lon": lon})
        render(meta, x, band_ids, train_mean(task, keep), sid)
        scenes.append(meta)

    with rasterio.open(args.hls_tif) as src:
        x = torch.from_numpy(src.read().astype(np.float32))
        cx, cy = src.transform * (src.width / 2, src.height / 2)
        lons, lats = warp(src.crs, "EPSG:4326", [cx], [cy])
        gsd = float(abs(src.transform.a))
    stem = args.hls_tif.stem  # ..._HLS.S30.T11SNS.2020276.v1.4_merged
    tile, yday = stem.split(".")[2], stem.split(".")[3]
    meta = {k: v for k, v in HLS.items() if k != "band_mean"}
    meta.update({"sample": f"tile {tile}", "gsd_m": gsd, "lat": round(lats[0], 4), "lon": round(lons[0], 4),
                 "date": datetime.strptime(yday, "%Y%j").date().isoformat()})
    render(meta, x, list(rb.get_contract("hls-burn-scars").bands), torch.tensor(HLS["band_mean"]), stem)
    scenes.append(meta)

    (OUT / "scenes.json").write_text(json.dumps(scenes, indent=1))
    print("wrote", OUT / "scenes.json")
    make_cards()


def make_cards(box: tuple[int, int, int] = CARD_BOX) -> None:
    """Card thumbnails: the same HLS crop clean and under each family at S3."""
    left, top, size = box
    out = OUT / "cards"
    out.mkdir(exist_ok=True)
    for fam, cond in [("clean", "clean"), ("p1", "p1_s3"), ("p2", "p2_gain_down_s3"), ("p3", "p3_s3"), ("p4", "p4_s3"),
                      ("c1", "c1_gain_down_s3"), ("c2", "c2_gain_down_s3")]:
        im = Image.open(OUT / "hls-burn-scars" / f"{cond}.jpg").crop((left, top, left + size, top + size))
        im.resize((192, 192), Image.NEAREST).save(out / f"{fam}.jpg", quality=90, optimize=True)
    print("wrote card thumbnails to", out)


if __name__ == "__main__":
    main()
