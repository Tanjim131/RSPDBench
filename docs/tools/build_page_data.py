#!/usr/bin/env python3
"""Build docs/assets/data.json for the project page from the shipped results.

Inputs: results/rspdbench_aggregate.csv, results/realism_s2_b08_b8a_summary.csv, and the
degradation-aware probe summary in docs/tools/inputs/.
Run from the repository root: python docs/tools/build_page_data.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
agg = pd.read_csv(ROOT / "results/rspdbench_aggregate.csv")
inp = ROOT / "docs/tools/inputs"

drops, clean, std = {}, {}, {}
for r in agg.itertuples():
    if r.condition == "clean":
        clean.setdefault(r.dataset, {})[r.model] = round(float(r.metric_mean) * 100, 2)
    else:
        drops.setdefault(r.dataset, {}).setdefault(r.model, {})[r.condition] = round(float(r.relative_drop_pct_mean), 2)
        std.setdefault(r.dataset, {}).setdefault(r.model, {})[r.condition] = round(float(r.relative_drop_pct_std), 2)

mit = pd.read_csv(inp / "mitigation_ben_family_s3.csv")
mitigation = {fam: {"original": round(g.original.mean(), 1), "aware": round(g.augprobe.mean(), 1), "n": int(len(g))} for fam, g in mit.groupby("family")}

real = pd.read_csv(ROOT / "results/realism_s2_b08_b8a_summary.csv")
realism = {k: [{"sigma": float(r.sigma), "r": float(r.median_r)} for r in g.itertuples()] for k, g in real.groupby("dataset")}

out = {"clean": clean, "drops": drops, "std": std, "mitigation": mitigation, "realism": realism}
(ROOT / "docs/assets/data.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote docs/assets/data.json", round(len(json.dumps(out)) / 1024, 1), "KB")
