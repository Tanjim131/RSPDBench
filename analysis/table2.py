#!/usr/bin/env python3
"""Paper Tab. 2: mean / median / max relative drop of each primitive over all model-dataset cells.

P2 uses the worse gain direction of each cell, as in Fig. 3.
Usage: python analysis/table2.py [results/rspdbench_aggregate.csv]
"""
import sys

import numpy as np
import pandas as pd

FAMILIES = {"P1": ["p1_s{s}"], "P2": ["p2_gain_down_s{s}", "p2_gain_up_s{s}"], "P3": ["p3_s{s}"], "P4": ["p4_s{s}"]}


def main(path: str = "results/rspdbench_aggregate.csv") -> None:
    a = pd.read_csv(path)
    print(f"{'':4s}{'':4s}{'n':>4s}{'mean':>8s}{'median':>8s}{'max':>8s}")
    for fam, pats in FAMILIES.items():
        for s in (1, 2, 3):
            vals = []
            for _, g in a.groupby(["dataset", "model"]):
                by = g.set_index("condition")["relative_drop_pct_mean"]
                conds = [p.format(s=s) for p in pats if p.format(s=s) in by.index]
                if conds:
                    vals.append(max(by[c] for c in conds))
            v = np.array(vals)
            print(f"{fam:4s}S{s:<3d}{len(v):>4d}{v.mean():>8.1f}{np.median(v):>8.1f}{v.max():>8.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
