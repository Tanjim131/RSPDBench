"""Robustness metrics used in the paper."""
from __future__ import annotations

from typing import Mapping


def relative_drop(clean: float, degraded: float) -> float:
    """Relative drop in percent, (M_clean - M_d) / M_clean x 100; negative values mean improvement."""
    return (clean - degraded) / clean * 100.0


C1_COMPONENTS = ("p1_s{s}", "p2_gain_{d}_s{s}", "p4_s{s}")
C2_COMPONENTS = ("c2_resolution_only_s{s}", "c2_gain_only_{d}_s{s}")


def compound_excess(drops: Mapping[str, float], chain: str) -> float:
    """Excess (pp) of a chain condition over its strongest matched component.

    `drops` maps condition names to relative drops of one model; `chain` is e.g.
    'c1_gain_down_s2' or 'c2_gain_up_s3'. Components are matched in severity and gain direction.
    """
    parts = chain.split("_")
    s = int(parts[-1][1:])
    d = parts[2]
    comps = C1_COMPONENTS if chain.startswith("c1_") else C2_COMPONENTS
    return drops[chain] - max(drops[c.format(s=s, d=d)] for c in comps)
