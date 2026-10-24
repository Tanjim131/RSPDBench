"""The 37 RSPDBench test conditions and their severity ladders (paper Tab. 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import BandContract

P1_SIGMA = {1: 0.5, 2: 1.0, 3: 2.0}      # PSF width in native pixels (sigma = N / 2)
P1_FACTOR = {1: 1, 2: 2, 3: 4}           # effective GSD factor N
P2_GAIN = {"down": {1: 0.97, 2: 0.92, 3: 0.85}, "up": {1: 1.03, 2: 1.08, 3: 1.15}}
P3_SHIFT_PX = {1: 0.3, 2: 1.0, 3: 2.0}   # displacement in native pixels
P4_NOISE_K = {1: 0.01, 2: 0.03, 3: 0.05}  # added-noise SNR = 1 / k


@dataclass(frozen=True)
class Condition:
    name: str
    family: str          # clean, P1, P2, P3, P4, C1, C2, C2-resolution, C2-gain
    severity: int = 0
    direction: str = ""  # down / up for gain families

    @property
    def sigma(self) -> float:
        return P1_SIGMA.get(self.severity, 0.0)

    @property
    def factor(self) -> int:
        return P1_FACTOR.get(self.severity, 1)

    @property
    def gain(self) -> float:
        return P2_GAIN[self.direction][self.severity] if self.direction else 1.0

    @property
    def shift_px(self) -> float:
        return P3_SHIFT_PX.get(self.severity, 0.0)

    @property
    def noise_k(self) -> float:
        return P4_NOISE_K.get(self.severity, 0.0)


def all_conditions(multispectral: bool = True) -> list[Condition]:
    """Clean plus every degraded condition: 37 for multispectral contracts, 19 otherwise."""
    out = [Condition("clean", "clean")]
    sev = (1, 2, 3)
    out += [Condition(f"p1_s{s}", "P1", s) for s in sev]
    out += [Condition(f"p2_gain_{d}_s{s}", "P2", s, d) for d in ("down", "up") for s in sev]
    if multispectral:
        out += [Condition(f"p3_s{s}", "P3", s) for s in sev]
    out += [Condition(f"p4_s{s}", "P4", s) for s in sev]
    out += [Condition(f"c1_gain_{d}_s{s}", "C1", s, d) for d in ("down", "up") for s in sev]
    if multispectral:
        out += [Condition(f"c2_gain_{d}_s{s}", "C2", s, d) for d in ("down", "up") for s in sev]
        out += [Condition(f"c2_resolution_only_s{s}", "C2-resolution", s) for s in sev]
        out += [Condition(f"c2_gain_only_{d}_s{s}", "C2-gain", s, d) for d in ("down", "up") for s in sev]
    return out


def conditions_for(contract: BandContract, band_ids: Sequence[str]) -> list[Condition]:
    """Conditions that are defined for a model consuming `band_ids` of this dataset."""
    return all_conditions(contract.multispectral(band_ids))


def get_condition(name: str) -> Condition:
    for c in all_conditions(True):
        if c.name == name:
            return c
    raise KeyError(f"Unknown condition {name!r}")
