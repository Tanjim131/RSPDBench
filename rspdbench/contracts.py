"""Dataset band contracts: which bands each RSPDBench operator may touch.

Band groups are defined on physical band identities. Operators intersect them
with the bands a model actually consumes, so a model that never reads a band is
never affected through it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

S2_13 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
S2_12 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
HLS_6 = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
RGBN = ("RED", "GREEN", "BLUE", "NIR")

# GEO-Bench stores m-EuroSAT bands in native EuroSAT order (B8A last) while
# reporting them under the standard Sentinel-2 names. Use this physical order
# whenever bands are selected by identity.
MEUROSAT_STORAGE_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A")

_S2_P3_SHIFTED = ("B05", "B06", "B07", "B8A", "B11", "B12")
_S2_NIR_SWIR = ("B08", "B8A", "B11", "B12")


@dataclass(frozen=True)
class BandContract:
    """Band groups of one dataset.

    reference: bands that P3 and C2 leave unchanged.
    p3_shifted: bands translated by P3.
    c2_geometric: bands resampled and translated by the C2 chain.
    c2_radiometric: bands rescaled by the C2 chain.
    """

    name: str
    bands: tuple[str, ...]
    reference: tuple[str, ...]
    p3_shifted: tuple[str, ...] = ()
    c2_geometric: tuple[str, ...] = ()
    c2_radiometric: tuple[str, ...] = ()
    native_gsd_m: float | None = None

    def multispectral(self, band_ids: Sequence[str]) -> bool:
        """True when P3 and C2 are defined for these consumed bands."""
        present = set(band_ids)
        return bool(present & set(self.p3_shifted)) and bool(present & (set(self.c2_geometric) | set(self.c2_radiometric)))


CONTRACTS: dict[str, BandContract] = {
    "m-eurosat": BandContract(
        name="m-eurosat",
        bands=MEUROSAT_STORAGE_ORDER,
        reference=("B02", "B03", "B04"),
        p3_shifted=_S2_P3_SHIFTED,
        c2_geometric=_S2_P3_SHIFTED,
        c2_radiometric=_S2_NIR_SWIR,
        native_gsd_m=10.0,
    ),
    "m-bigearthnet": BandContract(
        name="m-bigearthnet",
        bands=S2_12,
        reference=("B02", "B03", "B04"),
        p3_shifted=_S2_P3_SHIFTED,
        c2_geometric=_S2_NIR_SWIR,
        c2_radiometric=_S2_NIR_SWIR,
        native_gsd_m=10.0,
    ),
    "m-sa-crop-type": BandContract(
        name="m-sa-crop-type",
        bands=S2_12,
        reference=("B02", "B03", "B04"),
        p3_shifted=_S2_P3_SHIFTED,
        c2_geometric=_S2_NIR_SWIR,
        c2_radiometric=_S2_NIR_SWIR,
        native_gsd_m=10.0,
    ),
    "m-chesapeake": BandContract(
        name="m-chesapeake",
        bands=RGBN,
        reference=RGBN,
        native_gsd_m=1.0,
    ),
    "hls-burn-scars": BandContract(
        name="hls-burn-scars",
        bands=HLS_6,
        reference=("BLUE", "GREEN", "RED"),
        p3_shifted=("NIR_NARROW", "SWIR_1", "SWIR_2"),
        c2_geometric=("NIR_NARROW", "SWIR_1", "SWIR_2"),
        c2_radiometric=("NIR_NARROW", "SWIR_1", "SWIR_2"),
        native_gsd_m=30.0,
    ),
}


def get_contract(name: str) -> BandContract:
    key = name.strip().lower().replace("_", "-")
    aliases = {"m-sa-crop": "m-sa-crop-type", "hls-burn-scar": "hls-burn-scars", "hlsburnscars": "hls-burn-scars"}
    key = aliases.get(key, key)
    if key not in CONTRACTS:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(CONTRACTS)}")
    return CONTRACTS[key]


def canonical_band_id(name: str) -> str:
    """Map a Sentinel-2 band name such as '08A - Vegetation Red Edge' or 'B8A' to 'B8A'.

    Non-Sentinel-2 names (HLS, RGB-NIR) are returned upper-cased.
    """
    text = name.strip().upper().replace(" ", "")
    if text.startswith(("08A", "B8A")):
        return "B8A"
    for token in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"):
        if text.startswith(token) or text.startswith("B" + token):
            return "B" + token
    return name.strip().upper()
