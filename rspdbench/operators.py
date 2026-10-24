"""RSPDBench degradation operators.

All operators act on one image ``x`` of shape (C, H, W) in the dataset's native
units (e.g. reflectance x 10^4), before any model-specific band selection,
normalization, or resizing. Random components are deterministic functions of
(seed, sample_id, band), and stay fixed across severities: moving up a severity
ladder strengthens the same noise field or shift direction.

P1  effective spatial-resolution loss: Gaussian PSF, N x N area reduction, bilinear resize back.
P2  uniform radiometric gain on every band.
P3  coherent sub-pixel shift of an infrared band group against fixed reference bands.
P4  band-wise Gaussian noise with std = k * (clean band mean), i.e. added SNR = 1 / k.
C1  acquisition chain: gain -> PSF blur -> area reduction -> noise on the coarse grid -> resize.
C2  harmonization chain: P1 + P3-style shift on a geometric group, then gain on a radiometric group.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .conditions import Condition, get_condition
from .contracts import BandContract, get_contract


def _hash_u64(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _gaussian_kernel(radius: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(radius * 2 + 1, device=device, dtype=dtype) - radius
    k1 = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    k1 = k1 / k1.sum()
    k2 = k1[:, None] * k1[None, :]
    return k2 / k2.sum()


def noise_field(shape: Sequence[int], *, sample_id: str, seed: int, band: str) -> torch.Tensor:
    """Standard-normal field (exactly zero mean, unit std) for one band of one sample."""
    g = torch.Generator(device="cpu")
    g.manual_seed(_hash_u64("wacv_noise", seed, sample_id, band) % (2**63 - 1))
    z = torch.randn(tuple(shape), generator=g, dtype=torch.float32)
    return (z - z.mean()) / z.std(unbiased=False).clamp_min(1e-6)


def shift_direction(*, sample_id: str, seed: int) -> float:
    """Shift angle in radians; shared by every severity and every shifted band of a sample."""
    return (_hash_u64("wacv_shift_direction", seed, sample_id) / float(2**64)) * 2.0 * math.pi


def _band_noise(x: torch.Tensor, band_ids: Sequence[str], band_mean: torch.Tensor, k: float, *, sample_id: str, seed: int) -> torch.Tensor:
    means = band_mean.to(dtype=x.dtype, device=x.device).flatten()
    if means.numel() != x.shape[0]:
        raise ValueError(f"band_mean has {means.numel()} entries for {x.shape[0]} channels")
    parts = [(noise_field(x.shape[1:], sample_id=sample_id, seed=seed, band=b).to(x) * (float(k) * means[c])).unsqueeze(0) for c, b in enumerate(band_ids)]
    return torch.cat(parts, dim=0)


def resolution_loss(x: torch.Tensor, sigma: float, factor: int, *, noise: tuple | None = None) -> torch.Tensor:
    """P1: PSF blur, area reduction by `factor`, optional coarse-grid noise (for C1), bilinear resize back."""
    if sigma <= 0:
        return x
    y = x.unsqueeze(0)
    r = max(1, int(math.ceil(3.0 * sigma)))
    kernel = _gaussian_kernel(r, float(sigma), y.device, y.dtype).view(1, 1, 2 * r + 1, 2 * r + 1).repeat(y.shape[1], 1, 1, 1)
    y = F.conv2d(F.pad(y, (r, r, r, r), mode="reflect"), kernel, groups=y.shape[1])
    low = (max(1, int(round(x.shape[-2] / float(factor)))), max(1, int(round(x.shape[-1] / float(factor)))))
    y = F.interpolate(y, size=low, mode="area")
    if noise is not None:
        band_ids, band_mean, k, sample_id, seed = noise
        y = y + _band_noise(y.squeeze(0), band_ids, band_mean, k, sample_id=sample_id, seed=seed).unsqueeze(0)
    return F.interpolate(y, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False).squeeze(0)


def translate(x: torch.Tensor, shift_px: float, *, sample_id: str, seed: int) -> torch.Tensor:
    """Bilinear sub-pixel translation of every channel of x by shift_px in the sample's direction."""
    if shift_px <= 0:
        return x
    h, w = x.shape[-2:]
    theta = shift_direction(sample_id=sample_id, seed=seed)
    dx, dy = float(shift_px) * math.cos(theta), float(shift_px) * math.sin(theta)
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack([(xx - dx + 0.5) / w * 2.0 - 1.0, (yy - dy + 0.5) / h * 2.0 - 1.0], dim=-1).unsqueeze(0)
    return F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)


def _index(band_ids: Sequence[str], group: Sequence[str]) -> list[int]:
    wanted = set(group)
    return [i for i, b in enumerate(band_ids) if b in wanted]


def degrade(
    x: torch.Tensor,
    condition: Condition | str,
    *,
    band_ids: Sequence[str],
    contract: BandContract | str,
    sample_id: str,
    seed: int = 42,
    band_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one RSPDBench condition to a (C, H, W) image in native units.

    band_ids: physical identity of each channel of x (e.g. 'B8A', 'SWIR_1'); the
        contract's groups are intersected with them.
    band_mean: clean training-set mean of each channel, required by P4 and C1.
    """
    cond = get_condition(condition) if isinstance(condition, str) else condition
    contract = get_contract(contract) if isinstance(contract, str) else contract
    if len(band_ids) != x.shape[0]:
        raise ValueError(f"{len(band_ids)} band ids for {x.shape[0]} channels")
    if cond.family == "clean":
        return x
    y = x.clone()
    noisy = cond.family in ("P4", "C1")
    if noisy and band_mean is None:
        raise ValueError(f"{cond.name} needs band_mean (clean training mean per channel)")

    if cond.family == "P1":
        return resolution_loss(y, cond.sigma, cond.factor)
    if cond.family == "P2":
        return y * cond.gain
    if cond.family == "P3":
        idx = _index(band_ids, contract.p3_shifted)
        if idx:
            y[idx] = translate(y[idx], cond.shift_px, sample_id=sample_id, seed=seed)
        return y
    if cond.family == "P4":
        return y + _band_noise(y, band_ids, band_mean, cond.noise_k, sample_id=sample_id, seed=seed)
    if cond.family == "C1":
        return resolution_loss(y * cond.gain, cond.sigma, cond.factor, noise=(list(band_ids), band_mean, cond.noise_k, sample_id, seed))
    g = _index(band_ids, contract.c2_geometric)
    h = _index(band_ids, contract.c2_radiometric)
    if cond.family in ("C2", "C2-resolution") and g:
        y[g] = resolution_loss(y[g], cond.sigma, cond.factor)
    if cond.family == "C2" and g:
        y[g] = translate(y[g], cond.shift_px, sample_id=sample_id, seed=seed)
    if cond.family in ("C2", "C2-gain") and h:
        y[h] = y[h] * cond.gain
    return y
