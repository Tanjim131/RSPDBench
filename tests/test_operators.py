"""Properties of the RSPDBench operators stated in the paper. Run with `pytest -q`."""
import math

import pytest
import torch

import rspdbench as rb
from rspdbench.contracts import S2_12

IDS = list(S2_12)


@pytest.fixture
def image():
    torch.manual_seed(0)
    return torch.rand(len(IDS), 64, 64) * 3000 + 100


@pytest.fixture
def mean():
    return torch.linspace(500, 2500, len(IDS))


def run(x, name, mean=None, contract="m-bigearthnet", sid="s1", ids=IDS):
    return rb.degrade(x, name, band_ids=ids, contract=contract, sample_id=sid, seed=42, band_mean=mean)


def test_condition_counts():
    assert len(rb.all_conditions(True)) == 37
    assert len(rb.all_conditions(False)) == 19
    rgb = rb.conditions_for(rb.get_contract("m-bigearthnet"), ["B04", "B03", "B02"])
    assert len(rgb) == 19


def test_clean_is_identity(image):
    assert torch.equal(run(image, "clean"), image)


@pytest.mark.parametrize("name,gain", [("p2_gain_down_s3", 0.85), ("p2_gain_up_s1", 1.03)])
def test_p2_scales_every_band(image, name, gain):
    out = run(image, name)
    assert torch.allclose(out, image * gain)


def test_p3_moves_only_the_shifted_group(image):
    out = run(image, "p3_s3")
    moved = [IDS.index(b) for b in ("B05", "B06", "B07", "B8A", "B11", "B12")]
    fixed = [i for i in range(len(IDS)) if i not in moved]
    assert torch.equal(out[fixed], image[fixed])
    assert not torch.equal(out[moved], image[moved])


def test_p3_direction_is_fixed_across_severities():
    theta = rb.operators.shift_direction(sample_id="s1", seed=42)
    assert theta == rb.operators.shift_direction(sample_id="s1", seed=42)
    assert theta != rb.operators.shift_direction(sample_id="s2", seed=42)


def test_p4_noise_std_is_k_times_band_mean(image, mean):
    out = run(image, "p4_s2", mean)
    std = (out - image).flatten(1).std(dim=1, unbiased=False)
    assert torch.allclose(std, 0.03 * mean, rtol=1e-4)
    assert torch.equal(out, run(image, "p4_s2", mean))  # deterministic


def test_p1_keeps_shape_and_mean(image):
    for s in (1, 2, 3):
        out = run(image, f"p1_s{s}")
        assert out.shape == image.shape
        assert torch.allclose(out.mean(dim=(1, 2)), image.mean(dim=(1, 2)), rtol=1e-2)


def test_c2_never_touches_visible_bands(image):
    visible = [IDS.index(b) for b in ("B02", "B03", "B04")]
    for name in ("c2_gain_down_s3", "c2_resolution_only_s2", "c2_gain_only_up_s3"):
        assert torch.equal(run(image, name)[visible], image[visible])


def test_meurosat_c2_resamples_rededge_but_only_rescales_b08():
    ids = list(rb.MEUROSAT_STORAGE_ORDER)
    x = torch.rand(len(ids), 64, 64) * 3000
    out = rb.degrade(x, "c2_gain_down_s3", band_ids=ids, contract="m-eurosat", sample_id="e", seed=42)
    b08, b05 = ids.index("B08"), ids.index("B05")
    assert torch.allclose(out[b08], x[b08] * 0.85)          # radiometric group only
    assert not torch.allclose(out[b05], x[b05])             # geometric group only
    assert torch.allclose(out[b05].mean(), x[b05].mean(), rtol=1e-2)


def test_compound_excess():
    drops = {"c1_gain_down_s2": 61.7, "p1_s2": 23.6, "p2_gain_down_s2": 0.4, "p4_s2": 13.9}
    assert math.isclose(rb.compound_excess(drops, "c1_gain_down_s2"), 38.1)
    assert rb.relative_drop(0.8, 0.6) == pytest.approx(25.0)
