# RSPDBench

Degradation operators from the WACV 2027 paper *RSPDBench: Benchmarking Vision Foundation Models on Earth Observation Tasks Under Physically Grounded Remote-Sensing Product Degradations*.

Satellite images reach models as processed products, and those products can carry defects: lost resolution, calibration error, bands that do not line up, sensor noise, and residuals left when bands from different sensors are merged. This package applies each defect to the product in its native units, before any model-specific preprocessing.

[Repository](https://github.com/Tanjim131/RSPDBench) · [Project page](https://tanjim131.github.io/RSPDBench/) · [Paper](https://arxiv.org/abs/2609.23427)

## Install

```bash
pip install rspdbench
```

The package contains the operators, test conditions, per-dataset band contracts and metrics. The results, examples and analysis scripts are in the repository.

## Quick start

```python
import torch, rspdbench as rb

contract = rb.get_contract("hls-burn-scars")
bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]   # physical identity of each channel
x = torch.rand(6, 512, 512) * 0.4                                    # (C, H, W) in native units (HLS reflectance)
band_mean = torch.tensor([0.054, 0.079, 0.096, 0.212, 0.236, 0.173]) # clean training mean per band

for cond in rb.conditions_for(contract, bands):                      # 37 conditions for this input
    y = rb.degrade(x, cond, band_ids=bands, contract=contract, sample_id="tile_0001", band_mean=band_mean)
    # -> your model's own band selection, normalization and resizing -> frozen encoder -> clean-trained head
```

## The defects

| | Defect | What it does | S1 / S2 / S3 |
|---|---|---|---|
| P1 | Effective resolution loss | Gaussian blur, area reduction, bilinear resize back | blur 0.5, 1, 2 px; reduction 1, 2, 4 |
| P2 | Uniform gain error | Scales all bands up or down | 3, 8, 15 % |
| P3 | Band misregistration | Shifts the red-edge, NIR and SWIR bands against the visible bands | 0.3, 1, 2 px |
| P4 | Sensor noise | Adds Gaussian noise scaled by each band's clean training mean | SNR 100, 33, 20 |
| C1 | Acquisition chain | Gain error, blur and reduction, noise on the coarse grid, resize back | each part at the same level |
| C2 | Harmonization chain | Lower resolution and a shift on one band group, gain on another | each part at the same level |

Every defect runs at three severities, and the gain-based defects (P2, C1 and C2) run in both directions. P3 and C2 act on the red-edge, NIR and SWIR bands, so they are not defined for RGB or RGB + NIR input. `rb.conditions_for(contract, bands)` returns the conditions for a given input: 37 for multispectral input and 19 for RGB or RGB + NIR.

Random parts are deterministic functions of the seed, the sample and, for noise, the band. The P4 noise field and the P3 and C2 shift direction stay fixed across severities, so a higher severity strengthens the same defect. C1 draws its noise on the reduced grid, so its noise field differs between severities.

The repository has the full operator definitions, band contracts, results and evaluation protocol.

## Citation

```bibtex
@inproceedings{faruk2027rspdbench,
  title     = {{RSPDBench}: Benchmarking Vision Foundation Models on Earth Observation Tasks Under
               Physically Grounded Remote-Sensing Product Degradations},
  author    = {Faruk, Tanjim Bin and Reza, Khondaker Masfiq and Pallickara, Shrideep and Pallickara, Sangmi Lee},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027},
  note      = {arXiv:2609.23427}
}
```
