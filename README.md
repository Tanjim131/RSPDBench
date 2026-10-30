<div align="center">

# RSPDBench

### Benchmarking Vision Foundation Models on Earth Observation Tasks Under Physically Grounded Remote-Sensing Product Degradations

Tanjim Bin Faruk, Khondaker Masfiq Reza, Shrideep Pallickara, Sangmi Lee Pallickara

*Colorado State University*

**WACV 2027**

[Paper](https://arxiv.org/abs/2609.23427) · [Project page](https://tanjim131.github.io/RSPDBench/) · [PyPI](https://pypi.org/project/rspdbench/)

</div>

Satellite images reach models as processed products, and those products can carry defects: lost resolution, calibration error, bands that do not line up, sensor noise, and residuals left when bands from different sensors are merged. RSPDBench applies each defect to the product in its native units, before any model-specific preprocessing, and measures how much each frozen Earth-observation foundation model loses compared with its own clean score.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Tanjim131/RSPDBench/main/assets/readme/defects_dark.png">
  <img alt="A satellite crop of a burn scar and a reservoir, shown clean and under six defects, with a map below each showing how much every pixel changed" src="https://raw.githubusercontent.com/Tanjim131/RSPDBench/main/assets/readme/defects_light.png" width="100%">
</picture>

*A 1.9 km crop of a Harmonized Landsat Sentinel-2 scene (30 m pixels, a burn scar in San Diego County, false color with SWIR 2, NIR and red) under each defect at its strongest level, S3. The top row is what a model receives; the bottom row shows how much each pixel changed. Resolution loss and misregistration change the edges, the gain error grows with brightness, noise is spread evenly, and the harmonization chain changes only the NIR and SWIR bands.*

## What we found

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Tanjim131/RSPDBench/main/assets/readme/ranking_dark.png">
  <img alt="Slope chart of seven foundation models on m-EuroSAT: RAMEN ranks first under resolution loss and last under gain error, DINOv3-Sat ranks last under resolution loss and second under gain error" src="https://raw.githubusercontent.com/Tanjim131/RSPDBench/main/assets/readme/ranking_light.png" width="640">
</picture>
</p>

Each line is one foundation model on m-EuroSAT. RAMEN, which conditions on input resolution, has the smallest drop under resolution loss and the largest under a 15 % gain error, and DINOv3-Sat is close to the reverse. The two rankings are close to reversed on four of the five datasets (median Kendall $`\tau = -0.52`$), so the model you would pick for robustness to blur can be the one most hurt by calibration error. Combined defects also compound: the acquisition chain adds up to 38 percentage points of drop beyond its worst single part.

## What is in this repository

| Path | Contents |
|---|---|
| `rspdbench/` | The degradation operators (P1–P4, C1, C2), the test conditions, per-dataset band contracts and metrics |
| `results/` | Per-condition results for 7 foundation models and 2 supervised baselines on 5 datasets, 3 seeds each |
| `analysis/` | Scripts that regenerate the paper's Figs. 3–5 and Tab. 2, the Sentinel-2 realism check, and the figures in this README |
| `examples/` | Rendering a scene under every condition, and the evaluation protocol for your own frozen encoder |
| `tests/` | Tests of the operators |

## Install

```bash
pip install rspdbench                 # operators only: numpy, torch
pip install "rspdbench[examples]"     # + the data and plotting packages used by examples/ and analysis/
```

The package contains the operators, conditions, band contracts and metrics. The results, examples and analysis scripts are in this repository.

## Quick start

```python
import torch, rspdbench as rb

contract = rb.get_contract("hls-burn-scars")
bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]   # physical identity of each channel
x = torch.rand(6, 512, 512) * 0.4                                    # (C, H, W) in native units (HLS reflectance)
band_mean = torch.tensor([0.054, 0.079, 0.096, 0.212, 0.236, 0.173]) # clean training mean per band

for cond in rb.conditions_for(contract, bands):                      # 37 conditions, see "Test conditions"
    y = rb.degrade(x, cond, band_ids=bands, contract=contract, sample_id="tile_0001", band_mean=band_mean)
    # -> your model's own band selection, normalization and resizing -> frozen encoder -> clean-trained head
```

Random parts (noise fields, shift directions) are deterministic functions of `(seed, sample_id, band)` and stay fixed across severities, so a higher severity strengthens the same defect.

## The defects

| | Defect | Operator | S1 / S2 / S3 | Physical anchor |
|---|---|---|---|---|
| P1 | Effective resolution loss | $`\hat{x} = U_N\big(D_N(G_\sigma * x)\big)`$ | $`\sigma = 0.5,\ 1,\ 2`$ px; $`N = 1,\ 2,\ 4`$ | With $`\sigma = N/2`$, P1 keeps about 20 % of the contrast at the finest detail of the coarser grid at S2 and S3 (57 % at S1, which only blurs); Sentinel-2 specifies 15–30 % for its 10 m bands |
| P2 | Uniform gain error | $`\hat{x} = p\,x,\quad p = 1 \pm \alpha`$ | $`\alpha = 3,\ 8,\ 15\,\%`$ | Sentinel-2 absolute radiometric uncertainty below 5 % (goal 3 %); Landsat 8 OLI reflectance calibration ±3 % |
| P3 | Band misregistration | $`\hat{x}_c = T_{\Delta}(x_c),\quad c \in \mathcal{S}`$ | $`\Delta = 0.3,\ 1,\ 2`$ px | Sentinel-2 inter-band co-registration better than 0.3 of the coarser band's pixel at $`3\sigma`$, which is 0.6 px for a 10 m / 20 m pair |
| P4 | Sensor noise | $`\hat{x}_c = x_c + k\,\mu_c\,z_c`$ | $`\mathrm{SNR} = 1/k = 100,\ 33,\ 20`$ | Sentinel-2 SNR requirement at reference radiance: 72–174 |
| C1 | Acquisition chain | $`\hat{x} = U_N\big(D_N(G_\sigma * p\,x) + \epsilon\big)`$ | each part at the same level | A poor acquisition: gain error, blur and noise on the coarse grid |
| C2 | Harmonization chain | $`\hat{x} = A_p\big(T_{\Delta}(R(x;\mathcal{G});\mathcal{G});\mathcal{H}\big)`$ | each part at the same level | Bands merged from another product: lower resolution and a shift on group $`\mathcal{G}`$, gain on group $`\mathcal{H}`$, so band ratios change |

$`x`$ is the clean product, $`\mu_c`$ the clean training mean of band $`c`$, $`z_c`$ a Gaussian field standardized to zero mean and unit variance in each band, $`G_\sigma`$ a Gaussian PSF, $`D_N`$ area downsampling by $`N`$, $`U_N`$ bilinear resizing back to the input size, $`T_\Delta`$ a sub-pixel shift, $`\mathcal{S}`$ the shifted band group, $`\epsilon`$ the P4 noise added on the coarse grid, $`R`$ the P1 operator applied to group $`\mathcal{G}`$ only, and $`A_p`$ the gain applied to group $`\mathcal{H}`$ only.

## Test conditions

Every defect runs at three severities, S1 (mild), S2 (moderate) and S3 (severe). The gain-based defects (P2, C1 and C2) run in both directions, gain down and gain up. C2 is also run as two of its single-stage parts, resolution only and gain only, so a C2 loss can be compared with them. There is no shift-only part. With the clean input this gives 37 conditions:

| Conditions | Count |
|---|---|
| Clean | 1 |
| P1, P3, P4 at 3 severities | 9 |
| P2, C1, C2 at 3 severities × 2 gain directions | 18 |
| C2 resolution only, 3 severities | 3 |
| C2 gain only, 3 severities × 2 gain directions | 6 |
| **Total** | **37** |

P3 and C2 act on the red-edge, NIR and SWIR bands, so they are not defined for a model that reads only RGB (both DINOv3 models) or for the RGB + NIR imagery of m-Chesapeake. These cases get 19 conditions: clean, P1, P2, P4 and C1. `rb.conditions_for(contract, bands)` returns the list for a given input.

### Band contracts

| Dataset | Fixed reference | P3 shifted | C2 geometric | C2 radiometric |
|---|---|---|---|---|
| m-EuroSAT | B02–B04 | B05–B07, B8A, B11, B12 | B05–B07, B8A, B11, B12 | B08, B8A, B11, B12 |
| m-BigEarthNet, m-SA-crop | B02–B04 | B05–B07, B8A, B11, B12 | B08, B8A, B11, B12 | B08, B8A, B11, B12 |
| HLS Burn Scar | Blue, Green, Red | NIR, SWIR1, SWIR2 | NIR, SWIR1, SWIR2 | NIR, SWIR1, SWIR2 |
| m-Chesapeake | RGB + NIR: P3 and C2 not defined | | | |

Each group is intersected with the bands a model reads.

GEO-Bench stores the m-EuroSAT bands in the original EuroSAT order, with B8A last, but labels them in the standard Sentinel-2 order. From the ninth channel on, the label and the stored band differ: the channel labeled B8A holds B09, and the channel labeled B12 holds B8A. `rb.MEUROSAT_STORAGE_ORDER` lists the band stored in each channel; use it to select m-EuroSAT bands.

## Evaluate your own model

The protocol: freeze the encoder, train a light head on clean data with hyperparameters chosen on clean validation data, then evaluate the fixed head on every condition $`d`$ and report the relative drop

```math
\mathrm{drop}(d) = \frac{M_{\text{clean}} - M_d}{M_{\text{clean}}} \times 100\,\%
```

`examples/evaluate_frozen_encoder.py` implements it for m-EuroSAT; replace its `Encoder` class with your model. Datasets come from [GEO-Bench](https://github.com/ServiceNow/geo-bench) (m-EuroSAT, m-BigEarthNet, m-SA-crop-type, m-Chesapeake) and [PANGAEA](https://github.com/VMarsocci/pangaea-bench) (HLS Burn Scars).

## Reproduce the paper's numbers

```bash
git clone https://github.com/Tanjim131/RSPDBench && cd RSPDBench
pip install -e ".[examples]"
bash analysis/reproduce_paper.sh        # Figs. 3-5 -> figures/, Tab. 2 -> stdout
```

`results/rspdbench_aggregate.csv` holds every (dataset, model, condition) cell with 3-seed means and standard deviations and the compound-excess columns; `results/rspdbench_seed_level.csv` holds the per-seed scores. The excess columns use the mean per-seed drops and compare C1 with P1, P2 and P4, and C2 with its resolution-only and gain-only parts, as in Fig. 4. Metrics: Top-1 accuracy (m-EuroSAT), micro-F1 at a 0.5 threshold (m-BigEarthNet), mIoU (segmentation).

`analysis/realism_check.py` reruns the check of P1 and P3 against Sentinel-2's co-acquired 10 m (B08) and 20 m (B8A) near-infrared bands; its summary is in `results/realism_s2_b08_b8a_summary.csv`. `analysis/figures/plot_readme_figures.py` redraws the figures in this README from an HLS Burn Scars tile and the results file.

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

## License and data

Code: MIT (see `LICENSE`). The figures above show an HLS Burn Scars tile (NASA and IBM) obtained through PANGAEA. The project page also shows Copernicus Sentinel-2 data (m-EuroSAT, m-BigEarthNet, m-SA-crop) and USDA NAIP imagery (m-Chesapeake) obtained through GEO-Bench. The benchmark datasets keep their own licenses.
