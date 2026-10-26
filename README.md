# RSPDBench

**Benchmarking Vision Foundation Models on Earth Observation Tasks Under Physically Grounded Remote-Sensing Product Degradations**
Tanjim Bin Faruk, Khondaker Masfiq Reza, Shrideep Pallickara, Sangmi Lee Pallickara · Colorado State University · **WACV 2027**

[Paper (arXiv:2609.23427)](https://arxiv.org/abs/2609.23427) · [Project page](https://tanjim131.github.io/RSPDBench/)

![RSPDBench: one burn-scar scene under six product defects](https://raw.githubusercontent.com/Tanjim131/RSPDBench/main/docs/assets/og.png)

Satellite images reach models as processed products, and products carry defects: resolution loss, calibration drift, bands that do not line up, sensor noise, and the residuals left when sensors are harmonized. RSPDBench applies these defects to the product itself, in its native units and before any model-specific preprocessing, one at a time and in physically ordered chains, and measures what each frozen Earth-observation foundation model loses relative to its own clean score.

## What is in this repository

| Path | Contents |
|---|---|
| `rspdbench/` | The degradation operators (P1–P4, C1, C2), the 37 test conditions, per-dataset band contracts, metrics |
| `results/` | Per-condition results for 7 foundation models and 2 supervised baselines on 5 datasets (3 seeds) |
| `analysis/` | Scripts that regenerate the paper's Figs. 3–5 and Tab. 2, and the Sentinel-2 realism check |
| `examples/` | Rendering a scene under every condition; the evaluation protocol for your own frozen encoder |
| `tests/` | Property tests of the operators |
| `docs/` | The project page (GitHub Pages) |

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
band_mean = torch.tensor([0.052, 0.076, 0.094, 0.204, 0.232, 0.171]) # clean training mean per band

for cond in rb.conditions_for(contract, bands):                      # 37 conditions for multispectral inputs
    y = rb.degrade(x, cond, band_ids=bands, contract=contract, sample_id="tile_0001", band_mean=band_mean)
    # -> your model's own band selection, normalization and resizing -> frozen encoder -> clean-trained head
```

Random parts (noise fields, shift directions) are deterministic functions of `(seed, sample_id, band)` and stay fixed across severities, so a severity ladder strengthens the same defect.

## The defects

| | Defect | Operator | S1 / S2 / S3 | Physical anchor |
|---|---|---|---|---|
| P1 | Effective resolution loss | Gaussian PSF, N×N area integration, bilinear resize back | σ = 0.5/1/2 px, N = 1/2/4 | σ = N/2 passes ≈0.29 of the contrast at the coarse Nyquist frequency; Sentinel-2 10 m MTF spec is 0.15–0.30 |
| P2 | Uniform radiometric gain | every band × p | ±3 / 8 / 15 % | Sentinel-2 absolute radiometric uncertainty < 5 % (goal 3 %); Landsat 8 OLI ±3 % |
| P3 | Band misregistration | red-edge, narrow-NIR, SWIR group shifted against fixed visible bands | 0.3 / 1 / 2 px | Sentinel-2 inter-band co-registration < 0.3 coarse px at 3σ (0.6 px for 10 m/20 m pairs) |
| P4 | Signal-relative noise | N(0, (k·μ_band)²) per band | SNR 100 / 33 / 20 | Sentinel-2 SNR requirement at reference radiance: 72–174 |
| C1 | Acquisition chain | gain → PSF blur → area reduction → noise on the coarse grid → resize | matched severities | one poor acquisition |
| C2 | Harmonization chain | resolution mismatch + shift on a geometric group, gain on a radiometric group | matched severities | merged multi-sensor products; changes band ratios |

C2 is also evaluated through its resolution-only and gain-only arms, so compound effects can be attributed. Gain families run in both directions.

### Band contracts

| Dataset | Fixed reference | P3 shifted | C2 geometric | C2 radiometric |
|---|---|---|---|---|
| m-EuroSAT | B02–B04 | B05–B07, B8A, B11, B12 | B05–B07, B8A, B11, B12 | B08, B8A, B11, B12 |
| m-BigEarthNet, m-SA-crop | B02–B04 | B05–B07, B8A, B11, B12 | B08, B8A, B11, B12 | B08, B8A, B11, B12 |
| HLS Burn Scar | Blue, Green, Red | NIR, SWIR1, SWIR2 | NIR, SWIR1, SWIR2 | NIR, SWIR1, SWIR2 |
| m-Chesapeake | RGB + NIR: P3 and C2 not defined | | | |

Groups are intersected with the bands a model consumes. **GEO-Bench stores m-EuroSAT bands in native EuroSAT order (B8A last) under the standard Sentinel-2 names.** Select bands by physical identity using `rb.MEUROSAT_STORAGE_ORDER`, never by the reported names.

## Evaluate your own model

The protocol: freeze the encoder, train a light head on clean data with hyperparameters chosen on clean validation, then evaluate the fixed head on every condition and report `drop = (M_clean − M_d) / M_clean × 100 %`. `examples/evaluate_frozen_encoder.py` implements it for m-EuroSAT; replace its `Encoder` class with your model. Datasets come from [GEO-Bench](https://github.com/ServiceNow/geo-bench) (m-EuroSAT, m-BigEarthNet, m-SA-crop-type, m-Chesapeake) and [PANGAEA](https://github.com/VMarsocci/pangaea-bench) (HLS Burn Scars).

## Reproduce the paper's numbers

```bash
git clone https://github.com/Tanjim131/RSPDBench && cd RSPDBench
pip install -e ".[examples]"
bash analysis/reproduce_paper.sh        # Figs. 3-5 -> figures/, Tab. 2 -> stdout
```

`results/rspdbench_aggregate.csv` holds every (dataset, model, condition) cell with 3-seed means and standard deviations and the compound-excess columns; `results/rspdbench_seed_level.csv` holds the per-seed scores. Metrics: Top-1 accuracy (m-EuroSAT), micro-F1 at a 0.5 threshold (m-BigEarthNet), mIoU (segmentation).

`analysis/realism_check.py` reruns the check of P1 and P3 against Sentinel-2's co-acquired 10 m (B08) and 20 m (B8A) near-infrared bands; its summary is in `results/realism_s2_b08_b8a_summary.csv`.

### Fidelity of the released operators

The operators here are bit-identical to the implementation behind the m-BigEarthNet, m-SA-crop, m-Chesapeake and HLS Burn Scar results and the supervised baselines. The m-EuroSAT foundation-model runs used per-model extractors with the same operator definitions and band groups but a different random hash, so their noise and shift draws differ from what this code produces; deterministic conditions (P1, P2, C2 arms) agree to float precision. The per-model feature-extraction and probe-training pipelines are not part of this first release.

## Project page

`docs/index.html` is generated from `docs/tools/page.src.html` with the results inlined:

```bash
GEO_BENCH_DIR=/path/to/geobench python docs/tools/build_explorer_assets.py --hls-tif /path/to/<HLS tile>.tif
python docs/tools/build_page_data.py
python docs/tools/build_site.py --repo-url https://github.com/<owner>/RSPDBench --site-url https://<owner>.github.io/RSPDBench/
```

`build_explorer_assets.py` renders one scene per dataset under every condition with the released operators (clean training-set band means for P4/C1) and records each scene's location, date, resolution and false-color bands. The link-preview image comes from `docs/tools/make_og_image.py`.

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

Code: MIT (see `LICENSE`). The project page shows one scene per dataset: Copernicus Sentinel-2 data (m-EuroSAT, m-BigEarthNet, m-SA-crop) and USDA NAIP imagery (m-Chesapeake) via GEO-Bench, and an HLS Burn Scars tile (NASA and IBM) via PANGAEA; the benchmark datasets keep their own licenses.
