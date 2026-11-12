# UrbanFlood360: U-Net Flood Mapping and Building Damage Assessment

Valencia DANA flood, 29–30 October 2024, six Horta Sud towns (Alfafar, Catarroja, Massanassa, Paiporta, Picanya,
Sedaví; 41.6 km²). Sentinel-1 GRD pair from relative orbit 103: 19 Oct (before) and 31 Oct 2024 (after).

## 1. What was built

| Part | Where |
|---|---|
| U-Net written for this project (training, inference, losses) | `models/unet_v2/` |
| Flood-affected label option (`--trace-mode flood`) | `scripts/create_flood_labels.py` |
| Method comparison on identical pixels | `scripts/evaluate_flood_models.py` |
| Full-area flood maps (GeoTIFF + polygons) | `scripts/predict_flood_map.py` → `data/processed/predictions/unet_v2_{affected,water}/` |
| Building inventory (Catastro) | `scripts/prepare_buildings.py` → `data/processed/buildings/buildings_aoi.gpkg` |
| Building damage assessment | `scripts/assess_building_damage.py` → `data/processed/damage/`, `outputs/tables/` |
| Dashboard | `dashboard/app.py` (Streamlit) |
| One-command pipeline / tests | `run_pipeline.sh`, `tests/test_unet_v2.py` |

## 2. Model: U-Net v2

**Architecture.** A classic U-Net: 4 encoder levels (32→64→128→256 channels), a 512-channel bottleneck with
dropout 0.3, 4 decoder levels with transposed-convolution upsampling and skip connections, and a 1×1 output conv
that produces logits. Each block is two 3×3 conv → BatchNorm → ReLU. 7,763,905 parameters. Input: the six SAR
bands (VV/VH before, after, and the change in dB), z-scored with statistics fitted on training-region pixels only.

**Training, and what changed from the earlier baseline** (`models/unet`, `reports/unet_baseline_training_report.md`):

| | Earlier baseline | U-Net v2 |
|---|---|---|
| Training data | 301 fixed 64×64 patches (≥ 80 % labelled) | random 64×64 windows anywhere in the training region (199,834 candidate positions), so every labelled training pixel is used |
| Leakage control | 4×4 block split, 150 m buffer | same split (val = block 1,2; test = block 2,1); training windows lie entirely > 150 m (+2 px margin) from val/test, checked by a unit test |
| Loss | BCE | 0.5 · weighted BCE (pos_weight = non-flood/flood ratio, capped at 10) + 0.5 · soft Dice, ignoring label 255 |
| Augmentation | none (the original notebook's flip had an image/label sync bug) | optional rot90 + flip, one random draw shared by image and label (unit-tested) |
| Optimiser | Adam 1e-3 | AdamW 1e-3, weight decay 1e-4, cosine schedule, EMA of weights (decay 0.995) |
| Model selection | lowest val loss on 14 patches | best Dice on all labelled pixels of the val block; no selection before epoch 20; threshold chosen in 0.3–0.7 |
| Inference | non-overlapping 64×64 tiles | overlapping 128×128 tiles with tapered blending |

The selection rules (minimum 20 epochs, threshold range 0.3–0.7) were added during development, after the first
runs showed a "best" at epochs 1–3 and thresholds of 0.05. Both were artefacts of the single small validation block,
and the archived experiments show the same pattern (best epoch 1). The test block played no part in these decisions.

**Runtime.** Each run takes 5.6–7.0 minutes on the Apple M1 GPU (MPS), about 8 s per epoch. All four runs plus
evaluation took under 30 minutes, so Kaggle was not needed. The same `train.py` runs unchanged on CUDA (§7).

## 3. Two targets, and why

The first U-Net v2 was trained on the existing label, EMSR773 *Flooded area* = standing water. As a flood map it
works (§4.1). Plugged into the damage assessment, though, it flagged only **111 buildings** and detected only
**3.4 %** of the 1,426 EMSR773-damaged buildings in the area. Diagnosis on the label rasters:

* Only **150 of 71,375** building pixels (0.2 %) lie inside *Flooded area* polygons; **24,907** (35 %) lie inside
  *Flood trace* polygons. *Flooded area* is mostly fields and the Albufera rice paddies. The town centres are
  covered by *Flood trace* (mud and debris after the water drained). The radar image is from 31 Oct, two days after
  the peak.
* 77 % of that model's predicted water lay in the eastern strip (rice paddies), which Copernicus never imaged, so it
  cannot be checked.

So the same code was trained on a second target, **flood-affected = Flooded area + Flood trace**
(`create_flood_labels.py --trace-mode flood`, written to `data/processed/flood_affected_dataset/labels/`; the
default mode reproduces the original label exactly). This raises the positive pixels from 26,163 to 110,107. The
test block goes from 113 to 7,249 positive pixels, so test scores become meaningful.

## 4. Flood-mapping results

All numbers are pixel-pooled over labelled 10 m pixels (label 255 ignored). Thresholds: Otsu rules are fixed by
Otsu; earlier U-Nets use their documented thresholds; U-Net v2 uses the threshold chosen on the val block.
Full tables: `outputs/tables/model_comparison_{water,affected}.csv`. Error maps:
`reports/figures/model_comparison_{target}_{val,test}_block.png`.

### 4.1 Standing water (EMSR773 *Flooded area*)

| Method | Val block Dice | IoU | Precision | Recall | Val patches* Dice |
|---|--:|--:|--:|--:|--:|
| Otsu: low post-event VV | 0.570 | 0.399 | 0.844 | 0.431 | 0.460 |
| Otsu: VV decrease | 0.547 | 0.376 | 0.907 | 0.391 | 0.455 |
| Earlier U-Net, BCE baseline | 0.750 | 0.600 | 0.764 | 0.736 | 0.791 |
| Earlier U-Net, weighted BCE+Dice | **0.766** | **0.620** | 0.729 | 0.806 | **0.808** |
| U-Net v2, augmentation | 0.755 | 0.606 | 0.744 | 0.766 | 0.741 |
| **U-Net v2, no augmentation (selected)** | 0.764 | 0.618 | 0.816 | 0.718 | 0.794 |

\*the pixels of the original 14 validation patches, for comparison with earlier reports. Evaluated here with
full-raster inference (more spatial context), so the earlier models score higher than in their own reports.

**On standing water, U-Net v2 matches the earlier best model; it does not beat it.** The test block contains only
113 water pixels, and every U-Net misses them (Dice 0, specificity ≥ 0.9998), so the test block cannot rank the
methods for this target.

### 4.2 Flood-affected ground (*Flooded area* + *Flood trace*), used for the damage assessment

| Method | Val Dice | Val P / R | **Test Dice** | Test IoU | Test P / R | Test specificity |
|---|--:|--:|--:|--:|--:|--:|
| Otsu: low post-event VV | 0.126 | 0.86 / 0.07 | 0.006 | 0.003 | 0.04 / 0.00 | 0.973 |
| Otsu: VV decrease | 0.112 | 0.92 / 0.06 | 0.013 | 0.007 | 0.36 / 0.01 | 0.996 |
| U-Net v2 water model (no aug) | 0.204 | 0.84 / 0.12 | 0.000 | 0.000 | 0.00 / 0.00 | 1.000 |
| **U-Net v2 affected, augmentation (selected)** | **0.778** | 0.67 / 0.93 | **0.710** | **0.550** | 0.65 / 0.78 | 0.868 |
| U-Net v2 affected, no augmentation | 0.721 | 0.69 / 0.75 | 0.398 | 0.248 | 0.47 / 0.34 | 0.876 |

The held-out test block (7,249 affected pixels of 29,702 labelled) gives Dice **0.71** for the selected model.
Augmentation matters a lot here (0.71 vs 0.40 on test). Simple backscatter thresholds and the standing-water model
capture almost none of the flood-affected ground.

## 5. Building damage assessment

**Method.** For each of 12,767 Catastro buildings (6 municipal INSPIRE downloads, centroid inside the AOI; floors
aggregated from BuildingPart), take the footprint plus a 10 m ring. Read the flood-affected mask over every touched
pixel, then classify the building:
- **Affected:** ≥ 50 % of those pixels flood-affected.
- **Possibly affected:** ≥ 10 %.
- **Not affected:** otherwise.

Residents are estimated as dwellings × 2.5, an editable assumption.

**Headline numbers** (`outputs/tables/damage_summary.json`):

| | |
|---|--:|
| Flood-affected area (whole AOI) | 24.9 km² (60 %) |
| …inside the Copernicus-imaged part | 13.6 km², vs 11.0 km² mapped by EMSR773 |
| Standing water on 31 Oct (whole AOI) | 13.6 km² |
| Buildings affected / possibly affected | **2,733** / 626 (of 12,767) |
| Dwellings affected / possibly affected | **8,359** / 4,226 |
| Estimated residents in affected buildings | ~20,900 |
| Floor area of affected buildings | 3.87 km² |

| Municipality | Buildings | Affected | Possibly | % affected | Dwellings affected |
|---|--:|--:|--:|--:|--:|
| Alfafar | 1,873 | 406 | 59 | 21.7 | 1,631 |
| Catarroja | 3,831 | 826 | 262 | 21.6 | 2,632 |
| Massanassa | 1,981 | 375 | 104 | 18.9 | 640 |
| Paiporta | 2,294 | 663 | 63 | 28.9 | 2,666 |
| Picanya | 1,497 | 173 | 100 | 11.6 | 252 |
| Sedaví | 1,291 | 290 | 38 | 22.5 | 538 |

By use, industrial buildings are the most affected: 1,421 of 2,466 (58 %). Residential: 965 of 9,505 (10 %).

**Check 1: EMSR773 damaged buildings** (builtUpP, 1,426 points in the AOI, matched to buildings within 10 m).
**91.6 %** are flagged affected or possibly affected (86.0 % affected). By grade: 95.5 % of *Damaged* and 67.9 % of
*Possibly damaged*. EMSR lists only damaged buildings, so this is a detection rate. EMSR points are mostly
non-residential buildings.

**Check 2: buildings inside the EMSR773 flood extent** (building centre vs the flood-affected reference,
`outputs/tables/emsr_extent_building_check.csv`):

| Region | Buildings | Inside extent | Recall (Affected) | Recall (+possibly) | Precision | Flagged outside extent |
|---|---|--:|--:|--:|--:|--:|
| Test block (held out) | all | 431 | 0.81 | 0.90 | 0.60 | 6.9 % |
| Test block (held out) | residential | 19 | 0.74 | 0.90 | 0.08 | 5.7 % |
| Validation block | all | 545 | 0.93 | 0.96 | 0.73 | 7.5 % |
| Whole imaged area* | all | 1,953 | 0.86 | 0.91 | 0.65 | 8.7 % |
| Whole imaged area* | residential | 494 | 0.76 | 0.82 | 0.41 | 6.1 % |

\*includes the training area, so optimistic.

Residential precision is low because EMSR773 places very few homes inside its flood polygons (494 of 9,459). The
residential and resident counts here are therefore **higher than the Copernicus extent implies**. Which is closer to
reality cannot be settled with this data. Rapid optical mapping tends to under-record mud in narrow streets, while a
10 m radar model can over-extend flood signal into adjacent blocks.

## 6. Limitations

* **One event, one area.** About 67,000 flood-affected and 21,000 standing-water labelled pixels in the training
  region. Expect lower accuracy elsewhere.
* **Validation numbers are optimistic** (used for checkpoint, threshold and run selection). The standing-water test
  block has almost no water.
* **27 % of the AOI was never imaged** by the Copernicus optical scenes. Predictions there, mostly the eastern rice
  paddies, are unverified, and they make up much of the 24.9 km² flood-affected total.
* **"Flood trace" is a visual interpretation** of mud and debris on optical images. The radar does not see mud
  directly; the model learns correlated backscatter changes. Precision on the test block is 0.65.
* **"Affected" means flood evidence around the building**, not structural damage, water depth or interior flooding.
* **Labels and the EMSR damage check come from the same Copernicus activation**, so they are not independent.

## 7. Reproduce

```bash
# environment (created for this work)
~/Anaconda/anaconda3/bin/conda create -n urbanflood python=3.12
~/Anaconda/anaconda3/envs/urbanflood/bin/pip install torch rasterio geopandas pyogrio scipy scikit-image \
    matplotlib pyyaml streamlit streamlit-folium folium plotly pytest

./run_pipeline.sh                    # labels + 4 training runs + evaluation + maps + damage + dashboard assets
SKIP_TRAINING=1 ./run_pipeline.sh    # everything except training (~40 s)
~/Anaconda/anaconda3/envs/urbanflood/bin/streamlit run dashboard/app.py
~/Anaconda/anaconda3/envs/urbanflood/bin/python -m pytest tests -q
```

**On Kaggle (optional, not needed at this data size).** Upload `models/unet_v2/`, `data/processed/sar_features/sar_features_6ch.tif`
and the two label rasters as a dataset, keeping the same relative paths. Then run
`python -m models.unet_v2.train --config models/unet_v2/config_affected.yaml --run-name affected_aug`. The script
picks CUDA automatically. Copy the `runs/<name>/` folder back into `models/unet_v2/runs/`.
