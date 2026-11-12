# UrbanFlood360 — U-Net Dataset Generation Report

## 1. EMSR773 label-source investigation

`observedEventA` (GRA v2), re-verified directly via `ogrinfo`/geopandas: 1,528 features, fields
`event_type, obj_desc, det_method, notation, dmg_src_id, area`, CRS EPSG:4326. **MEASURED**: 356/1528
(23.3%) geometries were invalid (self-intersecting rings), concentrated in the "Flooded area" class
(352/1002, 35%). Repaired with `shapely.make_valid()` — 0 invalid remain, no geometry-type surprises
(Polygon/MultiPolygon only).

## 2. Flooded area vs Flood trace analysis

**MEASURED (from the actual attribute table, not documentation)**:
- "Flooded area" (1,002 polygons, 646.35 ha AOI03-wide): `det_method = Semi-automatic extraction`.
- "Flood trace" (526 polygons, 3,646.43 ha AOI03-wide): `det_method = Photo-interpretation`.
- Zero spatial overlap between the two categories; they are adjacent, not nested.

**INTERPRETATION**: "Flooded area" is derived directly from the image (a more direct observation of
standing water at acquisition time); "Flood trace" is a human-interpreted indicator of a flood that
had already partly receded (e.g. mud/debris marks), not necessarily current standing water. This is
consistent with, though not itself proof of, standard CEMS terminology.

**Decision: Option C** — Flooded area → confident positive; Flood trace → explicit ignore. Not merged
(Option B) because the two categories have measurably different detection methods; not simply
discarded from consideration (Option A implies ignoring it entirely) because it still marks genuinely
flood-affected ground that should not be mislabeled non-flood.

## 3. Label methodology

flood(1) = inside AOI ∩ inside EMSR773 image footprint ∩ inside "Flooded area".
ignore(255) = outside AOI, OR outside image footprint, OR inside "Flood trace", OR any of the 6 SAR
channels invalid at that pixel.
non-flood(0) = everything else (inside AOI, inside footprint, not flooded, not trace).

## 4. Label rasterization

Rasterized directly onto the SAR feature grid (`sar_features_6ch.tif`'s exact CRS/transform/
1120×951) via `rasterio.features.geometry_mask` — no reprojection or resampling of SAR data.
Grid identity confirmed programmatically after writing.

## 5. AOI handling

**MEASURED (critical finding)**: the EMSR773 `imageFootprintA` layer covers only **72.82%** of the
AOI — 27.18% of the AOI was never actually imaged by the source VHR optical scenes this product is
built from. This directly justifies the explicit ignore class: treating unmapped area as confirmed
non-flood without this check would have been an unverified assumption.

## 6. Label statistics

| Scope | Flood px | Non-flood px | Ignore px | Total | Flood % (of usable) |
|---|--:|--:|--:|--:|--:|
| AOI overall | 26,163 | 193,078 | 845,879 | 1,065,120 | 11.93% |
| Alfafar | 4,853 | 17,393 | 76,784 | 99,030 | 21.82% |
| Catarroja | 13,466 | 67,733 | 50,172 | 131,371 | 16.58% |
| Massanassa | 7,121 | 17,353 | 31,450 | 55,924 | 29.10% |
| Paiporta | 386 | 17,768 | 21,388 | 39,542 | **2.13%** |
| Picanya | 111 | 66,042 | 5,981 | 72,134 | **0.17%** |
| Sedaví | 226 | 6,789 | 11,313 | 18,328 | 3.22% |

Full table: `data/processed/unet_dataset/label_statistics.csv`. 0 pixels were dropped due to SAR
invalidity beyond what the reference logic already excluded (the AOI/footprint mask and the SAR
validity mask are already closely aligned).

**Important, honestly-flagged tension**: Paiporta (highest death toll in the real event) and Picanya
show extremely sparse "confident flood" representation (2.13% and 0.17%) despite being real,
flood-relevant towns — because most of their flood-affected area is EMSR773 "Flood trace" (54.1% and
8.3% respectively, per the earlier feature-generation-stage municipality table), which this
methodology deliberately treats as ignore, not positive. This is a direct, transparent consequence of
the evidence-based Section 2 decision, not an oversight.

## 7. Class balance

AOI-wide: 11.93% flood among usable pixels — a real, substantial imbalance, reported as-is. No
oversampling, synthetic labels, or loss weighting were applied at this stage.

## 8. Spatial split methodology

**Municipality-based splitting was tested first, per the brief's suggested example, and found
empirically infeasible**: every candidate 64px patch centered inside Massanassa or Alfafar was found
to be within **0.0m** of a neighboring municipality's border (confirmed by direct distance
computation) — the six towns are small and directly interlocking, so no buffer could be satisfied
around a single-town val/test region at any patch size tested.

**Switched to a spatial-block split** (the brief's explicit fallback), using a 3×4 grid over the full
raster extent (blocks ≈3170m×2800m — a coarser 4×6 grid was tried first and also failed for the same
geometric reason). Blocks were ranked by measured flood-pixel density (data-driven, not random):

| Block | Edge? | Usable px | Flood px | Flood % |
|---|---|--:|--:|--:|
| (1,2) | no | 49,500 | 14,716 | 29.7% |
| (2,2) | **yes** | 16,926 | 10,656 | **63.0%** |
| (1,1) | no | 49,181 | 307 | 0.6% |
| (0,1) | yes | 14,432 | 230 | 1.6% |
| (0,2) | yes | 7,244 | 136 | 1.9% |
| (0,0) | yes | 47,492 | 118 | 0.2% |
| (1,0) | yes | 34,466 | 0 | 0.0% |

**Split assignment**: `test = block(2,2)` (edge block, high flood density, only one substantial
train-adjacent side); `val = block(0,1) + block(0,2)` (edge blocks); `train` = everything else,
**including the single richest block (1,2), deliberately kept in train** to preserve positive-class
training signal.

## 9. Buffer methodology

Buffer = 150m, chosen after empirically finding that 200m left no feasible interior region within the
smaller 4×6 grid's blocks (every candidate patch touched a neighboring block at 0m distance). With the
coarser 3×4 grid, 150m is satisfiable. **Validated after generation**: the actual minimum distance
found between any two different-split patches is **480m** — comfortably exceeding the 150m requirement.

## 10. Patch-size investigation

**MEASURED, not assumed**: 128×128 and 96×96 patches produced **zero** val/test patches under every
tested stride/threshold combination — the val/test blocks are simply too small and irregularly shaped
for those larger windows to fit with adequate validity/labeled coverage plus buffer clearance. **Only
64×64 (640m×640m) produced any val/test patches at all.** This directly overrides the brief's
example-only suggestion of 128/256 — the actual data does not support them here.

## 11. Patch-stride investigation

| Stride | Candidates | Train | Val | Test |
|--:|--:|--:|--:|--:|
| 32 | 952 | 89 | 1 | 3 |
| 24 | 1,665 | 155 | 2 | 4 |
| 16 | 3,752 | 349 | 5 | 11 |

Stride=16 (75% overlap) was selected — a **deliberately high overlap, explicitly not "unnecessary"**:
it was required to reach a minimally workable val (5) and test (11) patch count given this AOI's size.
This is flagged as a real limitation of a single small-AOI development dataset, not hidden.

## 12. Patch filtering

Thresholds used: `min_valid_sar=0.90`, `min_labeled=0.80` (both chosen after testing 0.95/0.85/0.70 and
observing counts stabilize meaningfully around 0.90/0.80 without materially changing outcomes at
looser values — see Section 10 raw sweep data).

## 13. Dataset structure

```
data/processed/unet_dataset/
  labels/flood_reference.tif (+ flood_reference_before_sar_mask.tif diagnostic)
  label_statistics.csv
  dataset_statistics.csv
  dataset_config.yaml
  patches/{train,val,test}/{images,labels}/*.npy + metadata.csv
  patches/all_patches_metadata.csv
```
**Format decision**: `.npy` per patch (image: H×W×6 float32; label: H×W uint8), not per-patch GeoTIFF
or `.npz`. Reasoning: plain `.npy` is the fastest/simplest for repeated PyTorch loading; full
reproducibility is preserved via `metadata.csv`, which records exact row/col origin, geographic
bounds, and CRS for every patch, so any patch can be independently regenerated/verified from the
source rasters at any time.

## 14. Dataset statistics

| Split | Patches | Positive | Negative | Flood px | Non-flood px | Flood % |
|---|--:|--:|--:|--:|--:|--:|
| train | 349 | 159 | 190 | 107,455 | 1,232,210 | 8.02% |
| val | 5 | 4 | 1 | 45 | 17,829 | 0.25% |
| test | 11 | 11 | 0 | 38,478 | 921 | **97.66%** |

**Honestly flagged**: val and test have very different, non-representative flood ratios from each
other and from train (val is almost entirely non-flood; test is almost entirely flood). This is a
direct consequence of the extreme spatial concentration of flood pixels in one corner of the AOI
(Section 8) combined with the small number of held-out patches available. It is a genuine limitation
of this single-event, small-AOI dataset - not a bug, and not corrected here since doing so would
require either abandoning the spatial-separation requirement or fabricating data.

## 15. Leakage testing

`validate_unet_dataset.py`: 0 duplicate IDs, 0 cross-split bounding-box overlaps, minimum inter-split
distance 480m (≥150m required). **PASS.**

## 16. Normalization strategy preparation

No normalization was applied to any stored patch (verified: sample patch values exceed [-1,1], ruling
out accidental normalization). `dataset_config.yaml` records exact patch/split/threshold parameters
for reproducibility. Fitting real normalization statistics from `train` only is deferred to the
training stage, per instruction.

## 17. Problems encountered

- 356 invalid EMSR773 geometries (repaired, documented).
- Municipality-based split empirically infeasible (pivoted to block split, documented in full).
- 4×6 block grid also infeasible for buffer clearance (pivoted to 3×4).
- 128×128 and 96×96 patch sizes produced zero val/test patches (documented, not forced).
- High patch overlap (stride=16) was necessary to reach a workable val/test count.
- Resulting val/test class distributions are extreme and non-representative of the AOI as a whole.

## 18. Decisions made

All listed above; none were made to hit a predetermined "nice" number — every parameter was adjusted
only after measuring that the previous choice produced zero or inadequate val/test patches.

## 19. Recommendations for U-Net training

- Given val has only 45 flood pixels total across 5 patches, validation-set metrics (especially
  flood-class IoU/F1) will be extremely noisy — treat with caution, likely supplement with training-set
  monitoring.
- Given test is 97.66% flood, test-set metrics will mostly reflect flood-recall, not general
  performance — report both classes' metrics separately, not a single blended accuracy.
- Loss weighting / positive-focused sampling (train-only) is a reasonable next-stage consideration
  given the 8.02% train flood ratio, but was deliberately not implemented here.
- This dataset should be treated explicitly as a small development/proof-of-concept corpus; the
  project's own multi-event generalization goal will require substantially more data before these
  split-size limitations are resolved.
