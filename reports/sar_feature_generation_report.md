# UrbanFlood360 — SAR Feature Generation Report

Development case: Valencia DANA 2024, Horta Sud AOI. PRE = 2024-10-19, POST = 2024-10-31,
Sentinel-1A IW GRD, relative orbit 103, ascending, VV+VH.

## 1. Input datasets

- PRE:  `data/processed/sentinel1/prototype/20241019_R103/corrected/db/S1_PRE_20241019_R103_sigma0_db.tif`
- POST: `data/processed/sentinel1/post/20241031_R103/corrected/db/S1_POST_20241031_R103_sigma0_db.tif`

Both are the validated, co-registered, AOI-masked dB Sigma0 products from prior stages. VV/VH bands
were identified programmatically by band description (`find_band()` in `generate_sar_features.py`),
not assumed by position — both files confirmed `{'VH': band 1, 'VV': band 2}`.

## 2. Input raster metadata (both PRE and POST — confirmed identical)

CRS: EPSG:25830 · Transform: `|10,0,718825|0,-10,4370105|` · Width×Height: 1120×951 ·
dtype: float32 · NoData: NaN.

## 3. Feature definitions

| Feature | Definition |
|---|---|
| VV_pre_dB | PRE Sigma0_VV_db, unmodified |
| VH_pre_dB | PRE Sigma0_VH_db, unmodified |
| VV_post_dB | POST Sigma0_VV_db, unmodified |
| VH_post_dB | POST Sigma0_VH_db, unmodified |
| VV_change_dB | POST − PRE (VV), computed only where both valid |
| VH_change_dB | POST − PRE (VH), computed only where both valid |

## 4. NoData handling

Explicit per-pixel validity masks were built (`pre_valid`, `post_valid`) rather than relying only on
NaN arithmetic propagation, then cross-checked against plain NaN-propagating subtraction — **0
mismatches** confirmed for both VV and VH. Change = NaN unless both PRE and POST are valid at that
pixel. Measured: PRE valid = POST valid = both-valid = 416,329 pixels (39.09%) for both polarizations
— identical counts because PRE and POST already share the exact same AOI mask from prior stages.

## 5. Output products

All under `data/processed/sar_features/`:
`VV_pre_dB.tif`, `VH_pre_dB.tif`, `VV_post_dB.tif`, `VH_post_dB.tif`, `VV_change_dB.tif`,
`VH_change_dB.tif`, `sar_features_6ch.tif` (bands 1–6 in the exact specified order),
`sar_feature_statistics.csv`, `validation_summary.json`.

## 6. Metadata validation

Confirmed via `gdalinfo` on every output: CRS EPSG:25830, 1120×951, origin (718825, 4370105),
10m×10m pixels, Float32, NoData=nan, explicit band descriptions matching the feature names exactly.
`validate_sar_features.py` run against the outputs: **0 problems found, exit code 0**.

## 7. Statistical summary

See `sar_feature_statistics.csv` for the full table (count, min/max/mean/median/std, p1/p2/p5/p25/p75/p95/p98/p99).
All six channels report 416,329 valid pixels (39.09%), consistent with the AOI mask.

## 8. Change distribution analysis

| | VV_change_dB | VH_change_dB |
|---|---|---|
| mean | −1.85 | −2.48 |
| median | +0.22 | −0.11 |
| std | 5.62 | 6.18 |
| % below −3 dB | 29.9% | 30.8% |
| % below −5 dB | 27.0% | 27.2% |
| % below −10 dB | 15.0% | 18.3% |
| % above +3 dB | 14.9% | 12.7% |
| % above +5 dB | 3.8% | 2.9% |
| % above +10 dB | 0.21% | 0.06% |

**FACT (measured)**: both change histograms are clearly **bimodal** — a dominant peak near 0 dB plus a
distinct, well-separated secondary mode at roughly −10 to −12 dB (see
`reports/figures/sar_features/change_distributions.png`). This is not a long tail; it is a genuinely
separate population of pixels.

**INTERPRETATION (not a conclusion)**: bimodal change distributions of this kind are commonly reported
in SAR change-detection literature as consistent with a mixture of "changed" and "unchanged" surface
populations. This pattern is **not** being labeled as flood extent at this stage — no threshold,
label, or classification has been assigned to any pixel.

## 9. Extreme-value investigation

An initial extraction of "most positive" pixels had a bug — `np.argsort` on a NaN-containing array
places NaN at the end, so the naive "top-N" slice returned NaN pixel locations, not real extremes.
This was caught, explicitly reported (not silently fixed), and corrected by sorting only valid-pixel
indices.

**FACT (measured)**: extreme change pixels (beyond the 0.5th/99.5th percentile) have a median distance
to the raster edge of 237px (VV) / 193px (VH), versus 259px for all valid pixels, and a *minimum*
edge-distance of 13px (VV) / 9px (VH) — i.e., **extreme values are not concentrated at the raster
border or AOI mask boundary**; they are genuine interior values. The most negative VV/VH pixels cluster
tightly in small, spatially coherent groups (e.g., several adjacent pixels around map coordinates
(726390, 4362860) and (727940, 4363760)), and these locations fall within the same large-scale
southeastern region that earlier project stages already flagged as showing substantial PRE→POST
change. The most positive pixels cluster elsewhere (e.g., around (725790, 4363160) and (725320,
4365130)), in different, small, tight groups.

**INTERPRETATION**: tight spatial clustering (rather than random scatter) is more consistent with
genuine, spatially coherent SAR variation (water-related backscatter change, urban/building
scattering change, or vegetation change are all plausible categories per the brief's list) than with
random noise or a residual processing artifact. Distinguishing between these specific physical causes
is **not possible at this stage** without independent data (optical imagery, land cover, building
footprints) that is explicitly out of scope for this prompt. No pixel here is labeled as flood,
shadow/layover, or artifact — the evidence available only supports ruling out edge/masking artifacts.

## 10. Visualization summary

- `pre_post_quicklooks.png`: PRE/POST VV and VH, fixed display range consistent with prior stages.
- `change_maps_full_vs_robust.png`: full-range and robust symmetric-percentile (2nd/98th, ±13.6 dB VV /
  ±16.6 dB VH) versions of both change maps, AOI boundary overlaid. No rim/edge artifact is visible
  along the AOI boundary in any panel. The southeastern coherent change region follows real linear
  features (canals/roads remain visible as brighter streaks within it) rather than looking like a
  scan-line or block artifact.
- **New factual observation from this visualization**: VH_change_dB is positive across almost the
  entire western/central built-up core of the AOI (not just isolated points), while VV_change_dB in
  the same area is closer to neutral/mixed with scattered localized positive spots. This asymmetry
  between polarizations is reported as an observation only.
- `histograms_individual.png`, `pre_vs_post_comparison.png`, `change_distributions.png`: as specified.

## 11. Normalization investigation

Using the actual measured percentiles (not assumed):

| Channel | PRE p2/p98 | POST p2/p98 |
|---|---|---|
| VV | −15.09 / −0.13 | −22.29 / +1.59 |
| VH | −22.79 / −7.50 | −34.01 / −6.99 |

**FACT**: PRE and POST have measurably different percentile ranges for the same nominal channel —
POST's spread is wider on both bands. This is concrete evidence for why **independent per-image
percentile normalization is risky**: if each date were clipped/scaled to its own 2nd–98th percentile
range, the same physical backscatter value would be mapped to a different normalized value in PRE
vs. POST purely because of that date's own distribution — silently injecting a fake "change" signal
that has nothing to do with the real scene.

| Option | Mechanism | Cross-event generalization implication |
|---|---|---|
| A — Robust percentile normalization | Clip to (e.g.) 2nd–98th percentile, scale to ~0–1 | Must use percentiles derived from the **training corpus**, not per-image, per the reasoning above. A single Valencia-only percentile set is not yet a valid final choice for a system meant to generalize to other events. |
| B — Standardization (mean/std) | Subtract mean, divide by std | Same constraint — mean/std must come from the training dataset/multiple events, not this single development scene, or the network will implicitly learn Valencia-specific radiometry as "normal." |
| C — Fixed physical ranges | Fixed literature-typical Sigma0 dB bounds, same for every event | Most generalizable in principle (event-independent), but requires literature-grounded bounds validated across sensor conditions, not chosen from this one scene. |

**Decision for this stage**: none of A/B/C is committed. Raw six-channel features (in physical dB
units) were produced as the primary, authoritative output. The percentile/statistics tables in this
report are exactly what a future normalization-fitting stage (across multiple training events) will
need — they are not, themselves, a normalization commitment.

## 12. Problems encountered

- The extreme-value "most positive" extraction bug (Section 9) — caught and fixed, not hidden.
- No grid, CRS, band-identity, or NoData problems were found anywhere in this stage.

## 13. Decisions made

- Change computed as POST − PRE (not reversed), per specification.
- Explicit validity masks built and cross-checked against NaN-propagation rather than trusting either
  approach blindly.
- No normalization applied to the primary outputs.
- No extreme values were removed or altered.

## 14. Recommendations for the next stage

- Carry the bimodality finding (Section 8) and the SE spatial clustering finding (Section 9) forward
  as context when patches/labels are eventually introduced — but that comparison must happen in a
  later, explicitly-scoped stage (this one deliberately did not touch EMSR773 or any label source).
- When more training events are eventually available, revisit Section 11 to fit a real normalization
  strategy rather than deciding it from Valencia alone.
