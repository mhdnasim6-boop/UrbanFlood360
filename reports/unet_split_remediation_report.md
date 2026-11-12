# UrbanFlood360 — Spatial Split Remediation Report

## 1. Why the original split was inadequate

**MEASURED**: the prior 3×4 block split (val=block(0,1)+(0,2), test=block(2,2)) produced val=5 patches
(4 pos/1 neg) and test=11 patches (**11 pos / 0 neg**). A test set with zero negative examples cannot
support meaningful binary-segmentation evaluation — any threshold that predicts "flood everywhere"
scores perfectly on it. Val's 5 patches are too few for a stable signal.

## 2. Spatial distribution analysis

Per-municipality and per-grid-block flood/non-flood/ignore counts were recomputed and are consistent
with the prior stage's `label_statistics.csv`. **New finding this stage**: flood-pixel concentration is
so extreme that most individual grid blocks are either heavily flood-dominated (>60%) or nearly
flood-free (<2%) — genuinely "mixed" blocks (10–30% flood, at a size usable for evaluation) are rare
and grid-resolution-dependent, not present in every layout.

## 3. Candidate block layouts tested

Nine grid layouts were computed: 2×2, 2×3, 2×4, 3×3, 3×4, 3×5, 4×4, 4×5, 4×6 (statistics for all in
this session's working notes; full per-block table available on request). Block sizes ranged from
~4750m×5600m (2×2) down to ~2370m×1860m (4×6).

**Key measured finding**: block-level flood *percentage* does not reliably predict the realized
positive/negative *patch* mix after the min-valid-SAR/min-labeled filters are applied. E.g. grid
3×4 block(1,2) has a healthy 29.7% raw block flood ratio, yet **every surviving 64×64 patch sampled
from it turned out 100% positive** (confirmed in 3 separate candidate tests: c1, c6, c7) — the
filter-surviving sub-area of that block happens to concentrate in its flooded portion. This means
candidate blocks had to be validated by **actually running the patch pipeline**, not by trusting
block-level pixel ratios alone.

## 4. Buffer analysis

150m (carried over from the prior stage) was used as the baseline. **200m was tested on the winning
configuration and found to still avoid all overlap, but it eliminated val's negative patches entirely**
(val became 10 pos / 0 neg) — i.e. a larger buffer is geometrically feasible but not
methodologically better here, since it fails the harder requirement (both classes present in every
split). **150m was kept** as the buffer that satisfies both leakage-freedom and class-presence.

## 5. Candidate split table

Full data in `data/processed/unet_dataset/split_candidates.csv`. Summary:

| Candidate | Grid | Val region | Test region | Train | Val (pos/neg) | Test (pos/neg) | Verdict |
|---|---|---|---|--:|---|---|---|
| c1 | 3×4 | (2,2)+(0,1) | (1,2) | 260 | 15/1 | 81/0 | REJECTED - test 0% neg |
| c3 | 4×4 | (1,2) | (2,2) | 326 | 11/3 | 54/0 | REJECTED - test 0% neg |
| c5 | 4×5 | (1,3) | (2,3)+(2,2) | 313 | 7/0 | 55/4 | REJECTED - val 0% neg |
| c6 | 3×4 | (2,2) | (1,2) | 326 | 11/0 | 81/0 | REJECTED - both 0% neg |
| c7 | 3×4 | (1,2) | (2,2)+(0,0) | 164 | 81/0 | 49/36 | REJECTED - val 0% neg |
| **c9** | **4×4** | **(1,2)** | **(2,1)** | **301** | **11/3** | **3/10** | **SELECTED** |
| c9_buf200 | 4×4 | (1,2) | (2,1) | 301 | 10/0 | 3/8 | buffer=200 breaks val - not used |
| c10 | 3×5 | (1,3) | (2,2)+(0,1) | 235 | 50/0 | 14/2 | REJECTED - val 0% neg |
| c12 | 2×3 | (1,1) | (1,2) | 294 | 0/0 | 0/0 | REJECTED - test empty (buffer infeasible) |

No model-performance score was used to rank these — selection is based purely on the dataset-design
criteria in Section 16 of the brief (leakage-free, both classes present, adequate counts).

## 6. Selected split

Grid: **4×4** (blocks ≈2370m×2800m). `val = block(1,2)`, `test = block(2,1)`, buffer = 150m, all other
blocks → train.

## 7. Selection rationale

c9 is the **only** tested candidate where every split (train, val, test) contains both positive and
negative patches, satisfying Section 9's explicit hard requirement ("a test set with zero negative
patches is not acceptable"). It was not selected for a favorable flood percentage — its test set is in
fact negative-dominated (3 pos/10 neg) and its val set positive-dominated (11 pos/3 neg), an
asymmetry that is reported honestly, not hidden, per Section 9's acknowledgment that train/val/test
need not share identical flood prevalence.

## 8. Final patch statistics

| Split | Patches | Positive | Negative |
|---|--:|--:|--:|
| train | 301 | 167 | 134 |
| val | 14 | 11 | 3 |
| test | 13 | 3 | 10 |

## 9. Final flood/non-flood statistics

| Split | Flood px | Non-flood px | Flood % |
|---|--:|--:|--:|
| train | 106,751 | 1,037,220 | 9.3% |
| val | 3,189 | 48,704 | 6.1% |
| test | 51 | 48,684 | 0.1% |

## 10. Leakage results

`validate_unet_dataset.py` (updated to explicitly assert both classes present in val/test): 0
duplicate patch IDs, 0 cross-split bounding-box overlaps, minimum inter-split distance **480m**
(≥150m required). **PASS.**

## 11. Normalization leakage check

Confirmed: patch pixel values exceed [-1,1] range (i.e. not accidentally pre-normalized); no
normalization statistics have been computed anywhere in this pipeline; `dataset_config.yaml` contains
only patch/split/threshold parameters, no fitted scale/offset values.

## 12. Limitations

- **Val and test are small (14 and 13 patches)** and have very different, non-representative flood
  ratios from each other (6.1% vs 0.1%) and from train (9.3%) — a direct, unavoidable consequence of
  this AOI's extreme spatial flood concentration in one corner, not a methodology error.
- Test's 3 positive patches mean flood-class test metrics will still be statistically noisy, just no
  longer *undefined* (as they would be with 0 negative or 0 positive patches).
- The selected val/test blocks cross municipality boundaries (val overlaps Alfafar/Massanassa/Sedaví;
  test overlaps Catarroja/Massanassa) — recorded as metadata, not used as a splitting criterion, per
  instruction.
- This remains a small, single-event development dataset; these limitations are expected to ease once
  additional flood events are incorporated in later project stages.

## 13. Recommendation for U-Net training

- Report train/val/test flood-class metrics separately, never a single blended accuracy, given the
  large prevalence differences between splits.
- Treat val-based checkpoint selection cautiously given only 14 patches; consider tracking train-loss
  trends as a secondary signal.
- Fit any future normalization strictly on the 301 train patches, never on val/test.
