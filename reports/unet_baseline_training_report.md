# UrbanFlood360 — Baseline U-Net Training Report

## Dataset
Train: 301 patches. Validation: 14 patches. **Test: 13 patches — not accessed in this stage
(confirmed by code inspection: `scripts/train_unet_baseline.py` never instantiates a test dataset;
grep confirms zero references beyond a log line stating it was not loaded).**

## Model
Repository `Unet` architecture (chathumal93/Pytorch-UNet-Flood-Segmentation, commit `717e0ab`),
`in_channels=6`, `out_channels=1`. 7,760,961 parameters (unchanged from the integration stage).

## Normalization
Train-only z-score (`mean_std`), fitted from the 301 train patches, loaded from
`models/unet/normalization_stats.json`. Not refit; val/test never used for fitting.

## Training configuration
Optimizer: `Adam(lr=0.001)` (repository default, unmodified). Loss: `BCELoss` on `masked_select`-ed
valid pixels (repository default, unmodified, `y.ne(255)` ignore mask). Batch size: 8. Max epochs: 50,
early stopping patience: 10 (on validation loss). Seed: 42 (Python/NumPy/PyTorch). Device: MPS.
Threshold: 0.5 (baseline, not optimized).

**Augmentation: disabled.** Code inspection of the repository's `model-train.ipynb` found that its
flip-augmentation applies `self.transform(image)` and `self.target_transform(label)` as two
*separate* calls to the same stateful `RandomVerticalFlip(p=0.5)` transform — each call independently
samples its own random outcome, so image and label are not guaranteed to receive the same flip. This
is a latent correctness bug in the original notebook, not merely an SAR-inappropriateness issue.
Rather than silently re-implementing a fixed version (which the brief explicitly discourages as
"inventing a new augmentation" for this baseline), augmentation was disabled entirely and this
decision is documented here plainly.

## Results

| | Value |
|---|---|
| Best epoch (by val loss) | **8** |
| Epochs run | 18 (early-stopped, patience=10) |
| Best val loss | 0.1236 |
| Val Dice @ best epoch | 0.6752 |
| Val IoU @ best epoch | 0.5096 |
| Val precision @ best epoch | 0.8805 |
| Val recall @ best epoch | 0.5475 |
| Val specificity @ best epoch | 0.9951 |
| Val TP/TN/FP/FN @ best epoch | 1746 / 48467 / 237 / 1443 |

## Runtime
Mean epoch time: 8.79s. Total runtime: 158.2s (~2.6 min) for 18 epochs. No MPS errors, no operation
fallbacks to CPU, no memory pressure observed at batch size 8 — batch size 4 was not needed.

## Overfitting behavior (MEASURED, not merely asserted)
Train loss fell steadily to ~0.022 and train Dice/IoU/recall saturated near 0.94–0.97 by epoch ~4-6.
Validation loss did **not** show a clean monotonic increase after the epoch-8 minimum — it oscillated
in a noisy band (0.12–0.17) through epoch 18, consistent with the very small (14-patch) validation set
producing high-variance estimates rather than a textbook overfitting curve. The clearest, most
interpretable signal is the **persistent train/validation gap** itself (train Dice ≈0.97 vs. val Dice
≈0.65-0.68 throughout), which is a real and expected consequence of training a 7.76M-parameter model
on 301 patches, not evidence the pipeline is broken.

**New observation**: validation precision *declined* over training (≈0.95 → 0.78–0.82 by epoch 18)
while validation recall *increased* (≈0.37 → 0.54) — the model trended toward more liberal
(higher-recall, lower-precision) predictions on validation as training progressed, even as it became
increasingly confident and precise on train. Reported as an observation, not yet explained causally.

## Error analysis (validation set, best checkpoint, 4 representative patches — not cherry-picked to
only show successes)

Of 14 validation patches, only 4 contained any true-positive detections at all; several small-flood
patches were missed completely (predicted probability near-zero everywhere); two flood-free patches
received spurious false positives.

- **patch_00230 (IoU=0.79, relatively successful)**: reference flood region corresponds to a clearly
  visible, spatially coherent dark-blue (negative) region in `VV_change_dB`. The model's predicted
  probability closely tracks this region; the core is well detected (green/TP), with under-detection
  concentrated at thin, branching edges of the reference polygon (blue/FN) — plausibly a model
  resolution limitation on fine boundary shapes, or reference-label thin-boundary uncertainty; both
  are plausible, not distinguished here.
- **patch_00228 (IoU=0.23, partial success)**: reference shows scattered small flood specks plus one
  larger cluster. The model detected two coherent hotspots reasonably well but missed the scattered,
  spatially isolated single/few-pixel flood specks entirely. **Interpretation**: consistent with a
  CNN's inductive bias toward larger, spatially coherent patterns over sparse point-like signal —
  plausible, not proven.
- **patch_00212 (complete miss, 207 real flood px, 0 detected)**: predicted probability is uniformly
  near-zero across the entire patch. Its `VV_change_dB` panel shows only the routine mottled
  texture, without a strong, visually obvious negative-change blob like patch_00230's. **Interpretation
  (plausible, not confirmed)**: the reference-labeled flood here may correspond to weak backscatter
  change — consistent with the task's suggested categories of shallow water, vegetation-covered
  flooding, or narrow inundation below the sensor/model's effective sensitivity — or possibly
  reference-label uncertainty in this specific area (recall EMSR773's "Possibly damaged"-adjacent
  proximity-inference caveats documented in earlier project stages for the related Grading product).
- **patch_00226 (mixed failure)**: this patch actually contains 95 real flood pixels (not zero — an
  earlier internal plotting label mis-stated this and is corrected here), **all 95 missed** (FN), while
  the model additionally produced a confident, spatially distinct false-positive detection (40 px) at
  a *different* location with no mapped reference flood at all. That FP location does show a visible
  darker/reddish texture anomaly in `VV_change_dB`. **This is reported as an open, unresolved
  disagreement, not attributed definitively to either a model error or a reference-label gap** — per
  the explicit instruction not to assume EMSR773 is complete ground truth. The FP sits near the
  transition to the "outside image footprint" ignore region, which is also flagged as a plausible
  contributing factor (edge effect) without being asserted as the cause.

No false positive or false negative in this sample was attributed to a single cause with certainty;
multiple plausible explanations are given per instance, consistent with the brief's requirement not to
over-claim.

## Limitations (explicit)
- Validation has only 14 patches (11 positive / 3 negative pre-training-split-remediation-stage
  numbers were revised; current best-checkpoint evaluation used all 14) — metrics here carry
  substantial sampling variance and should not be treated as a precise, stable performance estimate.
- This remains a single-flood-event (Valencia DANA 2024), single-AOI development case; no claim of
  production or cross-event generalization is made or implied.
- The EMSR773 reference label itself carries documented uncertainty (Flood trace deliberately
  excluded as ambiguous; footprint coverage gaps; grading-adjacent proximity-inference caveats from
  earlier stages) — model/reference disagreements in this report are described as such, not
  automatically resolved in the model's favor or the reference's favor.
- Threshold (0.5) is a baseline convention, not tuned; a later validation-only experiment is expected
  to revisit it.
