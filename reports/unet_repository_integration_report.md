# UrbanFlood360 — U-Net Repository Integration Report

## 1. Repository inspected

https://github.com/chathumal93/Pytorch-UNet-Flood-Segmentation, commit `717e0ab` (cloned into
`external/Pytorch-UNet-Flood-Segmentation/`). Inspected: `models.py`, `helpers.py`,
`make-dataset.py`, `model-train.ipynb`, `model-inference.ipynb`, `requirements.txt`, `README.md`.
Full attribution record: `docs/UNET_REPOSITORY.md`.

## 2. Original architecture

**REPOSITORY CODE.** 4-level encoder/decoder U-Net (`models.py`). Each conv block: `Conv3x3 → ReLU →
Dropout(0.2) → Conv3x3 → ReLU`. Downsampling via `MaxPool2d(2)`, upsampling via
`ConvTranspose2d(k=2,s=2)`, skip connections via channel concatenation, final `Conv2d(k=1)` +
`sigmoid` (probability output, not raw logits).

## 3. Original input-channel configuration

`in_channels=4` (VV, VH, SRTM DEM, JRC permanent water), `out_channels=1`. The `Unet` class already
takes `in_channels`/`out_channels` as constructor arguments — **no code modification to `model.py`
was needed**; only the call site changed from `Unet(4,1)` to `Unet(6,1)`.

## 4. UrbanFlood360 modifications

| Component | Repository behavior | UrbanFlood360 change | Why |
|---|---|---|---|
| Input channels | 4 (VV,VH,DEM,water) | 6 (VV/VH pre/post/change dB) | Project's validated feature set |
| Dataset class | `Sentinel1Dataset`: reads 4 separate GeoTIFFs, fixed-range normalization | `UrbanFlood360PatchDataset`: reads pre-stacked 6-channel `.npy` patches | Data format differs; normalization strategy differs (see below) |
| Normalization | Fixed physical ranges (VV/VH clip[-50,25]→[0,1]; DEM/2000; water/100) | Train-only fitted mean/std (percentile also implemented, not active) | Project instruction: derive from train split, not hardcoded constants |
| Device selection | `cuda` else `cpu` | `mps` → `cuda` → `cpu` | Apple M1 target hardware |
| Batch size | 32 (256×256 patches) | 8 (64×64 patches, 8GB unified memory) | Memory safety |
| NaN handling | Not needed (repo's inputs have no NaN) | Explicit NaN→0.0 fill after normalization, in `dataset.py` | Our patches allow up to 10% invalid pixels per patch (min_valid_sar≥0.90) |
| Two-model ensemble | Trains 2 models split by water% | Not used — single model | Explicit project instruction |
| Ignore-label masking | `y.ne(255)` + `masked_select` (already present!) | **Reused unchanged** | Already exactly matches our requirement |
| Loss / optimizer | `BCELoss`, `Adam(lr=0.001)`, no scheduler | **Reused unchanged** | No reason to deviate at baseline stage |

## 5. Final input/output shapes

Input: `[B, 6, 64, 64]`. Output: `[B, 1, 64, 64]` (sigmoid probability map). Verified programmatically
in the smoke test, not assumed.

## 6. Parameter count

Total = trainable = **7,760,961**. Original 4-channel version = 7,760,385. **Difference = 576**,
verified to exactly equal `(6-4) input channels × 32 output channels × 3×3 kernel` — i.e. confirmed
analytically and empirically that the *only* architectural change is the first convolution's input
width; every other layer is untouched.

## 7. Dataset integration

`UrbanFlood360PatchDataset` loads directly from `data/processed/unet_dataset/patches/{split}/`,
verified: train=301, val=14, test=13 patches (unchanged from the split-remediation stage). No patches
were regenerated; no values were altered (checksums of the frozen SAR stack and label raster
reconfirmed identical before and after this integration work).

## 8. Ignore-label implementation

The repository's own `y.ne(255)` + `masked_select` pattern (in both its `train()`/`test()` functions)
already does exactly what UrbanFlood360 needs and was reused without modification. Verified in the
smoke test: of a 32,768-pixel batch, 2,214 (6.8%) were ignore-labeled and confirmed excluded from the
loss computation (loss is computed only on the `masked_select`-filtered tensor, which by construction
cannot include ignored pixels — the ignored count is separately reported for verification, not derived
from the loss itself).

## 9. Normalization implementation

`Normalizer` class (`models/unet/normalize.py`), fitted from the 301 **train** patches only
(`models/unet/normalization_stats.json`). Inspected statistics before choosing a method: all six
channels show mean ≈ median with no extreme skew (e.g. VV_pre_dB: mean=-10.09, median=-10.56),
supporting `mean_std` (z-score) as the active default; `percentile` (p2/p98 clip-and-scale) is
implemented and available but not active. Statistics were computed **only** from train — val/test were
never touched during fitting (verified: the fitting script globs only `train/images/*.npy`).

## 10. MPS configuration

`torch.backends.mps.is_available()=True`, `is_built()=True`, PyTorch 2.10.0, confirmed in the `base`
conda environment (Python 3.13.5 — matches the stated project environment exactly). Device priority
implemented as MPS → CUDA → CPU. All operations used (`Conv2d`, `MaxPool2d`, `ConvTranspose2d`,
`sigmoid`, `masked_select`, `BCELoss`) ran successfully on MPS in the smoke test with no fallback or
CPU round-trips required inside the training step.

## 11. Smoke-test results

All checks passed on first execution (see `models/unet/smoke_test_result.json`):
- Batch shapes: images `[8,6,64,64]`, labels `[8,1,64,64]` — as required.
- Label values in batch: `{0.0, 1.0, 255.0}` only.
- 1,974 NaN pixels found and filled (post-normalization) in this batch — confirmed zero NaN/Inf reach
  the model.
- Forward pass: output `[8,1,64,64]`.
- Loss: 0.7686 (finite, non-NaN).
- Backward pass: all 46 trainable parameter tensors received a gradient; no NaN gradients.
- Optimizer step: succeeded.
- Validation batch: forward pass succeeded, loss=0.7422, IoU=0.0101 (a near-random IoU is expected
  and correct for an untrained, freshly-initialized model — this is not a training result).

## 12. Files modified/created

New: `models/unet/{model.py (copied+annotated), dataset.py, normalize.py, train.py, config.yaml,
normalization_stats.json, smoke_test_result.json}`, `docs/UNET_REPOSITORY.md`,
`reports/unet_repository_integration_report.md`, `external/Pytorch-UNet-Flood-Segmentation/` (cloned
reference copy, not modified). No UrbanFlood360 data files were modified.

## 13. Repository attribution/license

See `docs/UNET_REPOSITORY.md` in full. Key point: **the repository has no LICENSE file** — this is
stated plainly rather than assuming a permissive license, and full attribution (URL, commit, author,
files reused) is recorded.

## 14. Known limitations

- This is a smoke test only — no epochs of real training have been run; the reported loss/IoU reflect
  an untrained, randomly-initialized model and must not be read as model performance.
- Val (14 patches) and test (13 patches) are small, per the previous split-remediation stage's own
  documented limitations — unchanged here, not re-litigated.
- `mean_std` normalization was chosen from a reasonable but not exhaustive inspection of train
  statistics; it is not claimed to be provably optimal, only well-justified and swappable
  (`percentile` remains available in the same config).
- The repository's flip-augmentation transform (`RandomHorizontalFlip`/`RandomVerticalFlip`) was
  identified during inspection but not enabled for the smoke test, for simplicity; it remains a
  documented, available option for the actual training stage.
