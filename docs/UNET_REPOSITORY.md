# External Repository — Attribution and Integration Record

## Repository
- **URL**: https://github.com/chathumal93/Pytorch-UNet-Flood-Segmentation
- **Author**: chathumal93 (as stated in the repository; no organization given)
- **Commit used**: `717e0abad6d71d6fd4059aad8ad8a98a6ebc1286` (2024-05-21), cloned into `external/Pytorch-UNet-Flood-Segmentation/`
- **License**: **No `LICENSE` file is present in the repository at the cloned commit.** No explicit
  open-source license is granted by the author anywhere in the repo. This is recorded honestly rather
  than assumed — UrbanFlood360's use of this code is a research/educational adaptation with full
  attribution, consistent with the repository's own README "Acknowledgments" convention (it itself
  credits the original U-Net paper without a formal license). Anyone redistributing UrbanFlood360
  more broadly should be aware no license terms have been granted by the upstream author and should
  consider contacting them directly if that matters for their use case.
- **Upstream acknowledgment** (from their README): "The implementation is based on the original U-Net
  paper: https://arxiv.org/abs/1505.04597."

## Files inspected
`models.py`, `helpers.py`, `make-dataset.py`, `make-features.ipynb`, `model-train.ipynb`,
`model-inference.ipynb`, `requirements.txt`, `README.md`.

## Files adapted into UrbanFlood360
- `models.py` → `models/unet/model.py` (copied near-verbatim; the `Unet` class already accepts
  `in_channels`/`out_channels` as constructor arguments, so **no architectural code change was
  required** to move from their 4-channel to our 6-channel input — only the call site changed to
  `Unet(6, 1)`).
- Training loop structure (`train()`, `test()`, `IoU()` functions, and their ignore-label masking
  pattern `y.ne(255)` + `masked_select`) from `model-train.ipynb` → reused with modifications in
  `models/unet/train.py` (device selection, batch size, dataset class, normalization).

## Files NOT reused
- `helpers.py`'s `tile256`/`restgee_data` (Sentinel-1 tiling + Google Earth Engine DEM/water download)
  — not applicable, UrbanFlood360 has its own validated preprocessing pipeline.
- `make-dataset.py`, `make-features.ipynb`, `gee-dem-data.py`, `gee-pwater-data.py` — their original
  data acquisition pipeline; not used at all.
- Their `Sentinel1Dataset` class — reads 4 separate per-band GeoTIFFs (VV, VH, DEM, JRC permanent
  water) per sample with fixed physical-range normalization. Not reused; UrbanFlood360 patches are
  already-stacked 6-channel `.npy` arrays, so a new, simpler dataset class was written instead.
- The two-model "water percentage grouping" ensemble strategy from `model-train.ipynb`/
  `model-inference.ipynb` — explicitly excluded per project instructions; UrbanFlood360 trains one
  model on one dataset.
- Their exact fixed-range normalization constants (VV/VH clipped to [-50,25] dB, DEM/[0,2000],
  water/[0,100]) — not reused; UrbanFlood360 fits normalization from its own train-split statistics
  instead (see `reports/unet_repository_integration_report.md`).

## Original architecture summary (unmodified)
4-level encoder/decoder U-Net, each conv block = `Conv3x3 → ReLU → Dropout(0.2) → Conv3x3 → ReLU`,
`MaxPool2d(2)` downsampling, `ConvTranspose2d(k=2,s=2)` upsampling, skip connections via channel
concatenation, final `Conv2d(k=1)` + `sigmoid` producing a per-pixel probability directly (not raw
logits). Original: `in_channels=4, out_channels=1`. Loss: `torch.nn.BCELoss()` on `sigmoid` output,
computed only over pixels where the label is not 255 (`y.ne(255)` mask). Optimizer: `Adam(lr=0.001)`,
no scheduler. Metric: IoU at a fixed 0.5 threshold, also `255`-aware.
