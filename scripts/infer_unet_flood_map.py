#!/usr/bin/env python3
"""
UrbanFlood360 - local inference: 6-channel SAR feature stack -> georeferenced
flood-map GeoTIFF, using the clean repository U-Net baseline checkpoint.

This is the missing final step of the local workflow (SAR stack -> U-Net ->
GeoTIFF). No training, no experimentation - it only runs the already-trained
baseline model over the full raster and writes a georeferenced prediction.

Tiling: the source raster is covered by non-overlapping 64x64 windows
(padding the last row/column of tiles as needed), matching the patch size
the model was trained on. Each tile is normalized with the same train-only
statistics used in training, run through the model, and stitched back into
the full-raster output array at its original position.

Output: a 2-band GeoTIFF, same CRS/transform/extent as the input SAR stack:
  Band 1 (float32): predicted flood probability, NaN where the input SAR
                     data itself was invalid (matches the "SAR-invalid" /
                     ignore convention used throughout this project).
  Band 2 (uint8):    binary flood mask at threshold 0.5 (the baseline's
                      un-optimized default - see reports/unet_baseline_training_report.md).
                      255 = ignore (invalid input), 1 = flood, 0 = non-flood.

Usage:
    python scripts/infer_unet_flood_map.py \
        --sar-stack data/processed/sar_features/sar_features_6ch.tif \
        --checkpoint models/unet/baseline/best_model.pt \
        --normalization-stats models/unet/normalization_stats.json \
        --output data/processed/predictions/flood_prediction.tif
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "unet"))
from model import Unet  # noqa: E402
from normalize import Normalizer  # noqa: E402

PATCH_SIZE = 64
THRESHOLD = 0.5  # baseline default, not optimized - see reports/unet_baseline_training_report.md
IGNORE_OUT = 255


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sar-stack", required=True, help="6-channel SAR GeoTIFF (VV_pre,VH_pre,VV_post,VH_post,VV_change,VH_change), band order must match training")
    p.add_argument("--checkpoint", default="models/unet/baseline/best_model.pt")
    p.add_argument("--normalization-stats", default="models/unet/normalization_stats.json")
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    with rasterio.open(args.sar_stack) as src:
        sar = src.read()  # (6,H,W)
        profile = src.profile
        H, W = src.height, src.width
        assert src.count == 6, f"Expected 6 bands, got {src.count}"

    normalizer = Normalizer.from_json(args.normalization_stats)
    model = Unet(6, 1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Raster: {H}x{W}, tiling in non-overlapping {PATCH_SIZE}x{PATCH_SIZE} windows")

    prob_full = np.full((H, W), np.nan, dtype=np.float32)
    sar_valid_full = np.all(~np.isnan(sar), axis=0)

    n_tiles = 0
    with torch.no_grad():
        for row0 in range(0, H, PATCH_SIZE):
            for col0 in range(0, W, PATCH_SIZE):
                row1 = min(row0 + PATCH_SIZE, H)
                col1 = min(col0 + PATCH_SIZE, W)
                tile = sar[:, row0:row1, col0:col1]  # (6,h,w), h/w may be < PATCH_SIZE at edges

                # pad to PATCH_SIZE x PATCH_SIZE (model expects a fixed tile size)
                h, w = tile.shape[1], tile.shape[2]
                if h < PATCH_SIZE or w < PATCH_SIZE:
                    padded = np.full((6, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
                    padded[:, :h, :w] = tile
                    tile = padded

                img = np.transpose(tile, (1, 2, 0))  # (H,W,6)
                img_n = normalizer.apply(img.copy())
                img_n = np.where(np.isnan(img_n), 0.0, img_n).astype("float32")
                img_t = torch.from_numpy(np.transpose(img_n, (2, 0, 1))).unsqueeze(0).to(device)

                prob_tile = model(img_t).cpu().squeeze(0).squeeze(0).numpy()  # (PATCH_SIZE,PATCH_SIZE)
                prob_full[row0:row1, col0:col1] = prob_tile[:h, :w]
                n_tiles += 1

    print(f"Ran inference on {n_tiles} tiles")

    prob_full = np.where(sar_valid_full, prob_full, np.nan)
    mask_full = np.where(sar_valid_full, (prob_full >= args.threshold).astype(np.uint8), IGNORE_OUT)

    out_profile = profile.copy()
    out_profile.update(count=2, dtype="float32", nodata=None)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(prob_full.astype("float32"), 1)
        dst.write(mask_full.astype("float32"), 2)
        dst.set_band_description(1, "flood_probability")
        dst.set_band_description(2, f"flood_mask_threshold_{args.threshold} (0=non-flood,1=flood,255=ignore)")

    n_valid = int(sar_valid_full.sum())
    n_flood = int(((mask_full == 1)).sum())
    print(f"\nWrote {out_path}")
    print(f"Valid pixels: {n_valid} / {H*W} | Predicted flood pixels: {n_flood} "
          f"({100*n_flood/max(n_valid,1):.2f}% of valid area) at threshold={args.threshold}")


if __name__ == "__main__":
    main()
