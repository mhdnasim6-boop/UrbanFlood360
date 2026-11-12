#!/usr/bin/env python3
"""
UrbanFlood360 - full-AOI flood map from a trained U-Net v2 checkpoint.

Runs overlapping sliding-window inference over the whole 6-channel SAR stack
(models/unet_v2/data.py::predict_raster) and writes, on the exact SAR grid:

  flood_probability.tif  float32, flood probability, NaN = no SAR data / outside AOI
  flood_mask.tif         uint8, 1 = flood, 0 = non-flood, 255 = no data
  flood_polygons.gpkg    the flood mask vectorised (one polygon per connected patch)
  prediction_meta.json   checkpoint, threshold, flooded area

The decision threshold defaults to the one chosen on the validation block
during training (metrics_val.json next to the checkpoint).

Output goes to data/processed/predictions/unet_v2_<target>/ where <target> is
"water" or "affected", read from the checkpoint's training label.

Usage:
    python scripts/predict_flood_map.py --checkpoint models/unet_v2/runs/affected_aug/best_model.pt
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet_v2.data import checkpoint_target, load_rasters, normalize, predict_raster  # noqa: E402
from models.unet_v2.model import UNet  # noqa: E402
from models.unet_v2.train import get_device  # noqa: E402


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    m = ckpt["config"]["model"]
    model = UNet(6, 1, m["base_channels"], m["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, help="default: threshold in metrics_val.json")
    ap.add_argument("--sar-stack", default=str(ROOT / "data/processed/sar_features/sar_features_6ch.tif"))
    ap.add_argument("--label", default=str(ROOT / "data/processed/unet_dataset/labels/flood_reference.tif"))
    ap.add_argument("--output-dir", help="default: data/processed/predictions/unet_v2_<target>")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    threshold = args.threshold
    if threshold is None:
        threshold = json.load(open(ckpt_path.parent / "metrics_val.json"))["threshold"]

    device = get_device()
    model, ckpt = load_model(ckpt_path, device)
    sar, _, profile, _ = load_rasters(args.sar_stack, args.label)
    valid = np.all(~np.isnan(sar), axis=0)
    image = normalize(sar, ckpt["normalization"])
    inf = ckpt["config"]["inference"]
    prob = predict_raster(model, image, device, inf["tile"], inf["overlap"], inf["batch_size"], valid=valid)

    mask = np.full(prob.shape, 255, dtype="uint8")
    mask[valid] = (prob[valid] >= threshold).astype("uint8")

    target = checkpoint_target(ckpt)
    out = Path(args.output_dir or ROOT / f"data/processed/predictions/unet_v2_{target}")
    out.mkdir(parents=True, exist_ok=True)
    base = dict(driver="GTiff", height=prob.shape[0], width=prob.shape[1], count=1,
                crs=profile["crs"], transform=profile["transform"], compress="deflate")
    with rasterio.open(out / "flood_probability.tif", "w", dtype="float32", nodata=np.nan, **base) as dst:
        dst.write(prob, 1)
        dst.set_band_description(1, "flood_probability")
    with rasterio.open(out / "flood_mask.tif", "w", dtype="uint8", nodata=255, **base) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, f"flood_mask_threshold_{threshold}")

    polys = [shape(g) for g, v in shapes(mask, mask=mask == 1, transform=profile["transform"]) if v == 1]
    gdf = gpd.GeoDataFrame({"area_m2": [p.area for p in polys]}, geometry=polys, crs=profile["crs"])
    gdf.to_file(out / "flood_polygons.gpkg", driver="GPKG", layer="flood")

    px_area = abs(profile["transform"].a * profile["transform"].e)
    flooded_km2 = float((mask == 1).sum() * px_area / 1e6)
    meta = dict(checkpoint=str(ckpt_path.relative_to(ROOT)), target=target, epoch=ckpt["epoch"], threshold=threshold,
                device=str(device), flooded_km2=round(flooded_km2, 3),
                valid_km2=round(float(valid.sum() * px_area / 1e6), 3),
                n_flood_polygons=len(gdf))
    json.dump(meta, open(out / "prediction_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Wrote flood_probability.tif, flood_mask.tif, flood_polygons.gpkg -> {out}")


if __name__ == "__main__":
    main()
