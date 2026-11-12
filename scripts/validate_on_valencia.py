#!/usr/bin/env python3
"""
UrbanFlood360 - score any flood map of the Valencia AOI against the Copernicus EMSR773 flood extent.

Use it to check how a model does on data the dashboard downloads by itself (different orbit, gamma0 instead of
sigma0), and to compare models (e.g. the Valencia-only U-Net v2 vs. a model trained on UrbanSARFloods).

The flood mask (1 = flood, 0 = not, 255 = no data, any CRS) is resampled onto the EMSR773 label grid
(nearest neighbour). Dice / precision / recall are reported for all Copernicus-imaged pixels and for the
held-out test block (never used for training or model selection).

Usage:
    python scripts/validate_on_valencia.py MASK.tif [MASK2.tif ...]

Reference results (flood-affected target, test block):
    SNAP sigma0, orbit 103, 19 -> 31 Oct 2024 (training data):     Dice 0.71
    Planetary Computer gamma0, orbit 110, 20 Oct -> 1 Nov 2024:   Dice 0.46  (same U-Net v2)
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet_v2.data import SPLIT_CODES, block_split_map  # noqa: E402

LABEL = ROOT / "data/processed/flood_affected_dataset/labels/flood_reference.tif"


def score(mask_path, label_path=LABEL):
    with rasterio.open(label_path) as L:
        lab, T, C = L.read(1), L.transform, L.crs
    with rasterio.open(mask_path) as r:
        m = np.full(lab.shape, 255, "uint8")
        reproject(r.read(1), m, src_transform=r.transform, src_crs=r.crs, src_nodata=255, dst_transform=T,
                  dst_crs=C, dst_nodata=255, resampling=Resampling.nearest)
    split = block_split_map(lab.shape)
    rows = []
    for region, sel in [("all imaged", np.ones_like(lab, bool)), ("test block", split == SPLIT_CODES["test"])]:
        ok = sel & (lab != 255) & (m != 255)
        p, t = m[ok] == 1, lab[ok] == 1
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        rows.append(dict(region=region, dice=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                         recall=tp / max(tp + fn, 1), predicted_flood_pct=100 * p.mean(),
                         reference_flood_pct=100 * t.mean(), n=int(ok.sum())))
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        print(path)
        for r in score(path):
            print(f"  {r['region']:11s} Dice {r['dice']:.3f}  P {r['precision']:.2f}  R {r['recall']:.2f}  "
                  f"flood predicted {r['predicted_flood_pct']:.0f}% vs reference {r['reference_flood_pct']:.0f}%"
                  f"  ({r['n']:,} px)")


if __name__ == "__main__":
    main()
