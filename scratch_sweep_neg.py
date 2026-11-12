import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

sar_stack = "data/processed/sar_features/sar_features_6ch.tif"
label_path = "data/processed/unet_dataset/labels/flood_reference.tif"

with rasterio.open(sar_stack) as src:
    sar = src.read(); transform = src.transform; H, W = src.height, src.width
with rasterio.open(label_path) as src:
    label = src.read(1)

sar_valid = np.all(~np.isnan(sar), axis=0)
LABEL_IGNORE = 255; LABEL_FLOOD = 1
block_assignment = {(1, 2): "val", (2, 1): "test"}
row_edges = np.linspace(0, H, 5).astype(int)
col_edges = np.linspace(0, W, 5).astype(int)
split_union = {"train": [], "val": [], "test": []}
for i in range(4):
    for j in range(4):
        r0, r1 = row_edges[i], row_edges[i+1]; c0, c1 = col_edges[j], col_edges[j+1]
        x0, y0 = transform * (c0, r0); x1, y1 = transform * (c1, r1)
        g = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        s = block_assignment.get((i, j), "train")
        split_union[s].append(g)
buffer_ref_geoms = {s: geoms for s, geoms in split_union.items()}
split_union_final = {s: gpd.GeoSeries(g).union_all() for s, g in split_union.items()}

ps, st, buffer_m = 128, 16, 150.0


def patch_bounds(row0, col0):
    x0, y0 = transform * (col0, row0); x1, y1 = transform * (col0 + ps, row0 + ps)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


rows = []
for row0 in range(0, H - ps + 1, st):
    for col0 in range(0, W - ps + 1, st):
        pbounds = patch_bounds(row0, col0); pcenter = pbounds.centroid
        owning = None
        for s, geom in split_union_final.items():
            if geom.contains(pcenter):
                owning = s; break
        if owning not in ("val", "test"):
            continue
        lbl_patch = label[row0:row0 + ps, col0:col0 + ps]
        is_pos = (lbl_patch == LABEL_FLOOD).sum() > 0
        if is_pos:
            continue
        valid_sar_frac = sar_valid[row0:row0 + ps, col0:col0 + ps].mean()
        labeled_frac = (lbl_patch != LABEL_IGNORE).mean()
        too_close = False
        for s2, geoms in buffer_ref_geoms.items():
            if s2 == owning:
                continue
            for g in geoms:
                if pbounds.distance(g) < buffer_m:
                    too_close = True; break
            if too_close:
                break
        rows.append((owning, row0, col0, round(valid_sar_frac, 3), round(labeled_frac, 3), too_close))

print(f"{'split':6} {'row0':5} {'col0':5} {'valid_sar':10} {'labeled':8} {'buffer_reject'}")
for r in rows:
    print(r)
