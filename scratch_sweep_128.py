import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

sar_stack = "data/processed/sar_features/sar_features_6ch.tif"
label_path = "data/processed/unet_dataset/labels/flood_reference.tif"

with rasterio.open(sar_stack) as src:
    sar = src.read()
    transform = src.transform
    H, W = src.height, src.width

with rasterio.open(label_path) as src:
    label = src.read(1)

sar_valid = np.all(~np.isnan(sar), axis=0)
LABEL_IGNORE = 255
LABEL_FLOOD = 1

block_grid = (4, 4)
block_assignment = {(1, 2): "val", (2, 1): "test"}
n_rows_grid, n_cols_grid = block_grid
row_edges = np.linspace(0, H, n_rows_grid + 1).astype(int)
col_edges = np.linspace(0, W, n_cols_grid + 1).astype(int)
block_geom = {}
split_union = {"train": [], "val": [], "test": []}
for i in range(n_rows_grid):
    for j in range(n_cols_grid):
        r0, r1 = row_edges[i], row_edges[i + 1]
        c0, c1 = col_edges[j], col_edges[j + 1]
        x0, y0 = transform * (c0, r0)
        x1, y1 = transform * (c1, r1)
        g = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        block_geom[(i, j)] = g
        s = block_assignment.get((i, j), "train")
        split_union[s].append(g)
buffer_ref_geoms = {s: geoms for s, geoms in split_union.items()}
split_union_final = {s: gpd.GeoSeries(geoms).union_all() for s, geoms in split_union.items()}

ps, st, buffer_m = 128, 16, 150.0


def patch_bounds(row0, col0):
    x0, y0 = transform * (col0, row0)
    x1, y1 = transform * (col0 + ps, row0 + ps)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


candidates = []
for row0 in range(0, H - ps + 1, st):
    for col0 in range(0, W - ps + 1, st):
        candidates.append((row0, col0))

owning = {}
for row0, col0 in candidates:
    pbounds = patch_bounds(row0, col0)
    pcenter = pbounds.centroid
    owning_split = None
    for s, geom in split_union_final.items():
        if geom.contains(pcenter):
            owning_split = s
            break
    owning[(row0, col0)] = owning_split

precomp = {}
for row0, col0 in candidates:
    s = owning[(row0, col0)]
    if s not in ("val", "test"):
        continue
    sar_patch_valid = sar_valid[row0:row0 + ps, col0:col0 + ps]
    lbl_patch = label[row0:row0 + ps, col0:col0 + ps]
    valid_sar_frac = sar_patch_valid.mean()
    labeled_frac = (lbl_patch != LABEL_IGNORE).mean()
    flood_frac_of_labeled = ((lbl_patch == LABEL_FLOOD).sum() / max(1, (lbl_patch != LABEL_IGNORE).sum()))
    pbounds = patch_bounds(row0, col0)
    too_close = False
    for os_, geoms in buffer_ref_geoms.items():
        if os_ == s:
            continue
        for g in geoms:
            if pbounds.distance(g) < buffer_m:
                too_close = True
                break
        if too_close:
            break
    precomp[(row0, col0)] = dict(split=s, valid_sar_frac=valid_sar_frac, labeled_frac=labeled_frac,
                                  too_close=too_close, is_positive=(lbl_patch == LABEL_FLOOD).sum() > 0)

for min_valid_sar, min_labeled in [(0.9, 0.8), (0.85, 0.75), (0.8, 0.7), (0.75, 0.6), (0.7, 0.5), (0.65, 0.45), (0.6, 0.4)]:
    counts = {"val": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}}
    for (row0, col0), d in precomp.items():
        if d["too_close"]:
            continue
        if d["valid_sar_frac"] < min_valid_sar or d["labeled_frac"] < min_labeled:
            continue
        counts[d["split"]]["pos" if d["is_positive"] else "neg"] += 1
    print(f"min_valid_sar={min_valid_sar} min_labeled={min_labeled}: "
          f"val(pos={counts['val']['pos']},neg={counts['val']['neg']})  "
          f"test(pos={counts['test']['pos']},neg={counts['test']['neg']})")
