#!/usr/bin/env python3
"""
UrbanFlood360 - U-Net patch dataset generation.

Generates fixed-size, spatially-split, deterministic training patches from
the validated 6-channel SAR stack and the flood reference label raster.
Does NOT resample/reproject the source rasters - patches are windows read
directly off the shared grid.

Spatial split is municipality-based (not random pixel/patch split), with a
geographic buffer enforced between patches assigned to different splits to
avoid spatial leakage from autocorrelated SAR/urban structure.

Usage:
    python create_unet_patches.py \
        --sar-stack data/processed/sar_features/sar_features_6ch.tif \
        --label data/processed/unet_dataset/labels/flood_reference.tif \
        --municipalities data/gis/aoi/municipalities_6.geojson \
        --split-config split_config.json \
        --output-dir data/processed/unet_dataset/patches \
        --patch-size 128 --stride 64 --buffer-m 200 \
        --min-valid-sar 0.95 --min-labeled 0.80 --seed 42
"""
import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

LABEL_FLOOD = 1
LABEL_NONFLOOD = 0
LABEL_IGNORE = 255

DEFAULT_SPLIT = {
    "train": ["Catarroja", "Paiporta", "Picanya", "Sedavi"],
    "val": ["Massanassa"],
    "test": ["Alfafar"],
}

# Spatial-BLOCK split (recommended default - see report). Municipality-based
# splitting was empirically found to be infeasible for this AOI: the six
# towns are small and directly interlocking, so every candidate patch
# centered in Massanassa or Alfafar was found to be 0m from a neighboring
# municipality's border - no buffer could be satisfied. Blocks are defined
# on a fixed 4-row x 6-column grid over the full raster extent; assignment
# below was chosen by ranking blocks by flood-pixel density (data-driven,
# not random) so that val/test each receive meaningful positive-class
# representation while train retains the majority of both area and flood
# pixels. See reports/unet_dataset_generation_report.md Section 8.
BLOCK_GRID = (3, 4)  # (n_rows, n_cols) - coarser grid chosen after a 4x6 grid was
# found to leave too little interior room for buffer clearance around small
# blocks (see report). Blocks are ~3170m x 2800m at this grid resolution.
BLOCK_SPLIT_ASSIGNMENT = {
    (2, 2): "test",              # edge block, 63% flood density, only 1 real train neighbor
    (0, 1): "val", (0, 2): "val",  # edge blocks, modest flood representation
    # everything else not listed defaults to "train" (includes the single
    # richest block (1,2) with 14,716 flood pixels, kept in train for signal)
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sar-stack", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--municipalities", required=True)
    p.add_argument("--split-config", default=None, help="JSON {train:[...],val:[...],test:[...]}; default built-in")
    p.add_argument("--split-mode", choices=["municipality", "block"], default="block",
                   help="'block' (recommended, default) uses the data-driven spatial-block split; "
                        "'municipality' is kept for comparison/documentation purposes only - it was "
                        "empirically found infeasible for val/test on this AOI (see report).")
    p.add_argument("--block-grid", default=None, help="Override BLOCK_GRID, e.g. '3,4' (rows,cols)")
    p.add_argument("--block-assignment-json", default=None,
                   help="JSON mapping 'i,j' -> 'val'|'test' overriding BLOCK_SPLIT_ASSIGNMENT "
                        "(blocks not listed default to train)")
    p.add_argument("--quiet", action="store_true", help="Suppress per-run verbose prints (for sweeps)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--buffer-m", type=float, default=200.0)
    p.add_argument("--min-valid-sar", type=float, default=0.95, help="min fraction of pixels with all 6 SAR channels valid")
    p.add_argument("--min-labeled", type=float, default=0.80, help="min fraction of pixels with label != ignore")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)

    with rasterio.open(args.sar_stack) as src:
        sar = src.read()
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width
        px_w, px_h = src.res

    with rasterio.open(args.label) as src:
        assert src.transform == transform and (src.width, src.height) == (W, H), "Label grid mismatch!"
        label = src.read(1)

    sar_valid = np.all(~np.isnan(sar), axis=0)

    muni = gpd.read_file(args.municipalities).to_crs(crs)
    name_to_geom = {row["municipality"]: row.geometry for _, row in muni.iterrows()}

    if args.split_mode == "municipality":
        split_cfg = DEFAULT_SPLIT
        if args.split_config:
            with open(args.split_config) as f:
                split_cfg = json.load(f)
        split_union = {}
        for split_name, names in split_cfg.items():
            geoms = [name_to_geom[n] for n in names]
            split_union[split_name] = gpd.GeoSeries(geoms).union_all()
        # buffer-check reference geometries = the OTHER splits' municipality polygons
        buffer_ref_geoms = {s: [name_to_geom[n] for n in names] for s, names in split_cfg.items()}
        print("Split assignment (municipality mode):")
        for s, names in split_cfg.items():
            print(f"  {s}: {names}")
    else:
        block_grid = BLOCK_GRID
        if args.block_grid:
            r, c = args.block_grid.split(",")
            block_grid = (int(r), int(c))
        block_assignment = BLOCK_SPLIT_ASSIGNMENT
        if args.block_assignment_json:
            with open(args.block_assignment_json) as f:
                raw = json.load(f)
            block_assignment = {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}

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
        split_union = {s: gpd.GeoSeries(geoms).union_all() for s, geoms in split_union.items()}
        if not args.quiet:
            print(f"Split assignment (block mode, {n_rows_grid}x{n_cols_grid} grid):")
            for (i, j), s in sorted(block_assignment.items()):
                print(f"  block({i},{j}) -> {s}")
            print("  all other blocks -> train")

    ps = args.patch_size
    st = args.stride
    buffer_m = args.buffer_m

    candidates = []
    for row0 in range(0, H - ps + 1, st):
        for col0 in range(0, W - ps + 1, st):
            candidates.append((row0, col0))
    print(f"\nCandidate patch positions (grid, before any filtering): {len(candidates)}")

    def patch_bounds(row0, col0):
        x0, y0 = transform * (col0, row0)
        x1, y1 = transform * (col0 + ps, row0 + ps)
        return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    rows_meta = []
    counts = {"train": 0, "val": 0, "test": 0, "rejected_valid_sar": 0,
              "rejected_labeled": 0, "rejected_no_split": 0, "rejected_buffer": 0}

    split_names = ["train", "val", "test"]
    imgs_dir = {s: out_dir / s / "images" for s in split_names}
    lbls_dir = {s: out_dir / s / "labels" for s in split_names}
    for s in split_names:
        imgs_dir[s].mkdir(parents=True, exist_ok=True)
        lbls_dir[s].mkdir(parents=True, exist_ok=True)

    patch_id = 0
    for row0, col0 in candidates:
        sar_patch = sar[:, row0:row0 + ps, col0:col0 + ps]
        lbl_patch = label[row0:row0 + ps, col0:col0 + ps]
        valid_patch = sar_valid[row0:row0 + ps, col0:col0 + ps]

        valid_sar_frac = valid_patch.mean()
        labeled_frac = (lbl_patch != LABEL_IGNORE).mean()

        if valid_sar_frac < args.min_valid_sar:
            counts["rejected_valid_sar"] += 1
            continue
        if labeled_frac < args.min_labeled:
            counts["rejected_labeled"] += 1
            continue

        pbounds = patch_bounds(row0, col0)
        pcenter = pbounds.centroid

        owning_split = None
        for s, geom in split_union.items():
            if geom.contains(pcenter):
                owning_split = s
                break
        if owning_split is None:
            counts["rejected_no_split"] += 1
            continue

        # Buffer check: reject if patch bounding box comes within buffer_m of
        # ANY municipality belonging to a DIFFERENT split.
        too_close = False
        for s, geoms in buffer_ref_geoms.items():
            if s == owning_split:
                continue
            for g in geoms:
                if pbounds.distance(g) < buffer_m:
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            counts["rejected_buffer"] += 1
            continue

        # Determine dominant municipality (by area) for metadata purposes
        dom_muni, dom_area = None, -1
        for name, geom in name_to_geom.items():
            a = pbounds.intersection(geom).area
            if a > dom_area:
                dom_area, dom_muni = a, name

        flood_px = int((lbl_patch == LABEL_FLOOD).sum())
        nonflood_px = int((lbl_patch == LABEL_NONFLOOD).sum())
        ignore_px = int((lbl_patch == LABEL_IGNORE).sum())
        is_positive = flood_px > 0

        pid = f"patch_{patch_id:05d}"
        img_arr = np.transpose(sar_patch, (1, 2, 0)).astype("float32")  # H,W,6
        np.save(imgs_dir[owning_split] / f"{pid}.npy", img_arr)
        np.save(lbls_dir[owning_split] / f"{pid}.npy", lbl_patch.astype("uint8"))

        rows_meta.append(dict(
            patch_id=pid, split=owning_split, source_raster=str(args.sar_stack),
            row0=row0, col0=col0, patch_size=ps,
            bounds_minx=pbounds.bounds[0], bounds_miny=pbounds.bounds[1],
            bounds_maxx=pbounds.bounds[2], bounds_maxy=pbounds.bounds[3],
            crs=str(crs), valid_sar_frac=round(float(valid_sar_frac), 4),
            labeled_frac=round(float(labeled_frac), 4),
            flood_px=flood_px, nonflood_px=nonflood_px, ignore_px=ignore_px,
            is_positive=is_positive, municipality=dom_muni,
            label_source="EMSR773_AOI03_GRA_v2",
        ))
        counts[owning_split] += 1
        patch_id += 1

    meta_path_all = out_dir / "all_patches_metadata.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows_meta:
        fieldnames = list(rows_meta[0].keys())
        with open(meta_path_all, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_meta)
        for s in split_names:
            split_rows = [r for r in rows_meta if r["split"] == s]
            with open(out_dir / s / "metadata.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(split_rows)

    print(f"\nPatch generation summary:")
    print(f"  Retained: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"  Rejected (valid_sar<{args.min_valid_sar}): {counts['rejected_valid_sar']}")
    print(f"  Rejected (labeled<{args.min_labeled}): {counts['rejected_labeled']}")
    print(f"  Rejected (no split / outside all municipalities): {counts['rejected_no_split']}")
    print(f"  Rejected (buffer violation): {counts['rejected_buffer']}")
    total_retained = counts['train'] + counts['val'] + counts['test']
    n_pos = sum(1 for r in rows_meta if r["is_positive"])
    print(f"  Total retained: {total_retained} ({n_pos} positive / {total_retained - n_pos} negative)")

    config = vars(args).copy()
    config["split_assignment"] = (
        {"mode": "block", "grid": list(block_grid),
         "block_to_split": {f"{k[0]},{k[1]}": v for k, v in block_assignment.items()}}
        if args.split_mode == "block" else {"mode": "municipality", **split_cfg}
    )
    config["candidate_patches"] = len(candidates)
    config["retained_patches"] = total_retained
    config["counts"] = counts
    with open(out_dir.parent / "dataset_config.yaml", "w") as f:
        try:
            import yaml
            yaml.safe_dump(config, f, default_flow_style=False)
        except ImportError:
            json.dump(config, f, indent=2)
    print(f"\nWrote {meta_path_all}")
    print(f"Wrote {out_dir.parent / 'dataset_config.yaml'}")
    print("\nPATCH GENERATION COMPLETE")


if __name__ == "__main__":
    main()
