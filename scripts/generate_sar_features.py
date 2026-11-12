#!/usr/bin/env python3
"""
UrbanFlood360 - SAR feature generation.

Takes validated, co-registered, already-terrain-corrected PRE and POST dB
GeoTIFFs (each with Sigma0_VH_db / Sigma0_VV_db bands, identified by band
description rather than assumed order) and produces the six-channel SAR
feature set used as the initial U-Net input representation:

    1. VV_pre_dB
    2. VH_pre_dB
    3. VV_post_dB
    4. VH_post_dB
    5. VV_change_dB = VV_post_dB - VV_pre_dB
    6. VH_change_dB = VH_post_dB - VH_pre_dB

This script does NOT modify its inputs, does NOT resample/reproject/shift,
and does NOT perform any ML-related work (no patches, no labels, no
training). It fails loudly (raises) if PRE/POST are not grid-compatible.

Usage:
    python generate_sar_features.py --pre <pre_db.tif> --post <post_db.tif> \
        --output-dir data/processed/sar_features
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio


FEATURE_NAMES = [
    "VV_pre_dB", "VH_pre_dB", "VV_post_dB", "VH_post_dB",
    "VV_change_dB", "VH_change_dB",
]

PERCENTILES = [1, 2, 5, 25, 50, 75, 95, 98, 99]


def find_band(descriptions, must_contain, must_not_contain=None):
    """Locate a 1-indexed band by (sub)string match on its description.
    Fails loudly (raises) if zero or more than one match is found."""
    matches = [
        i + 1 for i, d in enumerate(descriptions)
        if d and must_contain.upper() in d.upper()
        and (must_not_contain is None or must_not_contain.upper() not in d.upper())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one band matching '{must_contain}' "
            f"(excluding '{must_not_contain}') in {descriptions}, found {matches}"
        )
    return matches[0]


def load_scene(path):
    with rasterio.open(path) as src:
        desc = src.descriptions
        vh_band = find_band(desc, "VH")
        vv_band = find_band(desc, "VV")
        vh = src.read(vh_band)
        vv = src.read(vv_band)
        meta = dict(crs=src.crs, transform=src.transform, width=src.width,
                    height=src.height, nodata=src.nodata, dtype=src.dtypes[0])
    return vh, vv, meta, {"VH": vh_band, "VV": vv_band}


def assert_grid_compatible(meta_a, meta_b, label_a="PRE", label_b="POST"):
    checks = [
        ("CRS", meta_a["crs"] == meta_b["crs"], meta_a["crs"], meta_b["crs"]),
        ("width", meta_a["width"] == meta_b["width"], meta_a["width"], meta_b["width"]),
        ("height", meta_a["height"] == meta_b["height"], meta_a["height"], meta_b["height"]),
        ("transform", meta_a["transform"] == meta_b["transform"], meta_a["transform"], meta_b["transform"]),
    ]
    failed = [c for c in checks if not c[1]]
    if failed:
        msg = "\n".join(f"  {name}: {label_a}={a!r} != {label_b}={b!r}" for name, ok, a, b in failed)
        raise RuntimeError(f"GRID MISMATCH between {label_a} and {label_b}:\n{msg}")
    return checks


def compute_stats(arr, name):
    valid = arr[~np.isnan(arr)]
    total = arr.size
    row = {
        "channel": name,
        "count_valid": int(valid.size),
        "total_pixels": int(total),
        "valid_pct": 100.0 * valid.size / total,
    }
    if valid.size == 0:
        for k in ["min", "max", "mean", "median", "std"] + [f"p{p}" for p in PERCENTILES]:
            row[k] = float("nan")
        return row
    row["min"] = float(valid.min())
    row["max"] = float(valid.max())
    row["mean"] = float(valid.mean())
    row["median"] = float(np.median(valid))
    row["std"] = float(valid.std())
    pvals = np.percentile(valid, PERCENTILES)
    for p, v in zip(PERCENTILES, pvals):
        row[f"p{p}"] = float(v)
    return row


def write_geotiff(path, array, meta, band_name):
    profile = dict(
        driver="GTiff", height=meta["height"], width=meta["width"], count=1,
        dtype="float32", crs=meta["crs"], transform=meta["transform"],
        nodata=np.nan, compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
        dst.set_band_description(1, band_name)
        dst.update_tags(1, BAND_NAME=band_name)


def write_stack(path, arrays, meta, names):
    profile = dict(
        driver="GTiff", height=meta["height"], width=meta["width"], count=len(arrays),
        dtype="float32", crs=meta["crs"], transform=meta["transform"],
        nodata=np.nan, compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(arrays, names), start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)
            dst.update_tags(i, BAND_NAME=name)


def main():
    p = argparse.ArgumentParser(description="Generate UrbanFlood360 6-channel SAR feature set")
    p.add_argument("--pre", required=True, help="Path to validated PRE dB GeoTIFF")
    p.add_argument("--post", required=True, help="Path to validated POST dB GeoTIFF")
    p.add_argument("--output-dir", required=True, help="Directory to write feature outputs")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PRE  input: {args.pre}")
    print(f"POST input: {args.post}")

    vh_pre, vv_pre, meta_pre, bands_pre = load_scene(args.pre)
    vh_post, vv_post, meta_post, bands_post = load_scene(args.post)

    print(f"PRE  bands identified by description: {bands_pre}")
    print(f"POST bands identified by description: {bands_post}")

    checks = assert_grid_compatible(meta_pre, meta_post)
    for name, ok, a, b in checks:
        print(f"  grid check [{name}]: {'PASS' if ok else 'FAIL'}")
    print("Grid compatibility: CONFIRMED (fails loudly above if not)")

    # Validity masks - explicit, not assumed from NaN-propagation alone
    pre_vv_valid = ~np.isnan(vv_pre)
    pre_vh_valid = ~np.isnan(vh_pre)
    post_vv_valid = ~np.isnan(vv_post)
    post_vh_valid = ~np.isnan(vh_post)
    both_valid_vv = pre_vv_valid & post_vv_valid
    both_valid_vh = pre_vh_valid & post_vh_valid

    total_px = vv_pre.size
    print(f"\nValidity report (VV): PRE valid={pre_vv_valid.sum()} POST valid={post_vv_valid.sum()} "
          f"both-valid={both_valid_vv.sum()} total={total_px}")
    print(f"Validity report (VH): PRE valid={pre_vh_valid.sum()} POST valid={post_vh_valid.sum()} "
          f"both-valid={both_valid_vh.sum()} total={total_px}")

    # Explicit change computation: NaN unless BOTH pre and post valid at that pixel
    vv_change = np.full_like(vv_pre, np.nan, dtype="float32")
    vv_change[both_valid_vv] = vv_post[both_valid_vv] - vv_pre[both_valid_vv]
    vh_change = np.full_like(vh_pre, np.nan, dtype="float32")
    vh_change[both_valid_vh] = vh_post[both_valid_vh] - vh_pre[both_valid_vh]

    # Cross-check against plain NaN-propagating subtraction
    vv_change_naive = vv_post - vv_pre
    vh_change_naive = vh_post - vh_pre
    vv_mismatch = np.sum((~np.isnan(vv_change)) != (~np.isnan(vv_change_naive)))
    vh_mismatch = np.sum((~np.isnan(vh_change)) != (~np.isnan(vh_change_naive)))
    print(f"\nExplicit-mask vs NaN-propagation cross-check: VV mismatches={vv_mismatch}, VH mismatches={vh_mismatch} (must be 0)")
    assert vv_mismatch == 0 and vh_mismatch == 0, "Validity logic mismatch detected!"

    print(f"VV_change valid pixels: {int(both_valid_vv.sum())} ({100*both_valid_vv.sum()/total_px:.2f}%)")
    print(f"VH_change valid pixels: {int(both_valid_vh.sum())} ({100*both_valid_vh.sum()/total_px:.2f}%)")

    arrays = [vv_pre, vh_pre, vv_post, vh_post, vv_change, vh_change]

    # Individual GeoTIFFs
    for arr, name in zip(arrays, FEATURE_NAMES):
        out_path = out_dir / f"{name}.tif"
        write_geotiff(out_path, arr, meta_pre, name)
        print(f"Wrote {out_path}")

    # Six-band stack
    stack_path = out_dir / "sar_features_6ch.tif"
    write_stack(stack_path, arrays, meta_pre, FEATURE_NAMES)
    print(f"Wrote {stack_path}")

    # Statistics CSV
    import csv
    stats_rows = [compute_stats(arr, name) for arr, name in zip(arrays, FEATURE_NAMES)]
    csv_path = out_dir / "sar_feature_statistics.csv"
    fieldnames = list(stats_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(stats_rows)
    print(f"Wrote {csv_path}")

    # Validation summary JSON
    summary = {
        "pre_input": str(args.pre),
        "post_input": str(args.post),
        "bands_identified": {"pre": bands_pre, "post": bands_post},
        "grid": {
            "crs": str(meta_pre["crs"]),
            "width": meta_pre["width"],
            "height": meta_pre["height"],
            "transform": list(meta_pre["transform"])[:6],
        },
        "validity": {
            "total_pixels": int(total_px),
            "pre_vv_valid": int(pre_vv_valid.sum()),
            "pre_vh_valid": int(pre_vh_valid.sum()),
            "post_vv_valid": int(post_vv_valid.sum()),
            "post_vh_valid": int(post_vh_valid.sum()),
            "vv_change_valid": int(both_valid_vv.sum()),
            "vh_change_valid": int(both_valid_vh.sum()),
        },
        "stats": stats_rows,
    }
    summary_path = out_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    print("\nSAR FEATURE GENERATION COMPLETE")


if __name__ == "__main__":
    main()
