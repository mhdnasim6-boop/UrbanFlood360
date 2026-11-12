#!/usr/bin/env python3
"""
UrbanFlood360 - permanent QC component for generated SAR features.

Verifies that a feature-generation output directory (as produced by
generate_sar_features.py) is internally consistent: correct band names/
order, correct CRS/transform/dimensions, correct NoData, correct AOI
masking (individual files agree with the stack, and change layers are
NaN wherever either source is NaN), no unexpected Inf values, and that
the change calculation is exactly POST - PRE.

Usage:
    python validate_sar_features.py --pre <pre_db.tif> --post <post_db.tif> \
        --features-dir data/processed/sar_features

Exit codes:
    0 = PASS
    1 = FAIL
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

EXPECTED_ORDER = [
    "VV_pre_dB", "VH_pre_dB", "VV_post_dB", "VH_post_dB",
    "VV_change_dB", "VH_change_dB",
]


def fail(msg, problems):
    problems.append(msg)
    print(f"FAIL: {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    p.add_argument("--features-dir", required=True)
    args = p.parse_args()

    fdir = Path(args.features_dir)
    problems = []

    stack_path = fdir / "sar_features_6ch.tif"
    if not stack_path.exists():
        fail(f"stack file missing: {stack_path}", problems)
        sys.exit(1)

    with rasterio.open(stack_path) as stack:
        if stack.count != 6:
            fail(f"stack has {stack.count} bands, expected 6", problems)
        if list(stack.descriptions) != EXPECTED_ORDER:
            fail(f"stack band order/names = {stack.descriptions}, expected {EXPECTED_ORDER}", problems)
        if str(stack.crs) != "EPSG:25830":
            fail(f"stack CRS = {stack.crs}, expected EPSG:25830", problems)
        if (stack.width, stack.height) != (1120, 951):
            fail(f"stack dims = {(stack.width, stack.height)}, expected (1120, 951)", problems)
        if not (stack.nodata is not None and np.isnan(stack.nodata)):
            fail(f"stack nodata = {stack.nodata}, expected NaN", problems)
        stack_arrays = stack.read()
        stack_transform = stack.transform

    # Individual files must exactly match the stack, band for band
    for i, name in enumerate(EXPECTED_ORDER, start=1):
        ind_path = fdir / f"{name}.tif"
        if not ind_path.exists():
            fail(f"individual file missing: {ind_path}", problems)
            continue
        with rasterio.open(ind_path) as ind:
            if ind.descriptions[0] != name:
                fail(f"{name}.tif band description = {ind.descriptions[0]!r}, expected {name!r}", problems)
            if ind.transform != stack_transform:
                fail(f"{name}.tif transform differs from stack", problems)
            if (ind.width, ind.height) != (1120, 951):
                fail(f"{name}.tif dims = {(ind.width, ind.height)}, expected (1120, 951)", problems)
            arr = ind.read(1)
        stack_band = stack_arrays[i - 1]
        both_nan = np.isnan(arr) & np.isnan(stack_band)
        close = np.isclose(arr, stack_band, equal_nan=True)
        if not np.all(close | both_nan):
            fail(f"{name}.tif does NOT match stack band {i} values exactly", problems)

        if np.isinf(arr).any():
            fail(f"{name}.tif contains unexpected Inf values", problems)

    # Re-derive VV/VH bands from PRE/POST inputs by description (not assumed order)
    def load_by_desc(path):
        with rasterio.open(path) as src:
            desc = src.descriptions
            vh_i = [i for i, d in enumerate(desc) if d and "VH" in d.upper()][0]
            vv_i = [i for i, d in enumerate(desc) if d and "VV" in d.upper()][0]
            return src.read(vh_i + 1), src.read(vv_i + 1)

    vh_pre, vv_pre = load_by_desc(args.pre)
    vh_post, vv_post = load_by_desc(args.post)

    idx = {name: i for i, name in enumerate(EXPECTED_ORDER)}
    vv_change_stack = stack_arrays[idx["VV_change_dB"]]
    vh_change_stack = stack_arrays[idx["VH_change_dB"]]

    both_valid_vv = ~np.isnan(vv_pre) & ~np.isnan(vv_post)
    both_valid_vh = ~np.isnan(vh_pre) & ~np.isnan(vh_post)

    expected_vv_change = np.where(both_valid_vv, vv_post - vv_pre, np.nan)
    expected_vh_change = np.where(both_valid_vh, vh_post - vh_pre, np.nan)

    if not np.allclose(np.nan_to_num(vv_change_stack, nan=-9999),
                        np.nan_to_num(expected_vv_change.astype("float32"), nan=-9999), atol=1e-3):
        fail("VV_change_dB does not exactly equal POST-PRE where both valid", problems)
    if not np.allclose(np.nan_to_num(vh_change_stack, nan=-9999),
                        np.nan_to_num(expected_vh_change.astype("float32"), nan=-9999), atol=1e-3):
        fail("VH_change_dB does not exactly equal POST-PRE where both valid", problems)

    if not np.array_equal(np.isnan(vv_change_stack), ~both_valid_vv):
        fail("VV_change_dB NaN mask does not match 'both PRE and POST valid' logic exactly", problems)
    if not np.array_equal(np.isnan(vh_change_stack), ~both_valid_vh):
        fail("VH_change_dB NaN mask does not match 'both PRE and POST valid' logic exactly", problems)

    print(f"\nChecks completed. Problems found: {len(problems)}")
    if problems:
        print("VALIDATION: FAIL")
        for msg in problems:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print("VALIDATION: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
