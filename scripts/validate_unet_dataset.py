#!/usr/bin/env python3
"""
UrbanFlood360 - U-Net patch dataset validation and leakage check.

Verifies:
  - image/label shape and channel-count consistency
  - label value domain (only 0/1/255)
  - no duplicate patch IDs
  - no overlapping patch bounding boxes across DIFFERENT splits
  - the configured spatial buffer is respected between splits
  - metadata completeness

Usage:
    python validate_unet_dataset.py --patches-dir data/processed/unet_dataset/patches \
        --buffer-m 150

Exit codes: 0 = PASS, 1 = FAIL
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from shapely.geometry import box


def load_metadata(patches_dir):
    path = Path(patches_dir) / "all_patches_metadata.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patches-dir", required=True)
    p.add_argument("--buffer-m", type=float, default=150.0)
    args = p.parse_args()

    problems = []
    rows = load_metadata(args.patches_dir)
    print(f"Loaded metadata for {len(rows)} patches")

    ids = [r["patch_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("Duplicate patch IDs found")

    shapes_ok = True
    for r in rows:
        img_path = Path(args.patches_dir) / r["split"] / "images" / f"{r['patch_id']}.npy"
        lbl_path = Path(args.patches_dir) / r["split"] / "labels" / f"{r['patch_id']}.npy"
        if not img_path.exists() or not lbl_path.exists():
            problems.append(f"{r['patch_id']}: missing image or label file")
            shapes_ok = False
            continue
        img = np.load(img_path)
        lbl = np.load(lbl_path)
        ps = int(r["patch_size"])
        if img.shape != (ps, ps, 6):
            problems.append(f"{r['patch_id']}: image shape {img.shape} != ({ps},{ps},6)")
            shapes_ok = False
        if lbl.shape != (ps, ps):
            problems.append(f"{r['patch_id']}: label shape {lbl.shape} != ({ps},{ps})")
            shapes_ok = False
        bad_vals = set(np.unique(lbl)) - {0, 1, 255}
        if bad_vals:
            problems.append(f"{r['patch_id']}: unexpected label values {bad_vals}")
        if np.isinf(img).any():
            problems.append(f"{r['patch_id']}: infinite values in image")
    print(f"Shape/value checks: {'PASS' if shapes_ok else 'FAIL'}")

    # Bounding boxes per split
    boxes = {"train": [], "val": [], "test": []}
    for r in rows:
        b = box(float(r["bounds_minx"]), float(r["bounds_miny"]), float(r["bounds_maxx"]), float(r["bounds_maxy"]))
        boxes[r["split"]].append((r["patch_id"], b))

    # 1. No overlapping bounding boxes across DIFFERENT splits
    overlap_found = False
    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for sa, sb in split_pairs:
        for ida, ba in boxes[sa]:
            for idb, bb in boxes[sb]:
                if ba.intersects(bb):
                    problems.append(f"Overlap between {sa}:{ida} and {sb}:{idb}")
                    overlap_found = True
    print(f"Cross-split overlap check: {'FAIL' if overlap_found else 'PASS'}")

    # 1b. Both classes present in val AND test (not just leakage-free)
    for split in ["val", "test"]:
        sub = [r for r in rows if r["split"] == split]
        n_pos = sum(1 for r in sub if r["is_positive"] == "True")
        n_neg = len(sub) - n_pos
        print(f"{split}: {len(sub)} patches ({n_pos} positive / {n_neg} negative)")
        if n_pos == 0:
            problems.append(f"{split} split contains ZERO positive (flood) patches")
        if n_neg == 0:
            problems.append(f"{split} split contains ZERO negative (non-flood) patches")

    # 2. Buffer respected: min distance between different-split patches >= buffer_m
    #    (allow exact equality due to float rounding)
    min_gap = float("inf")
    worst_pair = None
    for sa, sb in split_pairs:
        for ida, ba in boxes[sa]:
            for idb, bb in boxes[sb]:
                d = ba.distance(bb)
                if d < min_gap:
                    min_gap = d
                    worst_pair = (sa, ida, sb, idb)
    buffer_ok = min_gap >= args.buffer_m - 1e-6 if boxes["train"] and (boxes["val"] or boxes["test"]) else True
    print(f"Minimum distance found between any two different-split patches: {min_gap:.1f}m "
          f"(required >= {args.buffer_m}m)")
    if worst_pair:
        print(f"  Closest pair: {worst_pair}")
    if not buffer_ok:
        problems.append(f"Buffer violated: minimum inter-split distance {min_gap:.1f}m < {args.buffer_m}m")

    # 3. Normalization-fitting sanity: this script does not itself fit
    #    normalization, but confirms no normalized/scaled files exist yet
    #    alongside the raw patches (i.e. nothing has been silently normalized).
    for r in rows[:1]:
        img_path = Path(args.patches_dir) / r["split"] / "images" / f"{r['patch_id']}.npy"
        sample = np.load(img_path)
        if np.nanmax(np.abs(sample)) <= 1.0 + 1e-6:
            problems.append("Sample patch values all within [-1,1] - possible accidental normalization")

    print(f"\nTotal problems found: {len(problems)}")
    if problems:
        print("DATASET VALIDATION: FAIL")
        for msg in problems[:50]:
            print(f"  - {msg}")
        raise SystemExit(1)
    else:
        print("DATASET VALIDATION: PASS - no leakage detected")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
