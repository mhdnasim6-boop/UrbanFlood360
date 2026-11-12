#!/usr/bin/env python3
"""
UrbanFlood360 - UrbanSARFloods READ-ONLY diagnostic (Kaggle, CPU is enough).

Performs NO model training, NO model instantiation, NO gradient computation.
Only reads GeoTIFFs and computes statistics on the OFFICIAL TRAINING SPLIT
(Train_dataset.txt, 4,501 chips) - the exact same file-resolution logic as
the already-successful scripts/kaggle_train_urbansarfloods.py run, so these
numbers describe precisely the data that run actually trained on.

Does not touch Valencia data anywhere. Does not modify any existing
checkpoint. Does not require a GPU - internet access only, to download the
official UrbanSARFloods train/val tarball (same ~38GB stream-extract to
/kaggle/temp as the training script; ~10-15 min, dominated by download, not
compute).
"""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
    import rasterio

DATA_ROOT = Path("/kaggle/temp/urban_sar_floods")
TARBALL_URL = "https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/main/urban_sar_floods.tar.gz"
OUT_DIR = Path("/kaggle/working/output")

# 1-indexed band positions in each 8-band SAR GeoTIFF, confirmed working in
# the training run (Train_dataset.txt/Valid_dataset.txt resolved 4501/1970
# pairs exactly matching the paper's official counts using this indexing).
INTENSITY_BANDS = {
    "pre_VH": 5,
    "pre_VV": 6,
    "post_VH": 7,
    "post_VV": 8,
}

SAMPLE_SIZE = 300   # chips sampled for per-pixel SAR band statistics (raise if you want more)
SEED = 42           # matches the training run's seed, for a reproducible sample


def print_disk_space():
    import os
    for path in ["/kaggle/working", "/kaggle/temp", "/"]:
        try:
            st = os.statvfs(path)
            free_gb = st.f_bavail * st.f_frsize / 1e9
            total_gb = st.f_blocks * st.f_frsize / 1e9
            print(f"  {path}: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        except FileNotFoundError:
            print(f"  {path}: does not exist")


def download_and_extract():
    print("Disk space before download:")
    print_disk_space()
    if DATA_ROOT.exists() and any(DATA_ROOT.iterdir()):
        print(f"{DATA_ROOT} already populated, skipping download.")
        return
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming + extracting {TARBALL_URL} -> {DATA_ROOT.parent} (no local .tar.gz retained)")
    t0 = time.time()
    cmd = f'curl -L "{TARBALL_URL}" | tar xz -C "{DATA_ROOT.parent}"'
    ret = subprocess.run(cmd, shell=True)
    print(f"Disk space after download attempt (exit code {ret.returncode}):")
    print_disk_space()
    if ret.returncode != 0:
        raise RuntimeError(f"Download/extract failed with exit code {ret.returncode}")
    print(f"Download+extract done in {time.time()-t0:.0f}s")


def resolve_split_files(list_path: Path):
    """IDENTICAL logic to scripts/kaggle_train_urbansarfloods.py's
    resolve_split_files - reusing it unchanged so this diagnostic is
    guaranteed to describe the exact same file set the training run used."""
    lines = [l.strip() for l in open(list_path) if l.strip()]
    print(f"{list_path.name}: {len(lines)} lines. First 3 raw: {lines[:3]}")

    pairs = []
    for line in lines:
        rel = line
        while rel.startswith("../"):
            rel = rel[3:]
        gt_path = DATA_ROOT / rel
        sar_path = Path(str(gt_path).replace("/GT/", "/SAR/").replace("_GT.tif", "_SAR.tif"))
        if gt_path.exists() and sar_path.exists():
            pairs.append((sar_path, gt_path))

    print(f"{list_path.name}: resolved {len(pairs)}/{len(lines)} pairs to existing SAR+GT files")
    return pairs


def inspect_sar_bands(train_pairs, sample_size, seed):
    rng = random.Random(seed)
    sample = train_pairs if len(train_pairs) <= sample_size else rng.sample(train_pairs, sample_size)
    print(f"\n=== SAR band inspection: sampling {len(sample)}/{len(train_pairs)} training chips (seed={seed}) ===")

    pooled = {name: [] for name in INTENSITY_BANDS}
    nan_counts = {name: 0 for name in INTENSITY_BANDS}
    inf_counts = {name: 0 for name in INTENSITY_BANDS}
    nodata_pixel_counts = {name: 0 for name in INTENSITY_BANDS}
    exact_zero_counts = {name: 0 for name in INTENSITY_BANDS}
    total_px_seen = {name: 0 for name in INTENSITY_BANDS}
    declared_nodata = {name: set() for name in INTENSITY_BANDS}
    shape_mismatches = []
    crs_mismatches = []
    transform_mismatches = []

    for sar_path, gt_path in sample:
        with rasterio.open(sar_path) as sar_src:
            sar_shape = (sar_src.height, sar_src.width)
            sar_crs = sar_src.crs
            sar_transform = sar_src.transform
            nodatavals = sar_src.nodatavals  # per-band tuple, may contain None

            for name, band_idx in INTENSITY_BANDS.items():
                arr = sar_src.read(band_idx).astype("float64")
                total_px_seen[name] += arr.size

                nd = nodatavals[band_idx - 1]
                if nd is not None:
                    declared_nodata[name].add(nd)
                    nodata_pixel_counts[name] += int((arr == nd).sum())

                nan_mask = np.isnan(arr)
                inf_mask = np.isinf(arr)
                nan_counts[name] += int(nan_mask.sum())
                inf_counts[name] += int(inf_mask.sum())
                exact_zero_counts[name] += int((arr == 0.0).sum())

                valid = arr[~nan_mask & ~inf_mask]
                pooled[name].append(valid)

        with rasterio.open(gt_path) as gt_src:
            gt_shape = (gt_src.height, gt_src.width)
            gt_crs = gt_src.crs
            gt_transform = gt_src.transform

        if sar_shape != gt_shape:
            shape_mismatches.append((str(sar_path), sar_shape, gt_shape))
        if sar_crs != gt_crs:
            crs_mismatches.append((str(sar_path), str(sar_crs), str(gt_crs)))
        if sar_transform != gt_transform:
            transform_mismatches.append(str(sar_path))

    band_stats = {}
    for name in INTENSITY_BANDS:
        vals = np.concatenate(pooled[name]) if pooled[name] else np.array([])
        n = len(vals)
        stats = dict(
            band=name,
            band_index_1based=INTENSITY_BANDS[name],
            n_pixels_sampled=int(total_px_seen[name]),
            n_valid_pixels_pooled=int(n),
            min=float(vals.min()) if n else None,
            max=float(vals.max()) if n else None,
            mean=float(vals.mean()) if n else None,
            std=float(vals.std()) if n else None,
            median=float(np.median(vals)) if n else None,
            p1=float(np.percentile(vals, 1)) if n else None,
            p99=float(np.percentile(vals, 99)) if n else None,
            nan_count=nan_counts[name],
            inf_count=inf_counts[name],
            declared_nodata_values=sorted(declared_nodata[name]),
            nodata_pixel_count=nodata_pixel_counts[name],
            nodata_pixel_pct=100.0 * nodata_pixel_counts[name] / total_px_seen[name] if total_px_seen[name] else 0.0,
            exact_zero_count=exact_zero_counts[name],
            exact_zero_pct=100.0 * exact_zero_counts[name] / total_px_seen[name] if total_px_seen[name] else 0.0,
            invalid_total_count=nan_counts[name] + inf_counts[name] + nodata_pixel_counts[name],
            invalid_total_pct=100.0 * (nan_counts[name] + inf_counts[name] + nodata_pixel_counts[name]) / total_px_seen[name] if total_px_seen[name] else 0.0,
        )
        band_stats[name] = stats
        print(f"\n--- {name} (band {INTENSITY_BANDS[name]}) ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    consistency = dict(
        n_chips_checked=len(sample),
        shape_mismatches=shape_mismatches,
        crs_mismatches=crs_mismatches,
        transform_mismatches=transform_mismatches,
    )
    print("\n--- SAR/GT spatial consistency (per sampled chip) ---")
    print(json.dumps(consistency, indent=2, default=str))

    return band_stats, consistency


def inspect_labels(train_pairs):
    print(f"\n=== GT label inspection: ALL {len(train_pairs)} training chips ===")
    class_counts = {0: 0, 1: 0, 2: 0}
    other_values = {}
    total_px = 0

    for i, (sar_path, gt_path) in enumerate(train_pairs):
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
        total_px += gt.size
        vals, counts = np.unique(gt, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            if v in class_counts:
                class_counts[v] += c
            else:
                other_values[v] = other_values.get(v, 0) + c
        if (i + 1) % 1000 == 0:
            print(f"  ...processed {i+1}/{len(train_pairs)} GT files")

    fo_fu = class_counts[1] + class_counts[2]
    summary = dict(
        n_chips=len(train_pairs),
        total_pixels=total_px,
        class_0_nonflood=class_counts[0],
        class_1_flooded_open=class_counts[1],
        class_2_flooded_urban=class_counts[2],
        fo_plus_fu=fo_fu,
        pct_class_0=100.0 * class_counts[0] / total_px,
        pct_class_1_fo=100.0 * class_counts[1] / total_px,
        pct_class_2_fu=100.0 * class_counts[2] / total_px,
        pct_fo_plus_fu=100.0 * fo_fu / total_px,
        pct_fu_within_flood=(100.0 * class_counts[2] / fo_fu) if fo_fu else None,
        unexpected_values=other_values,
    )
    print("\n--- Label class distribution (exact, full training split) ---")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== UrbanFlood360 - UrbanSARFloods READ-ONLY diagnostic ===")
    print("NO model is instantiated. NO training occurs. NO Valencia data is touched.\n")

    download_and_extract()

    train_pairs = resolve_split_files(DATA_ROOT / "Train_dataset.txt")
    print(f"\nSplit confirmation: resolved {len(train_pairs)} training pairs "
          f"(expected 4501, matching the earlier successful training run).")

    band_stats, consistency = inspect_sar_bands(train_pairs, SAMPLE_SIZE, SEED)
    label_stats = inspect_labels(train_pairs)

    # ---------- Final structured summary ----------
    confirmed = []
    possible_problems = []
    open_questions = []

    confirmed.append(f"Resolved {len(train_pairs)}/4501 official training pairs from Train_dataset.txt.")
    for name, s in band_stats.items():
        confirmed.append(f"{name}: range [{s['min']:.3f}, {s['max']:.3f}] dB-scale-plausible range, "
                          f"mean={s['mean']:.3f}, std={s['std']:.3f}, median={s['median']:.3f} "
                          f"(n={s['n_valid_pixels_pooled']:,} pooled valid pixels).")
        if s["nan_count"] > 0:
            confirmed.append(f"{name}: {s['nan_count']:,} NaN pixels found in sampled chips.")
        if s["inf_count"] > 0:
            possible_problems.append(f"{name}: {s['inf_count']:,} Inf pixels found - would break "
                                      f"np.nan_to_num(nan=0.0), which does NOT catch Inf.")
        if s["declared_nodata_values"]:
            confirmed.append(f"{name}: GeoTIFF declares nodata value(s) {s['declared_nodata_values']}, "
                              f"affecting {s['nodata_pixel_pct']:.4f}% of sampled pixels.")
        else:
            open_questions.append(f"{name}: no nodata value declared in GeoTIFF metadata - "
                                   f"does that mean there truly is no invalid data, or is it just undeclared?")
        if s["exact_zero_pct"] > 1.0:
            possible_problems.append(f"{name}: {s['exact_zero_pct']:.2f}% of sampled pixels are exactly 0.0 - "
                                      f"on a dB scale this sits at the BRIGHT end of the range, not a neutral "
                                      f"value; np.nan_to_num(nan=0.0) would map any real NaN to this same "
                                      f"bright-looking value.")

    if consistency["shape_mismatches"] or consistency["crs_mismatches"] or consistency["transform_mismatches"]:
        possible_problems.append(f"SAR/GT spatial mismatches found in sampled chips: "
                                  f"{len(consistency['shape_mismatches'])} shape, "
                                  f"{len(consistency['crs_mismatches'])} CRS, "
                                  f"{len(consistency['transform_mismatches'])} transform.")
    else:
        confirmed.append("No SAR/GT shape, CRS, or transform mismatches found in sampled chips.")

    confirmed.append(f"Label distribution (exact, all {label_stats['n_chips']} training chips): "
                      f"non-flood={label_stats['pct_class_0']:.3f}%, FO={label_stats['pct_class_1_fo']:.3f}%, "
                      f"FU={label_stats['pct_class_2_fu']:.3f}%, FO+FU={label_stats['pct_fo_plus_fu']:.3f}%.")
    if label_stats["pct_fu_within_flood"] is not None:
        confirmed.append(f"Within the 'flood' superclass, FU makes up {label_stats['pct_fu_within_flood']:.2f}% "
                          f"(FO makes up the rest) - i.e. the binary label>0 target is FO-dominated.")
    if label_stats["unexpected_values"]:
        possible_problems.append(f"Unexpected GT pixel values found (not in {{0,1,2}}): "
                                  f"{label_stats['unexpected_values']}")

    open_questions.append("Whether the observed numeric ranges are consistent with the paper's stated "
                           "dB preprocessing, or resemble a different scale - see printed min/max/mean above; "
                           "this diagnostic reports the numbers but does not assert an interpretation.")
    open_questions.append("Whether nodata/invalid pixels concentrate spatially at pre/post-event footprint "
                           "edges specifically (would require per-pixel spatial mapping, not attempted here).")

    print("\n" + "=" * 70)
    print("CONFIRMED OBSERVATIONS")
    print("=" * 70)
    for line in confirmed:
        print(f"  - {line}")
    print("\n" + "=" * 70)
    print("POSSIBLE PROBLEMS")
    print("=" * 70)
    if possible_problems:
        for line in possible_problems:
            print(f"  - {line}")
    else:
        print("  (none flagged)")
    print("\n" + "=" * 70)
    print("QUESTIONS THAT REMAIN UNRESOLVED")
    print("=" * 70)
    for line in open_questions:
        print(f"  - {line}")

    summary = dict(
        note="READ-ONLY diagnostic. No model trained, no Valencia data used.",
        n_train_pairs_resolved=len(train_pairs),
        sample_size_sar_bands=min(SAMPLE_SIZE, len(train_pairs)),
        seed=SEED,
        band_stats=band_stats,
        spatial_consistency=consistency,
        label_stats=label_stats,
        confirmed_observations=confirmed,
        possible_problems=possible_problems,
        open_questions=open_questions,
    )
    with open(OUT_DIR / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR / 'diagnostic_summary.json'}")
    print("\n=== DIAGNOSTIC COMPLETE (no training was performed) ===")


if __name__ == "__main__":
    main()
