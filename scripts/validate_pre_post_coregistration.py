#!/usr/bin/env python3
"""
UrbanFlood360 - permanent QC component.

Estimates residual sub-pixel content-level displacement between a PRE and
POST raster using FFT-based phase correlation over a set of user-specified
"stable" windows (areas expected NOT to have changed between acquisitions -
avoid water, vegetation, agricultural land, and known flood-affected areas).

This is a SEPARATE check from raster-grid metadata validation
(validate_pre_post_grid.py): two rasters can share an identical CRS/
transform/dimensions while their real-world content is still offset by a
small, sub-pixel amount. This script measures that.

Method note: phase correlation was chosen over spatial-domain
correlate2d()-based peak finding after a synthetic sign-convention test
revealed the latter can produce row/column cross-contamination on real
(non-toy) inputs. Phase correlation was validated on synthetic shifts of
known magnitude/sign before being trusted on real data (see project
report). Its sign convention: shift_estimate(ref, mov) = (dr, dc) means
mov's content is ref's content sampled from an offset location by (dr, dc)
in (row, col) index space - i.e. mov[i,j] ~= ref[i+dr, j+dc].

Usage:
    python validate_pre_post_coregistration.py <pre_raster> <post_raster> \
        [--band N] [--score-threshold 0.2] [--pixel-tolerance 0.5] \
        [--windows-json windows.json]

    windows.json (optional): {"name": [minx, miny, maxx, maxy], ...} in the
    raster's CRS units. If omitted, a small built-in default set covering
    the UrbanFlood360 Horta Sud AOI town centers is used.

Exit codes:
    0 = PASS (median residual displacement, over windows meeting the score
        threshold, is within --pixel-tolerance pixels in both row and col)
    1 = FAIL (residual displacement exceeds tolerance, or too few reliable
        windows to draw a conclusion)
"""
import argparse
import json
import sys

import numpy as np
import rasterio
from rasterio.windows import from_bounds

DEFAULT_WINDOWS = {
    "Alfafar":      (726890, 4364507, 727490, 4365107),
    "Catarroja":    (724086, 4363692, 724686, 4364292),
    "Massanassa":   (725923, 4363903, 726523, 4364503),
    "Paiporta":     (722437, 4366912, 723037, 4367512),
    "Picanya":      (720355, 4367778, 720955, 4368378),
    "Sedavi":       (725524, 4366825, 726124, 4367425),
    "Catarroja_2":  (723700, 4364100, 724300, 4364700),
    "DenseBlock_B": (722900, 4364700, 723500, 4365300),
}


def phase_corr_shift(ref: np.ndarray, mov: np.ndarray):
    """Returns ((row_shift, col_shift), peak_score)."""
    ref_f = np.nan_to_num(ref, nan=np.nanmean(ref)).astype(np.float64)
    mov_f = np.nan_to_num(mov, nan=np.nanmean(mov)).astype(np.float64)
    ref_f = ref_f - ref_f.mean()
    mov_f = mov_f - mov_f.mean()
    window = np.outer(np.hanning(ref_f.shape[0]), np.hanning(ref_f.shape[1]))
    F1 = np.fft.fft2(ref_f * window)
    F2 = np.fft.fft2(mov_f * window)
    R = F1 * np.conj(F2)
    R /= (np.abs(R) + 1e-12)
    corr = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    shifts = []
    for axis, size in enumerate(corr.shape):
        p = peak[axis]
        if p > size // 2:
            p -= size
        shifts.append(p)

    pr, pc = peak
    wrap = lambda i, n: i % n
    dr = 0.5 * (corr[wrap(pr - 1, corr.shape[0]), pc] - corr[wrap(pr + 1, corr.shape[0]), pc]) / (
        corr[wrap(pr - 1, corr.shape[0]), pc] - 2 * corr[pr, pc] + corr[wrap(pr + 1, corr.shape[0]), pc] + 1e-12
    )
    dc = 0.5 * (corr[pr, wrap(pc - 1, corr.shape[1])] - corr[pr, wrap(pc + 1, corr.shape[1])]) / (
        corr[pr, wrap(pc - 1, corr.shape[1])] - 2 * corr[pr, pc] + corr[pr, wrap(pc + 1, corr.shape[1])] + 1e-12
    )
    return (shifts[0] + dr, shifts[1] + dc), float(corr[peak])


def get_window(path, band, bounds):
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, src.transform)
        return src.read(band, window=win)


def run(pre_path, post_path, band, windows, score_threshold, pixel_tolerance):
    results = []
    for name, bounds in windows.items():
        pre = get_window(pre_path, band, bounds)
        post = get_window(post_path, band, bounds)
        if pre.size == 0 or pre.shape != post.shape:
            print(f"{name:<15} SKIP (empty or shape mismatch)")
            continue
        if np.isnan(pre).mean() > 0.02 or np.isnan(post).mean() > 0.02:
            print(f"{name:<15} SKIP (>2% NoData in window - too close to AOI edge)")
            continue
        mean_abs_change = float(np.nanmean(np.abs(post - pre)))
        shift, score = phase_corr_shift(pre, post)
        results.append(dict(name=name, shift=shift, score=score, mean_abs_change=mean_abs_change))

    print(f"\n{'Window':<15}{'Row shift':>12}{'Col shift':>12}{'Score':>10}{'MeanAbsChg':>12}")
    for r in results:
        print(f"{r['name']:<15}{r['shift'][0]:>12.4f}{r['shift'][1]:>12.4f}{r['score']:>10.4f}{r['mean_abs_change']:>12.3f}")

    reliable = [r for r in results if r["score"] >= score_threshold]
    print(f"\nReliable windows (score >= {score_threshold}): {len(reliable)}/{len(results)}")

    if len(reliable) < 3:
        print("\nFAIL: fewer than 3 reliable windows - cannot draw a confident conclusion.")
        return 1

    rows = np.array([r["shift"][0] for r in reliable])
    cols = np.array([r["shift"][1] for r in reliable])
    print(f"\nRow shift:    mean={rows.mean():+.4f} median={np.median(rows):+.4f} std={rows.std():.4f} "
          f"min={rows.min():+.4f} max={rows.max():+.4f}")
    print(f"Column shift: mean={cols.mean():+.4f} median={np.median(cols):+.4f} std={cols.std():.4f} "
          f"min={cols.min():+.4f} max={cols.max():+.4f}")

    ok = abs(np.median(rows)) <= pixel_tolerance and abs(np.median(cols)) <= pixel_tolerance
    print(f"\nOVERALL CO-REGISTRATION: {'PASS' if ok else 'FAIL'} "
          f"(median |shift| <= {pixel_tolerance} px required)")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pre_raster")
    p.add_argument("post_raster")
    p.add_argument("--band", type=int, default=2, help="1-indexed band (default 2 = VV in this project's convention)")
    p.add_argument("--score-threshold", type=float, default=0.20)
    p.add_argument("--pixel-tolerance", type=float, default=0.5)
    p.add_argument("--windows-json", default=None)
    args = p.parse_args()

    windows = DEFAULT_WINDOWS
    if args.windows_json:
        with open(args.windows_json) as f:
            windows = {k: tuple(v) for k, v in json.load(f).items()}

    sys.exit(run(args.pre_raster, args.post_raster, args.band, windows,
                 args.score_threshold, args.pixel_tolerance))
