#!/usr/bin/env python3
"""
UrbanFlood360 - flood reference label generation.

Builds a training-ready flood/non-flood/ignore label raster from the
EMSR773 AOI03 Grading v2 `observedEventA` delineation, rasterized onto the
exact SAR feature grid (same CRS/transform/dimensions as
sar_features_6ch.tif - no resampling/reprojection of the SAR data occurs
here).

Label philosophy (evidence-based, see reports/unet_dataset_generation_report.md):
  - "Flooded area" (det_method=Semi-automatic extraction, i.e. derived
    directly from the source image) -> FLOOD (1), the only class treated
    as a confident positive label.
  - "Flood trace" (det_method=Photo-interpretation, i.e. interpreted
    evidence of a flood that had already partly receded by acquisition
    time - not current standing water) -> IGNORE (255). It is not treated
    as equivalent to "Flooded area", and not treated as "non-flood" either,
    since the pixel genuinely cannot be confidently assigned either class
    for a SAR backscatter-change binary segmentation task.
  - Area outside the EMSR773 imageFootprintA (i.e. never actually
    observed by the source VHR imagery) -> IGNORE (255), regardless of
    AOI membership.
  - Area outside the true dissolved 6-municipality AOI polygon -> IGNORE (255).
  - Any pixel where ANY of the 6 SAR feature channels is NaN -> IGNORE (255),
    overriding any reference-based label, since such a pixel cannot be used
    for supervised training regardless of its reference label.
  - Everything else (inside AOI, inside footprint, not flooded, not trace) -> NON-FLOOD (0).

--trace-mode flood (added for the building damage assessment) instead labels
"Flood trace" as FLOOD (1), i.e. the target becomes "flood-affected ground"
(standing water OR receded-flood evidence) rather than "standing water". The
SAR image is from 31 Oct, two days after the peak: the urban cores carry
mostly flood trace, so this label is the one that describes affected buildings.
Default (ignore) reproduces the original label exactly.

Usage:
    python create_flood_labels.py \
        --sar-stack data/processed/sar_features/sar_features_6ch.tif \
        --emsr773-observed-event <path to observedEventA_v2.shp> \
        --emsr773-image-footprint <path to imageFootprintA_v2.shp> \
        --aoi data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson \
        --municipalities data/gis/aoi/municipalities_6.geojson \
        --output-dir data/processed/unet_dataset/labels
"""
import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.validation import make_valid

LABEL_FLOOD = 1
LABEL_NONFLOOD = 0
LABEL_IGNORE = 255


def repair(gdf):
    n_invalid_before = int((~gdf.geometry.is_valid).sum())
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)
    n_invalid_after = int((~gdf.geometry.is_valid).sum())
    return gdf, n_invalid_before, n_invalid_after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sar-stack", required=True)
    p.add_argument("--emsr773-observed-event", required=True)
    p.add_argument("--emsr773-image-footprint", required=True)
    p.add_argument("--aoi", required=True)
    p.add_argument("--municipalities", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trace-mode", choices=["ignore", "flood"], default="ignore",
                   help="'ignore' (original): Flood trace -> 255. 'flood': Flood trace -> 1 (flood-affected label)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.sar_stack) as src:
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
        sar = src.read()  # (6, H, W)
        sar_descriptions = src.descriptions

    print(f"SAR grid: CRS={crs}, shape={shape}, transform={transform}")
    print(f"SAR bands (order as stored): {sar_descriptions}")

    sar_valid = np.all(~np.isnan(sar), axis=0)  # True where ALL 6 channels valid
    print(f"SAR-valid pixels (all 6 channels): {sar_valid.sum()} / {sar_valid.size} "
          f"({100*sar_valid.sum()/sar_valid.size:.2f}%)")

    aoi = gpd.read_file(args.aoi).to_crs(crs)
    aoi_geom = aoi.geometry.iloc[0]

    ev_raw = gpd.read_file(args.emsr773_observed_event)
    ev_raw, n_inv_before, n_inv_after = repair(ev_raw)
    print(f"\nobservedEventA: {len(ev_raw)} features, {n_inv_before} invalid before repair, "
          f"{n_inv_after} invalid after make_valid() (must be 0)")
    assert n_inv_after == 0, "Could not repair all invalid geometries - stopping."
    ev = ev_raw.to_crs(crs)

    fp_raw = gpd.read_file(args.emsr773_image_footprint)
    fp_raw, n_fp_inv_before, n_fp_inv_after = repair(fp_raw)
    print(f"imageFootprintA: {len(fp_raw)} features, {n_fp_inv_before} invalid before, {n_fp_inv_after} after")
    fp = fp_raw.to_crs(crs)

    flooded_geoms = list(ev[ev["notation"] == "Flooded area"].geometry)
    trace_geoms = list(ev[ev["notation"] == "Flood trace"].geometry)
    footprint_geoms = list(fp.geometry)

    print(f"\nFlooded area polygons: {len(flooded_geoms)}")
    print(f"Flood trace polygons:  {len(trace_geoms)}")
    print(f"Image footprint polygons: {len(footprint_geoms)}")

    # Rasterize each layer as a boolean mask on the exact SAR grid.
    # geometry_mask(invert=True) -> True INSIDE the geometries.
    aoi_inside = geometry_mask([aoi_geom], out_shape=shape, transform=transform, invert=True)
    footprint_inside = geometry_mask(footprint_geoms, out_shape=shape, transform=transform, invert=True) if footprint_geoms else np.zeros(shape, dtype=bool)
    flooded_inside = geometry_mask(flooded_geoms, out_shape=shape, transform=transform, invert=True) if flooded_geoms else np.zeros(shape, dtype=bool)
    trace_inside = geometry_mask(trace_geoms, out_shape=shape, transform=transform, invert=True) if trace_geoms else np.zeros(shape, dtype=bool)

    # Reference-only label (before SAR-validity masking) - for transparency/statistics
    ref_label = np.full(shape, LABEL_IGNORE, dtype=np.uint8)
    observed = aoi_inside & footprint_inside
    ref_label[observed] = LABEL_NONFLOOD
    ref_label[observed & trace_inside] = LABEL_FLOOD if args.trace_mode == "flood" else LABEL_IGNORE
    ref_label[observed & flooded_inside] = LABEL_FLOOD  # highest priority

    # Final training label - also enforces SAR validity
    final_label = ref_label.copy()
    final_label[~sar_valid] = LABEL_IGNORE

    # --- Statistics ---
    total = shape[0] * shape[1]

    def counts(lbl):
        return dict(
            flood=int((lbl == LABEL_FLOOD).sum()),
            nonflood=int((lbl == LABEL_NONFLOOD).sum()),
            ignore=int((lbl == LABEL_IGNORE).sum()),
            total=total,
        )

    ref_counts = counts(ref_label)
    final_counts = counts(final_label)
    print(f"\nReference-only label counts (before SAR-validity mask): {ref_counts}")
    print(f"Final training label counts (after SAR-validity mask):   {final_counts}")

    n_lost_to_sar_invalidity = int(np.sum((ref_label != LABEL_IGNORE) & (final_label == LABEL_IGNORE)))
    print(f"Pixels that had a usable reference label but were dropped due to invalid SAR: {n_lost_to_sar_invalidity}")

    # Per-municipality breakdown
    muni = gpd.read_file(args.municipalities).to_crs(crs)
    muni_rows = []
    for _, row in muni.iterrows():
        m_mask = geometry_mask([row.geometry], out_shape=shape, transform=transform, invert=True)
        m_label = final_label[m_mask]
        m_total = m_mask.sum()
        muni_rows.append(dict(
            municipality=row["municipality"],
            total_px=int(m_total),
            flood_px=int((m_label == LABEL_FLOOD).sum()),
            nonflood_px=int((m_label == LABEL_NONFLOOD).sum()),
            ignore_px=int((m_label == LABEL_IGNORE).sum()),
        ))
        r = muni_rows[-1]
        usable = r["flood_px"] + r["nonflood_px"]
        flood_pct = 100 * r["flood_px"] / usable if usable else float("nan")
        print(f"  {row['municipality']:12s}: total={r['total_px']:6d} flood={r['flood_px']:5d} "
              f"nonflood={r['nonflood_px']:6d} ignore={r['ignore_px']:6d}  "
              f"flood%%(of usable)={flood_pct:.2f}%%")

    # --- Write label raster (uint8, matches SAR grid exactly) ---
    label_path = out_dir / "flood_reference.tif"
    profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1,
                   dtype="uint8", crs=crs, transform=transform, nodata=None,
                   compress="deflate")
    with rasterio.open(label_path, "w", **profile) as dst:
        dst.write(final_label, 1)
        dst.set_band_description(1, "flood_reference_label")
        dst.update_tags(1, LABEL_ENCODING="0=non-flood,1=flood,255=ignore", TRACE_MODE=args.trace_mode,
                         SOURCE="EMSR773_AOI03_GRA_v2_observedEventA + imageFootprintA + SAR-validity")

    ref_only_path = out_dir / "flood_reference_before_sar_mask.tif"
    with rasterio.open(ref_only_path, "w", **profile) as dst:
        dst.write(ref_label, 1)
        dst.set_band_description(1, "flood_reference_label_pre_sar_mask")

    print(f"\nWrote {label_path}")
    print(f"Wrote {ref_only_path} (diagnostic: reference label before SAR-validity masking)")

    # Grid identity check against SAR stack
    with rasterio.open(label_path) as check:
        assert check.crs == crs and check.transform == transform and (check.width, check.height) == (shape[1], shape[0])
        print("Grid identity vs SAR stack: CONFIRMED")

    # --- CSV outputs ---
    stats_path = out_dir.parent / "label_statistics.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "flood_px", "nonflood_px", "ignore_px", "total_px",
                    "flood_pct_of_total", "flood_pct_of_usable"])
        usable_total = final_counts["flood"] + final_counts["nonflood"]
        w.writerow(["AOI_overall", final_counts["flood"], final_counts["nonflood"], final_counts["ignore"],
                    final_counts["total"],
                    100*final_counts["flood"]/final_counts["total"],
                    100*final_counts["flood"]/usable_total if usable_total else float("nan")])
        for r in muni_rows:
            usable = r["flood_px"] + r["nonflood_px"]
            w.writerow([r["municipality"], r["flood_px"], r["nonflood_px"], r["ignore_px"], r["total_px"],
                        100*r["flood_px"]/r["total_px"],
                        100*r["flood_px"]/usable if usable else float("nan")])
    print(f"Wrote {stats_path}")

    print("\nLABEL GENERATION COMPLETE")


if __name__ == "__main__":
    main()
