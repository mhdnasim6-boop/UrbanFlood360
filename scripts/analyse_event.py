#!/usr/bin/env python3
"""
UrbanFlood360 - flood map + building damage assessment for a NEW flood event.

This is what the dashboard's "Analyse new event" page runs. It can also be used
from the command line.

Inputs
  --pre / --post   Sentinel-1 GeoTIFFs before and after the flood, each with a VV
                   and a VH band (calibrated Sigma0, terrain-corrected, e.g. the
                   SNAP output used for Valencia). Bands are found by their
                   description ("VV", "VH"); otherwise pass --band-order.
                   Linear Sigma0 is converted to dB automatically. If the two
                   images are not on the same grid, the "after" image is
                   resampled onto the "before" grid; geographic (lat/lon) or
                   non-10 m data are reprojected to UTM at 10 m, the model's
                   training resolution.
  --aoi            optional area boundary (GeoJSON / GPKG / zipped shapefile);
                   pixels outside it are ignored.
  --buildings      optional building footprints (GeoJSON / GPKG / zipped
                   shapefile / Catastro INSPIRE building.gml). Without it only
                   the flood map is produced.

Steps (same models and rules as the Valencia case study)
  1. 6-band stack: VV/VH before, after, change (dB)
  2. U-Net v2 "flood-affected" model (+ optionally the "standing water" model),
     selected checkpoints and thresholds from models/unet_v2/selected_model_*.json
  3. building class from the flood share of footprint + 10 m ring:
     Affected >= 0.5, Possibly affected >= 0.1 (scripts/assess_building_damage.py)

Outputs (in --output-dir)
  sar_features_6ch.tif, flood_probability.tif, flood_mask.tif,
  flood_polygons.geojson, [water_mask.tif], [buildings_impact.geojson / .csv],
  summary.json, results.zip

Usage:
    python scripts/analyse_event.py --pre pre.tif --post post.tif \
        [--aoi area.geojson] [--buildings buildings.gpkg] --output-dir outputs/user_runs/my_event
"""
import argparse
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import shape

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet_v2.data import normalize, predict_raster  # noqa: E402
from models.unet_v2.train import get_device  # noqa: E402
from scripts.assess_building_damage import building_flood_stats, classify  # noqa: E402
from scripts.predict_flood_map import load_model  # noqa: E402
from scripts.prepare_buildings import USE_LABELS  # noqa: E402

CHANNELS = ["VV_pre_dB", "VH_pre_dB", "VV_post_dB", "VH_post_dB", "VV_change_dB", "VH_change_dB"]
CLASSES = ["Affected", "Possibly affected", "Not affected", "No data"]
TRAINING_RES_M = 10.0
MAX_PIXELS = 40_000_000          # ~63 km x 63 km at 10 m; keeps memory and run time reasonable
DWELLING_COLUMNS = ["dwellings", "numberOfDwellings", "building:flats", "flats", "units"]
USE_COLUMNS = ["use", "currentUse", "building", "building_use", "type"]


class AnalysisError(ValueError):
    """Problem with the user's input, with a message meant for the user."""


# ------------------------------------------------------------------ SAR input
def _find(descriptions, pol):
    hits = [i for i, d in enumerate(descriptions) if d and pol in d.upper()]
    return hits[0] if len(hits) == 1 else None


def read_scene(path, band_order=None):
    """VV, VH (dB, NaN = no data) and the raster's grid. Notes are appended to a list."""
    notes = []
    with rasterio.open(path) as src:
        if src.crs is None:
            raise AnalysisError(f"{Path(path).name}: the GeoTIFF has no coordinate system (CRS).")
        if src.count < 2:
            raise AnalysisError(f"{Path(path).name}: needs a VV and a VH band, found {src.count} band(s).")
        desc = [d or "" for d in src.descriptions]
        if band_order:
            order = [p.strip().upper() for p in band_order.split(",")]
            iv, ih = order.index("VV"), order.index("VH")
        else:
            iv, ih = _find(desc, "VV"), _find(desc, "VH")
            if iv is None or ih is None:
                raise AnalysisError(
                    f"{Path(path).name}: could not tell which band is VV and which is VH from the band names "
                    f"{desc}. Choose the band order in the settings.")
        data = src.read([iv + 1, ih + 1]).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        data[data == 0] = np.nan                     # SNAP writes 0 outside the swath
        grid = dict(crs=src.crs, transform=src.transform, width=src.width, height=src.height, bounds=src.bounds)
    med = np.nanmedian(data[0]) if np.isfinite(data[0]).any() else np.nan
    if np.isnan(med):
        raise AnalysisError(f"{Path(path).name}: no valid pixels.")
    if med > 0:                                       # linear Sigma0 is positive, dB is mostly negative
        with np.errstate(divide="ignore", invalid="ignore"):
            data = 10 * np.log10(np.where(data > 0, data, np.nan))
        notes.append(f"{Path(path).name}: values looked like linear backscatter and were converted to dB.")
    return data[0], data[1], grid, notes


def utm_crs(grid):
    w, s, e, n = transform_bounds(grid["crs"], "EPSG:4326", *grid["bounds"])
    lon, lat = (w + e) / 2, (s + n) / 2
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def target_grid(grid, notes):
    """The grid the model runs on: the 'before' grid, unless it is lat/lon or not ~10 m."""
    res = abs(grid["transform"].a)
    if grid["crs"].is_geographic or not (8 <= res <= 12):
        dst = utm_crs(grid)
        transform, w, h = calculate_default_transform(grid["crs"], dst, grid["width"], grid["height"],
                                                      *grid["bounds"], resolution=TRAINING_RES_M)
        notes.append(f"Images were reprojected to {dst.to_string()} at 10 m, the resolution the model was trained on "
                     f"(input: {grid['crs'].to_string()}, pixel {res:g}).")
        return dict(crs=dst, transform=transform, width=w, height=h)
    return {k: grid[k] for k in ("crs", "transform", "width", "height")}


def to_grid(arr, src_grid, dst):
    same = (src_grid["crs"] == dst["crs"] and src_grid["transform"] == dst["transform"]
            and src_grid["width"] == dst["width"] and src_grid["height"] == dst["height"])
    if same:
        return arr
    out = np.full((dst["height"], dst["width"]), np.nan, dtype="float32")
    reproject(arr, out, src_transform=src_grid["transform"], src_crs=src_grid["crs"], src_nodata=np.nan,
              dst_transform=dst["transform"], dst_crs=dst["crs"], dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def read_vector(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        gdf = gpd.read_file(f"zip://{path}")
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise AnalysisError(f"{path.name}: the file has no coordinate system (CRS).")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def build_stack(pre_path, post_path, aoi_path=None, band_order=None):
    """(6,H,W) float32 feature stack, grid dict and notes."""
    vv0, vh0, g0, notes = read_scene(pre_path, band_order)
    vv1, vh1, g1, n1 = read_scene(post_path, band_order)
    notes += n1
    dst = target_grid(g0, notes)

    aoi = None
    if aoi_path:
        aoi = read_vector(aoi_path).to_crs(dst["crs"])
        # crop to the AOI (plus 2 pixels) before reprojecting anything
        t = dst["transform"]
        win = from_bounds(*aoi.total_bounds, transform=t).round_offsets().round_lengths()
        c0, r0 = max(int(win.col_off) - 2, 0), max(int(win.row_off) - 2, 0)
        c1 = min(int(win.col_off + win.width) + 2, dst["width"])
        r1 = min(int(win.row_off + win.height) + 2, dst["height"])
        if c1 <= c0 or r1 <= r0:
            raise AnalysisError("The area boundary does not overlap the 'before' image.")
        dst = dict(crs=dst["crs"], transform=t * rasterio.Affine.translation(c0, r0), width=c1 - c0, height=r1 - r0)

    if dst["width"] * dst["height"] > MAX_PIXELS:
        km = (dst["width"] * dst["height"]) ** 0.5 * TRAINING_RES_M / 1000
        raise AnalysisError(f"The area is too large (~{km:.0f} km × {km:.0f} km at 10 m). "
                            "Upload an area boundary to limit it to the city, or crop the images.")

    if g1["crs"] != g0["crs"] or g1["transform"] != g0["transform"] or g1["width"] != g0["width"]:
        notes.append("The 'after' image was on a different grid and was resampled onto the 'before' grid. "
                     "Both images should come from the same orbit track for reliable results.")
    vv0, vh0 = to_grid(vv0, g0, dst), to_grid(vh0, g0, dst)
    vv1, vh1 = to_grid(vv1, g1, dst), to_grid(vh1, g1, dst)
    stack = np.stack([vv0, vh0, vv1, vh1, vv1 - vv0, vh1 - vh0]).astype("float32")

    valid = np.all(np.isfinite(stack), axis=0)
    if aoi is not None:
        inside = ~geometry_mask(list(aoi.geometry), (dst["height"], dst["width"]), dst["transform"])
        valid &= inside
        stack[:, ~valid] = np.nan
    if valid.sum() == 0:
        raise AnalysisError("The 'before' and 'after' images do not overlap (no pixel has data in both).")
    if valid.mean() < 0.5 and aoi is None:
        notes.append(f"Only {100 * valid.mean():.0f}% of the area has data in both images.")
    return stack, dst, valid, aoi, notes


def input_check(stack, valid, norm):
    """Warn when inputs look unlike the training data (wrong calibration, wrong band order...)."""
    notes = []
    for c, name in enumerate(CHANNELS[:4]):
        s = norm["per_channel"][name]
        z = (float(np.nanmedian(stack[c][valid])) - s["mean"]) / s["std"]
        if abs(z) > 2:
            notes.append(f"{name} has a median of {np.nanmedian(stack[c][valid]):.1f} dB, far from the training data "
                         f"({s['mean']:.1f} ± {s['std']:.1f} dB). Check that the images are calibrated Sigma0 and that "
                         "VV and VH are not swapped. Results may be unreliable.")
    return notes


# ------------------------------------------------------------------ model
def selected(target):
    sel = json.load(open(ROOT / f"models/unet_v2/selected_model_{target}.json"))
    return ROOT / sel["checkpoint"], float(sel["threshold"])


def run_model(stack, valid, target, device, threshold=None):
    ckpt_path, default_thr = selected(target)
    thr = default_thr if threshold is None else threshold
    model, ckpt = load_model(ckpt_path, device)
    inf = ckpt["config"]["inference"]
    prob = predict_raster(model, normalize(stack, ckpt["normalization"]), device,
                          inf["tile"], inf["overlap"], inf["batch_size"], valid=valid)
    mask = np.full(prob.shape, 255, dtype="uint8")
    mask[valid] = (prob[valid] >= thr).astype("uint8")
    return prob, mask, thr, ckpt


def write_tif(path, arr, grid, dtype, nodata, name):
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"], width=grid["width"], count=arr.shape[0]
                       if arr.ndim == 3 else 1, crs=grid["crs"], transform=grid["transform"], dtype=dtype,
                       nodata=nodata, compress="deflate") as dst:
        if arr.ndim == 3:
            dst.write(arr.astype(dtype))
            for i, n in enumerate(name, start=1):
                dst.set_band_description(i, n)
        else:
            dst.write(arr.astype(dtype), 1)
            dst.set_band_description(1, name)


# ------------------------------------------------------------------ buildings
def pick(gdf, names):
    lower = {c.lower(): c for c in gdf.columns}
    return next((lower[n.lower()] for n in names if n.lower() in lower), None)


def prepare_buildings(path, grid):
    b = read_vector(path).to_crs(grid["crs"])
    t = grid["transform"]
    left, top = t.c, t.f
    right, bottom = left + t.a * grid["width"], top + t.e * grid["height"]
    b = b.cx[left:right, bottom:top].copy()
    b = b[b.geom_type.isin(["Polygon", "MultiPolygon", "Point", "MultiPoint"])]
    if len(b) == 0:
        raise AnalysisError("None of the buildings fall inside the radar images.")
    out = gpd.GeoDataFrame(geometry=b.geometry.values, crs=b.crs)
    idc = pick(b, ["building_id", "localId", "id", "osm_id", "@id", "fid"])
    out["building_id"] = b[idc].astype(str).values if idc else [f"B{i + 1}" for i in range(len(b))]
    dc = pick(b, DWELLING_COLUMNS)
    if dc and pd.to_numeric(b[dc], errors="coerce").notna().mean() < 0.5:
        dc = None        # e.g. OpenStreetMap building:flats, mapped on a few buildings only: too sparse to sum
    out["dwellings"] = pd.to_numeric(b[dc], errors="coerce").values if dc else np.nan
    uc = pick(b, USE_COLUMNS)
    use = b[uc].astype(str).map(lambda v: USE_LABELS.get(v, v)).replace({"yes": "Unknown", "None": "Unknown",
                                                                          "nan": "Unknown"}) if uc else "Unknown"
    out["use"] = use.values if uc else use
    fa = pick(b, ["floor_area_m2", "value", "area_m2"])
    out["floor_area_m2"] = pd.to_numeric(b[fa], errors="coerce").values if fa else np.nan
    out["footprint_m2"] = out.geometry.area.round(1)
    return out, dict(dwellings_column=dc, use_column=uc)


def assess_buildings(b, prob, mask, grid, water_mask=None, water_prob=None, ring_m=10.0, affected=0.5,
                     possible=0.1, persons_per_dwelling=2.5):
    zone = list(b.geometry.buffer(ring_m))
    s = building_flood_stats(zone, prob, mask, grid["transform"])
    b = b.copy()
    b["flood_fraction"] = np.round(s[:, 0], 3)
    b["mean_prob"] = np.round(s[:, 1], 3)
    b["impact"] = classify(s[:, 0], affected, possible)
    if water_mask is not None:
        b["water_fraction"] = np.round(building_flood_stats(zone, water_prob, water_mask, grid["transform"])[:, 0], 3)
    b["residents_est"] = (b["dwellings"] * persons_per_dwelling).round(1)
    return b


# ------------------------------------------------------------------ main entry
def analyse(pre, post, output_dir, aoi=None, buildings=None, band_order=None, with_water=False,
            threshold=None, ring_m=10.0, affected=0.5, possible=0.1, persons_per_dwelling=2.5,
            event_name="New event", progress=None):
    """Run the whole analysis; returns the summary dict. `progress(fraction, text)` is optional."""
    say = progress or (lambda f, t: print(f"[{f:4.0%}] {t}"))
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    say(0.05, "Reading the radar images")
    stack, grid, valid, aoi_gdf, notes = build_stack(pre, post, aoi, band_order)
    _, ckpt = load_model(selected("affected")[0], "cpu")
    notes += input_check(stack, valid, ckpt["normalization"])
    write_tif(out / "sar_features_6ch.tif", stack, grid, "float32", np.nan, CHANNELS)

    device = get_device()
    say(0.25, "Mapping flood-affected ground with the U-Net")
    prob, mask, thr, _ = run_model(stack, valid, "affected", device, threshold)
    write_tif(out / "flood_probability.tif", prob, grid, "float32", np.nan, "flood_probability")
    write_tif(out / "flood_mask.tif", mask, grid, "uint8", 255, f"flood_affected_threshold_{thr}")

    water_prob = water_mask = None
    if with_water:
        say(0.45, "Mapping standing water with the U-Net")
        water_prob, water_mask, water_thr, _ = run_model(stack, valid, "water", device)
        write_tif(out / "water_mask.tif", water_mask, grid, "uint8", 255, f"standing_water_threshold_{water_thr}")

    say(0.6, "Turning the flood map into polygons")
    polys = [shape(g) for g, v in shapes(mask, mask=mask == 1, transform=grid["transform"]) if v == 1]
    fp = gpd.GeoDataFrame({"area_m2": [round(p.area, 1) for p in polys]}, geometry=polys, crs=grid["crs"])
    fp.to_crs("EPSG:4326").to_file(out / "flood_polygons.geojson", driver="GeoJSON")

    px_km2 = abs(grid["transform"].a * grid["transform"].e) / 1e6
    summary = dict(
        event_name=event_name,
        created=time.strftime("%Y-%m-%d %H:%M"),
        inputs=dict(pre=Path(pre).name, post=Path(post).name, aoi=Path(aoi).name if aoi else None,
                    buildings=Path(buildings).name if buildings else None),
        grid=dict(crs=grid["crs"].to_string(), width=grid["width"], height=grid["height"],
                  pixel_m=abs(grid["transform"].a)),
        model=dict(checkpoint=str(selected("affected")[0].relative_to(ROOT)), threshold=thr, device=str(device)),
        area_with_data_km2=round(float(valid.sum() * px_km2), 2),
        flood_affected_km2=round(float((mask == 1).sum() * px_km2), 2),
        flood_affected_pct=round(100 * float((mask == 1).sum() / max(valid.sum(), 1)), 1),
        standing_water_km2=round(float((water_mask == 1).sum() * px_km2), 2) if with_water else None,
        n_flood_polygons=len(fp),
        parameters=dict(ring_m=ring_m, affected_threshold=affected, possible_threshold=possible,
                        persons_per_dwelling=persons_per_dwelling),
        notes=notes,
    )

    if buildings:
        say(0.7, "Checking every building against the flood map")
        b, cols = prepare_buildings(buildings, grid)
        b = assess_buildings(b, prob, mask, grid, water_mask, water_prob, ring_m, affected, possible,
                             persons_per_dwelling)
        if aoi_gdf is not None:
            b = b[b.geometry.representative_point().within(aoi_gdf.union_all())]
        b.to_crs("EPSG:4326").to_file(out / "buildings_impact.geojson", driver="GeoJSON")
        b.drop(columns="geometry").to_csv(out / "buildings_impact.csv", index=False)
        aff = b[b.impact == "Affected"]
        has_dw = cols["dwellings_column"] is not None
        summary["buildings"] = dict(
            total=int(len(b)),
            by_class={c: int((b.impact == c).sum()) for c in CLASSES},
            dwellings_affected=int(aff.dwellings.sum()) if has_dw else None,
            residents_affected_est=int(round(aff.residents_est.sum())) if has_dw else None,
            footprint_affected_m2=int(aff.footprint_m2.sum()),
            columns_used=cols,
        )
        if not has_dw:
            notes.append("The building data has no usable dwelling counts (missing, or filled in for fewer than half the buildings, as is usual in OpenStreetMap), so dwellings and residents are not estimated.")

    summary["runtime_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=str)

    say(0.95, "Packing the results")
    with zipfile.ZipFile(out / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f.name != "results.zip":
                z.write(f, f.name)
    say(1.0, "Done")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--aoi")
    ap.add_argument("--buildings")
    ap.add_argument("--band-order", help='e.g. "VH,VV" when bands have no names')
    ap.add_argument("--with-water", action="store_true", help="also run the standing-water model")
    ap.add_argument("--threshold", type=float, help="flood-affected threshold (default: the selected one)")
    ap.add_argument("--event-name", default="New event")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/user_runs" / time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()
    try:
        s = analyse(args.pre, args.post, args.output_dir, args.aoi, args.buildings, args.band_order,
                    args.with_water, args.threshold, event_name=args.event_name)
    except AnalysisError as e:
        sys.exit(f"Error: {e}")
    print(json.dumps(s, indent=2, default=str))
    print(f"Results in {args.output_dir}")


if __name__ == "__main__":
    main()
