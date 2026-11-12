#!/usr/bin/env python3
"""
UrbanFlood360 - building-level flood impact assessment.

Flood map used: the U-Net v2 "flood-affected" map (standing water OR flood
trace, data/processed/predictions/unet_v2_affected). The radar image is from
31 Oct, two days after the peak, when most urban streets held mud rather than
water, so a standing-water map alone misses most affected buildings. The
standing-water map (unet_v2_water), when present, is added per building as
`water_fraction`.

For every Catastro building in the AOI (data/processed/buildings/buildings_aoi.gpkg):
  1. Take the footprint plus a ring of `--ring-m` metres (default 10 m = one
     SAR pixel). At 10 m resolution the radar sees water in the streets and
     yards around a building, not inside it, so the ring is what carries the
     flood signal.
  2. Read every flood-map pixel touched by that shape and compute
       flood_fraction  share of pixels classed as flood
       mean_prob / max_prob  U-Net flood probability
  3. Classify:
       Affected           flood_fraction >= --affected  (default 0.5)
       Possibly affected  flood_fraction >= --possible  (default 0.1)
       Not affected       otherwise
       No data            no valid SAR pixel touches the building
  4. Exposure: dwellings, estimated residents (dwellings x --persons-per-dwelling,
     an assumption, default 2.5), floor area.

Validation against Copernicus EMSR773 building damage points (builtUpP):
each point is matched to the building containing it (or the nearest building
within 10 m), and the share of EMSR-damaged buildings that this method flags
is reported. EMSR only records damaged buildings and only inside its image
footprint, so this measures detection rate, not false alarms.

Building-level check against the EMSR773 flood extent (Flooded area + Flood
trace): each building's representative point is looked up in the reference
label, and recall / precision of our "Affected" class are reported for the
whole imaged area, the validation block and the held-out test block, for all
buildings and for residential ones (outputs/tables/emsr_extent_building_check.csv).

Outputs
  data/processed/damage/buildings_impact.gpkg      every building + impact fields
  outputs/tables/damage_by_municipality.csv
  outputs/tables/damage_by_use.csv
  outputs/tables/flood_extent_by_municipality.csv
  outputs/tables/emsr_agreement.csv                EMSR grade x our class
  outputs/tables/emsr_extent_building_check.csv    building-level recall/precision vs EMSR extent
  outputs/tables/priority_buildings.csv            affected buildings ranked by residents
  outputs/tables/damage_summary.json               headline numbers + parameters

Usage:
    python scripts/assess_building_damage.py
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet_v2.data import SPLIT_CODES, block_split_map  # noqa: E402

CRS = "EPSG:25830"
CLASSES = ["Affected", "Possibly affected", "Not affected", "No data"]


def building_flood_stats(geoms, prob, mask, transform):
    """flood_fraction / mean_prob / max_prob / n_pixels for each geometry (all_touched pixels)."""
    H, W = mask.shape
    inv = ~transform
    out = np.full((len(geoms), 4), np.nan)
    for i, g in enumerate(geoms):
        minx, miny, maxx, maxy = g.bounds
        c0, r0 = inv @ (minx, maxy)
        c1, r1 = inv @ (maxx, miny)
        c0, r0 = max(int(np.floor(c0)), 0), max(int(np.floor(r0)), 0)
        c1, r1 = min(int(np.ceil(c1)), W), min(int(np.ceil(r1)), H)
        if c1 <= c0 or r1 <= r0:
            continue
        win = Window(c0, r0, c1 - c0, r1 - r0)
        touched = rasterize([(g, 1)], out_shape=(r1 - r0, c1 - c0),
                            transform=window_transform(win, transform), all_touched=True, dtype="uint8") == 1
        m = mask[r0:r1, c0:c1][touched]
        p = prob[r0:r1, c0:c1][touched]
        ok = m != 255
        if ok.sum() == 0:
            continue
        out[i] = [(m[ok] == 1).mean(), np.nanmean(p[ok]), np.nanmax(p[ok]), ok.sum()]
    return out


def classify(fraction, affected, possible):
    return np.select([np.isnan(fraction), fraction >= affected, fraction >= possible],
                     ["No data", "Affected", "Possibly affected"], "Not affected")


def summarise(df, by):
    g = df.groupby(by)
    out = pd.DataFrame({
        "buildings": g.size(),
        "affected": g.apply(lambda x: (x.impact == "Affected").sum()),
        "possibly_affected": g.apply(lambda x: (x.impact == "Possibly affected").sum()),
        "not_affected": g.apply(lambda x: (x.impact == "Not affected").sum()),
        "no_data": g.apply(lambda x: (x.impact == "No data").sum()),
        "dwellings_affected": g.apply(lambda x: x.loc[x.impact == "Affected", "dwellings"].sum()),
        "dwellings_possibly": g.apply(lambda x: x.loc[x.impact == "Possibly affected", "dwellings"].sum()),
        "residents_affected_est": g.apply(lambda x: x.loc[x.impact == "Affected", "residents_est"].sum()),
        "floor_area_affected_m2": g.apply(lambda x: x.loc[x.impact == "Affected", "floor_area_m2"].sum()),
    })
    out["pct_affected"] = (100 * out.affected / out.buildings).round(1)
    out["pct_affected_or_possibly"] = (100 * (out.affected + out.possibly_affected) / out.buildings).round(1)
    return out.reset_index()


def extent_check(b, ref_affected, transform):
    """Recall/precision of 'Affected' buildings against the EMSR flood extent, by region and use."""
    split = block_split_map(ref_affected.shape)
    inv = ~transform
    rc = np.array([inv @ (p.x, p.y) for p in b.geometry.representative_point()])
    rows = np.clip(rc[:, 1].astype(int), 0, ref_affected.shape[0] - 1)
    cols = np.clip(rc[:, 0].astype(int), 0, ref_affected.shape[1] - 1)
    ref, region = ref_affected[rows, cols], split[rows, cols]
    out = []
    for region_name, in_region in [("all imaged area", np.ones(len(b), bool)),
                                   ("validation block", region == SPLIT_CODES["val"]),
                                   ("test block (held out)", region == SPLIT_CODES["test"])]:
        for use_name, in_use in [("all buildings", np.ones(len(b), bool)),
                                 ("residential", (b["use"] == "Residential").to_numpy())]:
            m = in_region & in_use & (ref != 255)
            inside = ref[m] == 1
            flagged = (b["impact"].to_numpy()[m] == "Affected")
            flagged_any = np.isin(b["impact"].to_numpy()[m], ["Affected", "Possibly affected"])
            out.append(dict(region=region_name, buildings=use_name, n=int(m.sum()), in_emsr_extent=int(inside.sum()),
                            recall_affected=round(float(flagged[inside].mean()), 3) if inside.any() else np.nan,
                            recall_affected_or_possibly=round(float(flagged_any[inside].mean()), 3) if inside.any() else np.nan,
                            precision_affected=round(float(inside[flagged].mean()), 3) if flagged.any() else np.nan,
                            flagged_outside_extent=round(float(flagged[~inside].mean()), 3) if (~inside).any() else np.nan))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-dir", default=str(ROOT / "data/processed/predictions/unet_v2_affected"))
    ap.add_argument("--water-dir", default=str(ROOT / "data/processed/predictions/unet_v2_water"),
                    help="standing-water prediction; skipped if missing")
    ap.add_argument("--buildings", default=str(ROOT / "data/processed/buildings/buildings_aoi.gpkg"))
    ap.add_argument("--ring-m", type=float, default=10.0)
    ap.add_argument("--affected", type=float, default=0.5)
    ap.add_argument("--possible", type=float, default=0.1)
    ap.add_argument("--persons-per-dwelling", type=float, default=2.5)
    args = ap.parse_args()

    pred_dir = Path(args.prediction_dir)
    with rasterio.open(pred_dir / "flood_probability.tif") as s:
        prob, transform = s.read(1), s.transform
        px_area = abs(transform.a * transform.e)
    with rasterio.open(pred_dir / "flood_mask.tif") as s:
        mask = s.read(1)

    b = gpd.read_file(args.buildings, layer="buildings").to_crs(CRS)
    zone = b.geometry.buffer(args.ring_m)
    stats = building_flood_stats(list(zone), prob, mask, transform)
    b["flood_fraction"] = np.round(stats[:, 0], 3)
    b["mean_prob"] = np.round(stats[:, 1], 3)
    b["max_prob"] = np.round(stats[:, 2], 3)
    b["n_pixels"] = stats[:, 3]
    b["impact"] = classify(stats[:, 0], args.affected, args.possible)
    water_dir = Path(args.water_dir)
    water_mask = None
    if (water_dir / "flood_mask.tif").exists():
        with rasterio.open(water_dir / "flood_probability.tif") as s:
            water_prob = s.read(1)
        with rasterio.open(water_dir / "flood_mask.tif") as s:
            water_mask = s.read(1)
        b["water_fraction"] = np.round(building_flood_stats(list(zone), water_prob, water_mask, transform)[:, 0], 3)
    b["residents_est"] = (b["dwellings"] * args.persons_per_dwelling).round(1)

    # --- EMSR773 damage points -> buildings
    emsr = gpd.read_file(ROOT / "data/raw/emsr773/extracted/GRA_v2/EMSR773_AOI03_GRA_PRODUCT_builtUpP_v2.shp").to_crs(CRS)
    aoi = gpd.read_file(ROOT / "data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson").to_crs(CRS)
    emsr = emsr[emsr.within(aoi.union_all())][["damage_gra", "obj_type", "geometry"]].reset_index(drop=True)
    hit = gpd.sjoin_nearest(emsr, b[["building_id", "impact", "geometry"]], how="left",
                            max_distance=10, distance_col="dist_m")
    hit = hit[~hit.index.duplicated()]
    emsr["building_id"] = hit["building_id"]
    emsr["our_impact"] = hit["impact"].fillna("No building within 10 m")
    b["emsr_damage"] = b["building_id"].map(
        emsr.dropna(subset=["building_id"]).sort_values("damage_gra").groupby("building_id")["damage_gra"].first()
    ).fillna("")

    out_dir = ROOT / "data/processed/damage"
    tables = ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    b.to_file(out_dir / "buildings_impact.gpkg", driver="GPKG", layer="buildings")
    emsr.to_file(out_dir / "buildings_impact.gpkg", driver="GPKG", layer="emsr_damage_points")

    by_muni = summarise(b, "municipality")
    by_use = summarise(b, "use")
    by_muni.to_csv(tables / "damage_by_municipality.csv", index=False)
    by_use.to_csv(tables / "damage_by_use.csv", index=False)

    # --- flooded area per municipality (prediction vs EMSR reference)
    muni = gpd.read_file(ROOT / "data/gis/aoi/municipalities_6.geojson").to_crs(CRS)
    with rasterio.open(ROOT / "data/processed/unet_dataset/labels/flood_reference.tif") as s:
        ref_water = s.read(1)
    with rasterio.open(ROOT / "data/processed/flood_affected_dataset/labels/flood_reference.tif") as s:
        ref_affected = s.read(1)
    muni_id = rasterize([(g, i + 1) for i, g in enumerate(muni.geometry)], out_shape=mask.shape,
                        transform=transform, dtype="uint8")
    extent = []
    for i, name in enumerate(muni.municipality, start=1):
        inside = muni_id == i
        valid = inside & (mask != 255)
        km2 = lambda m: round(float(m.sum()) * px_area / 1e6, 3)  # noqa: E731
        extent.append(dict(
            municipality=name,
            area_km2=km2(inside),
            predicted_flood_km2=km2(valid & (mask == 1)),
            predicted_flood_pct=round(100 * (valid & (mask == 1)).sum() / max(valid.sum(), 1), 1),
            predicted_flood_in_mapped_km2=km2(inside & (ref_affected != 255) & (mask == 1)),
            predicted_water_km2=km2(inside & (water_mask == 1)) if water_mask is not None else np.nan,
            emsr_flood_km2=km2(inside & (ref_affected == 1)),
            emsr_water_km2=km2(inside & (ref_water == 1)),
            emsr_trace_km2=km2(inside & (ref_affected == 1) & (ref_water != 1)),
            emsr_mapped_km2=km2(inside & (ref_affected != 255)),
        ))
    extent = pd.DataFrame(extent)
    extent.to_csv(tables / "flood_extent_by_municipality.csv", index=False)

    # --- agreement with EMSR
    agree = pd.crosstab(emsr["damage_gra"], emsr["our_impact"], margins=True, margins_name="Total")
    agree.to_csv(tables / "emsr_agreement.csv")
    check = extent_check(b, ref_affected, transform)
    check.to_csv(tables / "emsr_extent_building_check.csv", index=False)
    matched = emsr[emsr.building_id.notna()]
    flagged = matched.our_impact.isin(["Affected", "Possibly affected"])
    detection = {
        "emsr_points_in_aoi": int(len(emsr)),
        "matched_to_building": int(len(matched)),
        "flagged_affected_or_possibly_pct": round(100 * flagged.mean(), 1),
        "flagged_affected_pct": round(100 * (matched.our_impact == "Affected").mean(), 1),
        "by_grade_flagged_pct": {g: round(100 * x.our_impact.isin(["Affected", "Possibly affected"]).mean(), 1)
                                 for g, x in matched.groupby("damage_gra")},
    }

    affected = b[b.impact == "Affected"]
    priority = (affected.sort_values(["residents_est", "flood_fraction"], ascending=False)
                [["building_id", "municipality", "use", "dwellings", "residents_est", "floors_above",
                  "floor_area_m2", "flood_fraction", "mean_prob", "emsr_damage"]].head(500))
    priority.to_csv(tables / "priority_buildings.csv", index=False)

    meta = json.load(open(pred_dir / "prediction_meta.json"))
    summary = {
        "flood_map": meta,
        "parameters": {"ring_m": args.ring_m, "affected_threshold": args.affected,
                       "possible_threshold": args.possible, "persons_per_dwelling": args.persons_per_dwelling},
        "aoi_km2": round(float(extent.area_km2.sum()), 2),
        "flooded_km2": round(float(extent.predicted_flood_km2.sum()), 2),
        "flooded_pct_of_aoi": round(100 * float(extent.predicted_flood_km2.sum() / extent.area_km2.sum()), 1),
        "standing_water_km2": round(float(extent.predicted_water_km2.sum()), 2),
        "buildings_total": int(len(b)),
        "buildings_by_class": {c: int((b.impact == c).sum()) for c in CLASSES},
        "dwellings_affected": int(affected.dwellings.sum()),
        "dwellings_possibly_affected": int(b.loc[b.impact == "Possibly affected", "dwellings"].sum()),
        "residents_affected_est": int(round(affected.residents_est.sum())),
        "floor_area_affected_m2": int(affected.floor_area_m2.sum()),
        "emsr_validation": detection,
        "emsr_extent_check": check.to_dict(orient="records"),
    }
    json.dump(summary, open(tables / "damage_summary.json", "w"), indent=2)

    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in summary.items() if k != "flood_map"}, indent=2))
    print("\nBy municipality:\n" + by_muni.to_string(index=False))
    print("\nBy use:\n" + by_use.to_string(index=False))
    print("\nFlood extent:\n" + extent.to_string(index=False))
    print("\nEMSR agreement (rows = EMSR grade, cols = our class):\n" + agree.to_string())
    print("\nBuilding-level check vs EMSR flood extent:\n" + check.to_string(index=False))


if __name__ == "__main__":
    main()
