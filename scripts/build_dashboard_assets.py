#!/usr/bin/env python3
"""
UrbanFlood360 - prepare light-weight web assets for the Streamlit dashboard.

  outputs/dashboard/affected_mask.png       U-Net flood-affected area, Web Mercator RGBA
  outputs/dashboard/water_mask.png          U-Net standing water (31 Oct), Web Mercator RGBA
  outputs/dashboard/overlays.json           lat/lon bounds of the PNG overlays
  outputs/dashboard/emsr_reference.geojson  EMSR773 'Flooded area' + 'Flood trace' polygons in the AOI
  outputs/dashboard/buildings.geojson       buildings + impact attributes (EPSG:4326, simplified)
  outputs/dashboard/municipalities.geojson
  outputs/dashboard/emsr_points.geojson     EMSR773 damage points with our matched class

Usage:
    python scripts/build_dashboard_assets.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/processed/predictions"
DAMAGE = ROOT / "data/processed/damage/buildings_impact.gpkg"
OUT = ROOT / "outputs/dashboard"
WEB = "EPSG:3857"

AFFECTED_RGB = (74, 58, 167)     # #4a3aa7 violet
WATER_RGB = (42, 120, 214)       # #2a78d6 blue


def to_web_mercator(path):
    with rasterio.open(path) as src:
        transform, w, h = calculate_default_transform(src.crs, WEB, src.width, src.height, *src.bounds)
        dst = np.full((h, w), 255, dtype="uint8")
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs=WEB,
                  resampling=Resampling.nearest, dst_nodata=255)
        left, top = transform @ (0, 0)
        right, bottom = transform @ (w, h)
        west, south, east, north = transform_bounds(WEB, "EPSG:4326", left, bottom, right, top)
    return dst, [[south, west], [north, east]]


def save_overlay(mask_path, rgb, alpha, out_name, overlays, key):
    mask, bounds = to_web_mercator(mask_path)
    rgba = np.zeros(mask.shape + (4,), dtype="uint8")
    rgba[mask == 1] = (*rgb, alpha)
    Image.fromarray(rgba, mode="RGBA").save(OUT / out_name, optimize=True)
    overlays[key] = bounds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overlays = {}
    save_overlay(PREDICTIONS / "unet_v2_affected/flood_mask.tif", AFFECTED_RGB, 150, "affected_mask.png",
                 overlays, "affected")
    if (PREDICTIONS / "unet_v2_water/flood_mask.tif").exists():
        save_overlay(PREDICTIONS / "unet_v2_water/flood_mask.tif", WATER_RGB, 190, "water_mask.png",
                     overlays, "water")
    json.dump(overlays, open(OUT / "overlays.json", "w"), indent=2)

    aoi = gpd.read_file(ROOT / "data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson").to_crs("EPSG:25830")
    ev = gpd.read_file(ROOT / "data/raw/emsr773/extracted/GRA_v2/EMSR773_AOI03_GRA_PRODUCT_observedEventA_v2.shp")
    ev = ev.to_crs("EPSG:25830")
    ev["geometry"] = ev.geometry.make_valid()
    ev = gpd.clip(ev[["notation", "geometry"]], aoi)
    ev = ev.dissolve("notation").reset_index()
    ev["geometry"] = ev.geometry.simplify(2)
    ev.to_crs("EPSG:4326").to_file(OUT / "emsr_reference.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    b = gpd.read_file(DAMAGE, layer="buildings")
    b["geometry"] = b.geometry.simplify(0.5)
    keep = ["building_id", "municipality", "use", "dwellings", "residents_est", "floors_above",
            "floor_area_m2", "year_built", "condition", "flood_fraction", "mean_prob", "impact", "emsr_damage"]
    if "water_fraction" in b:
        keep.append("water_fraction")
    b = b[keep + ["geometry"]].to_crs("EPSG:4326")
    b["year_built"] = b["year_built"].astype("Int64")
    b.to_file(OUT / "buildings.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    muni = gpd.read_file(ROOT / "data/gis/aoi/municipalities_6.geojson")[["municipality", "area_km2", "geometry"]]
    muni.to_crs("EPSG:4326").to_file(OUT / "municipalities.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    pts = gpd.read_file(DAMAGE, layer="emsr_damage_points").to_crs("EPSG:4326")
    pts[["damage_gra", "obj_type", "our_impact", "geometry"]].to_file(
        OUT / "emsr_points.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    for f in sorted(OUT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
