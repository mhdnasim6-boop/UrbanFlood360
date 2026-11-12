#!/usr/bin/env python3
"""
UrbanFlood360 - building inventory for damage assessment.

Merges the six Catastro INSPIRE municipality downloads into one building
layer for the AOI:
  * building.gml      -> footprint, current use, dwellings, building units,
                         official floor area, condition, construction year
  * buildingpart.gml  -> floors above/below ground (only populated at part
                         level - aggregated per building as the max over parts)
  * municipality      -> spatial join of the footprint centroid with
                         data/gis/aoi/municipalities_6.geojson

Only buildings whose centroid lies inside the AOI are kept.

Output: data/processed/buildings/buildings_aoi.gpkg (EPSG:25830)

Usage:
    python scripts/prepare_buildings.py
"""
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*parsing datetimes.*")

ROOT = Path(__file__).resolve().parents[1]
CATASTRO = ROOT / "data/raw/catastro/extracted"
MUNICIPALITIES = ROOT / "data/gis/aoi/municipalities_6.geojson"
OUT = ROOT / "data/processed/buildings/buildings_aoi.gpkg"
CRS = "EPSG:25830"

USE_LABELS = {
    "1_residential": "Residential",
    "2_agriculture": "Agriculture",
    "3_industrial": "Industrial",
    "4_1_office": "Office",
    "4_2_retail": "Retail",
    "4_3_publicServices": "Public services",
}


def construction_year(value):
    try:
        year = int(str(value)[:4])
    except ValueError:
        return np.nan
    return year if 1700 <= year <= 2025 else np.nan  # a few records carry typo years like 3026


def load_municipality(folder):
    code = folder.name
    b = gpd.read_file(folder / f"A.ES.SDGC.BU.{code}.building.gml").to_crs(CRS)
    parts = gpd.read_file(folder / f"A.ES.SDGC.BU.{code}.buildingpart.gml", ignore_geometry=True)
    parts["building_id"] = parts["localId"].str.split("_part").str[0]
    floors = parts.groupby("building_id").agg(
        floors_above=("numberOfFloorsAboveGround", "max"),
        floors_below=("numberOfFloorsBelowGround", "max"),
        n_parts=("localId", "count"),
    )
    out = gpd.GeoDataFrame({
        "building_id": b["localId"],
        "catastro_code": code,
        "use_code": b["currentUse"],
        "use": b["currentUse"].map(USE_LABELS).fillna("Unknown"),
        "dwellings": pd.to_numeric(b["numberOfDwellings"], errors="coerce").fillna(0).astype(int),
        "building_units": pd.to_numeric(b["numberOfBuildingUnits"], errors="coerce").fillna(0).astype(int),
        "floor_area_m2": pd.to_numeric(b["value"], errors="coerce"),
        "condition": b["conditionOfConstruction"],
        "year_built": b["beginning"].map(construction_year),
    }, geometry=b.geometry, crs=CRS)
    out = out.merge(floors, left_on="building_id", right_index=True, how="left")
    return out


def main():
    folders = sorted(p for p in CATASTRO.iterdir() if p.is_dir())
    buildings = pd.concat([load_municipality(f) for f in folders], ignore_index=True)
    buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs=CRS)
    buildings["geometry"] = buildings.geometry.make_valid()
    buildings["footprint_m2"] = buildings.geometry.area.round(1)
    print(f"Catastro buildings loaded: {len(buildings)} from {len(folders)} municipalities")

    muni = gpd.read_file(MUNICIPALITIES).to_crs(CRS)[["municipality", "geometry"]]
    centroids = gpd.GeoDataFrame(buildings[["building_id"]], geometry=buildings.geometry.representative_point(), crs=CRS)
    joined = gpd.sjoin(centroids, muni, how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]
    buildings["municipality"] = joined["municipality"]
    outside = buildings["municipality"].isna().sum()
    buildings = buildings[buildings["municipality"].notna()].copy()
    print(f"Dropped {outside} buildings outside the AOI -> {len(buildings)} kept")

    buildings = buildings.drop_duplicates("building_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(OUT, driver="GPKG", layer="buildings")
    print(f"\nWrote {OUT}")
    print(buildings.groupby("municipality").agg(buildings=("building_id", "count"),
                                                dwellings=("dwellings", "sum")).to_string())
    print("\nBy use:\n" + buildings["use"].value_counts().to_string())
    print(f"\nMissing floors: {buildings['floors_above'].isna().sum()}, "
          f"missing floor area: {buildings['floor_area_m2'].isna().sum()}")


if __name__ == "__main__":
    main()
