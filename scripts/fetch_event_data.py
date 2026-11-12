#!/usr/bin/env python3
"""
UrbanFlood360 - download everything needed to analyse a flood: Sentinel-1 before/after images, the area boundary
and the buildings. Used by the dashboard's "Download and analyse" page; also works from the command line.

Data sources (no account needed)
  Area        OpenStreetMap Nominatim geocoder (place name -> boundary), or a bounding box you give
  Sentinel-1  Microsoft Planetary Computer, collection "sentinel-1-rtc": Sentinel-1 GRD, radiometrically
              terrain-corrected (gamma0, linear power), 10 m, cloud-optimised GeoTIFFs. Only the pixels inside the
              area are read, so a town is a few MB.
  Buildings   OpenStreetMap via the Overpass API (footprints, building type, building:flats when mapped)

Choosing the image pair
  For each orbit track that covers the area, the last image before the flood date and the first image on or
  after it are taken. Both must come from the same track (same viewing geometry), so the change between them is
  the flood, not a different look angle. The pair whose "after" image is closest to the flood wins; ties go to
  the pair that covers more of the area.

Output (in --output-dir): before.tif, after.tif (VV, VH in dB, gamma0), aoi.geojson, buildings.geojson,
download.json. Then run scripts/analyse_event.py on them (the dashboard does this for you).

Usage:
    python scripts/fetch_event_data.py --place "Paiporta, Spain" --flood-date 2024-10-29 --output-dir outputs/user_runs/paiporta
    python scripts/fetch_event_data.py --bbox -0.43 39.38 -0.36 39.44 --flood-date 2024-10-29 --output-dir ...
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from shapely.geometry import Polygon, box, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyse_event import AnalysisError  # noqa: E402

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
USER_AGENT = "UrbanFlood360/0.1 (urban flood mapping research dashboard)"
RES_M = 10.0
MAX_AREA_KM2 = 600           # keeps downloads, Overpass queries and inference reasonable
DAYS_BEFORE = 36             # search window: 3 x 12-day repeat cycle before the flood
DAYS_AFTER = 14


def log(progress, frac, text):
    (progress or (lambda f, t: print(f"[{f:4.0%}] {t}", flush=True)))(frac, text)


# ------------------------------------------------------------------ area
def geocode(place):
    """Boundary polygon (EPSG:4326) and display name for a place, from OpenStreetMap Nominatim."""
    r = requests.get(NOMINATIM, params=dict(q=place, format="json", limit=1, polygon_geojson=1),
                     headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    hits = r.json()
    if not hits:
        raise AnalysisError(f"Could not find '{place}'. Try a more specific name, e.g. 'Paiporta, Valencia, Spain'.")
    h = hits[0]
    geom = shape(h["geojson"]) if h.get("geojson") else None
    if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon"):
        s, n, w, e = map(float, h["boundingbox"])
        geom = box(w, s, e, n)
    return gpd.GeoDataFrame({"name": [h["display_name"]]}, geometry=[geom], crs="EPSG:4326")


def bbox_area(bbox):
    w, s, e, n = bbox
    return gpd.GeoDataFrame({"name": ["Selected box"]}, geometry=[box(w, s, e, n)], crs="EPSG:4326")


def utm_for(aoi):
    c = aoi.to_crs("EPSG:4326").union_all().centroid
    return CRS.from_epsg((32600 if c.y >= 0 else 32700) + int((c.x + 180) // 6) + 1)


def area_km2(aoi):
    return float(aoi.to_crs(utm_for(aoi)).area.sum() / 1e6)


def check_area(aoi):
    km2 = area_km2(aoi)
    if km2 > MAX_AREA_KM2:
        raise AnalysisError(f"The area is {km2:,.0f} km², above the {MAX_AREA_KM2} km² limit. Pick a town or district, "
                            "or draw a smaller box.")
    if km2 < 0.5:
        raise AnalysisError("The area is smaller than 0.5 km². Pick a larger area.")
    return km2


# ------------------------------------------------------------------ Sentinel-1 search
def _catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_scenes(aoi, flood_date, days_before=DAYS_BEFORE, days_after=DAYS_AFTER):
    """One row per acquisition (frames of the same pass merged) around the flood date."""
    flood_date = pd.Timestamp(flood_date).date()
    start, end = flood_date - dt.timedelta(days=days_before), flood_date + dt.timedelta(days=days_after)
    geom = aoi.to_crs("EPSG:4326").union_all()
    items = list(_catalog().search(collections=[COLLECTION], intersects=geom.__geo_interface__,
                                   datetime=f"{start}/{end}").items())
    rows = []
    for it in items:
        p = it.properties
        foot = shape(it.geometry)
        rows.append(dict(id=it.id, item=it, datetime=pd.Timestamp(p["datetime"]),
                         date=pd.Timestamp(p["datetime"]).date(), platform=p.get("platform"),
                         orbit=p.get("sat:relative_orbit"), direction=p.get("sat:orbit_state"),
                         polarizations=tuple(p.get("sar:polarizations", [])), footprint=foot))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df.polarizations.map(lambda x: "VV" in x and "VH" in x)]
    acq = []
    for (date, orbit, platform), g in df.groupby(["date", "orbit", "platform"]):
        foot = gpd.GeoSeries(list(g.footprint), crs="EPSG:4326").union_all()
        cover = float(geom.intersection(foot).area / geom.area)
        acq.append(dict(date=date, orbit=int(orbit), direction=g.direction.iloc[0], platform=platform,
                        coverage=round(cover, 3), n_frames=len(g), items=list(g["item"]),
                        when="before" if date < flood_date else "after",
                        days_from_flood=(date - flood_date).days))
    return pd.DataFrame(acq).sort_values(["date", "orbit"]).reset_index(drop=True)


def choose_pair(scenes, min_coverage=0.5):
    """(before_row, after_row) from the same orbit track: after closest to the flood, then best coverage."""
    if scenes.empty:
        raise AnalysisError("No Sentinel-1 images found for this area and dates.")
    best = None
    for orbit, g in scenes[scenes.coverage >= min_coverage].groupby("orbit"):
        pre, post = g[g.when == "before"], g[g.when == "after"]
        if pre.empty or post.empty:
            continue
        b, a = pre.iloc[-1], post.iloc[0]
        key = (a.days_from_flood, -min(a.coverage, b.coverage), -b.days_from_flood)
        if best is None or key < best[0]:
            best = (key, b, a)
    if best is None:
        have_after = not scenes[scenes.when == "after"].empty
        raise AnalysisError(
            "No before/after pair from the same orbit track covers this area. "
            + ("Try a larger search window." if have_after else
               "There is no Sentinel-1 image after this date yet (or within the search window)."))
    return best[1], best[2]


# ------------------------------------------------------------------ download
def target_grid(aoi, pad_m=200):
    crs = utm_for(aoi)
    minx, miny, maxx, maxy = aoi.to_crs(crs).total_bounds
    minx, miny = np.floor((minx - pad_m) / RES_M) * RES_M, np.floor((miny - pad_m) / RES_M) * RES_M
    maxx, maxy = np.ceil((maxx + pad_m) / RES_M) * RES_M, np.ceil((maxy + pad_m) / RES_M) * RES_M
    w, h = int(round((maxx - minx) / RES_M)), int(round((maxy - miny) / RES_M))
    return dict(crs=crs, transform=from_origin(minx, maxy, RES_M, RES_M), width=w, height=h)


def read_band(items, pol, grid):
    """Mosaic of one polarisation from the frames of one pass, resampled onto `grid`, linear power, NaN = no data."""
    import planetary_computer
    out = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    for it in items:
        href = planetary_computer.sign(it.assets[pol.lower()].href)
        with rasterio.Env(GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(href) as src, WarpedVRT(src, crs=grid["crs"], transform=grid["transform"],
                                                       width=grid["width"], height=grid["height"],
                                                       resampling=Resampling.bilinear, src_nodata=src.nodata,
                                                       nodata=np.nan, dtype="float32") as vrt:
                a = vrt.read(1)
        a[(a <= 0) | ~np.isfinite(a)] = np.nan
        fill = np.isnan(out) & np.isfinite(a)
        out[fill] = a[fill]
    return out


def download_scene(row, grid, path):
    vv, vh = read_band(row["items"], "VV", grid), read_band(row["items"], "VH", grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.stack([10 * np.log10(vv), 10 * np.log10(vh)]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"], width=grid["width"], count=2,
                       dtype="float32", crs=grid["crs"], transform=grid["transform"], nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        dst.write(db)
        dst.set_band_description(1, "Gamma0_VV_dB")
        dst.set_band_description(2, "Gamma0_VH_dB")
        dst.update_tags(source=f"Microsoft Planetary Computer {COLLECTION}", date=str(row["date"]),
                        relative_orbit=str(row["orbit"]), items=",".join(i.id for i in row["items"]))
    valid = float(np.isfinite(db).all(axis=0).mean())
    return valid


# ------------------------------------------------------------------ buildings
OSM_USE = {
    "house": "Residential", "residential": "Residential", "apartments": "Residential", "detached": "Residential",
    "semidetached_house": "Residential", "terrace": "Residential", "dormitory": "Residential", "bungalow": "Residential",
    "industrial": "Industrial", "warehouse": "Industrial", "factory": "Industrial", "manufacture": "Industrial",
    "commercial": "Retail", "retail": "Retail", "supermarket": "Retail", "kiosk": "Retail",
    "office": "Office", "school": "Public services", "university": "Public services", "college": "Public services",
    "hospital": "Public services", "public": "Public services", "civic": "Public services",
    "government": "Public services", "church": "Public services", "kindergarten": "Public services",
    "farm": "Agriculture", "barn": "Agriculture", "greenhouse": "Agriculture", "farm_auxiliary": "Agriculture",
    "garage": "Garage / shed", "garages": "Garage / shed", "shed": "Garage / shed", "roof": "Garage / shed",
}


def _overpass(query, timeout):
    """Run one Overpass query, retrying busy servers; returns the JSON elements."""
    errors = []
    for attempt in range(3):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT},
                                  timeout=timeout + 30)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                errors.append(f"{url.split('/')[2]}: HTTP {r.status_code}")
            except (requests.RequestException, ValueError) as e:  # try the next server
                errors.append(f"{url.split('/')[2]}: {type(e).__name__}")
        time.sleep(10 * (attempt + 1))             # servers answer 429/504 when busy
    raise AnalysisError("Could not download OpenStreetMap buildings (" + "; ".join(errors[-3:]) + "). "
                        "The OpenStreetMap servers may be busy: try again in a few minutes, or upload a building file.")


def fetch_buildings(aoi, timeout=180, tile_deg=0.05):
    """OpenStreetMap building footprints inside the area (EPSG:4326), fetched in tiles of ~5 km."""
    w, s, e, n = aoi.to_crs("EPSG:4326").total_bounds
    elements = {}
    for y0 in np.arange(s, n, tile_deg):
        for x0 in np.arange(w, e, tile_deg):
            y1, x1 = min(y0 + tile_deg, n), min(x0 + tile_deg, e)
            bb = f"{y0:.6f},{x0:.6f},{y1:.6f},{x1:.6f}"
            q = (f'[out:json][timeout:{timeout}];(way["building"]({bb});'
                 f'relation["building"]["type"="multipolygon"]({bb}););out tags geom;')
            for el in _overpass(q, timeout):
                elements[(el["type"], el["id"])] = el   # a building on a tile edge comes back twice
    rows = []
    for el in elements.values():
        tags = el.get("tags", {})
        try:
            if el["type"] == "way" and len(el.get("geometry", [])) >= 4:
                geom = Polygon([(p["lon"], p["lat"]) for p in el["geometry"]])
            elif el["type"] == "relation":
                outers = [Polygon([(p["lon"], p["lat"]) for p in m["geometry"]])
                          for m in el.get("members", []) if m.get("role") == "outer" and len(m.get("geometry", [])) >= 4]
                if not outers:
                    continue
                geom = gpd.GeoSeries(outers).union_all()
            else:
                continue
        except (KeyError, ValueError):
            continue
        b = tags.get("building", "yes")
        rows.append(dict(building_id=f"osm_{el['type']}_{el['id']}", building=b,
                         use=OSM_USE.get(b, "Unknown"), dwellings=pd.to_numeric(tags.get("building:flats"),
                                                                                 errors="coerce"),
                         levels=pd.to_numeric(tags.get("building:levels"), errors="coerce"),
                         name=tags.get("name"), geometry=geom))
    if not rows:
        return gpd.GeoDataFrame(columns=["building_id", "use", "geometry"], geometry="geometry", crs="EPSG:4326")
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    g["geometry"] = g.geometry.make_valid()
    inside = g.geometry.representative_point().within(aoi.to_crs("EPSG:4326").union_all())
    return g[inside].reset_index(drop=True)


# ------------------------------------------------------------------ main entry
def fetch(output_dir, flood_date, place=None, bbox=None, aoi_file=None, scenes=None, before_index=None,
          after_index=None, with_buildings=True, progress=None, aoi=None):
    """Download area, image pair and buildings into output_dir; returns the download record (dict)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(progress, 0.02, "Finding the area")
    if aoi is not None:
        aoi = aoi.to_crs("EPSG:4326")
    elif aoi_file:
        aoi = gpd.read_file(aoi_file).to_crs("EPSG:4326")[["geometry"]]
        aoi["name"] = Path(aoi_file).stem
    elif place:
        aoi = geocode(place)
    elif bbox:
        aoi = bbox_area(bbox)
    else:
        raise AnalysisError("Give a place name, a bounding box or an area file.")
    km2 = check_area(aoi)
    aoi.to_file(out / "aoi.geojson", driver="GeoJSON")

    log(progress, 0.08, "Searching Sentinel-1 images")
    if scenes is None:
        scenes = search_scenes(aoi, flood_date)
    if before_index is not None and after_index is not None:
        before, after = scenes.loc[before_index], scenes.loc[after_index]
        if before.orbit != after.orbit:
            raise AnalysisError("The before and after images must come from the same orbit track.")
    else:
        before, after = choose_pair(scenes)

    grid = target_grid(aoi)
    log(progress, 0.15, f"Downloading the before image ({before.date}, orbit {before.orbit})")
    v0 = download_scene(before, grid, out / "before.tif")
    log(progress, 0.35, f"Downloading the after image ({after.date}, orbit {after.orbit})")
    v1 = download_scene(after, grid, out / "after.tif")

    n_buildings = None
    if with_buildings:
        log(progress, 0.5, "Downloading OpenStreetMap buildings")
        b = fetch_buildings(aoi)
        n_buildings = len(b)
        if n_buildings:
            b.to_file(out / "buildings.geojson", driver="GeoJSON")

    record = dict(
        area=dict(name=str(aoi["name"].iloc[0]) if "name" in aoi else "area", km2=round(km2, 2), source="place" if place else
                  ("file" if aoi_file else ("map" if bbox is None else "bbox")), query=place),
        flood_date=str(pd.Timestamp(flood_date).date()),
        sentinel1=dict(source=f"Microsoft Planetary Computer, {COLLECTION} (GRD, radiometric terrain correction, "
                              "gamma0, 10 m)",
                       before=dict(date=str(before.date), orbit=int(before.orbit), direction=before.direction,
                                   platform=before.platform, coverage=float(before.coverage),
                                   items=[i.id for i in before["items"]], valid_fraction=round(v0, 3)),
                       after=dict(date=str(after.date), orbit=int(after.orbit), direction=after.direction,
                                  platform=after.platform, coverage=float(after.coverage),
                                  items=[i.id for i in after["items"]], valid_fraction=round(v1, 3),
                                  days_after_flood=int(after.days_from_flood))),
        grid=dict(crs=grid["crs"].to_string(), width=grid["width"], height=grid["height"], pixel_m=RES_M),
        buildings=dict(source="OpenStreetMap (Overpass API), © OpenStreetMap contributors, ODbL",
                       count=n_buildings) if with_buildings else None,
        downloaded=time.strftime("%Y-%m-%d %H:%M"),
    )
    json.dump(record, open(out / "download.json", "w"), indent=2)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--place")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    g.add_argument("--aoi")
    ap.add_argument("--flood-date", required=True)
    ap.add_argument("--no-buildings", action="store_true")
    ap.add_argument("--analyse", action="store_true", help="run scripts/analyse_event.py afterwards")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/user_runs" / time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()
    try:
        rec = fetch(args.output_dir, args.flood_date, args.place, args.bbox, args.aoi,
                    with_buildings=not args.no_buildings)
        print(json.dumps(rec, indent=2))
        if args.analyse:
            from scripts.analyse_event import analyse
            out = Path(args.output_dir)
            s = analyse(out / "before.tif", out / "after.tif", out, aoi=out / "aoi.geojson",
                        buildings=out / "buildings.geojson" if (out / "buildings.geojson").exists() else None,
                        event_name=rec["area"]["name"].split(",")[0])
            print(json.dumps({k: s[k] for k in ("flood_affected_km2", "flood_affected_pct", "buildings") if k in s},
                             indent=2))
    except AnalysisError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
