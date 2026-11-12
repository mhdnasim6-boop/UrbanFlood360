#!/usr/bin/env python3
"""
UrbanFlood360 - exposure data for the damage assessment, for any place in the world (no accounts needed).

    buildings   Overture Maps buildings: OpenStreetMap + Google Open Buildings + Microsoft ML footprints merged,
                so far more complete than OpenStreetMap alone (Porto Alegre: 139,834 vs 32,962 buildings)
    places      Overture Maps places (schools, hospitals, shops, ...): give buildings a type and a name, and
                flag critical facilities
    land_use    Overture Maps land-use zones (residential, industrial, ...): type for buildings with no tag
    population  GHS-POP 2025, 100 m (European Commission JRC, R2023A), spread over the homes in each cell
    dem         Copernicus DEM GLO-30 (Microsoft Planetary Computer): estimated water depth at each building

Damage: water depth -> damage fraction with the JRC global flood depth-damage curves (Huizinga, de Moel &
Szewczyk 2017, doi:10.2760/16510), per continent and building class; damage level Minor (< 0.5 m),
Moderate (0.5-1.5 m), Severe (>= 1.5 m).

Usage:
    python scripts/exposure.py --aoi area.geojson --output-dir demo/my_event   # downloads everything
"""
import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GHSL_CACHE = ROOT / "data/external/ghsl"
GHSL_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
            "GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
MOLLWEIDE = "ESRI:54009"
FILES = dict(buildings="buildings.parquet", places="places.parquet", land_use="land_use.parquet",
             population="population.tif", dem="dem.tif")

# ------------------------------------------------------------------ building types
CRITICAL_USES = {"Health care", "Education", "Emergency services"}
RESIDENT_WEIGHT = {"Residential": 1.0, "Unknown": 1.0, "Residential (likely)": 1.0, "Commercial": 0.3,
                   "Office": 0.3, "Hotel": 0.3, "Retail": 0.3}

# Overture / OpenStreetMap building class -> (use, detail)
BUILDING_CLASS = {
    **{k: "Residential" for k in ["residential", "house", "apartments", "detached", "semidetached_house", "terrace",
                                  "dormitory", "bungalow", "static_caravan", "hut", "cabin", "houseboat"]},
    **{k: "Commercial" for k in ["commercial", "retail", "supermarket", "kiosk", "service"]},
    "office": "Office", "hotel": "Hotel",
    **{k: "Industrial" for k in ["industrial", "warehouse", "factory", "manufacture", "storage_tank", "digester",
                                 "silo"]},
    **{k: "Education" for k in ["school", "university", "college", "kindergarten"]},
    **{k: "Health care" for k in ["hospital", "clinic"]},
    **{k: "Emergency services" for k in ["fire_station", "police"]},
    **{k: "Government / public" for k in ["civic", "government", "public", "townhall", "transportation",
                                          "train_station", "sports_hall", "stadium", "grandstand", "pavilion"]},
    **{k: "Religious" for k in ["church", "chapel", "mosque", "temple", "synagogue", "cathedral", "religious",
                                "shrine", "monastery"]},
    **{k: "Agriculture" for k in ["farm", "barn", "greenhouse", "farm_auxiliary", "stable", "cowshed", "sty"]},
    **{k: "Garage / shed" for k in ["garage", "garages", "shed", "carport", "roof", "parking", "boathouse"]},
}
# land-use class / subtype -> likely building use
LAND_USE = {"residential": "Residential", "commercial": "Commercial", "retail": "Commercial",
            "industrial": "Industrial", "works": "Industrial", "port": "Industrial", "depot": "Industrial",
            "school": "Education", "university": "Education", "college": "Education", "kindergarten": "Education",
            "education": "Education", "hospital": "Health care", "medical": "Health care",
            "religious": "Religious", "military": "Government / public", "farmyard": "Agriculture",
            "farmland": "Agriculture", "agriculture": "Agriculture", "greenhouse_horticulture": "Agriculture"}
CRITICAL_PLACES = {
    "hospital": ("Health care", "Hospital"), "emergency_department": ("Health care", "Emergency department"),
    "retirement_home": ("Health care", "Care home"), "nursing_home": ("Health care", "Care home"),
    "assisted_living_facility": ("Health care", "Care home"),
    "outpatient_care_facility": ("Health care", "Clinic"), "health_care": ("Health care", "Clinic"),
    "dialysis_clinic": ("Health care", "Dialysis clinic"), "pediatric_clinic": ("Health care", "Children's clinic"),
    "school": ("Education", "School"), "public_school": ("Education", "School"),
    "private_school": ("Education", "School"), "elementary_school": ("Education", "Primary school"),
    "middle_school": ("Education", "Secondary school"), "high_school": ("Education", "Secondary school"),
    "religious_school": ("Education", "School"), "preschool": ("Education", "Preschool"),
    "day_care_preschool": ("Education", "Day care"), "child_care_and_day_care": ("Education", "Day care"),
    "college_university": ("Education", "University / college"),
    "vocational_and_technical_school": ("Education", "Vocational school"),
    "police_station": ("Emergency services", "Police station"), "fire_station": ("Emergency services", "Fire station"),
    "fire_protection_service": ("Emergency services", "Fire station"),
    "homeless_shelter": ("Health care", "Shelter"),
}
PLACE_GROUP = {"shopping": "Commercial", "food_and_drink": "Commercial", "lifestyle_services": "Commercial",
               "services_and_business": "Commercial", "arts_and_entertainment": "Commercial",
               "travel_and_transportation": "Commercial", "sports_and_recreation": "Commercial",
               "lodging": "Hotel", "health_care": "Health care", "education": "Commercial",
               "community_and_government": "Government / public", "cultural_and_historic": "Religious"}
INDUSTRIAL_LEAVES = re.compile(r"manufactur|supplier|distributor|warehouse|storage|factory|wholesale|b2b|industrial")


def humanize(s):
    return str(s).replace("_", " ").replace(" or ", " / ").capitalize()


def place_type(taxonomy):
    """(use, detail, critical) of one Overture place from its taxonomy."""
    h = list(taxonomy["hierarchy"]) if taxonomy is not None and taxonomy.get("hierarchy") is not None else []
    if not h:
        return None, None, False
    leaf = h[-1]
    if leaf in CRITICAL_PLACES:
        use, detail = CRITICAL_PLACES[leaf]
        return use, detail, True
    if h[0] == "cultural_and_historic" and not any("worship" in x or "religious" in x for x in h):
        return "Commercial", humanize(leaf), False
    if INDUSTRIAL_LEAVES.search(leaf) or (len(h) > 1 and INDUSTRIAL_LEAVES.search(h[1])):
        return "Industrial", humanize(leaf), False
    use = PLACE_GROUP.get(h[0], "Commercial")
    if h[0] == "community_and_government" and leaf in ("government_office", "town_hall", "courthouse"):
        use = "Government / public"
    return use, humanize(leaf), False


# ------------------------------------------------------------------ downloads
def _bbox(aoi):
    return tuple(float(v) for v in aoi.to_crs("EPSG:4326").total_bounds)


def fetch_overture(kind, aoi):
    """Overture Maps layer ("building", "place", "land_use") clipped to the area, EPSG:4326."""
    import overturemaps.core as oc
    g = oc.geodataframe(kind, bbox=_bbox(aoi))
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    keep = {"building": ["id", "names", "sources", "subtype", "class", "height", "num_floors", "geometry"],
            "place": ["id", "names", "taxonomy", "basic_category", "confidence", "geometry"],
            "land_use": ["id", "subtype", "class", "geometry"]}[kind]
    g = g[[c for c in keep if c in g.columns]]
    if kind == "building":
        g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
        g = g[g.geometry.representative_point().within(aoi.to_crs("EPSG:4326").union_all())]
        g = g.assign(name=g.names.map(lambda n: n.get("primary") if n is not None else None),
                     source=g.sources.map(lambda s: s[0].get("dataset") if s is not None and len(s) else None))
        g = g.drop(columns=["names", "sources"])
    if kind == "place":
        g = g.assign(name=g.names.map(lambda n: n.get("primary") if n is not None else None)).drop(columns="names")
        t = g.taxonomy.map(place_type)
        g = g.assign(place_use=t.map(lambda x: x[0]), place_detail=t.map(lambda x: x[1]),
                     critical=t.map(lambda x: x[2])).drop(columns=["taxonomy"])
    return g.reset_index(drop=True)


def ghsl_tiles(aoi):
    """GHS-POP tile ids (row, col) covering the area: 1000 km tiles in World Mollweide."""
    x0, y0, x1, y1 = aoi.to_crs(MOLLWEIDE).total_bounds
    rows = range(int((9_000_000 - y1) // 1_000_000) + 1, int((9_000_000 - y0) // 1_000_000) + 2)
    cols = range(int((x0 + 18_041_000) // 1_000_000) + 1, int((x1 + 18_041_000) // 1_000_000) + 2)
    return [(r, c) for r in rows for c in cols]


def fetch_population(aoi, dest, pad_m=500):
    """GHS-POP 2025 (people per 100 m cell) clipped to the area, saved as dest (Mollweide)."""
    GHSL_CACHE.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = aoi.to_crs(MOLLWEIDE).total_bounds
    x0, y0, x1, y1 = x0 - pad_m, y0 - pad_m, x1 + pad_m, y1 + pad_m
    out = None
    for r, c in ghsl_tiles(aoi):
        tif = GHSL_CACHE / f"GHS_POP_E2025_R{r}_C{c}.tif"
        if not tif.exists():
            resp = requests.get(GHSL_URL.format(r=r, c=c), timeout=300)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                name = next(n for n in z.namelist() if n.endswith(".tif"))
                tif.write_bytes(z.read(name))
        with rasterio.open(tif) as src:
            win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            if out is None:
                transform = src.window_transform(win)
                out = np.zeros((int(win.height), int(win.width)), "float32")
                profile = dict(driver="GTiff", crs=src.crs, transform=transform, width=out.shape[1],
                               height=out.shape[0], count=1, dtype="float32", nodata=-1, compress="deflate")
            a = src.read(1, window=win, boundless=True, fill_value=0).astype("float32")
            a[a < 0] = 0
            out = np.maximum(out, a[:out.shape[0], :out.shape[1]])   # tiles do not overlap: max = mosaic
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "GHS_POP_E2025 people per 100 m cell")
    return dest


def fetch_dem(aoi, dest, pad_deg=0.01):
    """Copernicus DEM GLO-30 (m above sea level) for the area, EPSG:4326, from the Planetary Computer."""
    import planetary_computer
    from rasterio.merge import merge
    from scripts.fetch_event_data import _catalog
    w, s, e, n = _bbox(aoi)
    bounds = (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)
    items = list(_catalog().search(collections=["cop-dem-glo-30"], bbox=bounds).items())
    if not items:
        raise RuntimeError("no Copernicus DEM tiles for this area")
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="4", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        srcs = [rasterio.open(planetary_computer.sign(it.assets["data"].href)) for it in items]
        arr, transform = merge(srcs, bounds=bounds)
        crs = srcs[0].crs
        for s_ in srcs:
            s_.close()
    with rasterio.open(dest, "w", driver="GTiff", crs=crs, transform=transform, width=arr.shape[2],
                       height=arr.shape[1], count=1, dtype="float32", nodata=-9999, compress="deflate") as dst:
        dst.write(arr[0].astype("float32"), 1)
        dst.set_band_description(1, "Copernicus DEM GLO-30 elevation (m)")
    return dest


def prepare_exposure(aoi, folder, with_buildings=True, progress=None):
    """Download every exposure layer for the area into folder; returns notes for the layers that failed."""
    say = progress or (lambda f, t: print(f"[{f:4.0%}] {t}"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    aoi = aoi.to_crs("EPSG:4326")
    steps = ([("building", "buildings", "Downloading buildings (Overture Maps)")] if with_buildings else []) + [
        ("place", "places", "Downloading schools, hospitals and other places (Overture Maps)"),
        ("land_use", "land_use", "Downloading land-use zones (Overture Maps)")]
    notes = []
    for i, (kind, key, text) in enumerate(steps):
        say(i / (len(steps) + 2), text)
        try:
            fetch_overture(kind, aoi).to_parquet(folder / FILES[key])
        except Exception as e:  # keep going: each layer is optional
            notes.append(f"Could not download {key} from Overture Maps ({type(e).__name__}).")
    say(len(steps) / (len(steps) + 2), "Downloading population (GHS-POP 2025)")
    try:
        fetch_population(aoi, folder / FILES["population"])
    except Exception as e:
        notes.append(f"Could not download the population map ({type(e).__name__}); people are not estimated.")
    say((len(steps) + 1) / (len(steps) + 2), "Downloading terrain heights (Copernicus DEM)")
    try:
        fetch_dem(aoi, folder / FILES["dem"])
    except Exception as e:
        notes.append(f"Could not download terrain heights ({type(e).__name__}); water depth is not estimated.")
    return notes


# ------------------------------------------------------------------ building type
def classify_buildings(b, places=None, land_use=None, near_m=15.0):
    """Fill use / use_detail / name / critical / use_source for buildings (GeoDataFrame in a metric CRS)."""
    b = b.copy()
    for col, default in [("use", "Unknown"), ("use_detail", None), ("name", None)]:
        if col not in b:
            b[col] = default
    b["use"] = b["use"].fillna("Unknown").replace({"": "Unknown", "yes": "Unknown"})
    if "class" in b:     # Overture / OSM building class
        cls = b["class"].map(BUILDING_CLASS)
        fill = (b.use == "Unknown") & cls.notna()
        b.loc[fill, "use"] = cls[fill]
        b.loc[fill, "use_detail"] = b.loc[fill, "class"].map(humanize)
    b["use_source"] = np.where(b.use != "Unknown", "building tag", None)
    b["critical"] = b.use.isin(CRITICAL_USES)

    if places is not None and len(places):
        p = places[places.confidence.fillna(1) >= 0.5].to_crs(b.crs)
        p = p[p.place_use.notna()][["name", "place_use", "place_detail", "critical", "confidence", "geometry"]]
        p = p.rename(columns={"name": "place_name", "critical": "place_critical"})
        poly = b[["geometry"]].reset_index().rename(columns={"index": "bidx"})
        hit = gpd.sjoin(p, poly, predicate="within", how="inner")
        rest = p.loc[~p.index.isin(hit.index)]
        if len(rest):
            near = gpd.sjoin_nearest(rest, poly, how="inner", max_distance=near_m)
            hit = pd.concat([hit, near[~near.index.duplicated()]])
        # per building: a critical place wins, then the most confident one
        hit = hit.sort_values(["place_critical", "confidence"], ascending=False).drop_duplicates("bidx")
        hit = hit.set_index("bidx")
        cur = b.loc[hit.index, "use"]
        crit = hit.place_critical.astype(bool)
        take = crit | (cur == "Unknown")                       # critical places always win; else fill unknowns
        homes = ~take & (cur == "Residential") & (hit.place_use == "Commercial")   # shop on the ground floor
        t = hit.index[take]
        b.loc[t, "use"] = hit.loc[t, "place_use"]
        b.loc[t, "use_detail"] = hit.loc[t, "place_detail"]
        b.loc[t, "name"] = b.loc[t, "name"].fillna(hit.loc[t, "place_name"])
        b.loc[t, "use_source"] = "place inside"
        b.loc[t, "critical"] = b.loc[t, "critical"] | crit[take]
        hm = hit.index[homes]
        b.loc[hm, "use_detail"] = "Homes with " + hit.loc[hm, "place_detail"].str.lower()
        b.loc[hm, "name"] = b.loc[hm, "name"].fillna(hit.loc[hm, "place_name"])

    if land_use is not None and len(land_use):
        lu = land_use.to_crs(b.crs)
        lu = lu.assign(lu_use=lu["class"].map(LAND_USE).fillna(lu["subtype"].map(LAND_USE)))
        lu = lu[lu.lu_use.notna() & lu.geom_type.isin(["Polygon", "MultiPolygon"])]
        todo = b.index[b.use == "Unknown"]
        if len(todo) and len(lu):
            pts = gpd.GeoDataFrame(geometry=b.loc[todo].geometry.representative_point(), crs=b.crs)
            j = gpd.sjoin(pts, lu[["lu_use", "geometry"]], predicate="within", how="inner")
            j = j[~j.index.duplicated()]
            b.loc[j.index, "use"] = j.lu_use + " (likely)"
            b.loc[j.index, "use_detail"] = j.lu_use + " zone"
            b.loc[j.index, "use_source"] = "land-use zone"
            b.loc[j.index, "critical"] = j.lu_use.isin(CRITICAL_USES).values

    area = b.geometry.area
    todo = b.use == "Unknown"
    size_use = pd.Series(np.select([area < 25, area < 400, area < 1500], ["Garage / shed", "Residential (likely)",
                                    "Unknown"], "Large building (likely commercial / industrial)"), index=b.index)
    size_detail = pd.Series(np.select([area < 25, area < 400, area < 1500],
                                      ["small structure", "house-sized building", "medium building"],
                                      "large building"), index=b.index)
    b.loc[todo, "use"] = size_use[todo]
    b.loc[todo, "use_detail"] = size_detail[todo]
    b.loc[todo & (size_use != "Unknown"), "use_source"] = "building size"
    b["use_detail"] = b["use_detail"].fillna(b["use"])
    b["critical"] = b["critical"].astype(bool)
    return b


# ------------------------------------------------------------------ people
def residents_from_population(b, pop_tif):
    """Estimated residents per building: each 100 m cell's people spread over its buildings, weighted by
    floor area (footprint x floors) and by how likely the building is a home (RESIDENT_WEIGHT).
    Returns (residents Series, share of the cells' people that landed on a building)."""
    with rasterio.open(pop_tif) as src:
        pop, T, crs = src.read(1), src.transform, src.crs
    pts = b.geometry.representative_point().to_crs(crs)
    rows, cols = rasterio.transform.rowcol(T, pts.x.values, pts.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < pop.shape[0]) & (cols >= 0) & (cols < pop.shape[1])
    floors = pd.to_numeric(b.get("num_floors"), errors="coerce") if "num_floors" in b else None
    if floors is None or floors.isna().all():
        floors = pd.Series(np.nan, index=b.index)
    h = pd.to_numeric(b["height"], errors="coerce") if "height" in b else pd.Series(np.nan, index=b.index)
    floors = floors.fillna((h / 3).round()).fillna(1).clip(1, 60)
    w_use = b.use.map(lambda u: RESIDENT_WEIGHT.get(str(u).replace(" (likely)", ""), 0.0)).values
    w = b.geometry.area.values * floors.values * w_use
    cell = np.where(ok, rows * pop.shape[1] + cols, -1)
    df = pd.DataFrame(dict(cell=cell, w=w, w_any=b.geometry.area.values), index=b.index)
    df = df[df.cell >= 0]
    tot = df.groupby("cell").w.transform("sum")
    tot_any = df.groupby("cell").w_any.transform("sum")
    share = np.where(tot > 0, df.w / tot.replace(0, np.nan), df.w_any / tot_any)   # no homes: spread over all
    cell_pop = pop.ravel()[df.cell.values]
    res = pd.Series(0.0, index=b.index)
    res.loc[df.index] = np.nan_to_num(share * cell_pop)
    used = np.unique(df.cell.values)
    covered = float(pop.ravel()[used].sum())
    # share of the area's people that live in cells with at least one building
    aoi_pop = float(pop[pop > 0].sum())
    return res.round(1), covered / aoi_pop if aoi_pop else np.nan


# ------------------------------------------------------------------ water depth
def water_depth(mask, grid, dem_tif, permanent_water=None, factor=3, ground_px=5, min_dry_px=30, block_px=33,
                edge_quantile=0.8):
    """Estimated water depth (m) on the analysis grid, NaN outside the flood.

    Flood-edge method (like FwDET, Cohen et al. 2018): the water surface is the ground height along the flood
    edge (edges along rivers / lakes that were already water before the flood are skipped), carried into the
    flood from the nearest edge and smoothed with a median filter (~330 m); depth = water surface - ground.
    Done at 30 m (the DEM resolution). Copernicus DEM is a surface model (roofs, trees), so the ground is
    approximated with a morphological opening over ~150 m, which removes objects narrower than that.
    Depths are rough estimates, best read as shallow / deep."""
    H, W = mask.shape
    h, w = int(np.ceil(H / factor)), int(np.ceil(W / factor))
    t = grid["transform"]
    t30 = rasterio.Affine(t.a * factor, t.b, t.c, t.d, t.e * factor, t.f)
    pad = np.full((h * factor, w * factor), 255, "uint8")
    pad[:H, :W] = mask
    blocks = pad.reshape(h, factor, w, factor)
    valid = (blocks != 255).sum(axis=(1, 3))
    wet = (blocks == 1).sum(axis=(1, 3))
    flood = (valid > 0) & (wet * 2 >= valid)
    known = valid > 0
    dem = np.full((h, w), np.nan, "float32")
    with rasterio.open(dem_tif) as src:
        reproject(rasterio.band(src, 1), dem, src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=t30, dst_crs=grid["crs"], dst_nodata=np.nan, resampling=Resampling.bilinear)
    ok = np.isfinite(dem)
    if not ok.any():
        return None
    filled = np.where(ok, dem, np.nanmedian(dem))
    ground = ndimage.grey_opening(filled, size=(ground_px, ground_px))
    dem = np.where(ok, ground, np.nan).astype("float32")
    # clean radar speckle: small dry specks inside the flood are not real flood edges
    flood_c = ndimage.binary_closing(flood, structure=np.ones((3, 3), bool)) & known
    lab, n = ndimage.label(known & ~flood_c)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        flood_c |= np.isin(lab, np.flatnonzero(sizes < min_dry_px) + 1)
    dry = known & ~flood_c
    if permanent_water is not None:
        pw = np.zeros((h * factor, w * factor), bool)
        pw[:H, :W] = permanent_water
        pw = pw.reshape(h, factor, w, factor).mean(axis=(1, 3)) >= 0.5
        dry &= ~ndimage.binary_dilation(pw, iterations=1)
    edge = flood_c & ndimage.binary_dilation(dry, structure=np.ones((3, 3), bool)) & np.isfinite(dem)
    if edge.sum() < 10:
        return None
    # water surface per ~1 km block = upper edge height there (the flood reaches up to it); gaps: nearest block
    B = block_px
    hb, wb = int(np.ceil(h / B)), int(np.ceil(w / B))
    blk = np.full((hb, wb), np.nan, "float32")
    er, ec = np.nonzero(edge)
    df = pd.DataFrame(dict(k=(er // B) * wb + ec // B, z=dem[er, ec]))
    q = df.groupby("k").z.quantile(edge_quantile)
    blk.ravel()[q.index.values] = q.values
    miss = ~np.isfinite(blk)
    if miss.all():
        return None
    _, (ri, ci) = ndimage.distance_transform_edt(miss, return_indices=True)
    blk = ndimage.median_filter(blk[ri, ci], size=3)
    wse = ndimage.zoom(blk, B, order=1)[:h, :w]
    if wse.shape != (h, w):
        wse = np.pad(wse, ((0, h - wse.shape[0]), (0, w - wse.shape[1])), mode="edge")
    depth = np.where(flood & np.isfinite(dem), np.clip(wse - dem, 0, 10), np.nan).astype("float32")
    depth = np.repeat(np.repeat(depth, factor, axis=0), factor, axis=1)[:H, :W]
    depth[mask != 1] = np.nan
    return depth


def depth_at_buildings(geoms, depth, transform):
    """Deepest estimated water within ~10 m of each building (3 x 3 pixels around its centre); NaN if dry."""
    d = ndimage.maximum_filter(np.nan_to_num(depth, nan=-1.0), size=3)
    pts = geoms.representative_point()
    rows, cols = rasterio.transform.rowcol(transform, pts.x.values, pts.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < d.shape[0]) & (cols >= 0) & (cols < d.shape[1])
    out = np.full(len(pts), np.nan)
    out[ok] = d[rows[ok], cols[ok]]
    out[out < 0] = np.nan
    return out


def zonal_median(geoms, values, transform):
    """Median of the finite values under each geometry (all touched pixels); NaN if none."""
    H, W = values.shape
    inv = ~transform
    out = np.full(len(geoms), np.nan)
    for i, g in enumerate(geoms):
        minx, miny, maxx, maxy = g.bounds
        c0, r0 = inv @ (minx, maxy)
        c1, r1 = inv @ (maxx, miny)
        c0, r0 = max(int(np.floor(c0)), 0), max(int(np.floor(r0)), 0)
        c1, r1 = min(int(np.ceil(c1)), W), min(int(np.ceil(r1)), H)
        if c1 <= c0 or r1 <= r0:
            continue
        win = Window(c0, r0, c1 - c0, r1 - r0)
        m = rasterize([(g, 1)], out_shape=(r1 - r0, c1 - c0), transform=window_transform(win, transform),
                      all_touched=True, dtype="uint8") == 1
        v = values[r0:r1, c0:c1][m]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.median(v)
    return out


# ------------------------------------------------------------------ damage
DEPTHS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]
# JRC global depth-damage functions (Huizinga et al. 2017): damage fraction of the building at each depth.
# Values as published by physrisk (scaled to a maximum of 0.6) divided by 0.6 to give the JRC fractions.
_RAW = {
    "residential": {"South America": [0, .295, .427, .505, .57, .59, .6, .6, .6],
                    "Europe": [0, .15, .24, .3, .36, .45, .51, .57, .6],
                    "Africa": [0, .132, .227, .318, .381, .49, .542, .574, .6],
                    "Asia": [0, .196, .296, .37, .432, .522, .559, .59, .6],
                    "North America": [0, .266, .35, .41, .47, .513, .554, .575, .6],
                    "Oceania": [0, .285, .384, .429, .473, .557, .58, .59, .6]},
    "commercial": {"South America": [0, .367, .504, .554, .595, .6, .6, .6, .6],
                   "Europe": [0, .09, .18, .27, .33, .45, .54, .6, .6],
                   "Asia": [0, .226, .323, .396, .458, .53, .565, .588, .6],
                   "North America": [0, .144, .225, .28, .331, .412, .493, .545, .6],
                   "Oceania": [0, .143, .289, .404, .519, .6, .6, .6, .6],
                   "Global": [0, .194, .304, .381, .447, .518, .56, .587, .6]},
    "industrial": {"South America": [0, .4, .533, .568, .6, .6, .6, .6, .6],
                   "Europe": [0, .09, .162, .24, .312, .42, .51, .6, .6],
                   "Africa": [0, .038, .148, .242, .297, .411, .551, .6, .6],
                   "Asia": [0, .17, .289, .378, .43, .514, .545, .573, .6],
                   "North America": [0, .194, .307, .382, .444, .516, .562, .588, .6],
                   "Global": [0, .178, .288, .362, .417, .492, .554, .592, .6]},
}
_RAW["residential"]["Global"] = list(np.mean(list(_RAW["residential"].values()), axis=0))   # mean of continents
CURVES = {k: {c: np.array(v) / 0.6 for c, v in d.items()} for k, d in _RAW.items()}
CURVE_CLASS = {"Residential": "residential", "Residential (likely)": "residential", "Unknown": "residential",
               "Hotel": "commercial", "Commercial": "commercial", "Office": "commercial", "Retail": "commercial", "Public services": "commercial",
               "Health care": "commercial", "Education": "commercial", "Emergency services": "commercial",
               "Government / public": "commercial", "Religious": "commercial",
               "Large building (likely commercial / industrial)": "industrial", "Industrial": "industrial",
               "Agriculture": "industrial", "Garage / shed": "industrial"}
DAMAGE_LEVELS = ["Severe", "Moderate", "Minor"]


def continent(lon, lat):
    """Rough continent of a point, to pick the JRC curve."""
    if lon < -30:
        return "South America" if lat < 13 else "North America"
    if lon > 110 and lat < -10 or lon > 150:
        return "Oceania"
    if lat > 35 and lon < 45 or (lat > 50 and lon < 60):
        return "Europe"
    if lat < 37 and lon < 52 and not (lat > 12 and lon > 34):
        return "Africa"
    return "Asia"


def damage_fraction(depth, use, where):
    cls = use.map(lambda u: CURVE_CLASS.get(u.replace(" (likely)", "") if isinstance(u, str) else u,
                                            "residential")).fillna("residential")
    out = np.full(len(depth), np.nan)
    for k in CURVES:
        curve = CURVES[k].get(where, CURVES[k]["Global"])
        sel = (cls == k).values & np.isfinite(depth)
        out[sel] = np.interp(depth[sel], DEPTHS, curve)
    return np.round(out, 2)


def damage_level(depth, impact):
    lvl = np.select([depth >= 1.5, depth >= 0.5, depth >= 0], ["Severe", "Moderate", "Minor"], "")
    lvl = np.where(np.isfinite(depth) & (impact == "Affected"), lvl, None)
    return lvl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-buildings", action="store_true", help="keep your own building file")
    args = ap.parse_args()
    notes = prepare_exposure(gpd.read_file(args.aoi), args.output_dir, with_buildings=not args.no_buildings)
    print("\n".join(notes) or "All layers downloaded.")


if __name__ == "__main__":
    main()
