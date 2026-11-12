"""
UrbanFlood360 - simple dashboard: upload two Sentinel-1 images, get the flood map and the damage assessment.

    streamlit run dashboard/app.py

1. Upload a Sentinel-1 image from before and one from after the flood (GeoTIFF with VV and VH bands).
2. Press "Map the flood".
3. See the flood map, the buildings and people affected, critical facilities, estimated water depth and damage
   level of each building, and download everything.

Exposure data are downloaded for the area of the images (scripts/exposure.py): buildings, places and land use
from Overture Maps, population from GHS-POP, terrain from Copernicus DEM. A building file you add replaces the
Overture buildings. Every run is saved in outputs/user_runs/<time>_<name>/.
"""
import json
import sys
import time
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds
from shapely.geometry import box
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.results_view import (EXAMPLE, RUNS, buildings_png, depth_png, overlay_png, radar_png,  # noqa: E402
                                    save_upload)
from scripts.analyse_event import AnalysisError, analyse  # noqa: E402
from scripts.exposure import FILES, prepare_exposure  # noqa: E402
from scripts.fetch_event_data import MAX_AREA_KM2, area_km2, fetch_buildings  # noqa: E402

FLOOD_RGB = (42, 120, 214)
COLORS = {"Severe": "#7f1d1d", "Moderate": "#d03b3b", "Minor": "#f08a4b", "Affected": "#d03b3b",
          "Possibly affected": "#f2c14e"}
MAX_SHAPES = 5000          # above this, buildings are drawn as an image (fast) instead of hoverable shapes
ss = st.session_state

st.title("UrbanFlood360")
st.markdown("Upload a Sentinel-1 radar image from **before** the flood and one from **after** it. "
            "You get the **flood map**, the **buildings and people affected**, and an estimate of the **damage**.")

# ------------------------------------------------------------------ upload
c1, c2 = st.columns(2)
up_before = c1.file_uploader("Before the flood (GeoTIFF)", type=["tif", "tiff"])
up_after = c2.file_uploader("After the flood (GeoTIFF)", type=["tif", "tiff"])
with st.expander("Optional: your own building footprints or area boundary"):
    st.caption("Without a building file, buildings are downloaded from Overture Maps (OpenStreetMap + Google + "
               "Microsoft building footprints) for the area of the images.")
    o1, o2 = st.columns(2)
    up_buildings = o1.file_uploader("Building footprints", type=["geojson", "json", "gpkg", "zip", "gml", "parquet"])
    up_area = o2.file_uploader("Area boundary", type=["geojson", "json", "gpkg", "zip"])


def demo(folder, slug, buildings=None):
    d = ROOT / "demo" / folder
    return dict(before=d / "before.tif", after=d / "after.tif", buildings=buildings or d / FILES["buildings"],
                area=d / "aoi.geojson", exposure=d, slug=slug)


EXAMPLES = {
    "Porto Alegre, Brazil (May 2024)": demo("porto_alegre_2024", "porto_alegre_example"),
    "Conselice, Italy (May 2023)": demo("conselice_2023", "conselice_example"),
    "Valencia, Spain (Oct 2024)": dict(before=EXAMPLE["pre"], after=EXAMPLE["post"], buildings=EXAMPLE["buildings"],
                                       area=EXAMPLE["aoi"], exposure=ROOT / "demo/valencia_2024",
                                       slug="valencia_example"),
}
ICONS = [":material/location_city:", ":material/agriculture:", ":material/science:"]
run = st.button("Map the flood", type="primary", icon=":material/water:")
st.caption("Or try an example:")
example = None
shown = [(k, icon) for k, icon in zip(EXAMPLES, ICONS) if Path(EXAMPLES[k]["after"]).exists()]
for col, (k, icon) in zip(st.columns(len(shown)), shown):
    if col.button(k, icon=icon, width="stretch"):
        example = k


def image_area(path):
    with rasterio.open(path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return gpd.GeoDataFrame({"name": ["image area"]}, geometry=[box(w, s, e, n)], crs="EPSG:4326")


if run or example:
    if example:
        ex = EXAMPLES[example]
        before, after, area, exposure = ex["before"], ex["after"], ex["area"], ex["exposure"]
        buildings = ex["buildings"] if Path(ex["buildings"]).exists() else None
        name = example
    elif not (up_before and up_after):
        st.error("Please upload both images: one from before and one from after the flood.")
        st.stop()
    else:
        name = Path(up_after.name).stem[:40]
    run_dir = RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{EXAMPLES[example]['slug'] if example else 'upload'}"
    inputs = run_dir / "inputs"
    bar = st.progress(0.0, "Starting")
    try:
        if not example:
            before, after = save_upload(up_before, inputs), save_upload(up_after, inputs)
            area = save_upload(up_area, inputs) if up_area else None
            buildings = save_upload(up_buildings, inputs) if up_buildings else None
            zone = gpd.read_file(area).to_crs("EPSG:4326") if area else image_area(before)
            km2 = area_km2(zone)
            if km2 > MAX_AREA_KM2:
                raise AnalysisError(
                    f"The images cover {km2:,.0f} km², too large to download all its buildings and data "
                    f"(limit {MAX_AREA_KM2} km²). Crop the images to the town, or add an area boundary.")
            exposure = inputs / "exposure"
            notes = prepare_exposure(zone, exposure, with_buildings=buildings is None,
                                     progress=lambda f, t: bar.progress(0.2 * f, t))
            if buildings is None and (exposure / FILES["buildings"]).exists():
                buildings = exposure / FILES["buildings"]
            elif buildings is None:                      # Overture failed: fall back to OpenStreetMap
                bar.progress(0.2, "Downloading buildings from OpenStreetMap")
                b = fetch_buildings(zone)
                if len(b):
                    buildings = inputs / "osm_buildings.geojson"
                    b.to_file(buildings, driver="GeoJSON")
            for n in notes:
                st.warning(n)
        analyse(before, after, run_dir, aoi=area, buildings=buildings, event_name=name, exposure_dir=exposure,
                progress=lambda f, t: bar.progress(0.2 + 0.8 * f, t))
        ss["run_dir"] = str(run_dir)
        bar.empty()
    except AnalysisError as e:
        bar.empty()
        st.error(str(e))
        st.stop()
    except Exception as e:
        bar.empty()
        st.exception(e)
        st.stop()

# ------------------------------------------------------------------ results
if "run_dir" not in ss:
    st.stop()
run_dir = Path(ss["run_dir"])
summary = json.load(open(run_dir / "summary.json"))
bs = summary.get("buildings") or {}
bpath = run_dir / "buildings_impact.geojson"
b = gpd.read_file(bpath) if bpath.exists() else None
has_damage = b is not None and "damage_level" in b
people = bs.get("residents_affected_est")

st.divider()
st.subheader(f"Results · {summary['event_name']}")
for note in summary["notes"]:
    st.info(note)

k = st.columns(4)
k[0].metric("Flooded area", f"{summary['flood_affected_km2']:.1f} km²",
            f"{summary['flood_affected_pct']:.0f}% of the area", delta_color="off", delta_arrow="off", border=True)
if b is not None:
    n_aff, n_pos = int((b.impact == "Affected").sum()), int((b.impact == "Possibly affected").sum())
    k[1].metric("Buildings affected", f"{n_aff:,}", f"+ {n_pos:,} possibly · of {len(b):,}", delta_color="off",
                delta_arrow="off", border=True)
    k[2].metric("People affected (estimate)", f"{people:,}" if people is not None else "–",
                f"+ {bs.get('residents_possibly_affected_est') or 0:,} possibly" if people is not None
                else "no population data", delta_color="off", delta_arrow="off", border=True)
    k[3].metric("Critical facilities affected", f"{bs.get('critical_affected', 0):,}",
                f"of {bs.get('critical_total', 0):,} (schools, hospitals, care homes, police, fire)",
                delta_color="off", delta_arrow="off", border=True)
    if has_damage:
        lv = bs.get("damage_levels") or {}
        d = st.columns(4)
        d[0].metric("Severe damage", f"{lv.get('Severe', 0):,}", "water ≥ 1.5 m", delta_color="off",
                    delta_arrow="off", border=True)
        d[1].metric("Moderate damage", f"{lv.get('Moderate', 0):,}", "water 0.5–1.5 m", delta_color="off",
                    delta_arrow="off", border=True)
        d[2].metric("Minor damage", f"{lv.get('Minor', 0):,}", "water < 0.5 m", delta_color="off",
                    delta_arrow="off", border=True)
        p90 = bs.get("depth_p90_affected_m")
        if p90 is None and "depth_m" in b:
            p90 = float(b.loc[b.impact == "Affected", "depth_m"].quantile(0.9))
        d[3].metric("Deep water at buildings (est.)", f"{p90:.1f} m+" if p90 is not None else "–",
                    "1 in 10 affected buildings had this much or more", delta_color="off", delta_arrow="off",
                    border=True)
else:
    k[1].info("No buildings found for this area, so only the flood map is shown.")

# ------------------------------------------------------------------ map
png, bounds = overlay_png(run_dir / "flood_mask.tif", FLOOD_RGB, 160)
m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Tiles &copy; Esri", name="Satellite", max_zoom=19).add_to(m)
folium.TileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", name="Light map",
                 attr="&copy; OpenStreetMap contributors &copy; CARTO", max_zoom=19, show=False).add_to(m)
feats = run_dir / "sar_features_6ch.tif"
if feats.exists():  # bands: 1 VV before, 3 VV after (dB); dark = smooth surface such as standing water
    for band, label in [(1, "Radar image before the flood"), (3, "Radar image after the flood")]:
        rpng, rbounds = radar_png(feats, band, f"radar_vv_band{band}")
        ImageOverlay(str(rpng), bounds=rbounds, name=label, opacity=0.9, show=False).add_to(m)
ImageOverlay(str(png), bounds=bounds, name="Flooded area", opacity=0.75).add_to(m)
if (run_dir / "water_depth.tif").exists():
    dpng, dbounds = depth_png(run_dir / "water_depth.tif", run_dir / "water_depth.png")
    ImageOverlay(str(dpng), bounds=dbounds, name="Estimated water depth", opacity=0.85, show=False).add_to(m)


def status(row):
    return row["damage_level"] if has_damage and isinstance(row["damage_level"], str) and row["damage_level"] \
        else row["impact"]


if b is not None:
    hit = b[b.impact.isin(["Affected", "Possibly affected"])].copy()
    if len(hit):
        hit["status"] = hit.apply(status, axis=1)
        hit["color"] = hit.status.map(COLORS).fillna(COLORS["Affected"])
        hit["flooded_pct"] = (100 * hit.flood_fraction).round(0)
        if len(hit) <= MAX_SHAPES:
            fields = {"name": "", "use_detail": "Type", "status": "Status", "flooded_pct": "Flooded around it (%)"}
            if has_damage:
                fields["depth_m"] = "Water depth (m)"
            if people is not None:
                fields["residents_est"] = "People (est.)"
            folium.GeoJson(
                hit[["color", "geometry"] + list(fields)].fillna({"name": ""}), name="Affected buildings",
                style_function=lambda f: {"color": f["properties"]["color"], "fillColor": f["properties"]["color"],
                                          "weight": 0.6, "fillOpacity": 0.85},
                tooltip=folium.GeoJsonTooltip(fields=list(fields), aliases=list(fields.values())),
            ).add_to(m)
        else:
            bpng = run_dir / "buildings_affected.png"
            bb = buildings_png(hit, list(hit.color), bpng)
            ImageOverlay(str(bpng), bounds=bb, name="Affected buildings", opacity=1).add_to(m)
    crit = b[b.critical.astype(bool) & b.impact.isin(["Affected", "Possibly affected"])]
    if len(crit):
        fg = folium.FeatureGroup(name="Critical facilities")
        for _, r in crit.to_crs("EPSG:4326").iterrows():
            p = r.geometry.representative_point()
            tip = f"<b>{r['name'] or r['use_detail']}</b><br>{r['use_detail']}<br>{status(r)}" + (
                f"<br>water ≈ {r['depth_m']:.1f} m" if has_damage and pd.notna(r.get("depth_m")) else "")
            folium.CircleMarker([p.y, p.x], radius=4, color="white", weight=1, fill=True, fill_color="#6d28d9",
                                fill_opacity=1, tooltip=tip).add_to(fg)
        fg.add_to(m)
m.fit_bounds(bounds)
folium.LayerControl(collapsed=False).add_to(m)

legend = [f'<span style="color:rgb{FLOOD_RGB}">■</span> Flooded area']
if has_damage:
    legend += [f'<span style="color:{COLORS[k]}">■</span> {k} damage' for k in ["Severe", "Moderate", "Minor"]]
else:
    legend += [f'<span style="color:{COLORS["Affected"]}">■</span> Affected building']
legend += [f'<span style="color:{COLORS["Possibly affected"]}">■</span> Possibly affected',
           '<span style="color:#6d28d9">●</span> Critical facility']
st.markdown(" &nbsp; ".join(legend), unsafe_allow_html=True)
st.caption("Tip: the satellite background is from a normal day, not the flood. Switch on **Radar image after the "
           "flood** to see the water, or **Estimated water depth** (light = shallow, dark blue = 3 m or more). "
           "Hover over a purple dot to see the facility.")
st_folium(m, height=580, use_container_width=True, returned_objects=[], key=f"map_{run_dir.name}")

# ------------------------------------------------------------------ tables
if b is not None:
    t = b[b.impact.isin(["Affected", "Possibly affected"])].copy()
    pts = t.geometry.representative_point()
    t["latitude"], t["longitude"] = pts.y.round(6), pts.x.round(6)
    t["flooded_pct"] = (100 * t.flood_fraction).round(0)
    t["status"] = t.apply(status, axis=1) if len(t) else []
    order = {"Severe": 0, "Moderate": 1, "Minor": 2, "Affected": 3, "Possibly affected": 4}
    t["_o"] = t.status.map(order)
    sort = ["_o", "depth_m"] if has_damage else ["_o", "flooded_pct"]
    t = t.sort_values(sort, ascending=[True, False])
    cfg = {"building_id": "Building ID", "name": "Name", "use": "Type", "use_detail": "Details",
           "status": "Damage / status", "critical": st.column_config.CheckboxColumn("Critical"),
           "depth_m": st.column_config.NumberColumn("Water depth (m)", format="%.1f"),
           "damage_fraction": st.column_config.ProgressColumn("Damage (share of building value)", min_value=0,
                                                               max_value=1, format="percent"),
           "residents_est": st.column_config.NumberColumn("People (est.)", format="%.0f"),
           "flooded_pct": st.column_config.ProgressColumn("Flooded around it", min_value=0, max_value=100,
                                                          format="%d%%"),
           "latitude": st.column_config.NumberColumn("Latitude", format="%.5f"),
           "longitude": st.column_config.NumberColumn("Longitude", format="%.5f")}

    crit = t[t.critical.astype(bool)]
    if len(crit):
        st.subheader(f"Critical facilities in the flood ({len(crit):,} affected or possibly affected)")
        cc = ["name", "use_detail", "status"] + (["depth_m"] if has_damage else []) + ["latitude", "longitude"]
        st.dataframe(crit[cc], hide_index=True, width="stretch", height=min(38 * len(crit) + 40, 320),
                     column_config=cfg)

    by_use = pd.DataFrame(bs.get("by_use") or [])
    if len(by_use):
        st.subheader("Affected buildings by type")
        by_use = by_use[by_use.affected + by_use.possibly > 0]
        cols = ["use", "affected", "possibly"] + (["severe"] if "severe" in by_use else []) + \
               (["people"] if people is not None else [])
        st.dataframe(by_use[cols], hide_index=True, width="stretch", column_config={
            "use": "Type", "affected": st.column_config.NumberColumn("Affected", format="%d"),
            "possibly": st.column_config.NumberColumn("Possibly affected", format="%d"),
            "severe": st.column_config.NumberColumn("Severe damage", format="%d"),
            "people": st.column_config.NumberColumn("People (est.)", format="%d")})

    st.subheader(f"All affected buildings ({len(t):,})")
    cols = ["building_id", "name", "use", "use_detail", "status"] + \
           (["depth_m", "damage_fraction"] if has_damage else []) + \
           (["residents_est"] if people is not None else []) + ["flooded_pct", "critical", "latitude", "longitude"]
    if t["dwellings"].notna().any():
        cols.insert(5, "dwellings")
    t = t[cols]
    st.dataframe(t, hide_index=True, width="stretch", height=420, column_config=cfg)
    src = bs.get("building_sources")
    st.caption(
        "Affected: at least half of the ground in and around the building (10 m) is flooded; possibly affected: "
        "at least 10%. "
        + ("Damage level from the estimated water depth: Minor < 0.5 m, Moderate 0.5–1.5 m, Severe ≥ 1.5 m; "
           "damage share from the JRC global flood depth-damage curves. " if has_damage else "")
        + (f"People: {bs.get('people_source')}. " if people is not None else "")
        + "Types marked “(likely)” are inferred from the land-use zone or the building size. "
        + ("Buildings: " + ", ".join(f"{k} {v:,}" for k, v in src.items()) + ". " if src else "")
        + "These are estimates of flood exposure and likely damage, not confirmed damage.")

# ------------------------------------------------------------------ downloads
d1, d2, d3 = st.columns(3)
d1.download_button("Flood map (GeoTIFF)", (run_dir / "flood_mask.tif").read_bytes(), "flood_map.tif",
                   "image/tiff", icon=":material/download:", width="stretch")
if b is not None:
    d2.download_button("Affected buildings (CSV)", t.to_csv(index=False), "affected_buildings.csv", "text/csv",
                       icon=":material/download:", width="stretch")
d3.download_button("Everything (ZIP)", (run_dir / "results.zip").read_bytes(), f"{run_dir.name}.zip",
                   "application/zip", icon=":material/download:", width="stretch")

model_name = summary["model"].get("name", "U-Net v2 trained on Valencia 2024")
if "UrbanSARFloods" in model_name:
    st.caption(f"Flood map made by the {model_name}. It was not trained on this place, so check the map against "
               "local reports before acting on it.")
else:
    st.caption(f"Flood map made by the {model_name}. On other places, or images from another satellite track, it "
               "can show too much flooding. Check the map against local reports.")
