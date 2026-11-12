"""
UrbanFlood360 - simple dashboard: upload two Sentinel-1 images, get the flood map and the affected buildings.

    streamlit run dashboard/app.py

1. Upload a Sentinel-1 image from before and one from after the flood (GeoTIFF with VV and VH bands).
2. Press "Map the flood".
3. See the flood map, how many buildings were affected, the list of affected buildings, and download everything.

Buildings come from OpenStreetMap for the area of the images (or from a building file, if you add one).
Every run is saved in outputs/user_runs/<time>_<name>/.
"""
import json
import sys
import time
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds
from shapely.geometry import box
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.results_view import EXAMPLE, RUNS, overlay_png, radar_png, save_upload  # noqa: E402
from scripts.analyse_event import AnalysisError, analyse  # noqa: E402
from scripts.fetch_event_data import MAX_AREA_KM2, area_km2, fetch_buildings  # noqa: E402

RED, ORANGE = "#d03b3b", "#ec835a"
FLOOD_RGB = (42, 120, 214)
ss = st.session_state

st.title("UrbanFlood360")
st.markdown("Upload a Sentinel-1 radar image from **before** the flood and one from **after** it. "
            "You get the **flood map** and the list of **buildings affected**.")

# ------------------------------------------------------------------ upload
c1, c2 = st.columns(2)
up_before = c1.file_uploader("Before the flood (GeoTIFF)", type=["tif", "tiff"])
up_after = c2.file_uploader("After the flood (GeoTIFF)", type=["tif", "tiff"])
with st.expander("Optional: your own building footprints or area boundary"):
    st.caption("Without a building file, buildings are downloaded from OpenStreetMap for the area of the images.")
    o1, o2 = st.columns(2)
    up_buildings = o1.file_uploader("Building footprints", type=["geojson", "json", "gpkg", "zip", "gml"])
    up_area = o2.file_uploader("Area boundary", type=["geojson", "json", "gpkg", "zip"])

DEMO = ROOT / "demo/conselice_2023"
EXAMPLES = {
    "Valencia, Spain (Oct 2024)": dict(before=EXAMPLE["pre"], after=EXAMPLE["post"], buildings=EXAMPLE["buildings"],
                                       area=EXAMPLE["aoi"], slug="valencia_example"),
    "Conselice, Italy (May 2023)": dict(before=DEMO / "before.tif", after=DEMO / "after.tif",
                                        buildings=DEMO / "buildings.geojson", area=DEMO / "aoi.geojson",
                                        slug="conselice_example"),
}
PORTO = ROOT / "demo/porto_alegre_2024"
EXAMPLES["Porto Alegre, Brazil (May 2024)"] = dict(before=PORTO / "before.tif", after=PORTO / "after.tif",
                                                   buildings=PORTO / "buildings.geojson", area=PORTO / "aoi.geojson",
                                                   slug="porto_alegre_example")
run = st.button("Map the flood", type="primary", icon=":material/water:")
st.caption("Or try an example:")
example = None
shown = [(k, icon) for k, icon in [("Porto Alegre, Brazil (May 2024)", ":material/location_city:"),
                                   ("Conselice, Italy (May 2023)", ":material/agriculture:"),
                                   ("Valencia, Spain (Oct 2024)", ":material/science:")]
         if Path(EXAMPLES[k]["after"]).exists()]
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
        before, after, area = ex["before"], ex["after"], ex["area"]
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
        if buildings is None:
            zone = gpd.read_file(area).to_crs("EPSG:4326") if area else image_area(before)
            km2 = area_km2(zone)
            if km2 > MAX_AREA_KM2:
                raise AnalysisError(
                    f"The images cover {km2:,.0f} km², too large to download all its buildings "
                    f"(limit {MAX_AREA_KM2} km²). Crop the images to the town, or add an area boundary.")
            bar.progress(0.05, "Downloading buildings from OpenStreetMap")
            b = fetch_buildings(zone)
            if len(b):
                inputs.mkdir(parents=True, exist_ok=True)
                buildings = inputs / "osm_buildings.geojson"
                b.to_file(buildings, driver="GeoJSON")
        analyse(before, after, run_dir, aoi=area, buildings=buildings, event_name=name,
                progress=lambda f, t: bar.progress(0.15 + 0.85 * f, t))
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
bpath = run_dir / "buildings_impact.geojson"
b = gpd.read_file(bpath) if bpath.exists() else None

st.divider()
st.subheader(f"Results · {summary['event_name']}")
for note in summary["notes"]:
    st.warning(note)

k1, k2, k3 = st.columns(3)
k1.metric("Flooded area", f"{summary['flood_affected_km2']:.1f} km²",
          f"{summary['flood_affected_pct']:.0f}% of the area", delta_color="off", delta_arrow="off", border=True)
if b is not None:
    n_aff, n_pos = int((b.impact == "Affected").sum()), int((b.impact == "Possibly affected").sum())
    k2.metric("Buildings affected", f"{n_aff:,}", f"of {len(b):,} buildings", delta_color="off",
              delta_arrow="off", border=True)
    k3.metric("Possibly affected", f"{n_pos:,}", "some flooding around them", delta_color="off",
              delta_arrow="off", border=True)
else:
    k2.info("No buildings found for this area, so only the flood map is shown.")

# map
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
if b is not None:
    hit = b[b.impact.isin(["Affected", "Possibly affected"])].copy()
    if len(hit) > 15000:
        hit = hit.nlargest(15000, "flood_fraction")
    if len(hit):
        hit["flooded_pct"] = (100 * hit.flood_fraction).round(0)
        folium.GeoJson(
            hit[["building_id", "impact", "use", "flooded_pct", "geometry"]], name="Affected buildings",
            style_function=lambda f: {"color": RED if f["properties"]["impact"] == "Affected" else ORANGE,
                                      "fillColor": RED if f["properties"]["impact"] == "Affected" else ORANGE,
                                      "weight": 0.6, "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(fields=["impact", "use", "flooded_pct"],
                                          aliases=["", "Type", "Flooded around it (%)"]),
        ).add_to(m)
m.fit_bounds(bounds)
folium.LayerControl(collapsed=False).add_to(m)
st.markdown(
    f'<span style="color:rgb{FLOOD_RGB}">■</span> Flooded area &nbsp; '
    f'<span style="color:{RED}">■</span> Affected building &nbsp; '
    f'<span style="color:{ORANGE}">■</span> Possibly affected building', unsafe_allow_html=True)
st.caption("Tip: the satellite background is from a normal day, not the flood. Switch on **Radar image after the flood** (top right of the map): flood water shows as dark areas, which match the blue.")
st_folium(m, height=560, use_container_width=True, returned_objects=[], key=f"map_{run_dir.name}")

# affected buildings list
if b is not None:
    st.subheader("Affected buildings")
    t = b[b.impact.isin(["Affected", "Possibly affected"])].copy()
    pts = t.geometry.representative_point()
    t["latitude"], t["longitude"] = pts.y.round(6), pts.x.round(6)
    t["flooded_pct"] = (100 * t.flood_fraction).round(0)
    cols = ["building_id", "impact", "use", "flooded_pct", "latitude", "longitude"]
    if t["dwellings"].notna().any():
        cols.insert(3, "dwellings")
    t = t.sort_values(["impact", "flooded_pct"], ascending=[True, False])[cols]
    st.dataframe(t, hide_index=True, width="stretch", height=420, column_config={
        "building_id": "Building ID", "impact": "Status", "use": "Type", "dwellings": "Dwellings",
        "flooded_pct": st.column_config.ProgressColumn("Flooded around it", min_value=0, max_value=100,
                                                       format="%d%%"),
        "latitude": st.column_config.NumberColumn("Latitude", format="%.5f"),
        "longitude": st.column_config.NumberColumn("Longitude", format="%.5f")})
    st.caption("Affected: at least half of the ground in and around the building (10 m) is flooded. "
               "Possibly affected: at least 10%. This shows flood exposure, not confirmed structural damage.")

# downloads
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
