"""
UrbanFlood360 dashboard page: download Sentinel-1 data for a flood and analyse it.

1. Area: search a place (OpenStreetMap), draw a box, or upload a boundary.
2. Flood date: the page lists the Sentinel-1 passes around it (Microsoft Planetary Computer, sentinel-1-rtc) and
   picks a before/after pair from the same orbit track; the user can change the pair.
3. Download + analyse: scripts/fetch_event_data.py downloads the two images (clipped to the area, 10 m) and the
   OpenStreetMap buildings, then scripts/analyse_event.py makes the flood map and the damage assessment.

Each run is saved in outputs/user_runs/<time>_<name>/.
"""
import datetime as dt
import re
import sys
import time
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.results_view import RUNS, save_upload, show_results  # noqa: E402
from scripts.analyse_event import AnalysisError, analyse, selected  # noqa: E402
from scripts.fetch_event_data import (MAX_AREA_KM2, area_km2, check_area, fetch, geocode,  # noqa: E402
                                      search_scenes, choose_pair)  # noqa: F401

ss = st.session_state

st.title("UrbanFlood360 · Download and analyse a flood")
st.markdown(
    "Choose an **urban area** and the **flood date**. The app finds the Sentinel-1 radar images taken before and "
    "after the flood, downloads them with the area's **OpenStreetMap buildings**, and produces the **urban flood "
    "map** and the **building damage assessment**. No account or file preparation is needed.")

# ------------------------------------------------------------------ 1. area
st.subheader("1 · Area")
mode = st.radio("How do you want to choose the area?", ["Search a place", "Draw a box on the map",
                                                        "Upload a boundary"], horizontal=True)
if mode == "Search a place":
    c1, c2 = st.columns([4, 1])
    place = c1.text_input("Place", placeholder="e.g. Paiporta, Valencia, Spain", label_visibility="collapsed")
    if c2.button("Find", icon=":material/search:", width="stretch") and place:
        try:
            with st.spinner("Looking up the place"):
                ss["aoi"] = geocode(place)
                ss["aoi_label"] = place
                ss.pop("scenes", None)
        except Exception as e:
            st.error(str(e))
elif mode == "Draw a box on the map":
    st.caption("Use the square tool on the left of the map to draw a box around the town, then release.")
    centre = ss["aoi"].union_all().centroid if "aoi" in ss else None
    m = folium.Map(location=[centre.y, centre.x] if centre else [39.42, -0.40], zoom_start=12 if centre else 11,
                   tiles="CartoDB positron")
    Draw(draw_options=dict(polyline=False, polygon=False, circle=False, marker=False, circlemarker=False,
                           rectangle=True), edit_options=dict(edit=False)).add_to(m)
    drawn = st_folium(m, height=420, use_container_width=True, returned_objects=["last_active_drawing"],
                      key="draw_map")
    geom = (drawn or {}).get("last_active_drawing")
    if geom and geom.get("geometry"):
        g = gpd.GeoDataFrame({"name": ["Drawn box"]}, geometry=[shape(geom["geometry"])], crs="EPSG:4326")
        if "aoi_label" not in ss or ss.get("aoi_label") != str(g.total_bounds.round(5)):
            ss["aoi"], ss["aoi_label"] = g, str(g.total_bounds.round(5))
            ss.pop("scenes", None)
else:
    up = st.file_uploader("Area boundary", type=["geojson", "json", "gpkg", "zip"])
    if up is not None and ss.get("aoi_label") != up.name:
        try:
            p = save_upload(up, RUNS / "_uploads")
            g = gpd.read_file(f"zip://{p}" if p.suffix == ".zip" else p).to_crs("EPSG:4326")
            ss["aoi"] = gpd.GeoDataFrame({"name": [p.stem]}, geometry=[g.union_all()], crs="EPSG:4326")
            ss["aoi_label"] = up.name
            ss.pop("scenes", None)
        except Exception as e:
            st.error(f"Could not read the boundary: {e}")

aoi = ss.get("aoi")
if aoi is None:
    st.info("Choose an area to continue.")
    show_results("No analyses yet.")
    st.stop()

km2 = area_km2(aoi)
name = str(aoi["name"].iloc[0])
st.success(f"**{name}** · {km2:,.1f} km²", icon=":material/location_on:")
if km2 > MAX_AREA_KM2:
    st.error(f"That is larger than the {MAX_AREA_KM2} km² limit. Pick a town or district, or draw a smaller box.")
    st.stop()
if mode != "Draw a box on the map":
    w, s, e, n = aoi.total_bounds
    pm = folium.Map(tiles="CartoDB positron")
    folium.GeoJson(aoi, style_function=lambda f: {"color": "#2a78d6", "weight": 2, "fillOpacity": 0.08}).add_to(pm)
    pm.fit_bounds([[s, w], [n, e]])
    st_folium(pm, height=300, use_container_width=True, returned_objects=[], key=f"aoi_{ss.get('aoi_label')}")

# ------------------------------------------------------------------ 2. flood date and images
st.subheader("2 · Flood date and radar images")
c1, c2 = st.columns([1, 3])
flood_date = c1.date_input("Flood date", value=ss.get("flood_date", dt.date(2024, 10, 29)),
                           max_value=dt.date.today(), help="The day the flooding happened (or peaked).")
if ss.get("flood_date") != flood_date:
    ss["flood_date"] = flood_date
    ss.pop("scenes", None)
if c2.button("Search Sentinel-1 images", icon=":material/satellite_alt:"):
    try:
        with st.spinner("Searching the Sentinel-1 archive"):
            ss["scenes"] = search_scenes(aoi, flood_date)
    except Exception as e:
        st.error(f"Search failed: {e}")

scenes = ss.get("scenes")
if scenes is None:
    st.info("Search the images for the flood date to continue.")
    show_results("No analyses yet.")
    st.stop()
if scenes.empty:
    st.error("No Sentinel-1 images with VV and VH found around this date for this area.")
    st.stop()

table = scenes[["date", "when", "days_from_flood", "orbit", "direction", "platform", "coverage"]].copy()
table["coverage"] = (100 * table.coverage).round(0)
st.dataframe(table, hide_index=True, width="stretch", height=min(38 * (len(table) + 1), 320), column_config={
    "date": "Date", "when": "Before / after", "days_from_flood": "Days from flood", "orbit": "Orbit track",
    "direction": "Direction", "platform": "Satellite",
    "coverage": st.column_config.NumberColumn("Covers area (%)", format="%d")})

try:
    auto_b, auto_a = choose_pair(scenes)
except AnalysisError as e:
    st.error(str(e))
    st.stop()


def label(i):
    r = scenes.loc[i]
    return f"{r.date} · orbit {r.orbit} {r.direction} · {r.platform} · covers {100 * r.coverage:.0f}%"


c1, c2 = st.columns(2)
before_opts = list(scenes.index[scenes.when == "before"])
before_i = c1.selectbox("Before image", before_opts, index=before_opts.index(auto_b.name), format_func=label)
after_opts = list(scenes.index[(scenes.when == "after") & (scenes.orbit == scenes.loc[before_i].orbit)])
if not after_opts:
    c2.error("No after image from the same orbit track as this before image. Pick another before image.")
    st.stop()
after_i = c2.selectbox("After image (same orbit track)", after_opts,
                       index=after_opts.index(auto_a.name) if auto_a.name in after_opts else 0, format_func=label)
st.caption("Pre-selected: the last image before and the first image after the flood from the same orbit track, "
           "so both are seen from the same angle and the change between them is the flood.")
days = scenes.loc[after_i].days_from_flood
if days > 3:
    st.warning(f"The after image is {days} days after the flood date. In towns the water may have drained by then.")

# ------------------------------------------------------------------ 3. run
st.subheader("3 · Download and analyse")
with st.expander("Settings"):
    s1, s2, s3 = st.columns(3)
    with_buildings = s1.checkbox("Download OpenStreetMap buildings", value=True,
                                 help="Needed for the damage assessment.")
    with_water = s2.checkbox("Also map standing water", value=False)
    sel_thr = selected("affected")[1]
    threshold = s3.slider("Flood probability threshold", 0.1, 0.9, sel_thr, 0.05,
                          help=f"{sel_thr} was chosen on the Valencia validation area.")
    s1, s2, s3, s4 = st.columns(4)
    affected = s1.number_input("'Affected' if flooded share ≥", 0.05, 1.0, 0.5, 0.05)
    possible = s2.number_input("'Possibly affected' if ≥", 0.0, 1.0, 0.1, 0.05)
    ring_m = s3.number_input("Ring around building (m)", 0.0, 50.0, 10.0, 5.0)
    ppd = s4.number_input("Residents per dwelling", 0.5, 10.0, 2.5, 0.1)

if st.button("Download and analyse", type="primary", icon=":material/play_arrow:"):
    event = name.split(",")[0]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", event).strip("_")[:40] or "event"
    run_dir = RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
    bar = st.progress(0.0, "Starting")
    try:
        check_area(aoi)
        fetch(run_dir, flood_date, aoi=aoi, scenes=scenes, before_index=before_i, after_index=after_i,
              with_buildings=with_buildings, progress=lambda f, t: bar.progress(0.55 * f, t))
        bld = run_dir / "buildings.geojson"
        analyse(run_dir / "before.tif", run_dir / "after.tif", run_dir, aoi=run_dir / "aoi.geojson",
                buildings=bld if bld.exists() else None, with_water=with_water, threshold=threshold,
                ring_m=ring_m, affected=affected, possible=possible, persons_per_dwelling=ppd,
                event_name=f"{event} · {flood_date}",
                progress=lambda f, t: bar.progress(0.55 + 0.45 * f, t))
        ss["run_dir"] = str(run_dir)
        bar.empty()
    except AnalysisError as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:
        bar.empty()
        st.exception(e)

show_results("No analyses yet. Download and analyse a flood above.")
