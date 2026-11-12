"""
UrbanFlood360 dashboard page: analyse a NEW flood event.

The user uploads a Sentinel-1 image from before and one from after a flood (and optionally an area boundary and
building footprints). The page runs scripts/analyse_event.py, which applies the trained U-Net v2 models and the
building damage rules from the Valencia case study, and shows the flood map and the damage assessment.

Every run is saved in outputs/user_runs/<time>_<name>/ (inputs/, results, results.zip).
"""
import json
import re
import sys
import time
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.raster_layers import ImageOverlay
from PIL import Image
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyse_event import AnalysisError, analyse, selected  # noqa: E402
from scripts.build_dashboard_assets import to_web_mercator  # noqa: E402

RUNS = ROOT / "outputs/user_runs"
IMPACT_ORDER = ["Affected", "Possibly affected", "Not affected", "No data"]
IMPACT_COLORS = {"Affected": "#d03b3b", "Possibly affected": "#ec835a",
                 "Not affected": "#8f8e88", "No data": "#c9c8c2"}
VIOLET, BLUE = (74, 58, 167), (42, 120, 214)
EXAMPLE = dict(
    pre=ROOT / "data/processed/sentinel1/prototype/20241019_R103/corrected/db/S1_PRE_20241019_R103_sigma0_db.tif",
    post=ROOT / "data/processed/sentinel1/post/20241031_R103/corrected/db/S1_POST_20241031_R103_sigma0_db.tif",
    aoi=ROOT / "data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson",
    buildings=ROOT / "data/processed/buildings/buildings_aoi.gpkg",
)
BASEMAPS = {
    "Satellite (Esri)": dict(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                             attr="Tiles &copy; Esri, Maxar, Earthstar Geographics"),
    "Light (CARTO)": dict(tiles="CartoDB positron", attr=None),
    "OpenStreetMap": dict(tiles="OpenStreetMap", attr=None),
}
MAX_BUILDINGS_ON_MAP = 15_000


def fmt(n):
    return "–" if n is None or (isinstance(n, float) and np.isnan(n)) else f"{n:,.0f}"


def save_upload(upload, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(upload.name).name
    path.write_bytes(upload.getbuffer())
    return path


def overlay_png(mask_tif, rgb, alpha):
    """Web-Mercator RGBA PNG of a 0/1/255 mask next to the tif; returns (png path, lat/lon bounds)."""
    png = mask_tif.with_suffix(".png")
    mask, bounds = to_web_mercator(mask_tif)
    if not png.exists():
        rgba = np.zeros(mask.shape + (4,), dtype="uint8")
        rgba[mask == 1] = (*rgb, alpha)
        Image.fromarray(rgba, mode="RGBA").save(png, optimize=True)
    return png, bounds


@st.cache_data(show_spinner=False)
def load_run(run_dir):
    run = Path(run_dir)
    summary = json.load(open(run / "summary.json"))
    b = gpd.read_file(run / "buildings_impact.geojson") if (run / "buildings_impact.geojson").exists() else None
    return summary, b


def list_runs():
    if not RUNS.exists():
        return []
    return sorted((p for p in RUNS.iterdir() if (p / "summary.json").exists()), reverse=True)


# ------------------------------------------------------------------ header
st.title("UrbanFlood360 · Analyse a new flood event")
st.markdown(
    "Upload a **Sentinel-1 radar image from before** the flood and one from **after** it. The trained U-Net maps "
    "flood-affected ground, and every building is checked against that map. "
    "Add **building footprints** to get the damage assessment; without them you get the flood map only.")

with st.expander("What files do I need?"):
    st.markdown("""
| Input | Required | Format |
|---|---|---|
| Before image | yes | GeoTIFF with a **VV** and a **VH** band: Sentinel-1 GRD, calibrated to Sigma0 and terrain-corrected (for example with ESA SNAP). dB or linear values both work. |
| After image | yes | Same as above, ideally from the **same orbit track**, taken as soon as possible after the flood. |
| Area boundary | no | GeoJSON, GeoPackage or zipped shapefile. Limits the analysis to the city. Needed if the images are very large. |
| Buildings | no | GeoJSON, GeoPackage, zipped shapefile, or a Spanish Catastro INSPIRE `building.gml`. A column with the number of dwellings (`dwellings`, `numberOfDwellings` or `building:flats`) gives resident estimates. OpenStreetMap buildings work too. |

The radar images must already be processed (raw `.SAFE` / `.zip` downloads from Copernicus are not accepted).
Images that are in latitude/longitude or not at 10 m are reprojected to UTM at 10 m automatically.
""")

# ------------------------------------------------------------------ inputs
with st.form("inputs"):
    c1, c2 = st.columns([2, 1])
    event_name = c1.text_input("Event name", "My flood event")
    use_example = c2.toggle("Use the Valencia example files", value=False,
                            help="Fills in the Valencia 2024 images, area and Catastro buildings, to try the page.")
    c1, c2 = st.columns(2)
    up_pre = c1.file_uploader("Before-flood Sentinel-1 image (GeoTIFF)", type=["tif", "tiff"])
    up_post = c2.file_uploader("After-flood Sentinel-1 image (GeoTIFF)", type=["tif", "tiff"])
    c1, c2 = st.columns(2)
    up_aoi = c1.file_uploader("Area boundary (optional)", type=["geojson", "json", "gpkg", "zip"])
    up_bld = c2.file_uploader("Building footprints (optional)", type=["geojson", "json", "gpkg", "zip", "gml"])

    with st.expander("Settings"):
        sel_thr = selected("affected")[1]
        s1, s2, s3 = st.columns(3)
        band_order = s1.selectbox("Band order", ["Automatic (from band names)", "VV, VH", "VH, VV"],
                                  help="Only needed when the GeoTIFF bands have no names.")
        threshold = s2.slider("Flood probability threshold", 0.1, 0.9, sel_thr, 0.05,
                              help=f"{sel_thr} was chosen on the Valencia validation area.")
        with_water = s3.checkbox("Also map standing water", value=False,
                                 help="Runs the second U-Net, trained on standing water only.")
        s1, s2, s3, s4 = st.columns(4)
        affected = s1.number_input("'Affected' if flooded share ≥", 0.05, 1.0, 0.5, 0.05)
        possible = s2.number_input("'Possibly affected' if ≥", 0.0, 1.0, 0.1, 0.05)
        ring_m = s3.number_input("Ring around building (m)", 0.0, 50.0, 10.0, 5.0)
        ppd = s4.number_input("Residents per dwelling", 0.5, 10.0, 2.5, 0.1)

    run_clicked = st.form_submit_button("Run analysis", type="primary", icon=":material/play_arrow:")

if run_clicked:
    if use_example:
        pre, post, aoi, bld = EXAMPLE["pre"], EXAMPLE["post"], EXAMPLE["aoi"], EXAMPLE["buildings"]
        if event_name == "My flood event":
            event_name = "Valencia example"
    else:
        pre = post = aoi = bld = None
    slug = re.sub(r"[^A-Za-z0-9]+", "_", event_name).strip("_")[:40] or "event"
    run_dir = RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
    inputs = run_dir / "inputs"
    if up_pre:
        pre = save_upload(up_pre, inputs)
    if up_post:
        post = save_upload(up_post, inputs)
    if up_aoi:
        aoi = save_upload(up_aoi, inputs)
    if up_bld:
        bld = save_upload(up_bld, inputs)

    if pre is None or post is None:
        st.error("Upload both a before and an after image, or switch on the Valencia example files.")
    else:
        bar = st.progress(0.0, "Starting")
        try:
            analyse(pre, post, run_dir, aoi=aoi, buildings=bld,
                    band_order=None if band_order.startswith("Automatic") else band_order.replace(" ", ""),
                    with_water=with_water, threshold=threshold, ring_m=ring_m, affected=affected,
                    possible=possible, persons_per_dwelling=ppd, event_name=event_name,
                    progress=lambda f, t: bar.progress(min(f, 1.0), t))
            st.session_state["run_dir"] = str(run_dir)
            bar.empty()
        except AnalysisError as e:
            bar.empty()
            st.error(str(e))
        except Exception as e:  # unexpected: show it rather than a blank page
            bar.empty()
            st.exception(e)

# ------------------------------------------------------------------ results
runs = list_runs()
if not runs:
    st.info("No analyses yet. Upload images above, or switch on the Valencia example files, and press Run analysis.")
    st.stop()

current = st.session_state.get("run_dir", str(runs[0]))
names = [str(p) for p in runs]
run_dir = Path(st.selectbox("Results to show", names, index=names.index(current) if current in names else 0,
                            format_func=lambda p: Path(p).name))
summary, b = load_run(str(run_dir))

st.divider()
st.subheader(f"Results · {summary['event_name']}")
st.caption(f"Run {summary['created']} · before: {summary['inputs']['pre']} · after: {summary['inputs']['post']} · "
           f"model {summary['model']['checkpoint']} at threshold {summary['model']['threshold']} · "
           f"{summary['runtime_s']} s")
for note in summary["notes"]:
    st.warning(note)
st.info("The U-Net was trained on one flood (Valencia, October 2024). On other cities, seasons or flood types "
        "accuracy is unknown and probably lower. Check the map against local reports before acting on it.",
        icon=":material/info:")

bs = summary.get("buildings")
k = st.columns(5)
tiles = [("Flood-affected area", f"{summary['flood_affected_km2']:.1f} km²",
          f"{summary['flood_affected_pct']:.0f}% of {summary['area_with_data_km2']:.1f} km² with data")]
if bs:
    cls = bs["by_class"]
    tiles += [("Buildings affected", fmt(cls["Affected"]), f"+{fmt(cls['Possibly affected'])} possibly"),
              ("Dwellings affected", fmt(bs["dwellings_affected"]), "from the building file"),
              ("Residents affected", fmt(bs["residents_affected_est"]),
               f"est. {summary['parameters']['persons_per_dwelling']} per dwelling"),
              ("Buildings checked", fmt(bs["total"]),
               f"{100 * cls['Affected'] / max(bs['total'], 1):.0f}% affected")]
else:
    tiles += [("Standing water", f"{summary['standing_water_km2']:.1f} km²" if summary["standing_water_km2"] is not None
               else "–", "second U-Net")]
for col, (label, value, note) in zip(k, tiles):
    col.metric(label, value, note, delta_color="off", delta_arrow="off", border=True)

tab_names = ["Flood map"] + (["Damage assessment", "Priority buildings"] if bs else []) + ["Downloads"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

with tabs["Flood map"]:
    c1, c2, c3 = st.columns(3)
    basemap = c1.selectbox("Basemap", list(BASEMAPS))
    show_classes = c2.multiselect("Buildings shown", IMPACT_ORDER, default=["Affected", "Possibly affected"],
                                  disabled=b is None)
    show_water = c3.checkbox("Standing water layer", value=False, disabled=not (run_dir / "water_mask.tif").exists())

    png, bounds = overlay_png(run_dir / "flood_mask.tif", VIOLET, 150)
    (s, w), (n, e) = bounds
    bm = BASEMAPS[basemap]
    m = folium.Map(tiles=None, prefer_canvas=True, control_scale=True)
    folium.TileLayer(bm["tiles"], attr=bm["attr"], name=basemap, max_zoom=19).add_to(m)
    ImageOverlay(str(png), bounds=bounds, name="Flood-affected area (U-Net)", opacity=0.8).add_to(m)
    if show_water:
        wpng, wb = overlay_png(run_dir / "water_mask.tif", BLUE, 190)
        ImageOverlay(str(wpng), bounds=wb, name="Standing water (U-Net)", opacity=0.85).add_to(m)
    if b is not None and show_classes:
        mb = b[b.impact.isin(show_classes)].copy()
        if len(mb) > MAX_BUILDINGS_ON_MAP:
            st.caption(f"Showing the {fmt(MAX_BUILDINGS_ON_MAP)} buildings with the highest flood share "
                       f"of {fmt(len(mb))}.")
            mb = mb.nlargest(MAX_BUILDINGS_ON_MAP, "flood_fraction")
        if len(mb):
            mb["flood_pct"] = (100 * mb.flood_fraction).round(0)
            fields = ["impact", "use", "dwellings", "residents_est", "flood_pct", "building_id"]
            aliases = ["Impact", "Use", "Dwellings", "Residents (est.)", "Flooded share of footprint + ring (%)", "ID"]
            folium.GeoJson(
                mb[fields + ["geometry"]], name="Buildings",
                style_function=lambda f: {"fillColor": IMPACT_COLORS[f["properties"]["impact"]],
                                          "color": IMPACT_COLORS[f["properties"]["impact"]],
                                          "weight": 0.6, "fillOpacity": 0.85},
                tooltip=folium.GeoJsonTooltip(fields=["impact", "use"], aliases=["Impact", "Use"]),
                popup=folium.GeoJsonPopup(fields=fields, aliases=aliases),
            ).add_to(m)
    m.fit_bounds([[s, w], [n, e]])
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=620, use_container_width=True, returned_objects=[], pixelated=True,
              key=f"map_{run_dir.name}")
    st.caption("Violet: ground the U-Net classes as flood-affected (standing water or flood trace such as mud). "
               "Buildings are coloured by impact class; click one for details.")

if bs:
    with tabs["Damage assessment"]:
        c1, c2 = st.columns(2)
        counts = b.impact.value_counts().reindex(IMPACT_ORDER, fill_value=0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[IMPACT_COLORS[c] for c in counts.index],
                               hovertemplate="%{x}: %{y:,} buildings<extra></extra>"))
        fig.update_layout(title="Buildings by impact class", yaxis_title="Buildings", height=380,
                          margin=dict(l=10, r=10, t=50, b=10))
        c1.plotly_chart(fig, width="stretch")

        g = (b[b.impact.isin(["Affected", "Possibly affected"])].groupby(["use", "impact"]).size()
             .unstack(fill_value=0).reindex(columns=["Affected", "Possibly affected"], fill_value=0))
        g = g.loc[g.sum(axis=1).sort_values().index]
        fig = go.Figure()
        for c in ["Affected", "Possibly affected"]:
            fig.add_bar(y=g.index, x=g[c], name=c, orientation="h", marker_color=IMPACT_COLORS[c])
        fig.update_layout(barmode="stack", title="Affected buildings by use", xaxis_title="Buildings", height=380,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.08))
        c2.plotly_chart(fig, width="stretch")

        rows = []
        for use, x in b.groupby("use"):
            rows.append({"Use": use, "Buildings": len(x), "Affected": int((x.impact == "Affected").sum()),
                         "Possibly affected": int((x.impact == "Possibly affected").sum()),
                         "% affected": round(100 * (x.impact == "Affected").mean(), 1),
                         "Dwellings affected": x.loc[x.impact == "Affected", "dwellings"].sum(min_count=1),
                         "Residents affected (est.)": x.loc[x.impact == "Affected", "residents_est"].sum(min_count=1)})
        st.dataframe(pd.DataFrame(rows).sort_values("Affected", ascending=False), hide_index=True, width="stretch")
        p = summary["parameters"]
        st.caption(f"A building's class comes from the flood-affected share of its footprint plus a "
                   f"{p['ring_m']:.0f} m ring: Affected ≥ {p['affected_threshold']:.0%}, Possibly affected ≥ "
                   f"{p['possible_threshold']:.0%}. 'Affected' means flood evidence around the building, not "
                   "confirmed structural damage.")

    with tabs["Priority buildings"]:
        st.markdown("Affected buildings ranked by estimated residents, then by flooded share.")
        cols = ["building_id", "use", "dwellings", "residents_est", "footprint_m2", "flood_fraction"] + \
            (["water_fraction"] if "water_fraction" in b else [])
        pr = b[b.impact == "Affected"].sort_values(["residents_est", "flood_fraction"], ascending=False)[cols]
        st.dataframe(pr, hide_index=True, width="stretch", height=520, column_config={
            "building_id": "ID", "use": "Use", "dwellings": "Dwellings",
            "residents_est": st.column_config.NumberColumn("Residents (est.)", format="%.0f"),
            "footprint_m2": st.column_config.NumberColumn("Footprint (m²)", format="%d"),
            "flood_fraction": st.column_config.ProgressColumn("Flooded share", min_value=0, max_value=1, format="%.2f"),
            "water_fraction": st.column_config.NumberColumn("Standing water share", format="%.2f")})
        st.download_button("Download priority list (CSV)", pr.to_csv(index=False),
                           f"{run_dir.name}_priority.csv", "text/csv")

with tabs["Downloads"]:
    st.download_button("Download all results (ZIP)", (run_dir / "results.zip").read_bytes(),
                       f"{run_dir.name}.zip", "application/zip", type="primary", icon=":material/download:")
    st.markdown("""
| File | Contents |
|---|---|
| `flood_mask.tif` | 1 = flood-affected, 0 = not, 255 = no data (GIS-ready GeoTIFF) |
| `flood_probability.tif` | U-Net flood probability per 10 m pixel |
| `flood_polygons.geojson` | flood-affected areas as polygons |
| `water_mask.tif` | standing water (only if requested) |
| `buildings_impact.geojson` / `.csv` | every building with its flooded share and impact class |
| `sar_features_6ch.tif` | the six radar bands the model saw |
| `summary.json` | headline numbers, settings and notes |
""")
    st.caption(f"Saved on this computer in {run_dir.relative_to(ROOT)}/")
