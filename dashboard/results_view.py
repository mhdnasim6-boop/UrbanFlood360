"""
UrbanFlood360 dashboard: results view shared by the "Download and analyse" and "Analyse your own images" pages.

Shows one saved run from outputs/user_runs/<time>_<name>/ (summary.json, flood_mask.tif, buildings_impact.geojson,
results.zip, and download.json when the data was downloaded by the app).
"""
import json
import sys
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


def buildings_png(b, colors, png, max_px=6000, pixel_m=3.0):
    """Web-Mercator RGBA PNG with every building filled in its colour (colors: hex per row); returns bounds.

    Used when there are too many buildings for interactive shapes on the map (tens of thousands)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    g = b.to_crs("EPSG:3857")
    x0, y0, x1, y1 = g.total_bounds
    px = max(pixel_m, (x1 - x0) / max_px, (y1 - y0) / max_px)
    w, h = int(np.ceil((x1 - x0) / px)) + 1, int(np.ceil((y1 - y0) / px)) + 1
    T = from_origin(x0, y1, px, px)
    palette = list(dict.fromkeys(colors))
    code = pd.Series(colors, index=g.index).map({c: i + 1 for i, c in enumerate(palette)})
    lab = rasterize(zip(g.geometry, code), out_shape=(h, w), transform=T, fill=0, all_touched=True, dtype="uint8")
    rgba = np.zeros((h, w, 4), "uint8")
    for i, c in enumerate(palette, start=1):
        rgba[lab == i] = (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 235)
    Image.fromarray(rgba, mode="RGBA").save(png, optimize=True)
    west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", x0, y1 - h * px, x0 + w * px, y1)
    return [[south, west], [north, east]]


def depth_png(depth_tif, png, vmax=3.0):
    """Web-Mercator RGBA PNG of water depth (m): light blue (shallow) to dark blue (>= vmax); returns bounds."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    with rasterio.open(depth_tif) as src:
        tr, w, h = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs="EPSG:3857", src_nodata=src.nodata,
                  dst_nodata=np.nan, resampling=Resampling.nearest)
    left, top = tr @ (0, 0)
    right, bottom = tr @ (w, h)
    west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", left, bottom, right, top)
    if not png.exists():
        ok = np.isfinite(dst) & (dst >= 0)
        t = np.clip(np.nan_to_num(dst) / vmax, 0, 1)[..., None]
        light, dark = np.array([186, 228, 250]), np.array([8, 48, 107])
        rgb = (light * (1 - t) + dark * t).astype("uint8")
        rgba = np.zeros((h, w, 4), "uint8")
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(ok, 200, 0)
        Image.fromarray(rgba, mode="RGBA").save(png, optimize=True)
    return png, [[south, west], [north, east]]


def radar_png(features_tif, band, name, vmin=-25.0, vmax=0.0):
    """Web-Mercator greyscale PNG of one dB band (dark = low backscatter, e.g. water); returns (png, bounds)."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    png = features_tif.parent / f"{name}.png"
    with rasterio.open(features_tif) as src:
        tr, w, h = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        left, top = tr @ (0, 0)
        right, bottom = tr @ (w, h)
        west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", left, bottom, right, top)
        if not png.exists():
            dst = np.full((h, w), np.nan, dtype="float32")
            reproject(rasterio.band(src, band), dst, dst_transform=tr, dst_crs="EPSG:3857",
                      resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)
            ok = np.isfinite(dst)
            grey = np.clip((dst - vmin) / (vmax - vmin), 0, 1) * 255
            rgba = np.zeros((h, w, 4), dtype="uint8")
            rgba[..., :3] = np.where(ok, grey, 0)[..., None].astype("uint8")
            rgba[..., 3] = np.where(ok, 255, 0)
            Image.fromarray(rgba, mode="RGBA").save(png, optimize=True)
    return png, [[south, west], [north, east]]


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



def show_results(empty_message):
    """Pick a saved run (default: the one just made, in st.session_state["run_dir"]) and show its results."""
    runs = list_runs()
    if not runs:
        st.info(empty_message)
        return

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
    dl = json.load(open(run_dir / "download.json")) if (run_dir / "download.json").exists() else None
    if dl:
        s1 = dl["sentinel1"]
        st.markdown(
            f"**Area:** {dl['area']['name']} ({dl['area']['km2']:.1f} km²) · **Flood date:** {dl['flood_date']} · "
            f"**Before:** {s1['before']['date']} · **After:** {s1['after']['date']} "
            f"({s1['after']['days_after_flood']} days after the flood) · orbit {s1['after']['orbit']} "
            f"{s1['after']['direction']} · {s1['after']['platform']}")
        st.caption(f"Radar: {s1['source']}. Buildings: "
                   + (dl["buildings"]["source"] if dl.get("buildings") else "none") + ".")
        if s1["after"]["days_after_flood"] > 3:
            st.warning(f"The after image was taken {s1['after']['days_after_flood']} days after the flood date. "
                       "Water may already have drained; the map then shows flood traces at best and can miss "
                       "flooded streets.")
    for note in summary["notes"]:
        st.warning(note)
    st.info("The U-Net was trained on one flood (Valencia, October 2024, Sentinel-1 orbit 103 ascending, sigma0). "
            "On other cities, orbits or calibrations it can over-predict: on Valencia with the automatically "
            "downloaded data (orbit 110 descending, gamma0) its test-area Dice drops from 0.71 to 0.46. "
            "Treat the map as a first screening and check it against local reports.",
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

