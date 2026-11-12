"""
UrbanFlood360 dashboard page: Valencia DANA flood case study (29-30 Oct 2024), Horta Sud.

    streamlit run dashboard/app.py

Reads the outputs of the pipeline (run_pipeline.sh):
  outputs/dashboard/*        map layers built by scripts/build_dashboard_assets.py
  outputs/tables/*           damage and model-evaluation tables
  models/unet_v2/...         selected models and their training history
"""
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "outputs/dashboard"
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "reports/figures"

IMPACT_ORDER = ["Affected", "Possibly affected", "Not affected", "No data"]
# Impact is a severity state -> status colours (always shown with a text label).
IMPACT_COLORS = {"Affected": "#d03b3b", "Possibly affected": "#ec835a",
                 "Not affected": "#8f8e88", "No data": "#c9c8c2"}
BLUE, ORANGE, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#4a3aa7", "#eda100"
TARGET_LABELS = {"affected": "Flood-affected ground (water or flood trace) - used for the damage assessment",
                 "water": "Standing water on 31 Oct 2024"}
BASEMAPS = {
    "Satellite (Esri)": dict(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                             attr="Tiles &copy; Esri, Maxar, Earthstar Geographics"),
    "Light (CARTO)": dict(tiles="CartoDB positron", attr=None),
    "OpenStreetMap": dict(tiles="OpenStreetMap", attr=None),
}



# ---------------------------------------------------------------- data
@st.cache_data
def load_buildings():
    return gpd.read_file(ASSETS / "buildings.geojson")


@st.cache_data
def load_static():
    targets = [t for t in ("affected", "water") if (TABLES / f"model_comparison_{t}.csv").exists()]
    return dict(
        summary=json.load(open(TABLES / "damage_summary.json")),
        extent=pd.read_csv(TABLES / "flood_extent_by_municipality.csv"),
        comparison={t: pd.read_csv(TABLES / f"model_comparison_{t}.csv") for t in targets},
        selected={t: json.load(open(ROOT / f"models/unet_v2/selected_model_{t}.json")) for t in targets},
        agreement=pd.read_csv(TABLES / "emsr_agreement.csv", index_col=0).rename_axis("EMSR773 grade"),
        extent_check=pd.read_csv(TABLES / "emsr_extent_building_check.csv"),
        overlays=json.load(open(ASSETS / "overlays.json")),
        municipalities=gpd.read_file(ASSETS / "municipalities.geojson"),
        emsr_reference=gpd.read_file(ASSETS / "emsr_reference.geojson"),
        emsr_points=gpd.read_file(ASSETS / "emsr_points.geojson"),
    )


def fmt(n):
    return f"{n:,.0f}"


def swatch(color, label, opacity=1.0, outline=False):
    style = (f"border:2px solid {color};background:transparent" if outline
             else f"background:{color};opacity:{opacity}")
    return (f'<span style="display:inline-flex;align-items:center;margin-right:18px">'
            f'<span style="width:12px;height:12px;border-radius:3px;{style};margin-right:6px"></span>{label}</span>')


def map_legend():
    items = "".join(swatch(IMPACT_COLORS[k], k) for k in IMPACT_ORDER)
    items += swatch(VIOLET, "Flood-affected area (U-Net)", 0.6) + swatch(BLUE, "Standing water (U-Net)", 0.75)
    items += swatch(YELLOW, "EMSR773 reference", outline=True)
    st.markdown(f'<div style="font-size:0.85rem;margin:4px 0 8px">{items}</div>', unsafe_allow_html=True)


def style_chart(fig, height=380, legend=True):
    # Title at the very top, legend in its own row between title and plot.
    fig.update_layout(height=height, barcornerradius=4, bargap=0.3,
                      margin=dict(l=10, r=10, t=90 if legend else 50, b=10),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0) if legend else None,
                      showlegend=legend)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


# ---------------------------------------------------------------- page
data = load_static()
buildings = load_buildings()
summary = data["summary"]
params = summary["parameters"]

st.title("UrbanFlood360 · Valencia flood damage assessment")
st.caption("DANA flood of 29–30 October 2024 · Horta Sud (Alfafar, Catarroja, Massanassa, Paiporta, Picanya, Sedaví) · "
           "Sentinel-1 SAR (19 Oct → 31 Oct 2024) · U-Net flood segmentation · Catastro building inventory")

with st.sidebar:
    st.header("Filters")
    munis = sorted(buildings.municipality.unique())
    sel_muni = st.multiselect("Municipality", munis, default=munis)
    uses = sorted(buildings.use.unique())
    sel_use = st.multiselect("Building use", uses, default=uses)
    st.header("Map layers")
    basemap = st.selectbox("Basemap", list(BASEMAPS))
    show_classes = st.multiselect("Buildings shown on map", IMPACT_ORDER, default=["Affected", "Possibly affected"])
    show_affected = st.checkbox("Flood-affected area (U-Net)", value=True)
    show_water = st.checkbox("Standing water on 31 Oct (U-Net)", value=False, disabled="water" not in data["overlays"])
    show_ref = st.checkbox("EMSR773 reference polygons", value=False)
    show_points = st.checkbox("EMSR773 damage points", value=False)
    st.divider()
    st.caption(f"A building's class comes from the flood-affected share of its footprint + {params['ring_m']:.0f} m ring: "
               f"Affected ≥ {params['affected_threshold']:.0%}, Possibly affected ≥ {params['possible_threshold']:.0%}. "
               f"Residents = dwellings × {params['persons_per_dwelling']} (assumption).")

fb = buildings[buildings.municipality.isin(sel_muni) & buildings.use.isin(sel_use)]
ext = data["extent"][data["extent"].municipality.isin(sel_muni)]
aff = fb[fb.impact == "Affected"]
pos = fb[fb.impact == "Possibly affected"]

k = st.columns(5)
flood_km2, area_km2 = ext.predicted_flood_km2.sum(), ext.area_km2.sum()
tiles = [
    ("Flood-affected area", f"{flood_km2:.1f} km²", f"{100 * flood_km2 / max(area_km2, 1e-9):.0f}% of the area"),
    ("Buildings affected", fmt(len(aff)), f"+{fmt(len(pos))} possibly"),
    ("Dwellings affected", fmt(aff.dwellings.sum()), f"+{fmt(pos.dwellings.sum())} possibly"),
    ("Residents affected", fmt(aff.residents_est.sum()), f"est. {params['persons_per_dwelling']} per dwelling"),
    ("Affected floor area", f"{aff.floor_area_m2.sum() / 1e6:.2f} km²", f"{100 * len(aff) / max(len(fb), 1):.0f}% of buildings"),
]
for col, (label, value, note) in zip(k, tiles):
    col.metric(label, value, note, delta_color="off", delta_arrow="off", border=True)

tab_map, tab_damage, tab_priority, tab_model, tab_method = st.tabs(
    ["Map", "Damage analysis", "Priority buildings", "Model performance", "Method & limitations"])

# ---------------------------------------------------------------- map
with tab_map:
    map_legend()
    center = data["municipalities"].to_crs("EPSG:25830").union_all().centroid
    center = gpd.GeoSeries([center], crs="EPSG:25830").to_crs("EPSG:4326").iloc[0]
    bm = BASEMAPS[basemap]
    m = folium.Map(location=[center.y, center.x], zoom_start=14, tiles=None, prefer_canvas=True, control_scale=True)
    folium.TileLayer(bm["tiles"], attr=bm["attr"], name=basemap, max_zoom=19).add_to(m)

    ov = data["overlays"]
    if show_affected:
        ImageOverlay(str(ASSETS / "affected_mask.png"), bounds=ov["affected"],
                     name="Flood-affected area (U-Net)", opacity=0.8).add_to(m)
    if show_water and "water" in ov:
        ImageOverlay(str(ASSETS / "water_mask.png"), bounds=ov["water"],
                     name="Standing water (U-Net)", opacity=0.85).add_to(m)
    if show_ref:
        folium.GeoJson(data["emsr_reference"], name="EMSR773 reference",
                       style_function=lambda f: {"color": YELLOW, "weight": 1.5, "fillOpacity": 0},
                       tooltip=folium.GeoJsonTooltip(fields=["notation"], aliases=["EMSR773"])).add_to(m)

    folium.GeoJson(data["municipalities"], name="Municipalities",
                   style_function=lambda f: {"color": "#ffffff", "weight": 2, "fillOpacity": 0},
                   tooltip=folium.GeoJsonTooltip(fields=["municipality"], labels=False)).add_to(m)

    mb = fb[fb.impact.isin(show_classes)].copy()
    if len(mb):
        mb["flood_pct"] = (100 * mb.flood_fraction).round(0)
        mb["year_built"] = mb.year_built.astype("Int64").astype(str).replace("<NA>", "")  # no thousands separator
        fields = ["impact", "municipality", "use", "dwellings", "residents_est", "floors_above",
                  "floor_area_m2", "year_built", "flood_pct"]
        aliases = ["Impact", "Municipality", "Use", "Dwellings", "Residents (est.)", "Floors", "Floor area (m²)",
                   "Built", "Flood-affected share of footprint + ring (%)"]
        if "water_fraction" in mb:
            mb["water_pct"] = (100 * mb.water_fraction).round(0)
            fields.append("water_pct")
            aliases.append("Standing water on 31 Oct (%)")
        fields += ["emsr_damage", "building_id"]
        aliases += ["EMSR773 damage grade", "Catastro ID"]
        folium.GeoJson(
            mb[fields + ["geometry"]], name="Buildings",
            style_function=lambda f: {"fillColor": IMPACT_COLORS[f["properties"]["impact"]],
                                      "color": IMPACT_COLORS[f["properties"]["impact"]],
                                      "weight": 0.6, "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(fields=["impact", "use", "dwellings"], aliases=["Impact", "Use", "Dwellings"]),
            popup=folium.GeoJsonPopup(fields=fields, aliases=aliases),
        ).add_to(m)

    if show_points:
        folium.GeoJson(
            data["emsr_points"], name="EMSR773 damage points",
            marker=folium.CircleMarker(radius=3, weight=1, color="#ffffff", fill=True, fill_color=YELLOW, fill_opacity=0.95),
            tooltip=folium.GeoJsonTooltip(fields=["damage_gra", "our_impact"], aliases=["EMSR773 grade", "Our class"]),
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=640, use_container_width=True, returned_objects=[], pixelated=True)
    st.caption(f"{fmt(len(mb))} buildings drawn. Click a building for its details. The flood layers are U-Net "
               "predictions from Sentinel-1 radar; the EMSR773 reference exists only where Copernicus had optical "
               "images (about 73% of the area).")

# ---------------------------------------------------------------- damage analysis
with tab_damage:
    c1, c2 = st.columns(2)

    def stacked(df, by, title):
        g = (df[df.impact.isin(["Affected", "Possibly affected"])]
             .groupby([by, "impact"]).size().unstack(fill_value=0)
             .reindex(columns=["Affected", "Possibly affected"], fill_value=0))
        g = g.loc[g.sum(axis=1).sort_values().index]
        fig = go.Figure()
        for cls in ["Affected", "Possibly affected"]:
            fig.add_bar(y=g.index, x=g[cls], name=cls, orientation="h", marker_color=IMPACT_COLORS[cls],
                        hovertemplate=f"%{{y}}<br>{cls}: %{{x:,}} buildings<extra></extra>")
        fig.update_layout(barmode="stack", title=title, xaxis_title="Buildings", legend_traceorder="normal")
        return style_chart(fig)

    c1.plotly_chart(stacked(fb, "municipality", "Affected buildings by municipality"), width="stretch")
    c2.plotly_chart(stacked(fb, "use", "Affected buildings by use"), width="stretch")

    c3, c4 = st.columns(2)
    e = ext.sort_values("emsr_flood_km2", ascending=False)
    fig = go.Figure()
    fig.add_bar(x=e.municipality, y=e.predicted_flood_in_mapped_km2, name="U-Net flood-affected area", marker_color=BLUE,
                hovertemplate="%{x}<br>U-Net: %{y:.2f} km²<extra></extra>")
    fig.add_bar(x=e.municipality, y=e.emsr_flood_km2, name="EMSR773 flooded area + flood trace",
                marker_color=ORANGE, hovertemplate="%{x}<br>EMSR773: %{y:.2f} km²<extra></extra>")
    fig.update_layout(barmode="group", title="Flood-affected area, Copernicus-imaged part only", yaxis_title="km²")
    c3.plotly_chart(style_chart(fig), width="stretch")
    c3.caption("Like-for-like: both bars cover only the ~73% of each town that Copernicus imaged. "
               "The U-Net total in the headline also includes the unimaged part (mainly the eastern rice paddies).")

    hist = fb[fb.impact != "No data"]
    fig = go.Figure(go.Histogram(x=hist.flood_fraction, xbins=dict(start=0, end=1.0001, size=0.05),
                                 marker_color=BLUE, hovertemplate="Share %{x}<br>%{y:,} buildings<extra></extra>"))
    for x, label in [(params["possible_threshold"], "Possibly affected"), (params["affected_threshold"], "Affected")]:
        fig.add_vline(x=x, line_width=1.5, line_color="#8f8e88")
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{label} ≥ {x:.0%}", showarrow=False, xanchor="left",
                           xshift=4, font=dict(size=11))
    fig.update_layout(title="Flood-affected share of each building's footprint + ring", xaxis_title="Share",
                      yaxis_title="Buildings", xaxis_tickformat=".0%")
    c4.plotly_chart(style_chart(fig, legend=False), width="stretch")

    st.subheader("Summary by municipality")
    rows = []
    for name, x in fb.groupby("municipality"):
        rows.append({"Municipality": name, "Buildings": len(x),
                     "Affected": int((x.impact == "Affected").sum()),
                     "Possibly affected": int((x.impact == "Possibly affected").sum()),
                     "% affected": round(100 * (x.impact == "Affected").mean(), 1),
                     "Dwellings affected": int(x.loc[x.impact == "Affected", "dwellings"].sum()),
                     "Residents affected (est.)": int(x.loc[x.impact == "Affected", "residents_est"].sum()),
                     "Affected floor area (m²)": int(x.loc[x.impact == "Affected", "floor_area_m2"].sum())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button("Download filtered buildings (CSV)", fb.drop(columns="geometry").to_csv(index=False),
                       "urbanflood360_buildings_filtered.csv", "text/csv")

# ---------------------------------------------------------------- priority
with tab_priority:
    st.markdown("Affected buildings ranked by estimated residents, then by flood-affected share: a starting list for "
                "rescue and inspection teams.")
    cols = ["municipality", "use", "dwellings", "residents_est", "floors_above", "floor_area_m2",
            "flood_fraction"] + (["water_fraction"] if "water_fraction" in aff else []) + ["emsr_damage", "building_id"]
    pr = aff.sort_values(["residents_est", "flood_fraction"], ascending=False)[cols]
    st.dataframe(pr, hide_index=True, width="stretch", height=560,
                 column_config={
                     "municipality": "Municipality", "use": "Use", "dwellings": "Dwellings",
                     "residents_est": st.column_config.NumberColumn("Residents (est.)", format="%.0f"),
                     "floors_above": "Floors", "floor_area_m2": st.column_config.NumberColumn("Floor area (m²)", format="%d"),
                     "flood_fraction": st.column_config.ProgressColumn("Flood-affected share", min_value=0, max_value=1,
                                                                       format="%.2f"),
                     "water_fraction": st.column_config.NumberColumn("Standing water share", format="%.2f"),
                     "emsr_damage": "EMSR773 grade", "building_id": "Catastro ID"})
    st.download_button("Download priority list (CSV)", pr.to_csv(index=False), "urbanflood360_priority.csv", "text/csv")

# ---------------------------------------------------------------- model
with tab_model:
    targets = list(data["comparison"])
    target = st.radio("Segmentation target", targets, format_func=lambda t: TARGET_LABELS[t], horizontal=True)
    sel = data["selected"][target]
    comp = data["comparison"][target]
    st.markdown(f"**Selected model:** U-Net v2 run `{sel['run']}` · decision threshold **{sel['threshold']}**, "
                f"chosen on the validation block · {sel['selection_rule']}.")

    def region_table(region):
        t = comp[comp.region == region][["method", "trained_on", "threshold", "dice", "iou", "precision", "recall",
                                         "specificity"]]
        return t.rename(columns={"method": "Method", "trained_on": "Trained on", "threshold": "Threshold",
                                 "dice": "Dice (F1)", "iou": "IoU", "precision": "Precision", "recall": "Recall",
                                 "specificity": "Specificity"})

    val = comp[comp.region == "val_block"].iloc[0]
    test = comp[comp.region == "test_block"].iloc[0]
    st.markdown(f"**Validation block** · {int(val.tp + val.fn):,} flood px of {int(val[['tp', 'tn', 'fp', 'fn']].sum()):,} labelled")
    st.dataframe(region_table("val_block"), hide_index=True, width="stretch")
    st.markdown(f"**Test block (held out)** · {int(test.tp + test.fn):,} flood px of {int(test[['tp', 'tn', 'fp', 'fn']].sum()):,} labelled")
    st.dataframe(region_table("test_block"), hide_index=True, width="stretch")
    st.caption("Pixels are 10 m × 10 m. Validation scores are slightly optimistic because the validation block was used "
               "to pick the checkpoint and threshold. The test block was not used for any choice."
               + (" For the standing-water target the test block holds very little water, so its Dice is unstable."
                  if target == "water" else ""))

    fig = go.Figure()
    for region, label, color in [("val_block", "Validation block", BLUE), ("test_block", "Test block", ORANGE)]:
        t = comp[comp.region == region]
        fig.add_bar(x=t.method, y=t.dice, name=label, marker_color=color,
                    hovertemplate="%{x}<br>" + label + " Dice: %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", title="Dice (F1) by method", yaxis_title="Dice", yaxis_range=[0, 1])
    st.plotly_chart(style_chart(fig, height=380), width="stretch")

    hist_path = ROOT / "models/unet_v2/runs" / sel["run"] / "history.csv"
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=h.epoch, y=h.train_loss, name="Train loss", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=h.epoch, y=h.val_loss, name="Validation loss", line=dict(color=ORANGE, width=2))
        fig.update_layout(title=f"Training curves ({sel['run']}): loss", xaxis_title="Epoch", yaxis_title="Loss",
                          hovermode="x unified")
        c1.plotly_chart(style_chart(fig), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=h.epoch, y=h["val_dice@0.5"], name="Val Dice @0.5", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=h.epoch, y=h.val_best_dice, name="Val Dice @best threshold", line=dict(color=ORANGE, width=2))
        fig.update_layout(title=f"Training curves ({sel['run']}): validation Dice", xaxis_title="Epoch",
                          yaxis_title="Dice", hovermode="x unified")
        c2.plotly_chart(style_chart(fig), width="stretch")

    for region in ["val_block", "test_block"]:
        img = FIGURES / f"model_comparison_{target}_{region}.png"
        if img.exists():
            st.image(str(img), caption=f"Error maps, {region.replace('_', ' ')}", width="stretch")

    st.subheader("Check against Copernicus EMSR773 building damage")
    v = summary["emsr_validation"]
    c1, c2, c3 = st.columns(3)
    c1.metric("EMSR773 damaged buildings in the area", fmt(v["emsr_points_in_aoi"]))
    c2.metric("Flagged by us (affected or possibly)", f"{v['flagged_affected_or_possibly_pct']}%")
    c3.metric("Flagged as affected", f"{v['flagged_affected_pct']}%")
    st.dataframe(data["agreement"], width="stretch")
    st.markdown("**Buildings inside the EMSR773 flood extent (Flooded area + Flood trace)**")
    st.dataframe(data["extent_check"].rename(columns={
        "region": "Region", "buildings": "Buildings", "n": "N", "in_emsr_extent": "Inside EMSR extent",
        "recall_affected": "Recall (Affected)", "recall_affected_or_possibly": "Recall (Affected or possibly)",
        "precision_affected": "Precision (Affected)", "flagged_outside_extent": "Flagged outside extent"}),
        hide_index=True, width="stretch")
    st.caption("Rows: EMSR773 damage grade (photo-interpretation of very-high-resolution optical images). Columns: our "
               "class for the matched Catastro building. EMSR773 lists only damaged buildings, so this measures detection "
               "rate, not false alarms. EMSR773 flood polygons were also the training labels, so the two are not "
               "independent. The extent check uses each building's centre; the test block is the only region the "
               "model never saw during training or selection. EMSR773 places few homes inside its flood polygons, "
               "so residential precision is low: our residential counts are higher than the Copernicus extent implies.")

# ---------------------------------------------------------------- method
with tab_method:
    st.markdown(open(ROOT / "dashboard/method.md").read())
