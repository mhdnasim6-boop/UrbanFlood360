"""
UrbanFlood360 dashboard page: analyse a flood from images you upload.

The user uploads a Sentinel-1 image from before and one from after a flood (and optionally an area boundary and
building footprints). The page runs scripts/analyse_event.py, which applies the trained U-Net v2 models and the
building damage rules from the Valencia case study, and shows the flood map and the damage assessment.

Every run is saved in outputs/user_runs/<time>_<name>/ (inputs/, results, results.zip).
"""
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.results_view import EXAMPLE, RUNS, save_upload, show_results  # noqa: E402
from scripts.analyse_event import AnalysisError, analyse, selected  # noqa: E402

# ------------------------------------------------------------------ header
st.title("UrbanFlood360 · Analyse your own images")
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
show_results("No analyses yet. Upload images above, or switch on the Valencia example files, and press Run analysis.")
