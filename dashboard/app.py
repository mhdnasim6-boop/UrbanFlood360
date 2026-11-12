"""
UrbanFlood360 dashboard.

    streamlit run dashboard/app.py

One simple page (dashboard/simple.py): upload a Sentinel-1 image from before and one from after a flood, and get
the flood map and the list of affected buildings.

The earlier, more detailed pages are kept in this folder and can be added back to the list below if needed:
download.py (download Sentinel-1 automatically), analyse.py (upload with all settings), valencia.py (case study).
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="UrbanFlood360 · Urban flood map and damage", layout="wide")
st.navigation([st.Page(HERE / "simple.py", title="UrbanFlood360", icon=":material/water:")], position="hidden").run()
