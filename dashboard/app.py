"""
UrbanFlood360 dashboard.

    streamlit run dashboard/app.py

Pages
  Analyse a new flood event   upload before/after Sentinel-1 images (+ buildings) -> flood map + damage assessment
  Valencia case study         results of the Valencia DANA flood (29-30 Oct 2024) from run_pipeline.sh
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="UrbanFlood360 · Urban flood mapping and damage assessment", layout="wide")

nav = st.navigation([
    st.Page(HERE / "analyse.py", title="Analyse a new flood event", icon=":material/upload_file:", default=True),
    st.Page(HERE / "valencia.py", title="Valencia case study (2024)", icon=":material/map:"),
])
nav.run()
