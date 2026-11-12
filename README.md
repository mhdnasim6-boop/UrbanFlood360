# UrbanFlood360

AI-based urban flood mapping and building damage assessment from Sentinel-1 radar.
Case study: the Valencia DANA flood (29–30 October 2024) in Horta Sud: Alfafar, Catarroja, Massanassa, Paiporta,
Picanya and Sedaví (41.6 km²).

**Status:**
- **Flood maps:** U-Net flood mapping is trained and evaluated, with two maps: flood-affected ground and standing
  water.
- **Damage assessment:** done for 12,767 Catastro buildings, checked against Copernicus EMSR773.
- **Dashboard:** one simple page (`dashboard/simple.py`): upload a Sentinel-1 GeoTIFF from before and one from
  after the flood, press *Map the flood*, and get the flood map, the number and list of affected buildings, and
  downloads (flood map GeoTIFF, buildings CSV, everything as ZIP). Buildings come from OpenStreetMap for the image
  area, or from a building file you add. A *Valencia 2024 example* button runs the case study.
  More detailed pages are kept in `dashboard/` (download.py: automatic Sentinel-1 download; analyse.py: upload with
  all settings; valencia.py: case-study dashboard) and can be switched back on in `dashboard/app.py`.
- **Known limitation:** the current U-Net was trained on one Valencia image pair (orbit 103, SNAP sigma0). On the
  automatically downloaded Valencia pair (orbit 110, gamma0) its test-block Dice drops from 0.71 to 0.46 and it
  over-predicts flooding (`scripts/validate_on_valencia.py`). A model trained on UrbanSARFloods
  (`kaggle_kernel_v2/`) is meant to fix this.

Full write-up: [reports/unet_v2_flood_damage_report.md](reports/unet_v2_flood_damage_report.md).
The data-acquisition stage is documented in [docs/README_acquisition_stage.md](docs/README_acquisition_stage.md).

## Quick start

```bash
# dashboard (uses the results already in outputs/)
~/Anaconda/anaconda3/envs/urbanflood/bin/streamlit run dashboard/app.py

# rerun everything after training (~40 s), or everything including 4 training runs (~25-30 min on an M1)
SKIP_TRAINING=1 ./run_pipeline.sh
./run_pipeline.sh

# download + analyse a flood from the command line (same as the start page)
~/Anaconda/anaconda3/envs/urbanflood/bin/python scripts/fetch_event_data.py --place "Paiporta, Valencia, Spain" \
    --flood-date 2024-10-29 --analyse --output-dir outputs/user_runs/paiporta

# score a flood map of the Valencia area against Copernicus EMSR773
~/Anaconda/anaconda3/envs/urbanflood/bin/python scripts/validate_on_valencia.py outputs/user_runs/paiporta/flood_mask.tif

# analyse your own images from the command line (same as the upload page)
~/Anaconda/anaconda3/envs/urbanflood/bin/python scripts/analyse_event.py --pre before.tif --post after.tif \
    --aoi area.geojson --buildings buildings.gpkg --output-dir outputs/user_runs/my_event

# tests
~/Anaconda/anaconda3/envs/urbanflood/bin/python -m pytest tests -q
```

The `urbanflood` conda environment (Python 3.12, PyTorch with Apple MPS, rasterio, geopandas, Streamlit, folium,
plotly) was created for this work; the install command is in the report, §7. The download page also needs
`pip install pystac-client planetary-computer`.

## Results at a glance

| | |
|---|---|
| Flood-affected U-Net, held-out test block | Dice **0.71**, IoU 0.55, precision 0.65, recall 0.78 |
| Flood-affected U-Net, validation block | Dice 0.78 |
| Standing-water U-Net, validation block | Dice 0.76, on par with the earlier best repository U-Net (0.77) |
| Flood-affected area | 24.9 km², of which 13.6 km² inside the Copernicus-imaged part (EMSR773: 11.0 km²) |
| Buildings affected / possibly affected | **2,733** / 626 of 12,767 |
| Dwellings affected / estimated residents | 8,359 / ~20,900 (2.5 per dwelling, an assumption) |
| EMSR773-damaged buildings we flag | 91.6 % (affected or possibly) |

Why two flood maps? The radar image is from 31 October, two days after the peak. The towns then held mud
("flood trace"), not standing water: only 0.2 % of building pixels fall inside Copernicus's *Flooded area*, against
35 % inside *Flood trace*. A model of standing water alone detects only 3.4 % of the damaged buildings, so the
damage assessment uses the flood-affected model.

## Pipeline

```
Sentinel-1 6-band stack ─┐
EMSR773 labels ──────────┼─► U-Net v2 training (models/unet_v2/train.py)
                         │     ├─ target "water":    Flooded area
                         │     └─ target "affected": Flooded area + Flood trace
                         ├─► evaluation vs Otsu + earlier U-Nets (scripts/evaluate_flood_models.py)
                         ├─► full-area flood maps   (scripts/predict_flood_map.py)
Catastro buildings ──────┼─► building inventory     (scripts/prepare_buildings.py)
                         ├─► building impact + EMSR checks (scripts/assess_building_damage.py)
                         └─► dashboard assets → Streamlit dashboard (dashboard/app.py)
```

## Directory structure

```
UrbanFlood360/
├── data/
│   ├── raw/                         # original downloads (Sentinel-1, EMSR773, Catastro, DEM)
│   ├── processed/
│   │   ├── sar_features/            # 6-channel SAR stack (10 m, EPSG:25830)
│   │   ├── unet_dataset/            # standing-water labels + fixed patches (earlier stage)
│   │   ├── flood_affected_dataset/  # flood-affected labels (--trace-mode flood)
│   │   ├── predictions/unet_v2_{affected,water}/   # flood probability, mask, polygons
│   │   ├── buildings/               # Catastro building inventory for the AOI
│   │   └── damage/                  # buildings_impact.gpkg (+ EMSR773 damage points)
│   └── gis/aoi/                     # AOI and municipality boundaries
├── models/
│   ├── unet/                        # earlier repository U-Net baseline
│   └── unet_v2/                     # U-Net v2 code, configs, runs/, selected_model_*.json
├── scripts/                         # pipeline steps (see above) + earlier preprocessing scripts
├── dashboard/                       # Streamlit: app.py (router), download.py, analyse.py, valencia.py, results_view.py
├── outputs/
│   ├── tables/                      # model comparison, damage summaries, EMSR checks
│   ├── dashboard/                   # web-ready map layers
│   └── user_runs/                   # one folder per new-event analysis (inputs + results.zip)
├── reports/                         # stage reports and figures
├── tests/                           # pytest checks for U-Net v2 and damage logic
├── archive/                         # earlier U-Net experiments
└── run_pipeline.sh
```

## Key decisions

- **AOI:** dissolved union of six official OSM municipal boundaries, 41.63 km².
- **Sentinel-1 pair:** 19 Oct 2024 (before) and 31 Oct 2024 (after), relative orbit 103, ascending.
- **Labels:** EMSR773 AOI03 Grading v2 `observedEventA`. Areas outside the Copernicus image footprint (27 % of the
  AOI) and outside the AOI are ignored.
- **Split:** 4 × 4 spatial blocks: validation = block (1,2), test = block (2,1). Training windows stay more than 150 m
  from both.
- **Buildings:** Spanish Catastro INSPIRE. Floors come from BuildingPart. Residents = dwellings × 2.5 (assumption).

See `metadata/dataset_manifest.csv` for the dataset-by-dataset acquisition record.
