# UrbanFlood360

AI-based urban flood mapping, damage assessment and rescue response system.
Case study: Valencia DANA flood (29–30 October 2024), Horta Sud (Paiporta, Catarroja,
Alfafar, Picanya, Sedaví, Massanassa).

**Status: Dataset acquisition (core datasets) — complete. No preprocessing, no ML dataset,
no U-Net training has been performed.**

## Directory structure

```
UrbanFlood360/
├── data/
│   ├── raw/              # Original downloaded files, never modified. One subfolder per source.
│   │   ├── sentinel1/     # 4 GRD SAFE.zip products (2 primary + 2 secondary pair)
│   │   ├── sentinel2/     # (empty — deferred)
│   │   ├── emsr773/       # EMSR773 AOI03 GRA v2 + DEL v1 zips, + extracted/ subfolder
│   │   ├── catastro/      # 6 municipality INSPIRE building zips, + extracted/ subfolder
│   │   ├── osm/           # (empty — deferred)
│   │   ├── worldpop/      # (empty — deferred)
│   │   ├── ghs_pop/       # (empty — deferred)
│   │   ├── ine/           # (empty — deferred)
│   │   ├── gva/           # (empty — deferred)
│   │   ├── pnoa/          # (empty — deferred, access method unresolved)
│   │   └── ign_roads/     # (empty — deferred)
│   ├── reference/        # Ground-truth documentation (pointers into raw/, not copies)
│   ├── processed/        # (empty — future stage: ML-ready rasters/patches)
│   ├── interim/          # (empty — future stage: intermediate preprocessing outputs)
│   └── gis/
│       ├── aoi/           # AOI_HORTA_SUD_6MUN.geojson (analysis mask) + municipalities/ (6 individual boundaries)
│       └── vector_layers/ # (empty — future stage: derived layers, e.g. roads clipped to AOI)
├── scripts/               # (empty — future stage)
├── notebooks/             # (empty — future stage)
├── outputs/
│   ├── figures/, maps/, tables/  # (empty — future stage)
├── docs/                  # (empty — project documentation)
└── metadata/
    ├── dataset_manifest.csv    # Full acquisition manifest, one row per dataset ID
    ├── checksums/               # SHA256 checksums for all downloaded raw files
    └── logs/                    # (empty — future stage)
```

## Key decisions locked at acquisition stage

- **AOI**: dissolved union of 6 official OSM municipal boundaries (NOT a bounding box), 41.63 km².
  See `data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson`.
- **Sentinel-1 primary pair**: 19 Oct 2024 (pre) / 31 Oct 2024 (post), relative orbit 103, ascending.
- **Sentinel-1 secondary pair**: 20 Oct 2024 / 1 Nov 2024, relative orbit 110, descending.
- **Flood-extent ground truth**: EMSR773 AOI03 Grading v2's `observedEventA` layer (NOT the
  standalone Delineation v1 — see `data/reference/README.md` for why).
- **Building damage reference**: EMSR773 AOI03 Grading v2 `builtUpP` layer (points, 3-class
  `damage_gra` field). "Possibly damaged" is proximity-inferred, not confirmed.
- **Buildings**: Spanish Catastro INSPIRE, per-municipality GML, EPSG:25830. Floor counts only
  populated at BuildingPart level, not Building level — aggregation required later.

See `metadata/dataset_manifest.csv` for the complete dataset-by-dataset status and source record.
