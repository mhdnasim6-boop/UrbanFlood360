# Reference / Ground-Truth Data — UrbanFlood360

This directory documents (not duplicates) which files under `data/raw/` constitute the
project's flood-extent and building-damage reference data. Raw files are never modified;
this is a pointer/documentation layer only.

## Primary flood-extent reference (training label)

**File:** `data/raw/emsr773/extracted/GRA_v2/EMSR773_AOI03_GRA_PRODUCT_observedEventA_v2.shp`
(+ matching `.geojson`, `.prj`, `.dbf`, `.xml`, `.sld`)

- Source: Copernicus EMS EMSR773, AOI03 "Horta Sud", **Grading product v2** (corrected 2024-11-06)
- CRS: EPSG:4326 (WGS84 geographic)
- Feature count: 1528 polygons
- Field `notation`: **"Flooded area"** (1002 features) / **"Flood trace"** (526 features)
- **Why this file and not the standalone Delineation (DEL v1) package**: DEL v1's own
  `observedEventA` (617 features: 306 Flooded area + 311 Flood trace) was confirmed via the
  product's own version notes to be an **incomplete** delineation ("not all flood traces were
  delineated"). The GRA v2 correction was never re-issued as a standalone DEL v2 — the
  corrected extent only exists inside the Grading package. DEL v1 is kept in `data/raw/` for
  comparison only, not as the training label.
- **Not independent of the model input**: this product is derived from VHR optical imagery
  (WorldView-2/GeoEye-1, 31 Oct 2024), not Sentinel-1 SAR — so it is methodologically distinct
  from the SAR input, but it is still the single source used for training labels. See the
  ground-truth strategy note below.

## Primary building-damage reference

**File:** `data/raw/emsr773/extracted/GRA_v2/EMSR773_AOI03_GRA_PRODUCT_builtUpP_v2.shp`

- Geometry: **Points** (building centroids), not building footprint polygons — 3906 features
- Field `damage_gra`: **Damaged** (3258) / **Destroyed** (37) / **Possibly damaged** (611)
- **"Possibly damaged" is NOT confirmed structural damage** — per the product's own
  documentation this class means "presence of damage proxies and proximity with
  destroyed/damaged asset," i.e. inferred by nearness, not directly observed on the asset.
  Do not treat it as equivalent to "Damaged" in downstream analysis without flagging the
  confidence difference.
- To use with Catastro building footprints, these points require a **spatial join** (nearest
  building footprint), not an attribute join — there is no shared building ID between EMSR773
  and Catastro.

## What is NOT available as ground truth

No fully independent (different-methodology, different-source) flood-extent dataset exists
for this event — see project verification notes. The best candidate for an independent
validation subset is manual digitizing from `PNOA_POST_20241111` (IGN/CNIG 5cm aerial
orthophoto, independently captured, not yet downloaded — see manifest) for one or two held-out
towns. This has NOT been done yet; it is a planned future step, not a current dataset.

## Population — modeled vs. observed

Any population-exposure figure computed later (WorldPop/GHS-POP overlaid with the flood mask)
is a **modeled estimate**, not an observed count of affected/displaced/rescued people. Do not
conflate the two in any output.
