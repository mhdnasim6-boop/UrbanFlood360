"""UrbanFlood360 - Architecture B post-processing for the 31 Oct POST scene. Same logic as the PRE script."""
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np

AOI_PATH = "data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson"

JOBS = [
    {
        "src": "data/processed/sentinel1/post/20241031_R103/linear/S1_POST_20241031_R103_sigma0_linear.tif",
        "dst": "data/processed/sentinel1/post/20241031_R103/corrected/linear/S1_POST_20241031_R103_sigma0_linear.tif",
        "band_names": ["Sigma0_VH", "Sigma0_VV"],
    },
    {
        "src": "data/processed/sentinel1/post/20241031_R103/db/S1_POST_20241031_R103_sigma0_db.tif",
        "dst": "data/processed/sentinel1/post/20241031_R103/corrected/db/S1_POST_20241031_R103_sigma0_db.tif",
        "band_names": ["Sigma0_VH_db", "Sigma0_VV_db"],
    },
]

aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:25830")
aoi_geom = aoi.geometry.iloc[0]

for job in JOBS:
    print(f"\n=== {job['src']} -> {job['dst']} ===")
    with rasterio.open(job["src"]) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        data = src.read().astype("float32")
        crs_before, transform_before = src.crs, src.transform
        width_before, height_before = src.width, src.height

    outside = geometry_mask([aoi_geom], out_shape=shape, transform=transform, invert=False)
    n_out_before = outside.sum()
    n_out_valid_before = [int(np.sum((data[b][outside] != 0) & ~np.isnan(data[b][outside]))) for b in range(data.shape[0])]
    print(f"BEFORE fix: pixels outside true AOI = {n_out_before}, non-zero/non-NaN among them per band = {n_out_valid_before}")

    for b in range(data.shape[0]):
        data[b][outside] = np.nan

    n_out_valid_after = [int(np.sum((data[b][outside] != 0) & ~np.isnan(data[b][outside]))) for b in range(data.shape[0])]
    inside = ~outside
    n_inside_nan = [int(np.sum(np.isnan(data[b][inside]))) for b in range(data.shape[0])]
    print(f"AFTER fix:  non-zero/non-NaN among outside-AOI pixels per band = {n_out_valid_after} (must be 0)")
    print(f"Inside-AOI pixels accidentally turned NaN per band = {n_inside_nan} (must be 0)")

    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(job["dst"], "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(job["band_names"], start=1):
            dst.set_band_description(i, name)
            dst.update_tags(i, BAND_NAME=name)

    with rasterio.open(job["dst"]) as check:
        assert check.crs == crs_before
        assert check.transform == transform_before
        assert (check.width, check.height) == (width_before, height_before)
        print(f"Grid identity preserved: CRS={check.crs}, dims={check.width}x{check.height} -- CONFIRMED")
        print(f"NoData: {check.nodata}, Band descriptions: {check.descriptions}")
