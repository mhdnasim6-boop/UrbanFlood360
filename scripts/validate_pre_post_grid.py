#!/usr/bin/env python3
"""
UrbanFlood360 - permanent QC component.

Validates that two GeoTIFF rasters (typically a pre/post Sentinel-1 pair)
share an EXACTLY identical raster grid: CRS, width, height, pixel size,
affine transform, origin, and bounds.

Usage:
    python validate_pre_post_grid.py <pre_raster> <post_raster>

Exit codes:
    0 = exact grid match
    1 = mismatch (details printed to stdout)
"""
import sys
import rasterio


def validate_grid(pre_path: str, post_path: str) -> bool:
    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        checks = []

        checks.append(("CRS", pre.crs == post.crs, pre.crs, post.crs))
        checks.append(("Width", pre.width == post.width, pre.width, post.width))
        checks.append(("Height", pre.height == post.height, pre.height, post.height))
        checks.append(("Pixel width", pre.transform.a == post.transform.a, pre.transform.a, post.transform.a))
        checks.append(("Pixel height", pre.transform.e == post.transform.e, pre.transform.e, post.transform.e))
        checks.append(("Transform", pre.transform == post.transform, pre.transform, post.transform))
        checks.append((
            "Origin",
            (pre.transform.c, pre.transform.f) == (post.transform.c, post.transform.f),
            (pre.transform.c, pre.transform.f),
            (post.transform.c, post.transform.f),
        ))
        checks.append(("Bounds", tuple(pre.bounds) == tuple(post.bounds), tuple(pre.bounds), tuple(post.bounds)))

        overall = all(c[1] for c in checks)

        print(f"{'Property':<15}{'Status':<8}{'PRE':<45}{'POST'}")
        for name, ok, pre_val, post_val in checks:
            status = "PASS" if ok else "FAIL"
            print(f"{name:<15}{status:<8}{str(pre_val):<45}{str(post_val)}")

        print()
        print(f"OVERALL GRID MATCH: {'PASS' if overall else 'FAIL'}")
        return overall


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_pre_post_grid.py <pre_raster> <post_raster>")
        sys.exit(2)

    ok = validate_grid(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
