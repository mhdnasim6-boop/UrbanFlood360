"""Fast checks for U-Net v2 and the damage logic.

    ~/Anaconda/anaconda3/envs/urbanflood/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet_v2.data import (  # noqa: E402
    SPLIT_CODES, RandomWindowDataset, block_split_map, confusion, predict_raster,
    train_allowed_mask, train_window_positions,
)
from models.unet_v2.losses import masked_bce_dice  # noqa: E402
from models.unet_v2.model import UNet  # noqa: E402
from scripts.assess_building_damage import building_flood_stats, classify  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402


def test_unet_output_shape():
    out = UNet(6, 1, base=8)(torch.zeros(2, 6, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_loss_ignores_255():
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.full((1, 1, 4, 4), 255)
    target[0, 0, 0, 0] = 1
    loss = masked_bce_dice(logits, target)
    loss.backward()
    grad = logits.grad[0, 0]
    assert grad[0, 0] != 0 and torch.count_nonzero(grad) == 1


def test_training_windows_respect_buffer():
    shape = (951, 1120)
    split = block_split_map(shape)
    allowed = train_allowed_mask(split, buffer_m=150, pixel_size=10)
    label = np.zeros(shape, dtype="uint8")
    pos = train_window_positions(allowed, label, 64, 0.1)
    ys, xs = np.nonzero(split != SPLIT_CODES["train"])
    held_out = np.stack([ys, xs], 1)[::25]
    rng = np.random.default_rng(0)
    for r, c in pos[rng.choice(len(pos), 300, replace=False)]:
        # nearest held-out pixel to the window rectangle, edge to edge in metres
        dy = np.maximum(0, np.maximum(r - held_out[:, 0], held_out[:, 0] - (r + 63)))
        dx = np.maximum(0, np.maximum(c - held_out[:, 1], held_out[:, 1] - (c + 63)))
        assert (np.hypot(dy, dx) * 10 - 10).min() >= 150


def test_augmentation_keeps_image_and_label_aligned():
    img = np.zeros((6, 64, 64), dtype="float32")
    lab = np.zeros((64, 64), dtype="uint8")
    img[:, 3, 50] = 1.0
    lab[3, 50] = 1
    ds = RandomWindowDataset(img, lab, np.array([[0, 0]]), 64, 50, augment=True, seed=1)
    for i in range(50):
        x, y = ds[i]
        assert torch.equal((x[0] == 1), (y[0] == 1))


def test_predict_raster_covers_every_pixel():
    model = UNet(6, 1, base=8)
    prob = predict_raster(model, np.zeros((6, 100, 150), dtype="float32"), torch.device("cpu"), tile=64, overlap=16)
    assert prob.shape == (100, 150) and not np.isnan(prob).any()


def test_confusion_counts():
    prob = np.array([[0.9, 0.1], [0.8, np.nan]])
    label = np.array([[1, 1], [0, 1]], dtype="uint8")
    c = confusion(prob, label, np.ones((2, 2), bool), 0.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 0)


def test_building_stats_and_classes():
    transform = from_origin(0, 100, 10, 10)          # 10 x 10 grid of 10 m pixels
    mask = np.zeros((10, 10), dtype="uint8")
    mask[:, :5] = 1                                   # western half flooded
    prob = mask.astype("float32")
    west, east, edge = box(5, 55, 15, 65), box(75, 55, 85, 65), box(41, 55, 59, 65)
    stats = building_flood_stats([west, east, edge], prob, mask, transform)
    assert stats[0, 0] == 1.0 and stats[1, 0] == 0.0 and 0 < stats[2, 0] < 1
    assert list(classify(np.array([1.0, 0.2, 0.0, np.nan]), 0.5, 0.1)) == \
        ["Affected", "Possibly affected", "Not affected", "No data"]


def test_analyse_event_end_to_end(tmp_path):
    """New-event analysis on tiny synthetic linear lat/lon images: converts, reprojects, maps, assesses buildings."""
    import geopandas as gpd
    import rasterio
    from scripts.analyse_event import analyse

    rng = np.random.default_rng(0)
    t = from_origin(-0.40, 39.43, 0.0001, 0.0001)      # lat/lon, ~10 m pixels
    for name in ("pre", "post"):
        data = (10 ** (rng.normal(-12, 3, (2, 80, 80)) / 10)).astype("float32")  # linear Sigma0
        with rasterio.open(tmp_path / f"{name}.tif", "w", driver="GTiff", height=80, width=80, count=2,
                           dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)
            dst.descriptions = ("Sigma0_VH", "Sigma0_VV")
    b = gpd.GeoDataFrame({"numberOfDwellings": [4, 0]},
                         geometry=[box(-0.399, 39.425, -0.3985, 39.4255), box(-0.396, 39.428, -0.3955, 39.4285)],
                         crs="EPSG:4326")
    b.to_file(tmp_path / "b.geojson", driver="GeoJSON")
    s = analyse(tmp_path / "pre.tif", tmp_path / "post.tif", tmp_path / "out", buildings=tmp_path / "b.geojson",
                progress=lambda f, t: None)
    assert s["grid"]["crs"].startswith("EPSG:326") and s["grid"]["pixel_m"] == 10.0
    assert any("converted to dB" in n for n in s["notes"])
    assert s["buildings"]["total"] == 2 and sum(s["buildings"]["by_class"].values()) == 2
    for f in ("flood_mask.tif", "flood_probability.tif", "buildings_impact.geojson", "results.zip"):
        assert (tmp_path / "out" / f).exists()
