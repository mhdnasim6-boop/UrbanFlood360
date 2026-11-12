"""
Data handling for U-Net v2: full-raster loading, the spatial-block split as
pixel masks, random training windows, and sliding-window inference.

Why random windows instead of the fixed .npy patches
----------------------------------------------------
The fixed patch set (data/processed/unet_dataset/patches) keeps only windows
with >=80% labelled pixels, which leaves 301 training patches and discards
most labelled pixels. The loss already ignores label 255, so the purity filter
is not needed for correctness. Here every epoch draws fresh 64x64 windows from
anywhere inside the TRAIN region, so all labelled train pixels are used.

Leakage control (same split and buffer as the existing dataset)
---------------------------------------------------------------
Same 4x4 block grid and assignment as dataset_config.yaml: val = block(1,2),
test = block(2,1), all other blocks = train. A training window must lie
entirely inside the train region AND more than `buffer_m` away from any
val/test block, so the model never sees val/test pixels - not even as input
context - during training.

Evaluation is done on every labelled pixel of the val/test blocks (via
full-raster inference), not just the 14/13 fixed patches.
"""
import json

import numpy as np
import rasterio
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

IGNORE_LABEL = 255
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}

# The two segmentation targets (label rasters on the SAR grid, paths relative to the project root).
TARGETS = {
    "water": dict(label="data/processed/unet_dataset/labels/flood_reference.tif",
                  title="standing water (EMSR773 'Flooded area')"),
    "affected": dict(label="data/processed/flood_affected_dataset/labels/flood_reference.tif",
                     title="flood-affected ground (EMSR773 'Flooded area' + 'Flood trace')"),
}


def checkpoint_target(ckpt):
    """Target a U-Net v2 checkpoint was trained on, from the label path in its config."""
    label = ckpt["config"]["data"]["label"]
    return next((name for name, t in TARGETS.items() if t["label"] == label), "other")


def load_rasters(sar_path, label_path):
    with rasterio.open(sar_path) as src:
        sar = src.read().astype("float32")  # (C,H,W), NaN = invalid
        profile = src.profile
        channels = list(src.descriptions)
    with rasterio.open(label_path) as src:
        label = src.read(1)
        assert src.transform == profile["transform"], "label/SAR grid mismatch"
    return sar, label, profile, channels


def block_split_map(shape, grid=(4, 4), assignment=None):
    """(H,W) uint8 map of split codes, identical block edges to create_unet_patches.py."""
    assignment = assignment or {(1, 2): "val", (2, 1): "test"}
    H, W = shape
    rows = np.linspace(0, H, grid[0] + 1).astype(int)
    cols = np.linspace(0, W, grid[1] + 1).astype(int)
    split = np.zeros(shape, dtype="uint8")
    for (i, j), name in assignment.items():
        split[rows[i]:rows[i + 1], cols[j]:cols[j + 1]] = SPLIT_CODES[name]
    return split


def train_allowed_mask(split, buffer_m, pixel_size):
    """Train pixels farther than buffer_m from any val/test pixel.

    Distances are pixel-centre to pixel-centre; the 2-pixel margin makes the
    edge-to-edge distance exceed buffer_m even diagonally.
    """
    dist_px = distance_transform_edt(split == SPLIT_CODES["train"])
    return (split == SPLIT_CODES["train"]) & (dist_px * pixel_size > buffer_m + 2 * pixel_size)


def fit_normalization(sar, mask, channels):
    stats = {"method_active": "mean_std", "fitted_from": "train_region_pixels",
             "channels": channels, "per_channel": {}}
    for c, name in enumerate(channels):
        v = sar[c][mask & ~np.isnan(sar[c])]
        stats["per_channel"][name] = {"mean": float(v.mean()), "std": float(v.std())}
    return stats


def normalize(sar, stats):
    """z-score per channel, then NaN -> 0 (the channel mean, a neutral value)."""
    out = np.empty_like(sar)
    for c, name in enumerate(stats["channels"]):
        s = stats["per_channel"][name]
        out[c] = (sar[c] - s["mean"]) / (s["std"] + 1e-8)
    return np.nan_to_num(out, nan=0.0).astype("float32")


def _window_sums(mask, size):
    """Sum of `mask` over every size x size window, indexed by top-left corner."""
    ii = np.pad(mask.astype("int64").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]


def train_window_positions(allowed, label, size, min_labeled):
    """Top-left corners of windows fully inside `allowed` with >= min_labeled labelled pixels."""
    n = size * size
    inside = _window_sums(allowed, size) == n
    labelled = _window_sums(label != IGNORE_LABEL, size) >= min_labeled * n
    rows, cols = np.nonzero(inside & labelled)
    return np.stack([rows, cols], axis=1)


class RandomWindowDataset(Dataset):
    """Draws a random training window per item; `length` items = one epoch."""

    def __init__(self, image, label, positions, size, length, augment=False, seed=0):
        self.image = image          # (C,H,W) normalized float32
        self.label = label          # (H,W) uint8 {0,1,255}
        self.positions = positions
        self.size = size
        self.length = length
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        r, c = self.positions[self.rng.integers(len(self.positions))]
        x = self.image[:, r:r + self.size, c:c + self.size]
        y = self.label[r:r + self.size, c:c + self.size]
        if self.augment:
            # One random draw shared by image and label -> always identical transform.
            k = int(self.rng.integers(4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)).long().unsqueeze(0))


def _taper(size, overlap):
    w = np.ones(size, dtype="float32")
    if overlap > 0:
        ramp = np.linspace(0.05, 1.0, overlap, dtype="float32")
        w[:overlap], w[-overlap:] = ramp, ramp[::-1]
    return np.outer(w, w)


@torch.no_grad()
def predict_raster(model, image, device, tile=128, overlap=32, batch_size=16,
                   outputs_logits=True, valid=None):
    """Sliding-window flood probability for a full (C,H,W) normalized raster.

    Overlapping tiles are blended with a tapered weight so tile seams vanish.
    Returns (H,W) float32 probabilities, NaN where `valid` is False.
    """
    model.eval()
    C, H, W = image.shape
    stride = tile - overlap
    Hp = max(tile, int(np.ceil((H - tile) / stride)) * stride + tile)
    Wp = max(tile, int(np.ceil((W - tile) / stride)) * stride + tile)
    padded = np.zeros((C, Hp, Wp), dtype="float32")
    padded[:, :H, :W] = image

    weight = _taper(tile, overlap)
    acc = np.zeros((Hp, Wp), dtype="float32")
    wsum = np.zeros((Hp, Wp), dtype="float32")
    corners = [(r, c) for r in range(0, Hp - tile + 1, stride) for c in range(0, Wp - tile + 1, stride)]
    for i in range(0, len(corners), batch_size):
        chunk = corners[i:i + batch_size]
        x = torch.from_numpy(np.stack([padded[:, r:r + tile, c:c + tile] for r, c in chunk])).to(device)
        out = model(x)
        prob = (torch.sigmoid(out) if outputs_logits else out)[:, 0].float().cpu().numpy()
        for (r, c), p in zip(chunk, prob):
            acc[r:r + tile, c:c + tile] += p * weight
            wsum[r:r + tile, c:c + tile] += weight
    prob = (acc / np.maximum(wsum, 1e-6))[:H, :W]
    if valid is not None:
        prob = np.where(valid, prob, np.nan)
    return prob.astype("float32")


def confusion(prob, label, region, threshold):
    """Pixel-pooled TP/TN/FP/FN over labelled pixels inside `region` (prob >= threshold = flood)."""
    m = region & (label != IGNORE_LABEL) & ~np.isnan(prob)
    pred = prob[m] >= threshold
    true = label[m] == 1
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum()); tn = int((~pred & ~true).sum())
    eps = 1e-9
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                dice=2 * tp / (2 * tp + fp + fn + eps), iou=tp / (tp + fp + fn + eps),
                precision=tp / (tp + fp + eps), recall=tp / (tp + fn + eps),
                specificity=tn / (tn + fp + eps), accuracy=(tp + tn) / (tp + tn + fp + fn + eps))


SELECTABLE_THRESHOLDS = (0.3, 0.7)


def threshold_sweep(prob, label, region, thresholds=None, selectable=SELECTABLE_THRESHOLDS):
    """Metrics for each threshold; `best` = highest Dice among thresholds inside `selectable`.

    Thresholds far from 0.5 tend to fit the quirks of the single small val
    block and transfer badly to the rest of the AOI, so the choice is limited
    to a moderate range (all thresholds are still reported).
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = [dict(threshold=float(t), **confusion(prob, label, region, t)) for t in thresholds]
    lo, hi = selectable
    best = max((r for r in rows if lo - 1e-9 <= r["threshold"] <= hi + 1e-9), key=lambda r: r["dice"])
    return rows, best


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
