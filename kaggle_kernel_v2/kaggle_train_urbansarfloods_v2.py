#!/usr/bin/env python3
"""
UrbanFlood360 - U-Net trained on UrbanSARFloods with the paper's settings (runs on a Kaggle GPU).

Paper: J. Zhao, Z. Xiong, X. X. Zhu, "UrbanSARFloods: Sentinel-1 SLC-based benchmark dataset for urban and
open-area flood mapping", CVPR Workshops 2024, pp. 419-429, doi:10.1109/CVPRW63382.2024.00047.
Data: https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1 (CC-BY-4.0)

The authors released no training code ("All baseline models were trained using standard segmentation pipelines
implemented in the segmentation_models_pytorch (SMP) library ... training scripts are not included", repo README).
This script rebuilds that pipeline from Section 4.1 of the paper.

From the paper (Sec. 4.1)                           Here
  SMP Unet                                            smp.Unet (encoder not stated -> SMP default resnet34)
  ImageNet-pretrained or from scratch                 ENCODER_WEIGHTS = "imagenet" (pretrained scored better on FO)
  Weighted cross-entropy, 3 classes (weights n/s)     inverse-sqrt class frequency from the training chips
  Adam, lr 1e-5, weight decay 1e-4                    same
  batch size 12, 100 epochs                           same (stops early only if Kaggle's 12 h limit is near)
  512x512 chips randomly cropped to 256x256           same
  random h/v flips + 90/180/270 rotations             same, one draw shared by image and label
  input: 8 bands (intensity + coherence)              CHANGED: 6 bands, see below
  normalisation: not stated                           z-score per channel, fitted on training chips
  model selection: not stated                         best validation mean F1 of the two flood classes

Input change: the dashboard works with Sentinel-1 GRD images, which have no coherence. So the model gets the same
six bands as UrbanFlood360's U-Net v2: VV/VH before, VV/VH after, and the change (after - before) in dB. The
four intensity bands are found from the data (dB intensity is negative, coherence lies in 0..1), because the
paper does not give the band index order. The detection is printed and saved in output/band_check.json.

Labels: 0 = not flooded, 1 = flooded open area (FO), 2 = flooded urban area (FU); anything else and pixels
without SAR data are ignored. The checkpoint also gives P(flood) = P(FO) + P(FU) for the dashboard.

Outputs (/kaggle/working/output): best_model.pt, last_model.pt, history.csv, metrics.json, band_check.json
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    import rasterio
except ImportError:
    pip("rasterio")
    import rasterio
try:
    import segmentation_models_pytorch as smp
except ImportError:
    pip("segmentation-models-pytorch")
    import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------ settings (paper, Sec. 4.1)
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
LR = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 12
EPOCHS = 100
CROP = 256
N_CLASSES = 3
CLASS_NAMES = ["NF", "FO", "FU"]
IGNORE = 255
SEED = 42
TIME_BUDGET_S = 11.0 * 3600          # Kaggle GPU sessions stop at 12 h
CHANNELS = ["VV_pre_dB", "VH_pre_dB", "VV_post_dB", "VH_post_dB", "VV_change_dB", "VH_change_dB"]

TARBALL_URL = "https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/main/urban_sar_floods.tar.gz"
DATA_ROOT = Path(os.environ.get("USF_DATA_ROOT", "/kaggle/temp/urban_sar_floods"))
CACHE = Path(os.environ.get("USF_CACHE", "/kaggle/temp/usf_cache"))
OUT_DIR = Path(os.environ.get("USF_OUT", "/kaggle/working/output"))
SMOKE = os.environ.get("USF_SMOKE") == "1"   # tiny local test run


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data
def download():
    if DATA_ROOT.exists() and any(DATA_ROOT.iterdir()):
        log(f"{DATA_ROOT} already populated")
        return
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    log("Streaming + extracting the training/validation tarball (~38 GB)")
    ret = subprocess.run(f'curl -sL "{TARBALL_URL}" | tar xz -C "{DATA_ROOT.parent}"', shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"download failed ({ret.returncode})")


def split_pairs(list_name):
    pairs = []
    for line in open(DATA_ROOT / list_name):
        rel = line.strip()
        while rel.startswith("../"):
            rel = rel[3:]
        if not rel:
            continue
        gt = DATA_ROOT / rel
        sar = Path(str(gt).replace("/GT/", "/SAR/").replace("_GT.tif", "_SAR.tif"))
        if gt.exists() and sar.exists():
            pairs.append((sar, gt))
    log(f"{list_name}: {len(pairs)} SAR/GT pairs")
    return pairs


def detect_bands(pairs, n=60):
    """Indices (0-based) of VV_pre, VH_pre, VV_post, VH_post among the 8 bands, from the data."""
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
    with rasterio.open(pairs[0][0]) as s:
        desc, count = list(s.descriptions), s.count
    vals = [[] for _ in range(count)]
    for i in pick:
        with rasterio.open(pairs[i][0]) as s:
            a = s.read().astype("float32")
        for b in range(count):
            v = a[b][np.isfinite(a[b])]
            if v.size:
                vals[b].append(np.median(v))
    med = [float(np.median(v)) if v else float("nan") for v in vals]
    p01 = [float(np.mean([(0 <= x <= 1) for x in v])) if v else 0.0 for v in vals]
    log(f"band descriptions: {desc}")
    log("band medians: " + ", ".join(f"b{b + 1}={m:.2f}" for b, m in enumerate(med)))

    named = {}
    for b, d in enumerate(desc):
        d = (d or "").upper()
        if "COH" in d:
            continue
        for key, pol, when in [("VV_pre", "VV", "PRE"), ("VH_pre", "VH", "PRE"),
                               ("VV_post", "VV", "POST"), ("VH_post", "VH", "POST")]:
            if pol in d and when in d:
                named[key] = b
    if len(named) == 4:
        order, how = [named[k] for k in ("VV_pre", "VH_pre", "VV_post", "VH_post")], "band descriptions"
    else:
        intensity = [b for b in range(count) if med[b] < -1 and p01[b] < 0.5]   # dB: negative; coherence: 0..1
        if len(intensity) != 4:
            raise RuntimeError(f"Expected 4 intensity (dB) bands, found {intensity} from medians {med}")
        pre, post = intensity[:2], intensity[2:]                    # paper lists pre-event intensity first
        vv_vh = lambda pair: sorted(pair, key=lambda b: -med[b])    # VV backscatter is higher than VH  # noqa: E731
        order, how = vv_vh(pre) + vv_vh(post), "value ranges (dB < 0, coherence in 0..1)"
    check = dict(method=how, order_0based=order, descriptions=desc, medians=med,
                 mapping={k: f"band {b + 1}" for k, b in zip(CHANNELS[:4], order)})
    log(f"intensity bands by {how}: {check['mapping']}")
    return order, check


def build_cache(pairs, order, name):
    """float16 (N,4,512,512) intensity + uint8 (N,512,512) labels on local disk, read once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    xs, ys = CACHE / f"{name}_x.npy", CACHE / f"{name}_y.npy"
    if xs.exists() and ys.exists():
        return np.load(xs, mmap_mode="r"), np.load(ys, mmap_mode="r")
    with rasterio.open(pairs[0][0]) as s:
        H, W = s.height, s.width
    X = np.lib.format.open_memmap(xs, "w+", np.float16, (len(pairs), 4, H, W))
    Y = np.lib.format.open_memmap(ys, "w+", np.uint8, (len(pairs), H, W))
    t0 = time.time()
    for i, (sar, gt) in enumerate(pairs):
        with rasterio.open(sar) as s:
            a = s.read([b + 1 for b in order]).astype("float32")
        with rasterio.open(gt) as s:
            g = s.read(1)
        lab = np.where(np.isin(g, (0, 1, 2)), g, IGNORE).astype("uint8")
        lab[~np.all(np.isfinite(a), axis=0)] = IGNORE
        X[i], Y[i] = a, lab
        if i % 500 == 0:
            log(f"caching {name}: {i}/{len(pairs)} ({time.time() - t0:.0f}s)")
    X.flush(); Y.flush()
    return np.load(xs, mmap_mode="r"), np.load(ys, mmap_mode="r")


def six_bands(x4):
    """(4,H,W) VV_pre, VH_pre, VV_post, VH_post -> (6,H,W) with the two change bands."""
    return np.concatenate([x4, x4[2:4] - x4[0:2]], axis=0)


def fit_stats(X, n=300):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(X), min(n, len(X)), replace=False))
    sums = np.zeros(6); sq = np.zeros(6); cnt = 0
    for i in idx:
        x = six_bands(np.asarray(X[i], dtype="float32"))[:, ::4, ::4]
        ok = np.all(np.isfinite(x), axis=0)
        v = x[:, ok].astype("float64")
        sums += v.sum(1); sq += (v ** 2).sum(1); cnt += v.shape[1]
    mean = sums / cnt
    std = np.sqrt(sq / cnt - mean ** 2)
    return {"method_active": "mean_std", "fitted_from": "UrbanSARFloods train chips", "channels": CHANNELS,
            "per_channel": {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(CHANNELS, mean, std)}}


def class_weights(Y, n=600):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(Y), min(n, len(Y)), replace=False))
    counts = np.zeros(N_CLASSES)
    for i in idx:
        y = np.asarray(Y[i])
        counts += np.bincount(y[y != IGNORE].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    freq = counts / counts.sum()
    w = 1 / np.sqrt(np.maximum(freq, 1e-6))
    w = w / w[0]
    log(f"class pixel shares {dict(zip(CLASS_NAMES, freq.round(4)))} -> weights {dict(zip(CLASS_NAMES, w.round(2)))}")
    return torch.tensor(w, dtype=torch.float32), freq


class Chips(Dataset):
    def __init__(self, X, Y, stats, train):
        self.X, self.Y, self.train = X, Y, train
        self.mean = np.array([stats["per_channel"][c]["mean"] for c in CHANNELS], "float32")[:, None, None]
        self.std = np.array([stats["per_channel"][c]["std"] for c in CHANNELS], "float32")[:, None, None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = six_bands(np.asarray(self.X[i], dtype="float32"))
        y = np.asarray(self.Y[i]).astype("int64")
        if self.train:
            H, W = y.shape
            r, c = np.random.randint(0, H - CROP + 1), np.random.randint(0, W - CROP + 1)
            x, y = x[:, r:r + CROP, c:c + CROP], y[r:r + CROP, c:c + CROP]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
        x = np.nan_to_num((x - self.mean) / (self.std + 1e-8), nan=0.0)
        return torch.from_numpy(np.ascontiguousarray(x, "float32")), torch.from_numpy(np.ascontiguousarray(y))


# ------------------------------------------------------------------ training
def confusion(pred, y, cm):
    m = y != IGNORE
    idx = y[m] * N_CLASSES + pred[m]
    cm += torch.bincount(idx, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES).cpu().numpy()


def scores(cm):
    out = {}
    for k, n in enumerate(CLASS_NAMES):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        out[n] = dict(precision=float(p), recall=float(r), f1=float(2 * p * r / max(p + r, 1e-9)))
    # binary flood (FO or FU) vs not - what the dashboard uses
    tp = cm[1:, 1:].sum(); fp = cm[0, 1:].sum(); fn = cm[1:, 0].sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    out["flood_any"] = dict(precision=float(p), recall=float(r), f1=float(2 * p * r / max(p + r, 1e-9)))
    out["mean_flood_f1"] = (out["FO"]["f1"] + out["FU"]["f1"]) / 2
    return out


def run_epoch(model, loader, loss_fn, opt, device, train):
    model.train(train)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype="int64")
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += loss.item(); n += 1
            confusion(logits.argmax(1), y, cm)
    return total / max(n, 1), scores(cm), cm


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if not SMOKE:
        download()
    train_pairs, val_pairs = split_pairs("Train_dataset.txt"), split_pairs("Valid_dataset.txt")
    order, check = detect_bands(train_pairs)
    json.dump(check, open(OUT_DIR / "band_check.json", "w"), indent=2)
    Xtr, Ytr = build_cache(train_pairs, order, "train")
    Xva, Yva = build_cache(val_pairs, order, "val")
    stats = fit_stats(Xtr)
    weights, freq = class_weights(Ytr)
    log("normalisation: " + json.dumps({c: [round(v["mean"], 2), round(v["std"], 2)]
                                        for c, v in stats["per_channel"].items()}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 0 if SMOKE else 4
    train_dl = DataLoader(Chips(Xtr, Ytr, stats, True), BATCH_SIZE, shuffle=True, num_workers=workers,
                          pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    val_dl = DataLoader(Chips(Xva, Yva, stats, False), 4, shuffle=False, num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    model = smp.Unet(ENCODER, encoder_weights=None if SMOKE else ENCODER_WEIGHTS, in_channels=6,
                     classes=N_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=IGNORE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    config = dict(paper="UrbanSARFloods, CVPRW 2024, Sec. 4.1", encoder=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                  lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, epochs=EPOCHS, crop=CROP,
                  loss="weighted cross-entropy", class_weights=weights.tolist(), class_pixel_share=freq.tolist(),
                  channels=CHANNELS, classes=CLASS_NAMES, resolution_m=20, band_check=check,
                  n_train=len(train_pairs), n_val=len(val_pairs), seed=SEED)
    log(f"device {device}, {sum(p.numel() for p in model.parameters()):,} parameters")

    history, best, best_epoch = [], -1.0, 0
    epochs = 2 if SMOKE else EPOCHS
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr, _ = run_epoch(model, train_dl, loss_fn, opt, device, True)
        va_loss, va, cm = run_epoch(model, val_dl, loss_fn, opt, device, False)
        row = dict(epoch=epoch, seconds=round(time.time() - t0, 1), train_loss=tr_loss, val_loss=va_loss,
                   val_f1_NF=va["NF"]["f1"], val_f1_FO=va["FO"]["f1"], val_f1_FU=va["FU"]["f1"],
                   val_f1_flood_any=va["flood_any"]["f1"], val_mean_flood_f1=va["mean_flood_f1"])
        history.append(row)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        ckpt = dict(state_dict=model.state_dict(), config=config, normalization=stats, epoch=epoch,
                    val=va, val_confusion=cm.tolist(), architecture="smp.Unet")
        torch.save(ckpt, OUT_DIR / "last_model.pt")
        if va["mean_flood_f1"] > best:
            best, best_epoch = va["mean_flood_f1"], epoch
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            json.dump(dict(best_epoch=epoch, val=va, val_confusion=cm.tolist(), config=config),
                      open(OUT_DIR / "metrics.json", "w"), indent=2)
        with open(OUT_DIR / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader(); w.writerows(history)
        elapsed = time.time() - t_start
        if elapsed + 1.5 * row["seconds"] > TIME_BUDGET_S:
            log(f"stopping after epoch {epoch}: Kaggle time limit is near ({elapsed / 3600:.1f} h used)")
            break
    log(f"done: best epoch {best_epoch}, val mean flood F1 {best:.4f}, total {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
