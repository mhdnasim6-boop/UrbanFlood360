#!/usr/bin/env python3
"""
UrbanFlood360 - UrbanSARFloods baseline training (runs on Kaggle GPU).

Self-contained: downloads the official UrbanSARFloods train/val tarball
(stream-extracted, no local copy of the .tar.gz kept), builds a minimal
PyTorch Dataset from the official Train_dataset.txt/Valid_dataset.txt
split-membership files, and trains the UNMODIFIED
chathumal93/Pytorch-UNet-Flood-Segmentation Unet architecture (embedded
verbatim below, commit 717e0ab) with in_channels=4.

Input channels (decision: 4-intensity-only, so the resulting checkpoint
stays directly compatible with Valencia's GRD-only SAR pipeline later):
  band 5 = pre-event intensity VH
  band 6 = pre-event intensity VV
  band 7 = post-event intensity VH
  band 8 = post-event intensity VV
(the dataset's other 4 bands - interferometric coherence - require SLC
data Valencia does not have, so they are not used here; see the paper's
Fig.1 band order for the full 8-band layout this indexing follows.)

Label: UrbanSARFloods GT is 3-class {0=non-flood, 1=flooded open, 2=flooded
urban}. Collapsed to binary flood/non-flood (label>0 -> 1) - the minimum
change needed to reuse the existing binary sigmoid+BCE Unet(in,1) interface
unchanged, consistent with every prior UrbanFlood360 stage.

No augmentation, no loss/threshold/architecture experiments - a single,
simple baseline run: Adam(lr=1e-3), BCELoss, checkpoint by min val loss.

Valencia is never referenced anywhere in this script.
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
    import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# REPOSITORY CODE - copied verbatim from
# https://github.com/chathumal93/Pytorch-UNet-Flood-Segmentation
# (commit 717e0abad6d71d6fd4059aad8ad8a98a6ebc1286), models.py.
# No modification to this class's logic anywhere below.
# ----------------------------------------------------------------------
from torch import cat, sigmoid


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_conv_01 = Unet.conv_block(in_channels, 32)
        self.down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_02 = Unet.conv_block(32, 64)
        self.down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_03 = Unet.conv_block(64, 128)
        self.down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_04 = Unet.conv_block(128, 256)
        self.down_sample_04 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.base = Unet.conv_block(256, 512)
        self.up_sample_04 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.dec_conv_04 = Unet.conv_block(512, 256)
        self.up_sample_03 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.dec_conv_03 = Unet.conv_block(256, 128)
        self.up_sample_02 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dec_conv_02 = Unet.conv_block(128, 64)
        self.up_sample_01 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dec_conv_01 = Unet.conv_block(64, 32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc_conv_1 = self.enc_conv_01(x)
        enc_conv_2 = self.enc_conv_02(self.down_sample_01(enc_conv_1))
        enc_conv_3 = self.enc_conv_03(self.down_sample_02(enc_conv_2))
        enc_conv_4 = self.enc_conv_04(self.down_sample_03(enc_conv_3))
        base_block = self.base(self.down_sample_04(enc_conv_4))
        dec_conv_4 = self.up_sample_04(base_block)
        dec_conv_4 = cat((enc_conv_4, dec_conv_4), dim=1)
        dec_conv_4 = self.dec_conv_04(dec_conv_4)
        dec_conv_3 = self.up_sample_03(dec_conv_4)
        dec_conv_3 = cat((enc_conv_3, dec_conv_3), dim=1)
        dec_conv_3 = self.dec_conv_03(dec_conv_3)
        dec_conv_2 = self.up_sample_02(dec_conv_3)
        dec_conv_2 = cat((enc_conv_2, dec_conv_2), dim=1)
        dec_conv_2 = self.dec_conv_02(dec_conv_2)
        dec_conv_1 = self.up_sample_01(dec_conv_2)
        dec_conv_1 = cat((enc_conv_1, dec_conv_1), dim=1)
        dec_conv_1 = self.dec_conv_01(dec_conv_1)
        return sigmoid(self.final_conv(dec_conv_1))

    @staticmethod
    def conv_block(_in, _out):
        return nn.Sequential(
            nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=_out, out_channels=_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
# ----------------------------------------------------------------------
# END REPOSITORY CODE
# ----------------------------------------------------------------------

DATA_ROOT = Path("/kaggle/temp/urban_sar_floods")  # scratch disk, NOT /kaggle/working:
# /kaggle/working is capped (~20GB, and is what `kaggle kernels output` downloads in
# full) - the extracted dataset (~38GB+) must not live there. /kaggle/temp has much
# more headroom and is never synced back, which is exactly what we want for a large,
# disposable-after-training input. Only small artifacts (checkpoints/csv/json) below
# go to /kaggle/working/output, so `kernels output` only ever pulls those back.
OUT_DIR = Path("/kaggle/working/output")
TARBALL_URL = "https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/main/urban_sar_floods.tar.gz"
INTENSITY_BAND_INDICES = [5, 6, 7, 8]  # 1-indexed: pre_VH, pre_VV, post_VH, post_VV
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 15
PATIENCE = 5
SEED = 42


def print_disk_space():
    for path in ["/kaggle/working", "/kaggle/temp", "/"]:
        try:
            st = os.statvfs(path)
            free_gb = st.f_bavail * st.f_frsize / 1e9
            total_gb = st.f_blocks * st.f_frsize / 1e9
            print(f"  {path}: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        except FileNotFoundError:
            print(f"  {path}: does not exist")


def download_and_extract():
    print("Disk space before download:")
    print_disk_space()
    if DATA_ROOT.exists() and any(DATA_ROOT.iterdir()):
        print(f"{DATA_ROOT} already populated, skipping download.")
        return
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming + extracting {TARBALL_URL} -> {DATA_ROOT.parent} (no local .tar.gz retained)")
    t0 = time.time()
    cmd = f'curl -L "{TARBALL_URL}" | tar xz -C "{DATA_ROOT.parent}"'
    ret = subprocess.run(cmd, shell=True)
    print(f"Disk space after download attempt (exit code {ret.returncode}):")
    print_disk_space()
    if ret.returncode != 0:
        raise RuntimeError(f"Download/extract failed with exit code {ret.returncode}")
    print(f"Download+extract done in {time.time()-t0:.0f}s")


def resolve_split_files(list_path: Path):
    """Reads a Train_dataset.txt/Valid_dataset.txt. Confirmed format (from an
    actual run's diagnostic printout): each line is a GT-relative path with
    one extra leading '../', e.g. '../03_FU/GT/20190329_Iran_ID_17_19_GT.tif'
    - i.e. relative to a subfolder that doesn't exist in the release, so one
    '../' is stripped and the remainder resolved against DATA_ROOT. The SAR
    counterpart replaces '/GT/' with '/SAR/' and '_GT.tif' with '_SAR.tif'."""
    lines = [l.strip() for l in open(list_path) if l.strip()]
    print(f"{list_path.name}: {len(lines)} lines. First 3 raw: {lines[:3]}")

    pairs = []
    for line in lines:
        rel = line
        while rel.startswith("../"):
            rel = rel[3:]
        gt_path = DATA_ROOT / rel
        sar_path = Path(str(gt_path).replace("/GT/", "/SAR/").replace("_GT.tif", "_SAR.tif"))
        if gt_path.exists() and sar_path.exists():
            pairs.append((sar_path, gt_path))

    print(f"{list_path.name}: resolved {len(pairs)}/{len(lines)} pairs to existing SAR+GT files")
    return pairs


class UrbanSARFloodsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sar_path, gt_path = self.pairs[idx]
        with rasterio.open(sar_path) as src:
            bands = src.read(INTENSITY_BAND_INDICES).astype("float32")  # (4,H,W)
        with rasterio.open(gt_path) as src:
            gt = src.read(1).astype("float32")  # (H,W), values {0,1,2}

        bands = np.nan_to_num(bands, nan=0.0)
        label = (gt > 0).astype("float32")  # collapse FO(1)+FU(2) -> flood=1

        image = torch.from_numpy(bands)
        label = torch.from_numpy(label).unsqueeze(0)
        return image, label


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, optimizer, device, train):
    model.train() if train else model.eval()
    running_loss, n_batches = 0.0, 0
    tp = tn = fp = fn = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            pred_bin = (pred.detach() > 0.5).float()
            tp += int(((pred_bin == 1) & (labels == 1)).sum().item())
            tn += int(((pred_bin == 0) & (labels == 0)).sum().item())
            fp += int(((pred_bin == 1) & (labels == 0)).sum().item())
            fn += int(((pred_bin == 0) & (labels == 1)).sum().item())
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return dict(loss=running_loss / max(n_batches, 1), iou=iou, dice=dice,
                tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    download_and_extract()

    train_pairs = resolve_split_files(DATA_ROOT / "Train_dataset.txt")
    val_pairs = resolve_split_files(DATA_ROOT / "Valid_dataset.txt")
    if not train_pairs or not val_pairs:
        raise RuntimeError("Could not resolve train/val pairs - inspect the printed raw list format above.")

    train_ds = UrbanSARFloodsDataset(train_pairs)
    val_ds = UrbanSARFloodsDataset(val_pairs)
    print(f"Dataset: train={len(train_ds)}  val={len(val_ds)}")

    device = get_device()
    print(f"Device: {device}")

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = Unet(4, 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} | in_channels=4 out_channels=1")

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_m = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={train_m['loss']:.4f} train_dice={train_m['dice']:.4f}  "
              f"val_loss={val_m['loss']:.4f} val_dice={val_m['dice']:.4f} val_iou={val_m['iou']:.4f}  ({dt:.0f}s)")

        row = {"epoch": epoch, "epoch_time_s": round(dt, 1)}
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_m.items():
            row[f"val_{k}"] = v
        history.append(row)

        torch.save(model.state_dict(), OUT_DIR / "last_model.pt")
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best={best_epoch})")
            break

    with open(OUT_DIR / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    best_row = next(r for r in history if r["epoch"] == best_epoch)
    metrics = dict(
        dataset="UrbanSARFloods", in_channels=4, channels_used="pre_VH,pre_VV,post_VH,post_VV",
        out_channels=1, label_convention="binary (FO+FU collapsed to flood=1)",
        n_train=len(train_ds), n_val=len(val_ds), best_epoch=best_epoch, epochs_run=len(history),
        best_val_loss=best_row["val_loss"], best_val_dice=best_row["val_dice"], best_val_iou=best_row["val_iou"],
        best_val_tp=best_row["val_tp"], best_val_tn=best_row["val_tn"], best_val_fp=best_row["val_fp"], best_val_fn=best_row["val_fn"],
        total_runtime_s=round(time.time() - t_start, 1), device=str(device),
        batch_size=BATCH_SIZE, lr=LR, seed=SEED, valencia_used=False,
    )
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
