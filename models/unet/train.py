"""
UrbanFlood360 U-Net training entry point.

REPOSITORY CODE (adapted from model-train.ipynb, chathumal93/Pytorch-UNet-
Flood-Segmentation, commit 717e0ab): the train(), test()/evaluate(), and IoU()
function bodies below reuse that notebook's structure and its ignore-label
masking pattern (`y.ne(255)` + `masked_select`) essentially unchanged.

URBANFLOOD360 MODIFICATIONS (see reports/unet_repository_integration_report.md
for the full list):
  - device selection: MPS -> CUDA -> CPU (repo only had CUDA -> CPU)
  - dataset: UrbanFlood360PatchDataset (6-channel .npy patches), not the
    repo's 4-channel multi-GeoTIFF Sentinel1Dataset
  - normalization: train-fitted Normalizer (mean_std or percentile), not the
    repo's fixed physical-range constants
  - batch size: 8 (memory-safe for 8GB M1), not the repo's 32
  - no two-model water-percentage ensemble (single model only)
  - explicit NaN-fill after normalization (dataset.py) - not present in repo
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import UrbanFlood360PatchDataset
from model import Unet
from normalize import Normalizer

IGNORE_LABEL = 255


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- REPOSITORY CODE (adapted) ---
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for images, labels, n_nan, pids in dataloader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        valid_pixel_mask = labels.ne(IGNORE_LABEL)
        pred_valid = pred.masked_select(valid_pixel_mask)
        y_valid = labels.masked_select(valid_pixel_mask)
        loss = loss_fn(pred_valid, y_valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


def IoU(pred, true):
    # REPOSITORY CODE (unchanged logic): ignore-aware IoU at threshold 0.5
    valid_pixel_mask = true.ne(IGNORE_LABEL)
    pred_arr = pred.masked_select(valid_pixel_mask).detach().cpu().numpy()
    pred_arr = np.where(pred_arr > 0.5, 1, 0)
    true_arr = true.masked_select(valid_pixel_mask).detach().cpu().numpy()
    intersection = np.logical_and(true_arr, pred_arr)
    union = np.logical_or(true_arr, pred_arr)
    return intersection.sum() / (union.sum() + 1e-7)


@torch.no_grad()
def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    running_loss, running_iou, n_batches = 0.0, 0.0, 0
    for images, labels, n_nan, pids in dataloader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        valid_pixel_mask = labels.ne(IGNORE_LABEL)
        pred_valid = pred.masked_select(valid_pixel_mask)
        y_valid = labels.masked_select(valid_pixel_mask)
        loss = loss_fn(pred_valid, y_valid)
        running_loss += loss.item()
        running_iou += IoU(pred, labels)
        n_batches += 1
    return running_loss / max(n_batches, 1), running_iou / max(n_batches, 1)
# --- END REPOSITORY CODE (adapted) ---


def smoke_test(patches_dir, normalization_stats_path, batch_size, seed):
    torch.manual_seed(seed)
    device = get_device()
    print("Device:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS available:   {torch.backends.mps.is_available()}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    print(f"  Selected device: {device}")

    normalizer = Normalizer.from_json(normalization_stats_path)

    train_ds = UrbanFlood360PatchDataset(patches_dir, "train", normalizer)
    val_ds = UrbanFlood360PatchDataset(patches_dir, "val", normalizer)
    test_ds = UrbanFlood360PatchDataset(patches_dir, "test", normalizer)
    print(f"\nDataset:")
    print(f"  train patches = {len(train_ds)}")
    print(f"  val patches   = {len(val_ds)}")
    print(f"  test patches  = {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    images, labels, n_nan, pids = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  images = {list(images.shape)}  dtype={images.dtype}")
    print(f"  labels = {list(labels.shape)}  dtype={labels.dtype}")
    assert images.shape[1] == 6, f"Expected 6 input channels, got {images.shape[1]}"
    assert images.shape[2] == 64 and images.shape[3] == 64, f"Expected 64x64, got {images.shape[2:]}"
    label_vals = set(torch.unique(labels).tolist())
    print(f"  unique label values in batch: {sorted(label_vals)}")
    assert label_vals.issubset({0.0, 1.0, 255.0}), f"Unexpected label values: {label_vals}"
    assert not torch.isnan(images).any(), "NaN present in model input after preprocessing!"
    assert not torch.isinf(images).any(), "Inf present in model input after preprocessing!"
    print(f"  NaN pixels filled (this batch, sum across samples): {sum(n_nan.tolist())}")

    model = Unet(6, 1).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel:")
    print(f"  input  = {list(images.shape)}")
    with torch.no_grad():
        out_shape_check = model(images.to(device)).shape
    print(f"  output = {list(out_shape_check)}")
    assert list(out_shape_check) == [images.shape[0], 1, 64, 64], f"Unexpected output shape {out_shape_check}"
    print(f"  total parameters:     {n_total:,}")
    print(f"  trainable parameters: {n_trainable:,}")

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    images, labels = images.to(device), labels.to(device)
    model.train()
    pred = model(images)
    valid_pixel_mask = labels.ne(IGNORE_LABEL)
    n_ignored = int((~valid_pixel_mask).sum().item())
    n_total_px = labels.numel()
    pred_valid = pred.masked_select(valid_pixel_mask)
    y_valid = labels.masked_select(valid_pixel_mask)
    loss = loss_fn(pred_valid, y_valid)
    print(f"\nLoss: {loss.item():.6f}")
    assert not torch.isnan(loss).any(), "Loss is NaN!"
    assert not torch.isinf(loss).any(), "Loss is Inf!"
    print(f"Ignored pixels: {n_ignored} / {n_total_px} ({100*n_ignored/n_total_px:.1f}%)")

    optimizer.zero_grad()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    n_params_with_grad = len(grad_norms)
    n_params_total = sum(1 for p in model.parameters() if p.requires_grad)
    any_nan_grad = any(np.isnan(g) for g in grad_norms)
    print(f"\nGradient:")
    print(f"  params with grad: {n_params_with_grad}/{n_params_total}")
    print(f"  any NaN grad: {any_nan_grad}")
    assert not any_nan_grad, "NaN gradient detected!"
    assert n_params_with_grad == n_params_total, "Some parameters received no gradient!"

    optimizer.step()
    print("Optimizer step: SUCCESS")

    # One validation batch
    model.eval()
    with torch.no_grad():
        v_images, v_labels, v_n_nan, v_pids = next(iter(val_loader))
        v_images, v_labels = v_images.to(device), v_labels.to(device)
        v_pred = model(v_images)
        v_mask = v_labels.ne(IGNORE_LABEL)
        v_loss = loss_fn(v_pred.masked_select(v_mask), v_labels.masked_select(v_mask))
        v_iou = IoU(v_pred, v_labels)
    print(f"\nValidation:")
    print(f"  forward pass SUCCESS, val batch loss={v_loss.item():.6f}, val batch IoU={v_iou:.6f}")
    assert not torch.isnan(v_loss).any(), "Validation loss is NaN!"

    print("\nSMOKE TEST: ALL CHECKS PASSED")
    return dict(device=str(device), n_total_params=n_total, n_trainable_params=n_trainable,
                train_loss=loss.item(), val_loss=v_loss.item(), val_iou=float(v_iou),
                n_ignored_px=n_ignored, n_total_px=n_total_px)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patches-dir", default="../../data/processed/unet_dataset/patches")
    p.add_argument("--normalization-stats", default="normalization_stats.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        result = smoke_test(args.patches_dir, args.normalization_stats, args.batch_size, args.seed)
        with open("smoke_test_result.json", "w") as f:
            json.dump(result, f, indent=2)
    else:
        raise SystemExit("Full training loop intentionally not invoked in this stage.")
