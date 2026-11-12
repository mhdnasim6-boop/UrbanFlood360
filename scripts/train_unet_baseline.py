#!/usr/bin/env python3
"""
UrbanFlood360 - baseline U-Net training run.

REPOSITORY CODE (adapted from chathumal93/Pytorch-UNet-Flood-Segmentation,
commit 717e0ab): model architecture (models/unet/model.py, unmodified except
call-site channel count), loss (BCELoss), optimizer (Adam, lr=0.001), and the
ignore-label masking pattern (y.ne(255) + masked_select) in the train/eval
loops below.

URBANFLOOD360 MODIFICATIONS: MPS device priority, 6-channel/.npy dataset,
train-only normalization, batch size 8, early stopping (repo has none),
full epoch-accumulated metrics (Dice/IoU/precision/recall/specificity +
TP/TN/FP/FN, repo only had loss+IoU), augmentation DISABLED for this baseline
(see reports/unet_baseline_training_report.md - a synchronization bug was
found in the repo's image/label augmentation pattern), CSV/JSON logging,
checkpointing of both best (by val loss) and last epoch.

Usage:
    python scripts/train_unet_baseline.py --config models/unet/config.yaml
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "unet"))
from dataset import UrbanFlood360PatchDataset  # noqa: E402
from metrics import ConfusionAccumulator  # noqa: E402
from model import Unet  # noqa: E402
from normalize import Normalizer  # noqa: E402

IGNORE_LABEL = 255


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE (documented, not silently ignored): MPS does not currently
    # guarantee bit-for-bit reproducibility for every op even with a fixed
    # seed (unlike CUDA's cudnn.deterministic path). This is a known
    # PyTorch/MPS limitation, not an UrbanFlood360 bug.


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool, threshold: float):
    model.train() if train else model.eval()
    acc = ConfusionAccumulator()
    running_loss, n_batches, n_ignored, n_total_px = 0.0, 0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels, n_nan, pids in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            valid_mask = labels.ne(IGNORE_LABEL)
            n_ignored += int((~valid_mask).sum().item())
            n_total_px += labels.numel()
            pred_valid = pred.masked_select(valid_mask)
            y_valid = labels.masked_select(valid_mask)
            loss = loss_fn(pred_valid, y_valid)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            acc.update(pred.detach(), labels, threshold=threshold)

    metrics = acc.compute()
    metrics["loss"] = running_loss / max(n_batches, 1)
    metrics["n_ignored_px"] = n_ignored
    metrics["n_total_px"] = n_total_px
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="models/unet/config.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default="models/unet/baseline")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["random_seed"]
    set_seed(seed)
    device = get_device()
    threshold = 0.5
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]

    print("=== UrbanFlood360 Baseline U-Net Training ===")
    print(f"Device: {device} | PyTorch {torch.__version__} | MPS available={torch.backends.mps.is_available()} | CUDA available={torch.cuda.is_available()}")
    print(f"Batch size: {batch_size} | LR: {lr} | Seed: {seed} | Threshold: {threshold} (baseline, not optimized)")
    print("Augmentation: DISABLED for this baseline run (see report - repo's image/label flip "
          "synchronization was found to be unreliable across two independent transform calls)")

    normalizer = Normalizer.from_json(cfg["normalization"]["stats_file"])
    patches_dir = cfg["dataset"]["patches_dir"]

    train_ds = UrbanFlood360PatchDataset(patches_dir, "train", normalizer)
    val_ds = UrbanFlood360PatchDataset(patches_dir, "val", normalizer)
    print(f"Dataset: train={len(train_ds)}  val={len(val_ds)}  (test intentionally NOT loaded)")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = Unet(6, 1).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True, threshold=threshold)
        val_m = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False, threshold=threshold)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_m['loss']:.4f} train_iou={train_m['iou']:.4f} train_dice={train_m['dice']:.4f}  |  "
              f"val_loss={val_m['loss']:.4f} val_iou={val_m['iou']:.4f} val_dice={val_m['dice']:.4f}  "
              f"({epoch_time:.1f}s)")

        row = {"epoch": epoch, "epoch_time_s": round(epoch_time, 2)}
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_m.items():
            row[f"val_{k}"] = v
        history.append(row)

        torch.save(model.state_dict(), out_dir / "last_model.pt")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={args.patience}, "
                  f"best epoch={best_epoch}, best val_loss={best_val_loss:.4f})")
            break

    total_time = time.time() - total_start
    n_epochs_run = len(history)

    fieldnames = list(history[0].keys())
    with open(out_dir / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(history)

    best_row = next(r for r in history if r["epoch"] == best_epoch)
    metrics_summary = {
        "best_epoch": best_epoch,
        "epochs_run": n_epochs_run,
        "early_stopped": n_epochs_run < args.epochs,
        "total_runtime_s": round(total_time, 1),
        "mean_epoch_time_s": round(total_time / n_epochs_run, 2),
        "best_val_loss": best_row["val_loss"],
        "best_val_dice": best_row["val_dice"],
        "best_val_iou": best_row["val_iou"],
        "best_val_precision": best_row["val_precision"],
        "best_val_recall": best_row["val_recall"],
        "best_val_specificity": best_row["val_specificity"],
        "best_val_tp": best_row["val_tp"], "best_val_tn": best_row["val_tn"],
        "best_val_fp": best_row["val_fp"], "best_val_fn": best_row["val_fn"],
        "device": str(device), "batch_size": batch_size, "threshold": threshold,
        "n_train_patches": len(train_ds), "n_val_patches": len(val_ds),
        "test_set_accessed": False,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Copy config + normalization stats alongside the run for full reproducibility
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    import shutil
    shutil.copy("models/unet/normalization_stats.json", out_dir / "normalization_stats.json")

    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
