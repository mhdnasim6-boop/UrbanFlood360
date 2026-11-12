"""
Train UrbanFlood360 U-Net v2.

Run from the project root:
    python -m models.unet_v2.train --run-name main
    python -m models.unet_v2.train --run-name no_aug --no-augment

Outputs (models/unet_v2/runs/<run-name>/):
    best_model.pt              checkpoint with the best val Dice (state_dict + config + norm stats)
    last_model.pt
    normalization_stats.json   fitted on train-region pixels only
    history.csv                per-epoch train loss and val metrics
    val_threshold_sweep.csv    val-block metrics for thresholds 0.05..0.95 (best checkpoint)
    metrics_val.json           chosen threshold + val metrics. The TEST block is not touched here;
                               scripts/evaluate_flood_models.py evaluates it once at the end.
    training_curves.png

The evaluated/saved weights are an exponential moving average (EMA) of the
training weights, which smooths the epoch-to-epoch noise of a small dataset.

Works on Apple MPS, CUDA (e.g. Kaggle) or CPU. --max-minutes stops training
early so a run can never tie up the laptop for long.
"""
import argparse
import csv
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from models.unet_v2.data import (
    SPLIT_CODES, RandomWindowDataset, block_split_map, fit_normalization, load_rasters,
    normalize, predict_raster, save_json, threshold_sweep, train_allowed_mask,
    train_window_positions,
)
from models.unet_v2.losses import masked_bce_dice
from models.unet_v2.model import UNet, count_parameters

ROOT = Path(__file__).resolve().parents[2]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(cfg):
    d = cfg["data"]
    sar, label, profile, channels = load_rasters(ROOT / d["sar_stack"], ROOT / d["label"])
    split = block_split_map(label.shape, tuple(d["block_grid"]),
                            {tuple(d["val_block"]): "val", tuple(d["test_block"]): "test"})
    allowed = train_allowed_mask(split, d["buffer_m"], profile["transform"].a)
    valid = np.all(~np.isnan(sar), axis=0)
    stats = fit_normalization(sar, allowed & valid, channels)
    image = normalize(sar, stats)
    positions = train_window_positions(allowed, label, d["window"], d["min_labeled"])
    return dict(image=image, label=label, split=split, allowed=allowed, valid=valid,
                stats=stats, positions=positions, profile=profile)


def plot_history(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, [h["val_dice@0.5"] for h in history], label="val Dice @0.5")
    ax[1].plot(ep, [h["val_best_dice"] for h in history], label="val Dice @best threshold (0.3-0.7)")
    ax[1].plot(ep, [h["val_iou@0.5"] for h in history], label="val IoU @0.5")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Validation block"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def val_loss_from_prob(prob, label, region, pos_weight, bce_weight):
    m = region & (label != 255) & ~np.isnan(prob)
    p = np.clip(prob[m].astype("float64"), 1e-6, 1 - 1e-6)
    y = (label[m] == 1).astype("float64")
    bce = -(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return float(bce_weight * bce + (1 - bce_weight) * dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "models/unet_v2/config.yaml"))
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.no_augment:
        cfg["train"]["augment"] = False
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    t, inf = cfg["train"], cfg["inference"]
    run_dir = ROOT / "models/unet_v2/runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(t["seed"])
    device = get_device()
    data = prepare_data(cfg)
    label, split, valid = data["label"], data["split"], data["valid"]
    val_region = split == SPLIT_CODES["val"]

    train_px = data["allowed"] & (label != 255)
    n_flood, n_non = int((label[train_px] == 1).sum()), int((label[train_px] == 0).sum())
    pos_weight = min(n_non / max(n_flood, 1), 10.0) if t["pos_weight"] == "auto" else float(t["pos_weight"])
    cfg["train"]["pos_weight_resolved"] = round(pos_weight, 4)

    print(f"Device: {device} | torch {torch.__version__}")
    print(f"Train region: {int(data['allowed'].sum())} px | labelled {int(train_px.sum())} "
          f"(flood {n_flood}, non-flood {n_non}) | pos_weight {pos_weight:.2f}")
    print(f"Candidate training windows: {len(data['positions'])} | augment: {t['augment']}")

    save_json(data["stats"], run_dir / "normalization_stats.json")
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)

    ds = RandomWindowDataset(data["image"], label, data["positions"], cfg["data"]["window"],
                             t["windows_per_epoch"], augment=t["augment"], seed=t["seed"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=0)

    model = UNet(6, 1, cfg["model"]["base_channels"], cfg["model"]["dropout"]).to(device)
    print(f"U-Net parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(t["ema_decay"]), use_buffers=True)

    history, best_score, best_epoch, bad_epochs = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, t["epochs"] + 1):
        e0 = time.time()
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = masked_bce_dice(model(x), y, pos_weight, t["bce_weight"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update_parameters(model)
            losses.append(loss.item())
        sched.step()

        prob = predict_raster(ema.module, data["image"], device, inf["tile"], inf["overlap"],
                              inf["batch_size"], valid=valid)
        sweep, best = threshold_sweep(prob, label, val_region)
        at05 = next(r for r in sweep if r["threshold"] == 0.5)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                   val_loss=val_loss_from_prob(prob, label, val_region, pos_weight, t["bce_weight"]),
                   **{"val_dice@0.5": at05["dice"], "val_iou@0.5": at05["iou"],
                      "val_precision@0.5": at05["precision"], "val_recall@0.5": at05["recall"]},
                   val_best_dice=best["dice"], val_best_threshold=best["threshold"],
                   lr=sched.get_last_lr()[0], epoch_s=time.time() - e0)
        history.append(row)

        # Early epochs are not eligible: a half-trained model can match the single small
        # val block by luck (the archived experiments picked epoch 1-3 this way).
        improved = epoch >= t["min_epochs"] and best["dice"] > best_score
        if improved:
            best_score, best_epoch, bad_epochs = best["dice"], epoch, 0
            torch.save({"state_dict": ema.module.state_dict(), "config": cfg, "epoch": epoch,
                        "normalization": data["stats"]}, run_dir / "best_model.pt")
        elif epoch >= t["min_epochs"]:
            bad_epochs += 1
        print(f"ep {epoch:3d} | loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
              f"Dice@0.5 {at05['dice']:.3f} | best-thr Dice {best['dice']:.3f} (t={best['threshold']}) | "
              f"{row['epoch_s']:.1f}s{' *' if improved else ''}", flush=True)

        elapsed_min = (time.time() - t0) / 60
        if bad_epochs >= t["patience"]:
            print(f"Early stop: no val improvement for {t['patience']} epochs.")
            break
        if elapsed_min > args.max_minutes:
            print(f"Stopping: time budget of {args.max_minutes} min reached.")
            break

    torch.save({"state_dict": ema.module.state_dict(), "config": cfg, "epoch": epoch,
                "normalization": data["stats"]}, run_dir / "last_model.pt")
    if best_epoch == 0:  # stopped before min_epochs: fall back to the last weights
        best_epoch = epoch
        shutil.copy(run_dir / "last_model.pt", run_dir / "best_model.pt")
    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    plot_history(history, run_dir / "training_curves.png")

    # Re-evaluate the best checkpoint on the val block and fix the decision threshold.
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    prob = predict_raster(model, data["image"], device, inf["tile"], inf["overlap"],
                          inf["batch_size"], valid=valid)
    sweep, best = threshold_sweep(prob, label, val_region)
    with open(run_dir / "val_threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)
    save_json({"run": args.run_name, "best_epoch": best_epoch, "epochs_run": epoch,
               "threshold": best["threshold"], "val": best,
               "total_minutes": round((time.time() - t0) / 60, 2), "device": str(device),
               "parameters": count_parameters(model), "pos_weight": pos_weight,
               "augment": t["augment"], "test_block_accessed": False},
              run_dir / "metrics_val.json")
    print(f"\nBest epoch {best_epoch}: val Dice {best['dice']:.4f} IoU {best['iou']:.4f} "
          f"P {best['precision']:.3f} R {best['recall']:.3f} at threshold {best['threshold']}")
    print(f"Total time {(time.time() - t0) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
