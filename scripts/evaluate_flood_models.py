#!/usr/bin/env python3
"""
UrbanFlood360 - compare flood-mapping methods on identical pixels.

Two targets (both from Copernicus EMSR773 AOI03 grading v2, same ignore rules):
  water     "Flooded area" only = standing water on 31 Oct 2024
            (data/processed/unet_dataset/labels/flood_reference.tif)
  affected  "Flooded area" + "Flood trace" = ground the flood reached
            (data/processed/flood_affected_dataset/labels/flood_reference.tif,
             create_flood_labels.py --trace-mode flood)

Methods
  1. Otsu threshold on post-event VV backscatter (classic "dark water" rule)
  2. Otsu threshold on the VV change (post - pre), flood = strong decrease
  3. Repository U-Net, BCE baseline           (water only; models/unet/baseline, threshold 0.5)
  4. Repository U-Net, weighted BCE+Dice      (water only; archived previous best, threshold 0.4)
  5. Every U-Net v2 run in models/unet_v2/runs/ (threshold chosen on its own val block)

Evaluation regions (labelled pixels only, label 255 ignored)
  val_block    every labelled pixel of block (1,2)  - used for model/threshold selection
  test_block   every labelled pixel of block (2,1)  - held out, evaluated only here
  val_patches / test_patches (water only)  pixels of the original 14 / 13 fixed patches,
               for comparison with the numbers in the earlier reports

Among the U-Net v2 runs trained on the chosen target, the one with the best
val_block Dice is written to models/unet_v2/selected_model_<target>.json
(the test block plays no part in that choice).

Usage:
    python scripts/evaluate_flood_models.py --target water
    python scripts/evaluate_flood_models.py --target affected

Outputs: outputs/tables/model_comparison_<target>.csv,
         reports/figures/model_comparison_<target>_{val,test}_block.png
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet_v2.data import (  # noqa: E402
    SPLIT_CODES, TARGETS, block_split_map, checkpoint_target, confusion, load_rasters, normalize, predict_raster,
)
from models.unet_v2.train import get_device  # noqa: E402
from scripts.predict_flood_map import load_model  # noqa: E402

SAR = ROOT / "data/processed/sar_features/sar_features_6ch.tif"
PATCH_META = ROOT / "data/processed/unet_dataset/patches/all_patches_metadata.csv"
FIG_DIR = ROOT / "reports/figures"


def load_repo_unet(checkpoint, device):
    spec = importlib.util.spec_from_file_location("repo_unet_model", ROOT / "models/unet/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Unet(6, 1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


def repo_norm_stats(path):
    s = json.load(open(path))
    return {"channels": s["channels"],
            "per_channel": {k: {"mean": v["mean"], "std": v["std"]} for k, v in s["per_channel"].items()}}


def patch_mask(shape, split):
    meta = pd.read_csv(PATCH_META)
    m = np.zeros(shape, dtype=bool)
    for _, r in meta[meta.split == split].iterrows():
        m[r.row0:r.row0 + r.patch_size, r.col0:r.col0 + r.patch_size] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), default="affected")
    args = ap.parse_args()
    target = TARGETS[args.target]

    device = get_device()
    sar, label, profile, channels = load_rasters(SAR, ROOT / target["label"])
    valid = np.all(~np.isnan(sar), axis=0)
    split = block_split_map(label.shape)
    regions = {"val_block": split == SPLIT_CODES["val"], "test_block": split == SPLIT_CODES["test"]}
    if args.target == "water":
        regions["val_patches"] = patch_mask(label.shape, "val")
        regions["test_patches"] = patch_mask(label.shape, "test")

    methods = {}  # name -> (prob raster, threshold, trained_on, note)
    vv_post, vv_change = sar[channels.index("VV_post_dB")], sar[channels.index("VV_change_dB")]
    t_post, t_change = threshold_otsu(vv_post[valid]), threshold_otsu(vv_change[valid])
    methods["Otsu: low post-event VV"] = (np.where(valid, (vv_post <= t_post).astype("float32"), np.nan), 0.5,
                                          "-", f"VV_post <= {t_post:.2f} dB")
    methods["Otsu: VV decrease"] = (np.where(valid, (vv_change <= t_change).astype("float32"), np.nan), 0.5,
                                    "-", f"VV_change <= {t_change:.2f} dB")

    if args.target == "water":
        repo_models = {
            "Repository U-Net (BCE baseline)": (ROOT / "models/unet/baseline", 0.5),
            "Repository U-Net (weighted BCE+Dice)": (
                ROOT / "archive/unet_experiments/models_unet/loss_experiments/weighted_bce_dice", 0.4),
        }
        for name, (folder, thr) in repo_models.items():
            model = load_repo_unet(folder / "best_model.pt", device)
            image = normalize(sar, repo_norm_stats(folder / "normalization_stats.json"))
            prob = predict_raster(model, image, device, tile=64, overlap=16, outputs_logits=False, valid=valid)
            methods[name] = (prob, thr, "water", "301 fixed patches")

    v2_runs = {}
    for run_dir in sorted((ROOT / "models/unet_v2/runs").glob("*/")):
        if not (run_dir / "metrics_val.json").exists():
            continue
        mv = json.load(open(run_dir / "metrics_val.json"))
        model, ckpt = load_model(run_dir / "best_model.pt", device)
        inf = ckpt["config"]["inference"]
        prob = predict_raster(model, normalize(sar, ckpt["normalization"]), device,
                              inf["tile"], inf["overlap"], inf["batch_size"], valid=valid)
        name = f"U-Net v2 ({run_dir.name})"
        trained_on = checkpoint_target(ckpt)
        methods[name] = (prob, mv["threshold"], trained_on, f"epoch {mv['best_epoch']}, augment={mv['augment']}")
        if trained_on == args.target:
            v2_runs[name] = run_dir

    rows = []
    for name, (prob, thr, trained_on, note) in methods.items():
        for region_name, region in regions.items():
            c = confusion(prob, label, region, thr)
            rows.append(dict(method=name, trained_on=trained_on, region=region_name, threshold=thr, note=note,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}))
    df = pd.DataFrame(rows)
    table = ROOT / f"outputs/tables/model_comparison_{args.target}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table, index=False)

    pd.set_option("display.width", 220)
    print(f"Target: {target['title']}")
    for region_name in regions:
        sub = df[df.region == region_name].set_index("method")
        print(f"\n=== {region_name} (flood px {int(sub.iloc[0].tp + sub.iloc[0].fn)}, "
              f"labelled px {int(sub.iloc[0][['tp', 'tn', 'fp', 'fn']].sum())})")
        print(sub[["trained_on", "threshold", "dice", "iou", "precision", "recall", "specificity", "fp", "fn"]].to_string())

    if v2_runs:
        val = df[(df.region == "val_block") & df.method.isin(v2_runs)]
        best = val.loc[val.dice.idxmax()]
        run_dir = v2_runs[best.method]
        out = ROOT / f"models/unet_v2/selected_model_{args.target}.json"
        json.dump({"target": args.target, "target_description": target["title"], "run": run_dir.name,
                   "checkpoint": str((run_dir / "best_model.pt").relative_to(ROOT)),
                   "threshold": float(best.threshold), "val_block_dice": float(best.dice),
                   "selection_rule": "highest val_block Dice among U-Net v2 runs trained on this target "
                                     "(test block not used)"},
                  open(out, "w"), indent=2)
        print(f"\nSelected model: {best.method} (val block Dice {best.dice:.4f}) -> {out.relative_to(ROOT)}")

    plot_regions(args.target, methods, sar, channels, label, split)


def plot_regions(target, methods, sar, channels, label, split):
    """Per-region panel: VV change, reference, and each method's TP/FP/FN map."""
    cmap = ListedColormap(["#e9e8e4", "#2b8a3e", "#e8590c", "#1c7ed6", "#ffffff"])  # TN, TP, FP, FN, ignore
    vv_change = sar[channels.index("VV_change_dB")]
    for region_name, code in [("val_block", SPLIT_CODES["val"]), ("test_block", SPLIT_CODES["test"])]:
        rows, cols = np.nonzero(split == code)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        lab = label[r0:r1, c0:c1]
        n = len(methods) + 2
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow))
        axes = axes.ravel()
        axes[0].imshow(vv_change[r0:r1, c0:c1], cmap="RdBu", vmin=-8, vmax=8)
        axes[0].set_title("VV change (dB)\nblue = increase, red = decrease", fontsize=9)
        ref = np.where(lab == 255, np.nan, lab.astype(float))
        axes[1].imshow(ref, cmap=ListedColormap(["#d0d0d0", "#1c7ed6"]), vmin=0, vmax=1)
        axes[1].set_title(f"EMSR773 reference ({target})\nblue = flood, white = ignore", fontsize=9)
        for ax, (name, (prob, thr, _, _)) in zip(axes[2:], methods.items()):
            p = prob[r0:r1, c0:c1]
            pred = p >= thr
            cat = np.full(lab.shape, 4)
            known = (lab != 255) & ~np.isnan(p)
            cat[known & ~pred & (lab == 0)] = 0
            cat[known & pred & (lab == 1)] = 1
            cat[known & pred & (lab == 0)] = 2
            cat[known & ~pred & (lab == 1)] = 3
            c = confusion(prob, label, split == code, thr)
            ax.imshow(cat, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            ax.set_title(f"{name}\nDice {c['dice']:.2f}  P {c['precision']:.2f}  R {c['recall']:.2f}", fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"{target} target, {region_name}: green = correct flood, orange = false alarm, "
                     f"blue = missed flood, grey = correct non-flood", fontsize=10)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"model_comparison_{target}_{region_name}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
