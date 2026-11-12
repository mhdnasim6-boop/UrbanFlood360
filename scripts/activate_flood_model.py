#!/usr/bin/env python3
"""
UrbanFlood360 - compare the UrbanSARFloods U-Net (Kaggle) with the Valencia U-Net v2 on Valencia, and make the
better one the dashboard's flood model.

Steps
  1. copy the Kaggle checkpoint (default kaggle_kernel_v2/output/best_model.pt) to models/urbansarfloods/
  2. map the Valencia AOI with both models on two image pairs:
       snap  19 -> 31 Oct 2024, orbit 103, SNAP sigma0  (the pair U-Net v2 was trained on)
       pc    20 Oct -> 1 Nov 2024, orbit 110, Planetary Computer gamma0 (what the download page gets; downloaded
             automatically if missing)
  3. score each map against Copernicus EMSR773 (scripts/validate_on_valencia.py) on the held-out test block
  4. with --activate: write models/active_flood_model.json so the dashboard uses the UrbanSARFloods model
     (only if it beats U-Net v2 on the "pc" pair, unless --force)

The UrbanSARFloods model never saw Valencia, so its Valencia scores measure how it does on a new city.
Its threshold is 0.5 (P(flooded open) + P(flooded urban) >= 0.5); it is not tuned on Valencia.

Usage:
    python scripts/activate_flood_model.py [--checkpoint path/to/best_model.pt] [--activate] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyse_event as ae  # noqa: E402
from scripts.validate_on_valencia import score  # noqa: E402

DEST = ROOT / "models/urbansarfloods/best_model.pt"
WORK = ROOT / "outputs/model_check"
AOI = ROOT / "data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson"
PAIRS = {
    "snap": (ROOT / "data/processed/sentinel1/prototype/20241019_R103/corrected/db/S1_PRE_20241019_R103_sigma0_db.tif",
             ROOT / "data/processed/sentinel1/post/20241031_R103/corrected/db/S1_POST_20241031_R103_sigma0_db.tif"),
    "pc": (WORK / "pc_pair/before.tif", WORK / "pc_pair/after.tif"),
}


def save_compact(src, dest, limit_bytes=95 * 2**20):
    """Copy a Kaggle checkpoint as weights + config + normalisation only (no optimiser state).

    Full float32 weights (~98 MB) stay under GitHub's 100 MiB file limit; if a file would not, the weights are
    stored in float16 instead (half the size, probabilities differ by a few hundredths at most).
    """
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    slim = dict(architecture=ckpt["architecture"], config=ckpt["config"], normalization=ckpt["normalization"],
                epoch=ckpt["epoch"], val=ckpt.get("val"), state_dict=ckpt["state_dict"], dtype="float32")
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dest)
    if dest.stat().st_size > limit_bytes:
        slim["state_dict"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()}
        slim["dtype"] = "float16"
        torch.save(slim, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 2**20:.0f} MiB, {slim['dtype']}, epoch {ckpt['epoch']})")


def ensure_pc_pair():
    if not PAIRS["pc"][1].exists():
        from scripts.fetch_event_data import fetch
        print("Downloading the Planetary Computer pair for Valencia")
        fetch(WORK / "pc_pair", "2024-10-29", aoi_file=AOI, with_buildings=False, progress=lambda f, t: None)


def run(model_json, pair, tag):
    """Map the Valencia AOI with the model described by model_json (None = U-Net v2); return test/all scores."""
    out = WORK / tag
    active = ae.ACTIVE_MODEL
    backup = active.read_text() if active.exists() else None
    try:
        if model_json is None:
            active.unlink(missing_ok=True)
        else:
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(json.dumps(model_json))
        ae.analyse(*PAIRS[pair], out, aoi=AOI, event_name=tag, progress=lambda f, t: None)
    finally:
        if backup is None:
            active.unlink(missing_ok=True)
        else:
            active.write_text(backup)
    return {r["region"]: r for r in score(out / "flood_mask.tif")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "kaggle_kernel_v2/output/best_model.pt"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--activate", action="store_true")
    ap.add_argument("--force", action="store_true", help="activate even if it does not beat U-Net v2")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    if src.resolve() != DEST.resolve():
        save_compact(src, DEST)
    usf = dict(name="U-Net trained on UrbanSARFloods (18 flood events)", checkpoint=str(DEST.relative_to(ROOT)),
               threshold=args.threshold)
    ensure_pc_pair()

    rows = []
    for pair in PAIRS:
        for model, cfg in [("U-Net v2 (Valencia)", None), ("U-Net UrbanSARFloods", usf)]:
            s = run(cfg, pair, f"{pair}_{'v2' if cfg is None else 'usf'}")
            for region in ("test block", "all imaged"):
                r = s[region]
                rows.append(dict(pair=pair, model=model, region=region, dice=round(r["dice"], 3),
                                 precision=round(r["precision"], 2), recall=round(r["recall"], 2),
                                 flood_pct=round(r["predicted_flood_pct"]), reference_pct=round(r["reference_flood_pct"])))
    df = pd.DataFrame(rows)
    WORK.mkdir(parents=True, exist_ok=True)
    df.to_csv(WORK / "model_comparison_valencia.csv", index=False)
    print(df.to_string(index=False))

    test = df[df.region == "test block"].set_index(["pair", "model"]).dice
    better = test[("pc", "U-Net UrbanSARFloods")] > test[("pc", "U-Net v2 (Valencia)")]
    print(f"\nOn the downloaded (pc) pair, test-block Dice: UrbanSARFloods {test[('pc', 'U-Net UrbanSARFloods')]:.3f} "
          f"vs U-Net v2 {test[('pc', 'U-Net v2 (Valencia)')]:.3f} -> {'better' if better else 'not better'}")
    if args.activate and (better or args.force):
        usf["valencia_test_dice"] = {p: float(test[(p, "U-Net UrbanSARFloods")]) for p in PAIRS}
        ae.ACTIVE_MODEL.write_text(json.dumps(usf, indent=2))
        print(f"Activated: {ae.ACTIVE_MODEL.relative_to(ROOT)} -> the dashboard now uses the UrbanSARFloods U-Net.")
    elif args.activate:
        print("Not activated (use --force to activate anyway).")


if __name__ == "__main__":
    main()
