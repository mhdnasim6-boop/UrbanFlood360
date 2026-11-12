#!/usr/bin/env bash
# UrbanFlood360 - end-to-end: labels -> U-Net v2 training -> evaluation -> flood maps -> building damage -> dashboard assets.
#
#   ./run_pipeline.sh            # full pipeline, including 4 training runs (~25-40 min on an M1, MPS)
#   SKIP_TRAINING=1 ./run_pipeline.sh   # reuse the trained checkpoints in models/unet_v2/runs/
#
# Then start the dashboard with:
#   ~/Anaconda/anaconda3/envs/urbanflood/bin/streamlit run dashboard/app.py
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/Anaconda/anaconda3/envs/urbanflood/bin/python}"

if [[ "${SKIP_TRAINING:-0}" != "1" ]]; then
  echo "== 1/7 Flood-affected labels (EMSR773 Flooded area + Flood trace)"
  E=data/raw/emsr773/extracted/GRA_v2/EMSR773_AOI03_GRA_PRODUCT
  "$PY" scripts/create_flood_labels.py --sar-stack data/processed/sar_features/sar_features_6ch.tif \
    --emsr773-observed-event "${E}_observedEventA_v2.shp" --emsr773-image-footprint "${E}_imageFootprintA_v2.shp" \
    --aoi data/gis/aoi/AOI_HORTA_SUD_6MUN.geojson --municipalities data/gis/aoi/municipalities_6.geojson \
    --output-dir data/processed/flood_affected_dataset/labels --trace-mode flood

  echo "== 2/7 Training U-Net v2: 2 targets x (with / without augmentation)"
  mkdir -p models/unet_v2/logs
  for target in water affected; do
    cfg=models/unet_v2/config.yaml; [[ $target == affected ]] && cfg=models/unet_v2/config_affected.yaml
    "$PY" -m models.unet_v2.train --config "$cfg" --run-name "${target}_aug" | tee "models/unet_v2/logs/train_${target}_aug.log"
    "$PY" -m models.unet_v2.train --config "$cfg" --run-name "${target}_no_aug" --no-augment | tee "models/unet_v2/logs/train_${target}_no_aug.log"
  done
fi

echo "== 3/7 Comparing methods and selecting one U-Net v2 run per target (validation block only)"
for target in water affected; do
  "$PY" scripts/evaluate_flood_models.py --target "$target"
  CKPT=$("$PY" -c "import json; print(json.load(open('models/unet_v2/selected_model_${target}.json'))['checkpoint'])")
  THR=$("$PY" -c "import json; print(json.load(open('models/unet_v2/selected_model_${target}.json'))['threshold'])")
  echo "== 4/7 Full-area $target map from $CKPT (threshold $THR)"
  "$PY" scripts/predict_flood_map.py --checkpoint "$CKPT" --threshold "$THR"
done

echo "== 5/7 Building inventory"
"$PY" scripts/prepare_buildings.py

echo "== 6/7 Building damage assessment (flood-affected map)"
"$PY" scripts/assess_building_damage.py

echo "== 7/7 Dashboard assets"
"$PY" scripts/build_dashboard_assets.py

echo "Done. Start the dashboard: $(dirname "$PY")/streamlit run dashboard/app.py"
