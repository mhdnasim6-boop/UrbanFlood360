#!/bin/bash
SLUG="nasimudheen/urbansarfloods-u-net-baseline"
MAX_CHECKS=240   # 240 * 60s = 4 hours max
i=0
while [ $i -lt $MAX_CHECKS ]; do
  STATUS=$(kaggle kernels status "$SLUG" 2>&1)
  echo "[$(date +%H:%M:%S)] $STATUS"
  if echo "$STATUS" | grep -qE "COMPLETE|ERROR|CANCEL"; then
    break
  fi
  sleep 60
  i=$((i+1))
done
echo "=== FINAL STATUS: $STATUS ==="
echo "=== Downloading output (checkpoints/metrics only expected now) ==="
rm -rf ./final_output_v2
kaggle kernels output "$SLUG" -p ./final_output_v2 2>&1
echo "=== Output files ==="
find ./final_output_v2 -maxdepth 3 -type f 2>&1
echo "=== Disk usage of downloaded output ==="
du -sh ./final_output_v2 2>&1
