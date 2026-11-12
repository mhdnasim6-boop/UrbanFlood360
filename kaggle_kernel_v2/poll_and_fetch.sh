#!/bin/bash
SLUG="nasimudheen/urbansarfloods-u-net-paper-settings"
MAX_CHECKS=720   # 720 * 60s = 12 hours max
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
rm -rf ./output
kaggle kernels output "$SLUG" -p ./output 2>&1
echo "=== Output files ==="
find ./output -maxdepth 3 -type f 2>&1
echo "=== Disk usage of downloaded output ==="
du -sh ./output 2>&1
