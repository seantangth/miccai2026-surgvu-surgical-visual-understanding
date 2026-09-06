#!/usr/bin/env bash
# 一台機器一個分片：抽幀 → 弱標籤 → 雙 teacher 偽標籤 → 打包上 gDrive data/pseudo_scale_r1_shard<N>.tar.gz
# 用法：SHARD=0 NSHARDS=2 DEADLINE_MIN=120 nohup bash job_extract_shard.sh > ~/scaleup/job_shard0.log 2>&1 &
set -uo pipefail
cd ~
SHARD="${SHARD:-0}"; NSHARDS="${NSHARDS:-2}"; DEADLINE_MIN="${DEADLINE_MIN:-120}"; MAXV="${MAXV:-10000}"
TAG="pseudo_scale_r1_shard${SHARD}"; GDRIVE="gdrive:MICCAI_2026_SurgVU"
source ~/venv/bin/activate
echo "=== [1] extract shard $SHARD/$NSHARDS deadline ${DEADLINE_MIN} min  $(date -u) ==="
python3 ~/scaleup/extract_scale.py --shard "$SHARD" --nshards "$NSHARDS" --out ~/scaleup/frames640 \
  --labels ~/labels --manifest ~/videos_zip_manifest.json --done ~/rare_frames_done_videos.json \
  --deadline-min "$DEADLINE_MIN" --max-videos "$MAXV" || echo "EXTRACT EXIT $?"
NF=$(find ~/scaleup/frames640 -name "*.jpg" | wc -l); echo "frames640: $NF"
[ "$NF" -ge 1000 ] || { echo "FATAL: too few frames ($NF)"; exit 2; }
echo "=== [2] weak labels  $(date -u) ==="
python3 ~/scaleup/weak_labels_scale.py ~/scaleup/frames640 ~/labels ~/scaleup/weak_scale.json || { echo "FATAL weak"; exit 3; }
echo "=== [3] pseudo-label (s1 + R1v3 WBF)  $(date -u) ==="
python3 ~/scaleup/pseudo_label_scale.py ~/scaleup/frames640 ~/scaleup/weak_scale.json ~/scaleup/$TAG ~/weights/s1/best.pt ~/weights/R1v3/best.pt || { echo "FATAL pseudo"; exit 4; }
echo "=== [4] pack + upload  $(date -u) ==="
cd ~/scaleup && tar -czf $TAG.tar.gz $TAG && rclone copy $TAG.tar.gz $GDRIVE/data/ && rclone copy $TAG/stats.json $GDRIVE/data/${TAG}_stats.json 2>/dev/null; rclone lsl $GDRIVE/data/$TAG.tar.gz
echo "SHARD $SHARD DONE $(date -u)"
