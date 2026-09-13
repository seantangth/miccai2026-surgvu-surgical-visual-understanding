#!/usr/bin/env bash
# Round B（Lambda，一台一個分片）：等 Round A 兩個權重上雲 → 以 WBF(v7_rA, v7_rA2) 當老師重標本機分片
# （M1 另外把原始 20 支影片的 rare_frames 裁切後一併重標）→ 上傳 pseudo_scale_r2_shard<N> → 等另一分片 → 訓練 Round B。
# 用法：SHARD=0 RUN_TAG=v8_rB CLOSE_MOSAIC=3 BATCH=32 WITH_ORIG=1 setsid nohup bash ~/scaleup/job_roundB.sh > ~/scaleup/job_roundB.log 2>&1 < /dev/null &
set -uo pipefail
cd ~
SHARD="${SHARD:-0}"; OTHER=$((1 - SHARD)); GDRIVE="gdrive:MICCAI_2026_SurgVU"
RUN_TAG="${RUN_TAG:-v8_rB}"; WITH_ORIG="${WITH_ORIG:-0}"
source ~/venv/bin/activate
echo "=== [B0] wait Round A weights  $(date -u) ==="
while :; do
  a=$(rclone lsl $GDRIVE/runs/v7_rA_rtdetr-l/last.pt 2>/dev/null); b=$(rclone lsl $GDRIVE/runs/v7_rA2_rtdetr-l/last.pt 2>/dev/null)
  [ -n "$a" ] && [ -n "$b" ] && break; sleep 300
done
# 本機 Round A 訓練若仍在收尾，等它結束（避免 GPU 撞車）
while pgrep -f "^/home/ubuntu/venv/bin/python3 /home/ubuntu/venv/bin/yolo" >/dev/null; do sleep 60; done
# guard 延長 12 小時
for p in $(sudo pgrep -f "^/bin/bash /root/selfkill_core.sh"); do sudo kill $p; done
(sudo nohup setsid /root/selfkill_core.sh 43200 >/dev/null 2>&1 < /dev/null & disown); echo "guard re-armed 12h $(date -u)"
mkdir -p ~/weights/rA ~/weights/rA2
rclone copy $GDRIVE/runs/v7_rA_rtdetr-l/last.pt ~/weights/rA/ && rclone copy $GDRIVE/runs/v7_rA2_rtdetr-l/last.pt ~/weights/rA2/
ls -la ~/weights/rA/last.pt ~/weights/rA2/last.pt || { echo "FATAL: weights missing"; exit 2; }
if [ "$WITH_ORIG" = "1" ] && [ ! -d ~/scaleup/frames640_orig ]; then
  echo "=== [B1] original 20 videos: download rare_frames.tar.gz + crop  $(date -u) ==="
  rclone copy $GDRIVE/data/rare_frames.tar.gz ~/scaleup/ && mkdir -p ~/scaleup/orig_720 && tar -xzf ~/scaleup/rare_frames.tar.gz -C ~/scaleup/orig_720
  python3 - <<'PY'
import cv2, os, glob
from concurrent.futures import ThreadPoolExecutor
src_root = os.path.expanduser("~/scaleup/orig_720"); dst_root = os.path.expanduser("~/scaleup/frames640_orig")
files = glob.glob(os.path.join(src_root, "**", "*.jpg"), recursive=True); print("orig frames:", len(files))
def one(f):
    im = cv2.imread(f)
    if im is None or im.shape[0] < 720 or im.shape[1] < 1088: return 0
    vid = os.path.basename(os.path.dirname(f)); d = os.path.join(dst_root, vid); os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, os.path.basename(f)), cv2.resize(im[0:720, 192:1088], (640, 512), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95]); return 1
with ThreadPoolExecutor(24) as ex: print("cropped:", sum(ex.map(one, files)))
PY
  rm -rf ~/scaleup/orig_720 ~/scaleup/rare_frames.tar.gz
fi
echo "=== [B2] relabel shard $SHARD with WBF(rA, rA2)  $(date -u) ==="
TAG="pseudo_scale_r2_shard${SHARD}"
python3 ~/scaleup/pseudo_label_scale.py ~/scaleup/frames640 ~/scaleup/weak_scale.json ~/scaleup/$TAG ~/weights/rA/last.pt ~/weights/rA2/last.pt || { echo "FATAL pseudo r2"; exit 4; }
if [ "$WITH_ORIG" = "1" ]; then
  python3 ~/scaleup/weak_labels_scale.py ~/scaleup/frames640_orig ~/labels ~/scaleup/weak_orig.json
  python3 ~/scaleup/pseudo_label_scale.py ~/scaleup/frames640_orig ~/scaleup/weak_orig.json ~/scaleup/pseudo_scale_r2_orig ~/weights/rA/last.pt ~/weights/rA2/last.pt || echo "WARN orig relabel failed"
  (cd ~/scaleup && tar -czf pseudo_scale_r2_orig.tar.gz pseudo_scale_r2_orig && rclone copy pseudo_scale_r2_orig.tar.gz $GDRIVE/data/)
fi
(cd ~/scaleup && tar -czf $TAG.tar.gz $TAG && rclone copy $TAG.tar.gz $GDRIVE/data/ && rclone copy $TAG/stats.json $GDRIVE/data/${TAG}_stats.json); rclone lsl $GDRIVE/data/$TAG.tar.gz
echo "=== [B3] wait other shard r2  $(date -u) ==="
T0=$(date +%s)
while :; do rclone lsl "$GDRIVE/data/pseudo_scale_r2_shard${OTHER}.tar.gz" 2>/dev/null | grep -q . && break; [ $(( ($(date +%s) - T0) / 60 )) -ge 120 ] && { echo "other shard r2 timeout, train with own shard only"; break; }; sleep 120; done
EXTRA="pseudo_agybox_v1.tar.gz pseudo_llm_v1.tar.gz pseudo_scale_r2_shard0.tar.gz pseudo_scale_r2_shard1.tar.gz"
rclone lsl "$GDRIVE/data/pseudo_scale_r2_orig.tar.gz" 2>/dev/null | grep -q . && EXTRA="$EXTRA pseudo_scale_r2_orig.tar.gz"
HAVE=""; for f in $EXTRA; do rclone lsl "$GDRIVE/data/$f" 2>/dev/null | grep -q . && HAVE="$HAVE $f"; done
echo "=== [B4] train Round B with:$HAVE  $(date -u) ==="
rm -rf ~/cat1_yolo ~/cat1_yolo_v3 ~/pseudo_rare_v3 ~/pseudo_agy_v3 ~/pseudo_agybox_v1 ~/pseudo_scale_r1_shard* ~/pseudo_scale_r2_* ~/pseudo_llm_v1; rm -f ~/pseudo_scale_r*.tar.gz ~/pseudo_llm_v1.tar.gz
# 若 r2_orig 存在則不再併入 pseudo_rare_v3（同一批影片的舊標籤），避免重複
PSEUDO_TAR=pseudo_rare_v3.tar.gz; PSEUDO_DIR=pseudo_rare_v3
echo "$HAVE" | grep -q r2_orig && { PSEUDO_TAR=pseudo_agy_v3.tar.gz; PSEUDO_DIR=pseudo_agy_v3; }
export RUN_TAG MODELS="${MODELS:-rtdetr-l.pt}" IMGSZ=640 EPOCHS="${EPOCHS:-16}" BATCH="${BATCH:-32}" PATIENCE="${PATIENCE:-16}" CLOSE_MOSAIC="${CLOSE_MOSAIC:-3}" CACHE=ram WORKERS=12 \
       ALLGT=1 USE_PSEUDO=1 PSEUDO_TAR="$PSEUDO_TAR" PSEUDO_DIR="$PSEUDO_DIR" AGY_TAR=pseudo_agy_v3.tar.gz AGY_DIR=pseudo_agy_v3 EXTRA_TARS="$HAVE" UIBLUR_P=0 UIBLUR_VAL=0
bash ~/train_v5.sh; echo "ROUND B TRAIN EXIT $?  $(date -u)"
