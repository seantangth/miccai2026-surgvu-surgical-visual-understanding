#!/usr/bin/env bash
# Round A 訓練（Lambda）：等 gDrive 上的偽標籤包到齊 → 全新組資料 → train_v5.sh（s1 系配方＋ALLGT）→ 上傳權重
# 用法：RUN_TAG=v7_rA MODELS=rtdetr-l.pt EPOCHS=30 PATIENCE=30 CLOSE_MOSAIC=5 BATCH=32 WAIT_TARS="pseudo_scale_r1_shard0.tar.gz pseudo_scale_r1_shard1.tar.gz pseudo_opus_v1.tar.gz" WAIT_MIN=90 \
#       setsid nohup bash ~/scaleup/job_train_roundA.sh > ~/scaleup/job_train.log 2>&1 < /dev/null &
set -uo pipefail
cd ~
GDRIVE="gdrive:MICCAI_2026_SurgVU"
WAIT_TARS="${WAIT_TARS:-pseudo_scale_r1_shard0.tar.gz pseudo_scale_r1_shard1.tar.gz}"
OPT_TARS="${OPT_TARS:-pseudo_opus_v1.tar.gz}"        # 等不到也照跑
WAIT_MIN="${WAIT_MIN:-90}"
echo "=== wait for $WAIT_TARS (max $WAIT_MIN min)  $(date -u) ==="
T0=$(date +%s)
while :; do
  missing=""
  for f in $WAIT_TARS; do rclone lsl "$GDRIVE/data/$f" 2>/dev/null | grep -q . || missing="$missing $f"; done
  [ -z "$missing" ] && break
  [ $(( ($(date +%s) - T0) / 60 )) -ge "$WAIT_MIN" ] && { echo "WAIT TIMEOUT, missing:$missing"; break; }
  sleep 60
done
OPT_WAIT_MIN="${OPT_WAIT_MIN:-30}"; T1=$(date +%s)
while :; do
  ok=1; for f in $OPT_TARS; do rclone lsl "$GDRIVE/data/$f" 2>/dev/null | grep -q . || ok=0; done
  [ "$ok" = 1 ] && break
  [ $(( ($(date +%s) - T1) / 60 )) -ge "$OPT_WAIT_MIN" ] && { echo "optional tars not all present after $OPT_WAIT_MIN min, continue"; break; }
  sleep 60
done
HAVE=""; for f in $WAIT_TARS $OPT_TARS; do rclone lsl "$GDRIVE/data/$f" 2>/dev/null | grep -q . && HAVE="$HAVE $f"; done
echo "using extra tars:$HAVE  $(date -u)"
# 全新資料目錄（舊 cat1_yolo 已被 ALLGT 併過，不可重用）
rm -rf ~/cat1_yolo ~/cat1_yolo_v3 ~/pseudo_rare_v3 ~/pseudo_rare_v5 ~/pseudo_agy_v3 ~/pseudo_agy_v5 ~/pseudo_agybox_v1 ~/pseudo_scale_r1_shard* ~/pseudo_opus_v1
rm -f ~/pseudo_scale_r1_shard*.tar.gz ~/pseudo_opus_v1.tar.gz
export RUN_TAG="${RUN_TAG:-v7_rA}" MODELS="${MODELS:-rtdetr-l.pt}" IMGSZ=640 EPOCHS="${EPOCHS:-30}" BATCH="${BATCH:-32}" \
       PATIENCE="${PATIENCE:-30}" CLOSE_MOSAIC="${CLOSE_MOSAIC:-5}" CACHE=ram WORKERS="${WORKERS:-12}" \
       ALLGT=1 USE_PSEUDO=1 PSEUDO_TAR=pseudo_rare_v3.tar.gz PSEUDO_DIR=pseudo_rare_v3 AGY_TAR=pseudo_agy_v3.tar.gz AGY_DIR=pseudo_agy_v3 \
       EXTRA_TARS="pseudo_agybox_v1.tar.gz$HAVE" UIBLUR_P=0 UIBLUR_VAL=0
bash ~/train_v5.sh
echo "ROUND A TRAIN EXIT $?  $(date -u)"
