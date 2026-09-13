#!/usr/bin/env bash
# v12_rA2_800：rA2 的完整配方（同資料 93,660 幀、ALLGT、AdamW、16 ep、close_mosaic 6）只把 imgsz 640 → 800，batch 24 → 20（40 GB 顯存）。
# 依據 EXP-28：同資料 640 vs 800 受控對照 mean +0.0131、AP75 +0.0275。用途：Final #3 常見類 rA2+rA2_800 WBF（容器 v7 routing 指定 imgsz 800）。
# 鐵律 G：先保存、後關機——train_v5.sh 練完立刻 rclone 上傳；本腳本結尾以 rclone lsl 驗證 last.pt 在雲端後才自毀。
set -uo pipefail
cd ~
cat > ~/train_v5.sh <<'TRAINV5EOF'
#!/usr/bin/env bash
# ============================================================================
# SurgVU 2026 Cat 1 — train_v5.sh（草案，尚未執行）
#
# 相對 v4 的四項修正（全部有源碼／results.csv 證據，見 agentC 稽核）：
#   1. optimizer 顯式化：ultralytics 8.4.137 `optimizer=auto` 在
#      iterations = ceil(N_train/max(batch,nbs=64)) * epochs > 10000 時
#      會從 AdamW(lr≈0.000556) 靜默翻成 **MuSGD(lr=0.01)**（trainer.py:1116）。
#      v4 資料 10,007 幀 ×60ep = 9,420（AdamW）；一旦 ALLGT(+1,929) 或 epochs↑
#      就越過 10,000 → 配方靜默大變。本腳本強制顯式並列印 iterations。
#   2. warmup_bias_lr=0.0 必帶：auto 分支會順手設 warmup_bias_lr=0（同上行下一行）。
#      只寫 `optimizer=AdamW lr0=...` 而漏掉它 → bias 群組 warmup 峰值 lr=0.1。
#      實證：runs/rtdetr_v3_s2p/results.csv epoch1 lr/pg2 = **0.0669**（比目標 1e-4 大 669×），
#      mAP 崩到 0.018 → 0.083 → 0.067，best 只回到 0.482。s2(auto，pg2 乾淨)=0.533。
#      ⇒ EXP-12「低 LR finetune 更差」有一半是這個 bug，不是分佈假設。
#   3. 讓 schedule 跑完：s1 設 epochs=60/patience=12，實際 **epoch 34 早停**，
#      close_mosaic 在 epoch 50 才會觸發 ⇒ **s1 從頭到尾 mosaic 全開、LR 只退到 0.455×lr0**。
#      v5 用 epochs=40 + close_mosaic=10（第 30 epoch 關 mosaic）+ patience>=epochs（不早停）。
#   4. UI 帶模糊增強（域衛生，收益小但幾乎零成本）：
#      官方 val_public/訓練幀底部有 da Vinci UI **逐字寫著工具名**
#      （LARGE NEEDLE DRIVER / CADIERE FORCEPS…），但官方測試集 UI 被模糊
#      （6_papers/references/UI_blur_RTOhCor.png）⇒ 該資訊在測試端消失。
#      ★ 已實測（ui_blur_probe.py，s1 on val 1,929 幀）：把 val 的上下 26px 帶
#        高斯模糊後 mAP50-95 0.5660 → 0.5618（-0.0042）⇒ **模型並沒有大量抄 UI 捷徑**，
#        本項的期望增益上限只有 ~0.004。保留是因為「測試端 UI 一律模糊」，
#        對齊域幾乎免費；不要把它當主要槓桿。
#
# 用法（全部用環境變數；不帶參數＝run #1 建議配方）：
#   RUN_TAG=v5_a IMGSZ=640 MODELS="rtdetr-l.pt" EPOCHS=40 BATCH=24 \
#   ALLGT=0 UIBLUR_P=0.5 bash train_v5.sh
#
# 硬性規則：本腳本只在 Lambda 機器上跑；不改 ROOT/3_src 既有檔案。
# ============================================================================
set -euo pipefail
cd ~

# ---------------------------- 可調參數 --------------------------------------
RUN_TAG="${RUN_TAG:-v5_a}"                 # gDrive runs/<RUN_TAG>_<model> 命名前綴
MODELS="${MODELS:-rtdetr-l.pt}"            # 空白分隔：rtdetr-l.pt rtdetr-x.pt yolo11l.pt yolo11x.pt
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-24}"                       # -1 = ultralytics AutoBatch
PATIENCE="${PATIENCE:-$EPOCHS}"            # 預設不早停（讓 close_mosaic + LR anneal 跑完）
CLOSE_MOSAIC="${CLOSE_MOSAIC:-10}"
CACHE="${CACHE:-ram}"                      # imgsz>=960 建議改 disk（960²×3×12k≈33GB RAM）
WORKERS="${WORKERS:-8}"

ALLGT="${ALLGT:-0}"                        # 1 = 把 val(影片1,7) 併入 train（final 用；val 分數變記憶分數）
USE_PSEUDO="${USE_PSEUDO:-1}"              # 1 = 併入 ${PSEUDO_DIR:-pseudo_rare_v5}（淨化版，3,747 幀）+ ${AGY_DIR:-pseudo_agy_v5}

UIBLUR_P="${UIBLUR_P:-1.0}"                # train 幀中做 UI 帶模糊的比例（0=關閉）
                                           # 測試端 UI 一律模糊 ⇒ 預設 1.0；留 0.5 只為當正則化實驗
UIBLUR_TALL="${UIBLUR_TALL:-0}"            # 1 = 隨機用 62/92px 高帶（會毀標籤，預設關）
UIBLUR_VAL="${UIBLUR_VAL:-1}"              # 1 = val 幀 100% 模糊（讓訓練期 val 貼近測試域）
UIBLUR_TOP="${UIBLUR_TOP:-26}"             # 上方橫幅高度 px（640×512 座標）
UIBLUR_BOT="${UIBLUR_BOT:-26}"             # 下方 UI 面板高度 px；面板展開態可到 ~92

# 顯式 optimizer —— 預設值＝忠實複刻 s1 實跑的 auto 分支（AdamW, nc=14）
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-0.000556}"                     # = round(0.002*5/(4+14), 6)
LRF="${LRF:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"                # auto 分支用 0.9（不是 default.yaml 的 0.937）
WARMUP_BIAS_LR="${WARMUP_BIAS_LR:-0.0}"    # ★ 漏掉會炸 bias 群組（見上方 #2）
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
COS_LR="${COS_LR:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"               # 例：scale=0.75 degrees=5 multi_scale=0.3

GDRIVE="${GDRIVE:-gdrive:MICCAI_2026_SurgVU}"

VOCAB=(needle_driver monopolar_curved_scissor force_bipolar clip_applier \
       tip_up_fenestrated_grasper cadiere_forceps bipolar_forceps vessel_sealer \
       suction_irrigator bipolar_dissector prograsp_forceps stapler \
       permanent_cautery_hook_spatula grasping_retractor)

echo "=========================================================="
echo " train_v5  RUN_TAG=$RUN_TAG  MODELS='$MODELS'"
echo " IMGSZ=$IMGSZ EPOCHS=$EPOCHS BATCH=$BATCH PATIENCE=$PATIENCE CLOSE_MOSAIC=$CLOSE_MOSAIC"
echo " ALLGT=$ALLGT USE_PSEUDO=$USE_PSEUDO UIBLUR_P=$UIBLUR_P UIBLUR_VAL=$UIBLUR_VAL"
echo " OPT=$OPTIMIZER lr0=$LR0 lrf=$LRF momentum=$MOMENTUM warmup_bias_lr=$WARMUP_BIAS_LR"
echo "=========================================================="

# ---------------------------- [1] 環境 --------------------------------------
echo "=== [1/7] 環境 ==="
command -v rclone >/dev/null || { curl -fsSL https://rclone.org/install.sh | sudo bash; }
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
# ★ 釘死版本：容器與訓練必須同版。8.4.x 的 RTDETRDecoder.postprocess 已是
#   query×class 攤平 top-k（多標籤），舊版是 per-query argmax——換版會靜默改變語意。
pip install -q "ultralytics==8.4.137"
python3 -c "import ultralytics,torch;print('ultralytics',ultralytics.__version__,'torch',torch.__version__)"

# ---------------------------- [2] 拉資料 ------------------------------------
echo "=== [2/7] 拉資料 ==="
FILES="cat1_yolo_v3.tar.gz"
[ "$USE_PSEUDO" = "1" ] && FILES="$FILES ${PSEUDO_TAR:-pseudo_rare_v5.tar.gz} ${AGY_TAR:-pseudo_agy_v5.tar.gz}"
EXTRA_TARS="${EXTRA_TARS:-}"                # 例：pseudo_agybox_v1.tar.gz（agy 重標 retractor/stapler）
[ -n "$EXTRA_TARS" ] && FILES="$FILES $EXTRA_TARS"
for f in $FILES; do
  [ -f "$f" ] || rclone copy "$GDRIVE/data/$f" .
  tar xzf "$f"
done
[ -d cat1_yolo ] || mv cat1_yolo_v3 cat1_yolo

# ---------------------------- [3] 組資料 ------------------------------------
echo "=== [3/7] 組資料 ==="
if [ "$USE_PSEUDO" = "1" ]; then
  EXTRA_DIRS=""; for x in $EXTRA_TARS; do EXTRA_DIRS="$EXTRA_DIRS ${x%.tar.gz}"; done
  for d in ${PSEUDO_DIR:-pseudo_rare_v5} ${AGY_DIR:-pseudo_agy_v5} $EXTRA_DIRS; do
    # find -exec：避免 5 萬檔案的 glob 超過 ARG_MAX（2026-09-03 02:00 事故：分片靜默未併入，train 只剩 11,508 幀）
    n0=$(ls cat1_yolo/images/train | wc -l)
    [ -d "$d/images" ] && find "$d/images" -maxdepth 1 -name "*.jpg" -exec cp -n -t cat1_yolo/images/train/ {} +
    [ -d "$d/labels" ] && find "$d/labels" -maxdepth 1 -name "*.txt" -exec cp -n -t cat1_yolo/labels/train/ {} +
    echo "  [merge] $d: +$(( $(ls cat1_yolo/images/train | wc -l) - n0 )) imgs (dir has $(ls "$d/images" 2>/dev/null | wc -l))"
  done
fi

# --- ALLGT：把官方 val(影片1,7) 併進 train ---------------------------------
# 風險（必讀）：併入後 val 目錄＝訓練資料的子集，訓練期 val 分數是「記憶分數」，
#   只能當 sanity（NaN/崩潰/類別分佈異常偵測），**絕不可拿來選模型或比較 run**。
#   ⇒ ALLGT=1 時本腳本額外上傳 last.pt，並在報告中明示「請用 last.pt，不要用 best.pt」。
VAL_DIR="$HOME/cat1_yolo/images/val"
if [ "$ALLGT" = "1" ]; then
  echo "  [ALLGT] 併入 val(影片1,7) → train（hardlink，保留 val 目錄供 sanity 監控）"
  # ★ 必須 cp 不能 ln：步驟[4] 的 UI 模糊是 in-place 覆寫，hardlink 會經同一個 inode
  #   把 val 影格一起改掉（且高度隨機），val 目錄就再也不是乾淨的監控集。
  n_before=$(ls cat1_yolo/images/train | wc -l)
  for f in cat1_yolo/images/val/*.jpg; do
    bn=$(basename "$f"); cp "$f" "cat1_yolo/images/train/allgt_$bn"
  done
  for f in cat1_yolo/labels/val/*.txt; do
    bn=$(basename "$f"); cp "$f" "cat1_yolo/labels/train/allgt_${bn}"
  done
  n_after=$(ls cat1_yolo/images/train | wc -l)
  n_pair_i=$(ls cat1_yolo/images/train/allgt_*.jpg | wc -l)
  n_pair_l=$(ls cat1_yolo/labels/train/allgt_*.txt | wc -l)
  echo "  [ALLGT] train $n_before -> $n_after (allgt imgs=$n_pair_i labels=$n_pair_l)"
  [ "$((n_after - n_before))" -eq 1929 ] || { echo "FATAL: ALLGT 併入數不是 1929（實得 $((n_after-n_before)))" >&2; exit 6; }
  [ "$n_pair_i" -eq "$n_pair_l" ] || { echo "FATAL: ALLGT image/label 不成對 ($n_pair_i vs $n_pair_l)" >&2; exit 6; }
fi

N_TRAIN=$(ls cat1_yolo/images/train | wc -l)
N_VAL=$(ls cat1_yolo/images/val | wc -l)
N_LBL_T=$(ls cat1_yolo/labels/train | wc -l)
echo "train imgs: $N_TRAIN (labels $N_LBL_T), val imgs: $N_VAL"
MIN_TRAIN=3000; [ "$USE_PSEUDO" = "1" ] && MIN_TRAIN=6500
[ "$ALLGT" = "1" ] && MIN_TRAIN=$((MIN_TRAIN + 1900))
if [ "$N_TRAIN" -lt "$MIN_TRAIN" ] || [ "$N_VAL" -lt 1900 ]; then
  echo "FATAL: dataset incomplete (train=$N_TRAIN < $MIN_TRAIN, val=$N_VAL)" >&2; exit 2
fi
[ "$N_TRAIN" -eq "$N_LBL_T" ] || { echo "FATAL: image/label 數不等 ($N_TRAIN vs $N_LBL_T)" >&2; exit 2; }

# --- fail-fast：解析度抽查（域對齊鐵證，EXP-10 制度化）---------------------
python3 - <<'PYEOF'
import glob, os, random, struct, sys
def jpg_size(p):
    d = open(p, "rb").read(64000); i = 2
    while i < len(d) - 8:
        if d[i] != 0xFF: i += 1; continue
        if d[i+1] in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", d[i+5:i+9]); return w, h
        i += 2 + struct.unpack(">H", d[i+2:i+4])[0]
    return None, None
files = glob.glob(os.path.expanduser("~/cat1_yolo/images/train/*.jpg"))
bad = [f for f in random.sample(files, min(80, len(files))) if jpg_size(f) != (640, 512)]
if bad:
    print(f"FATAL: {len(bad)}/80 train frames not 640x512 — domain mismatch! e.g. {bad[:3]}", file=sys.stderr)
    sys.exit(3)
print(f"resolution check OK ({len(files)} train frames, 80 sampled, all 640x512)")
PYEOF

# ---------------------------- [4] UI 帶模糊 ---------------------------------
# 為什麼：val_public / 訓練幀底部 UI 面板逐字印著工具名（"CADIERE FORCEPS"…），
#   RT-DETR 的 AIFI 是全域 self-attention，完全有能力抄這個捷徑；
#   官方測試集把 UI 模糊掉 ⇒ 捷徑消失。本步驟把訓練域拉向測試域。
# 安全性：band y>486px 只碰到 4.2% 的 GT train 框、把 >50% 面積吃掉的只有 0.15%。
if [ "$(python3 -c "print(1 if float('$UIBLUR_P')>0 else 0)")" = "1" ]; then
echo "=== [4/7] UI 帶模糊（train p=$UIBLUR_P, val=$UIBLUR_VAL, top=$UIBLUR_TOP bot=$UIBLUR_BOT）==="
UIBLUR_P="$UIBLUR_P" UIBLUR_VAL="$UIBLUR_VAL" UIBLUR_TOP="$UIBLUR_TOP" UIBLUR_BOT="$UIBLUR_BOT" \
UIBLUR_TALL="${UIBLUR_TALL:-0}" \
python3 - <<'PYEOF'
import cv2, glob, os, random
p = float(os.environ["UIBLUR_P"]); do_val = os.environ["UIBLUR_VAL"] == "1"
tall = os.environ.get("UIBLUR_TALL", "0") == "1"
top = int(os.environ["UIBLUR_TOP"]); bot = int(os.environ["UIBLUR_BOT"])
random.seed(0)
def blur(f, tp, bp):
    im = cv2.imread(f)
    if im is None: return False
    h = im.shape[0]
    if tp: im[:tp] = cv2.GaussianBlur(im[:tp], (31, 31), 12)
    if bp: im[h-bp:] = cv2.GaussianBlur(im[h-bp:], (31, 31), 12)
    cv2.imwrite(f, im, [cv2.IMWRITE_JPEG_QUALITY, 95]); return True
home = os.path.expanduser("~")
n = 0
for f in sorted(glob.glob(f"{home}/cat1_yolo/images/train/*.jpg")):
    # 面板展開態高度不定 → 隨機 band 高度，讓模型對兩種狀態都不敏感
    # 高帶（62/92px，面板展開態）預設關閉：y>420 會把 8.1% 的 train 框吃掉一半以上面積
    # 卻仍保留標籤 ⇒ 教模型「往模糊區塞框」。要開請設 UIBLUR_TALL=1 並自行處理標籤。
    bp = random.choice([bot, bot, 62, 92]) if tall else bot
    if random.random() < p and blur(f, top, bp): n += 1
print(f"[uiblur] train blurred {n} frames")
if do_val:
    m = sum(blur(f, top, bot) for f in sorted(glob.glob(f"{home}/cat1_yolo/images/val/*.jpg")))
    print(f"[uiblur] val blurred {m} frames (訓練期 val 因此貼近測試域)")
PYEOF
else
  echo "=== [4/7] UI 帶模糊：關閉（UIBLUR_P=0）==="
fi

# ---------------------------- [5] dataset.yaml + 逐字校驗 -------------------
echo "=== [5/7] dataset.yaml ==="
{
  echo "path: $HOME/cat1_yolo"
  echo "train: images/train"
  echo "val: images/val"
  echo "names:"
  i=0; for n in "${VOCAB[@]}"; do echo "  $i: $n"; i=$((i+1)); done
} > cat1_yolo/dataset.yaml
# 逐字校驗（類名錯一個字，官方評分該類直接 0）
python3 - <<'PYEOF'
import sys, yaml
V = ["needle_driver","monopolar_curved_scissor","force_bipolar","clip_applier",
     "tip_up_fenestrated_grasper","cadiere_forceps","bipolar_forceps","vessel_sealer",
     "suction_irrigator","bipolar_dissector","prograsp_forceps","stapler",
     "permanent_cautery_hook_spatula","grasping_retractor"]
import os
d = yaml.safe_load(open(os.path.expanduser("~/cat1_yolo/dataset.yaml")))
got = [d["names"][i] for i in range(len(d["names"]))]
if got != V:
    print("FATAL: vocabulary mismatch\n got=%s\n exp=%s" % (got, V), file=sys.stderr); sys.exit(7)
print("vocabulary check OK: 14/14 逐字相符（底線格式、單數）")
PYEOF

# ---------------------------- [6] 訓練 --------------------------------------
echo "=== [6/7] 訓練 ==="
# optimizer 護欄：auto 一律拒絕（見檔頭 #1）
[ "$OPTIMIZER" = "auto" ] && { echo "FATAL: OPTIMIZER=auto 被禁止（>10000 iterations 會靜默翻 MuSGD）" >&2; exit 8; }
NBS=64
EFF_B=$BATCH; [ "$BATCH" -lt 1 ] 2>/dev/null && EFF_B=16   # AutoBatch 時用 16 估算
ITER=$(python3 -c "import math;print(math.ceil($N_TRAIN/max($EFF_B,$NBS))*$EPOCHS)")
echo "  iterations = ceil($N_TRAIN/max($EFF_B,$NBS))*$EPOCHS = $ITER  （auto 分支在 >10000 會翻 MuSGD；本 run 已顯式指定 $OPTIMIZER）"
echo "  close_mosaic 會在 epoch $((EPOCHS - CLOSE_MOSAIC)) 觸發（patience=$PATIENCE，$( [ "$PATIENCE" -ge "$EPOCHS" ] && echo '不早停 ⇒ 一定會觸發' || echo '⚠ 可能早停於 close_mosaic 之前' )）"

for M in $MODELS; do
  BASE=$(basename "$M" .pt)
  NAME="${RUN_TAG}_${BASE}"
  echo "----- train $NAME (model=$M imgsz=$IMGSZ) -----"
  EXTRA_M=""
  case "$BASE" in
    yolo11*) EXTRA_M="mixup=0.1 degrees=10 hsv_v=0.5" ;;   # v4 的 y11l 配方
  esac
  yolo detect train \
    data="$HOME/cat1_yolo/dataset.yaml" model="$M" \
    epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" patience="$PATIENCE" \
    close_mosaic="$CLOSE_MOSAIC" cos_lr="$COS_LR" \
    optimizer="$OPTIMIZER" lr0="$LR0" lrf="$LRF" momentum="$MOMENTUM" \
    warmup_bias_lr="$WARMUP_BIAS_LR" weight_decay="$WEIGHT_DECAY" \
    project="$HOME/runs" name="$NAME" exist_ok=True workers="$WORKERS" cache="$CACHE" \
    $EXTRA_M $EXTRA_ARGS 2>&1 | tail -6
  # 練完立刻搶救（selfkill 開槍也不陪葬）；ALLGT 時 last.pt 才是可用權重
  rclone copy "$HOME/runs/$NAME/weights/best.pt" "$GDRIVE/runs/$NAME/"
  rclone copy "$HOME/runs/$NAME/weights/last.pt" "$GDRIVE/runs/$NAME/"
  rclone copy "$HOME/runs/$NAME/results.csv"     "$GDRIVE/runs/$NAME/" || true
  rclone copy "$HOME/runs/$NAME/args.yaml"       "$GDRIVE/runs/$NAME/" || true
  # 立即回報實際生效的 optimizer / lr（防止「以為設了其實沒生效」）
  grep -E "^(optimizer|lr0|lrf|momentum|warmup_bias_lr|close_mosaic|epochs|imgsz|batch):" \
       "$HOME/runs/$NAME/args.yaml" || true
  python3 - "$HOME/runs/$NAME/results.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
k = "metrics/mAP50-95(B)"
b = max(rows, key=lambda r: float(r[k]))
print(f"  [{sys.argv[1]}] epochs_run={len(rows)} best_ep={b['epoch']} best_mAP={float(b[k]):.4f} "
      f"last_mAP={float(rows[-1][k]):.4f} ep1_lr_pg2={rows[0]['lr/pg2']} (應≈0，非 0.0669)")
PYEOF
done

# ---------------------------- [7] val 報告 ----------------------------------
echo "=== [7/7] val 報告 ==="
if [ "$ALLGT" = "1" ]; then
  echo "  ⚠⚠ ALLGT=1：以下 val 分數是【記憶分數】，只當 sanity（不崩、類別分佈正常）。"
  echo "  ⚠⚠ 提交請用 last.pt（best.pt 是在記憶資料上挑的）。"
fi
for M in $MODELS; do
  NAME="${RUN_TAG}_$(basename "$M" .pt)"
  for W in best last; do
    echo "--- $NAME/$W ---"
    yolo detect val data="$HOME/cat1_yolo/dataset.yaml" \
        model="$HOME/runs/$NAME/weights/$W.pt" imgsz="$IMGSZ" 2>&1 | tail -20
  done
done
echo "ALL DONE $(date -u)"
TRAINV5EOF
chmod +x ~/train_v5.sh
GDRIVE="gdrive:MICCAI_2026_SurgVU"
echo "########## RUN v12_rA2_800: rtdetr-l 800 rA2 data ALLGT 16ep  $(date -u) ##########"
RUN_TAG=v12_rA2_800 MODELS=rtdetr-l.pt IMGSZ=800 EPOCHS=16 BATCH=20 PATIENCE=16 CLOSE_MOSAIC=6 CACHE=ram WORKERS=12 \
  ALLGT=1 USE_PSEUDO=1 PSEUDO_TAR=pseudo_rare_v3.tar.gz PSEUDO_DIR=pseudo_rare_v3 AGY_TAR=pseudo_agy_v3.tar.gz AGY_DIR=pseudo_agy_v3 \
  EXTRA_TARS="pseudo_agybox_v1.tar.gz pseudo_scale_r1_shard0.tar.gz pseudo_scale_r1_shard1.tar.gz pseudo_llm_v1.tar.gz" \
  UIBLUR_P=0 UIBLUR_VAL=0 bash ~/train_v5.sh
echo "TRAIN EXIT $?  $(date -u)"
# 保存驗證（鐵律 G）：last.pt 必須在雲端且 >50 MB，否則不關機、留給人工處理
cp ~/job.log ~/train_v12_rA2_800.log 2>/dev/null; rclone copy ~/train_v12_rA2_800.log "$GDRIVE/runs/v12_rA2_800_rtdetr-l/" || true
SZ=$(rclone lsl "$GDRIVE/runs/v12_rA2_800_rtdetr-l/last.pt" 2>/dev/null | awk '{print $1}')
echo "cloud last.pt size: ${SZ:-none}"
if [ -n "${SZ:-}" ] && [ "$SZ" -gt 50000000 ]; then
  echo "SAVED_OK → self-terminate in 60s  $(date -u)"; sleep 60; sudo /root/selfkill_core.sh 1
else
  echo "SAVE_NOT_VERIFIED → NOT terminating; guard deadline will handle it  $(date -u)"
fi
