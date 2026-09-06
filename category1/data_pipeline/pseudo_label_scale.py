#!/usr/bin/env python3
"""雙 teacher WBF 偽標籤（Lambda GPU）— 沿用 v4 strict 邏輯，加三條框級規則、不做尺寸過濾。
  規則：(1) prograsp(10)/stapler(11) 為忽略類：從目標與預測同時移除（老師對它們不可靠）
        (2) UI 橫幅假框逐框剔除：框中心 cy<34 或 >470（512px 座標）
        (3) 跨類同框（不同類 IoU>0.7）→ 整幀丟（老師自相矛盾）
  strict = 每類框數 == 安裝數 且 全部框 conf>=0.35
用法：python3 pseudo_label_scale.py <frames_dir> <weak.json> <out_dir> <teacher1.pt> [teacher2.pt ...]
"""
import json, shutil, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
from ultralytics import RTDETR
from ensemble_boxes import weighted_boxes_fusion

SRC, WEAK, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); TEACHERS = sys.argv[4:]
CONF_PRED, CONF_MIN, CONF_STRICT, WBF_IOU = 0.05, 0.20, 0.35, 0.7
IGNORE = {10, 11}; UI_TOP, UI_BOT = 34, 470; BATCH = 32
GT2ID = {"needle driver": 0, "monopolar curved scissors": 1, "force bipolar": 2, "clip applier": 3,
         "tip-up fenestrated grasper": 4, "cadiere forceps": 5, "bipolar forceps": 6, "vessel sealer": 7,
         "suction irrigator": 8, "bipolar dissector": 9, "prograsp forceps": 10, "stapler": 11,
         "permanent cautery hook/spatula": 12, "grasping retractor": 13}
weak = json.load(open(WEAK)); models = [RTDETR(t) for t in TEACHERS]
dev = "cuda" if torch.cuda.is_available() else "cpu"
(OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(parents=True, exist_ok=True)

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

stats, cls_kept = Counter(), Counter()
frames = sorted(SRC.rglob("*.jpg")); print(f"frames {len(frames)} | teachers {TEACHERS} | {dev}", flush=True); t0 = time.time()
for i in range(0, len(frames), BATCH):
    batch = frames[i:i+BATCH]; paths = [str(p) for p in batch]
    per_model = [m.predict(paths, conf=CONF_PRED, imgsz=640, max_det=300, device=dev, half=True, verbose=False) for m in models]
    for j, p in enumerate(batch):
        vid = p.parent.name; sec = int(p.stem.split("_")[1])
        installed = weak.get(vid, {}).get(str(sec)) or []
        target = Counter(GT2ID[t] for t in installed if t in GT2ID and GT2ID[t] not in IGNORE)
        if not target: stats["no_weak_label"] += 1; continue
        H, W = per_model[0][j].orig_shape
        BL, SL, LL = [], [], []
        for res in per_model:
            r = res[j]
            if len(r.boxes) == 0: BL.append(np.zeros((0, 4))); SL.append(np.zeros(0)); LL.append(np.zeros(0)); continue
            xyxy = r.boxes.xyxy.cpu().numpy() / np.array([W, H, W, H]); BL.append(np.clip(xyxy, 0, 1))
            SL.append(r.boxes.conf.cpu().numpy()); LL.append(r.boxes.cls.cpu().numpy())
        if len(models) > 1:
            b, s, l = weighted_boxes_fusion(BL, SL, LL, iou_thr=WBF_IOU, skip_box_thr=CONF_PRED, conf_type="max")
        else:
            b, s, l = BL[0], SL[0], LL[0]
        kept = []
        for bb, sc, lc in zip(b, s, l):
            cid = int(lc)
            if sc < CONF_MIN: continue
            if cid in IGNORE: stats["box_ignored_class"] += 1; continue
            if cid not in target: stats["box_dropped_not_installed"] += 1; continue
            cy = (bb[1] + bb[3]) / 2 * H
            if cy < UI_TOP or cy > UI_BOT: stats["box_dropped_ui_band"] += 1; continue
            kept.append((cid, float(sc), bb))
        if not kept: stats["frame_no_kept_box"] += 1; continue
        kept.sort(key=lambda k: -k[1]); by_cls = Counter(); trimmed = []
        for k in kept:
            if by_cls[k[0]] < target[k[0]]: trimmed.append(k); by_cls[k[0]] += 1
            else: stats["box_dropped_dup_over_target"] += 1
        if Counter(k[0] for k in trimmed) != target or any(k[1] < CONF_STRICT for k in trimmed):
            stats["frame_not_strict"] += 1; continue
        if any(iou(a[2], c[2]) > 0.7 for x, a in enumerate(trimmed) for c in trimmed[x+1:] if a[0] != c[0]):
            stats["frame_cross_class_same_box"] += 1; continue
        stats["frame_strict"] += 1
        stem = f"{vid}_sec{sec:06d}_strict"; shutil.copy(p, OUT / "images" / f"{stem}.jpg")
        lines = []
        for cid, sc, bb in trimmed:
            cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2; w, h = bb[2]-bb[0], bb[3]-bb[1]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"); cls_kept[cid] += 1
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    if (i // BATCH) % 100 == 0:
        print(f"progress {i}/{len(frames)} | {(time.time()-t0)/60:.1f} min | strict {stats['frame_strict']} | {dict(stats)}", flush=True)
print("FINAL STATS:", dict(stats), flush=True)
print("PER-CLASS kept boxes:", {k: cls_kept[k] for k in sorted(cls_kept)}, flush=True)
print(f"strict 幀率: {stats['frame_strict']}/{len(frames)} | {(time.time()-t0)/60:.1f} min", flush=True)
json.dump({"stats": dict(stats), "per_class": {int(k): v for k, v in cls_kept.items()}, "teachers": TEACHERS, "n_frames": len(frames)}, open(OUT / "stats.json", "w"), indent=1)
