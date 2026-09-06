"""validate_elim_holdout.py — 在「器械在畫面上、但老師叫不出名字」這個正確區間測消去法。

做法（hold-one-class-out）：
  用 s1（訓練時排除官方影片 1、7）當老師，對影片 1、7 的 GT 幀推論。
  把某個類別 X 假裝成死類：把老師對 X 的所有預測整個丟掉，模擬「老師對 X 全盲」。
  其餘已知類照常各取 top-n 佔位，剩下未被認領的框跑消去法指派給 X，
  再與 X 的真實框比 IoU。X 一定在畫面上（GT 有框），所以這正是 retractor 的處境。

用法：bsenv/bin/python validate_elim_holdout.py [--models s1=path ...] [--classes 5 3 ...]
"""
import argparse, collections, json, os, sys
from pathlib import Path
import numpy as np, torch
from ultralytics import RTDETR

ROOT = Path(os.environ.get("SURGVU_ROOT", "."))
DATA = ROOT / "1_data/processed/cat1_yolo"          # val = 影片 1、7（s1 沒訓練過）
KNOWN_ALL = {0, 1, 2, 3, 5, 6, 7, 12}
KNOWN_CONF, LEFT_CONF = 0.35, 0.25
UI_TOP, UI_BOT = 34, 470
NAMES = ["needle_driver", "monopolar_curved_scissor", "force_bipolar", "clip_applier", "tip_up_fenestrated_grasper",
         "cadiere_forceps", "bipolar_forceps", "vessel_sealer", "suction_irrigator", "bipolar_dissector",
         "prograsp_forceps", "stapler", "permanent_cautery_hook_spatula", "grasping_retractor"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "4_models/v3/rtdetr_v3_s1/best.pt"))
ap.add_argument("--split", default="val")
ap.add_argument("--classes", type=int, nargs="+", default=[5, 3, 7, 1])   # cadiere / clip / vessel / mono
ap.add_argument("--batch", type=int, default=16)
A = ap.parse_args()

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

# 讀 GT
items = []
ld = DATA / "labels" / A.split
for lf in sorted(ld.iterdir()):
    if lf.suffix != ".txt": continue
    rows = [l.split() for l in lf.read_text().splitlines() if l.strip()]
    img = DATA / "images" / A.split / (lf.stem + ".jpg")
    if rows and img.exists(): items.append((img, rows))
print(f"{A.split} frames with GT: {len(items)}  model={Path(A.model).parent.name}", flush=True)

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = RTDETR(A.model)
cache = []
for i in range(0, len(items), A.batch):
    chunk = items[i:i+A.batch]
    res = model.predict([str(p) for p, _ in chunk], conf=0.05, imgsz=640, max_det=300, device=dev,
                        half=(dev != "cpu"), verbose=False)
    for (img, rows), r in zip(chunk, res):
        H, W = r.orig_shape
        gt = collections.defaultdict(list)
        for row in rows:
            c = int(row[0]); cx, cy, w, h = (float(x) for x in row[1:5])
            gt[c].append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
        dets = []
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            ycen = (b[1] + b[3]) / 2
            if ycen < UI_TOP or ycen > UI_BOT: continue
            dets.append([int(c), float(s), b.tolist()])
        dets.sort(key=lambda d: -d[1])
        cache.append((dict(gt), dets))
    if i % 320 == 0: print(f"  infer {i}/{len(items)}", flush=True)

out = {}
for X in A.classes:
    n_frames = sum(1 for gt, _ in cache if X in gt)
    if n_frames == 0: print(f"skip {NAMES[X]}: 0 GT frames"); continue
    KNOWN = KNOWN_ALL - {X}
    st = collections.Counter(); hits = []
    for gt, dets_all in cache:
        if X not in gt: continue
        T = collections.Counter({c: len(v) for c, v in gt.items()})
        dets = [d for d in dets_all if d[0] != X]              # ★ 老師對 X 全盲
        assigned, used = [], set(); short = False
        for c in sorted(T):
            if c not in KNOWN: continue
            cand = [k for k, d in enumerate(dets) if d[0] == c and d[1] >= KNOWN_CONF and k not in used]
            if len(cand) < T[c]: short = True; break
            for k in cand[:T[c]]: assigned.append((c, dets[k][2])); used.add(k)
        if short: st["known_short"] += 1; continue
        left = [k for k, d in enumerate(dets) if k not in used and d[1] >= LEFT_CONF
                and all(iou(d[2], a[1]) < 0.5 for a in assigned)]
        merged = []
        for k in left:
            if all(iou(dets[k][2], dets[m][2]) < 0.5 for m in merged): merged.append(k)
        need = T[X]
        if need == 1 and len(merged) == 1:
            best = max(iou(dets[merged[0]][2], g) for g in gt[X]); hits.append(best); st["assigned"] += 1
        else:
            st[f"unresolved_need{need}_left{len(merged)}"] += 1; st["unresolved"] += 1
    a = np.array(hits) if hits else np.array([0.0])
    cov = st["assigned"] / max(1, n_frames)
    out[NAMES[X]] = {"gt_frames": n_frames, "assigned": st["assigned"], "coverage": round(cov, 3),
                     "iou>=0.5": round(float((a >= 0.5).mean()), 3) if hits else None,
                     "iou>=0.3": round(float((a >= 0.3).mean()), 3) if hits else None,
                     "median_iou": round(float(np.median(a)), 3) if hits else None,
                     "zero_overlap": round(float((a == 0).mean()), 3) if hits else None,
                     "stats": dict(st)}
    print(f"\n== 假裝 {NAMES[X]} 是死類（老師對它全盲）==")
    print(f"  含該類 GT 幀 {n_frames} | 消去法指派成功 {st['assigned']} ({cov:.1%}) | 無解 {st['unresolved']} | 已知類湊不齊 {st['known_short']}")
    if hits:
        print(f"  指派框 vs 真實框：IoU>=0.5 {(a>=0.5).mean():.3f} | IoU>=0.3 {(a>=0.3).mean():.3f} | 中位 {np.median(a):.3f} | 完全落空 {(a==0).mean():.3f}")
    print(f"  細節：{dict(st)}")

json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/validate_elim_holdout.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved validate_elim_holdout.json")
