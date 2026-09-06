"""validate_assign_hungarian.py — 類別無關偵測＋整體最佳指派，對照「已知類先佔位＋消去法」。

動機（validate_elim_holdout.py 的發現）：消去法最常見的失敗是「剩餘候選 0 個」，
代表老師看見了那個物體、只是叫錯名字，然後在已知類佔位那步被吃掉。
所以改成：框歸框、類別歸類別。

方法：
  1. 老師以 conf>=0.02、max_det=300 推論。RT-DETR 8.4.x 的 postprocess 是 query×class 攤平 top-k，
     同一個物體會以多個類別重複出現 ⇒ 用類別無關的 IoU>=CLUSTER_IOU 分群，得到「物件」清單，
     每個物件保留一個代表框（分數最高者）與 per-class 最高分向量。
  2. 介面列（此處用 GT 類別多重集合模擬）說在場的是 K 支工具 ⇒ 取分數最高的 K 個物件。
  3. 用匈牙利演算法做整體指派：cost = -log(score[物件][工具])，
     老師全盲的類別給一個固定的低分 FLOOR（它對任何物件都一樣，等於「剩下的給它」）。
  4. 與真實框比 IoU。

用法：bsenv/bin/python validate_assign_hungarian.py [--classes 5 3 7 1]
"""
import argparse, collections, json, os
from pathlib import Path
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from ultralytics import RTDETR

ROOT = Path(os.environ.get("SURGVU_ROOT", "."))
DATA = ROOT / "1_data/processed/cat1_yolo"
UI_TOP, UI_BOT = 34, 470
CLUSTER_IOU, MIN_CONF, FLOOR = 0.6, 0.02, 0.01
NAMES = ["needle_driver", "monopolar_curved_scissor", "force_bipolar", "clip_applier", "tip_up_fenestrated_grasper",
         "cadiere_forceps", "bipolar_forceps", "vessel_sealer", "suction_irrigator", "bipolar_dissector",
         "prograsp_forceps", "stapler", "permanent_cautery_hook_spatula", "grasping_retractor"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "4_models/v3/rtdetr_v3_s1/best.pt"))
ap.add_argument("--split", default="val"); ap.add_argument("--classes", type=int, nargs="+", default=[5, 3, 7, 1])
ap.add_argument("--batch", type=int, default=16)
A = ap.parse_args()

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

items = []
for lf in sorted((DATA / "labels" / A.split).iterdir()):
    if lf.suffix != ".txt": continue
    rows = [l.split() for l in lf.read_text().splitlines() if l.strip()]
    img = DATA / "images" / A.split / (lf.stem + ".jpg")
    if rows and img.exists(): items.append((img, rows))
print(f"{A.split} frames {len(items)}  model={Path(A.model).parent.name}", flush=True)

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = RTDETR(A.model); cache = []
for i in range(0, len(items), A.batch):
    chunk = items[i:i+A.batch]
    res = model.predict([str(p) for p, _ in chunk], conf=MIN_CONF, imgsz=640, max_det=300, device=dev,
                        half=(dev != "cpu"), verbose=False)
    for (img, rows), r in zip(chunk, res):
        H, W = r.orig_shape
        gt = collections.defaultdict(list)
        for row in rows:
            c = int(row[0]); cx, cy, w, h = (float(x) for x in row[1:5])
            gt[c].append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
        raw = []
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            yc = (b[1] + b[3]) / 2
            if yc < UI_TOP or yc > UI_BOT: continue
            raw.append([int(c), float(s), b.tolist()])
        raw.sort(key=lambda d: -d[1])
        # 類別無關分群 → 物件
        objs = []            # [{box, score, cls_score: {c: s}}]
        for c, s, b in raw:
            hit = None
            for o in objs:
                if iou(b, o["box"]) >= CLUSTER_IOU: hit = o; break
            if hit is None: objs.append({"box": b, "score": s, "cls_score": {c: s}})
            else: hit["cls_score"][c] = max(hit["cls_score"].get(c, 0.0), s)
        cache.append((dict(gt), objs))
    if i % 320 == 0: print(f"  infer {i}/{len(items)}", flush=True)
print(f"平均每幀物件數 {np.mean([len(o) for _, o in cache]):.2f}", flush=True)

out = {}
for X in A.classes:
    n_frames = sum(1 for gt, _ in cache if X in gt); st = collections.Counter(); hits = []
    for gt, objs in cache:
        if X not in gt: continue
        tools = []                                   # 在場工具（模擬介面列讀到的）
        for c, v in gt.items(): tools += [c] * len(v)
        K = len(tools)
        cand = [o for o in objs if o["score"] >= MIN_CONF][:max(K, 1) * 3]
        if len(cand) < K: st["objects_fewer_than_tools"] += 1; continue
        cand = cand[:K] if len(cand) > K else cand   # 取分數最高的 K 個物件
        C = np.zeros((K, len(cand)))
        for ti, t in enumerate(tools):
            for oi, o in enumerate(cand):
                s = FLOOR if t == X else o["cls_score"].get(t, FLOOR)   # ★ 老師對 X 全盲
                C[ti, oi] = -np.log(max(s, 1e-6))
        ri, ci = linear_sum_assignment(C)
        for ti, oi in zip(ri, ci):
            if tools[ti] != X: continue
            best = max(iou(cand[oi]["box"], g) for g in gt[X]); hits.append(best); st["assigned"] += 1
    a = np.array(hits) if hits else np.array([0.0])
    out[NAMES[X]] = {"gt_frames": n_frames, "assigned": st["assigned"],
                     "coverage": round(st["assigned"] / max(1, n_frames), 3),
                     "iou>=0.5": round(float((a >= 0.5).mean()), 3) if hits else None,
                     "iou>=0.3": round(float((a >= 0.3).mean()), 3) if hits else None,
                     "median_iou": round(float(np.median(a)), 3) if hits else None,
                     "stats": dict(st)}
    print(f"\n== 假裝 {NAMES[X]} 是死類｜匈牙利整體指派 ==")
    print(f"  含該類 GT 幀 {n_frames} | 指派 {st['assigned']} ({st['assigned']/max(1,n_frames):.1%}) | 物件少於工具數 {st['objects_fewer_than_tools']}")
    if hits: print(f"  IoU>=0.5 {(a>=0.5).mean():.3f} | IoU>=0.3 {(a>=0.3).mean():.3f} | 中位 {np.median(a):.3f} | 完全落空 {(a==0).mean():.3f}")

json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/validate_assign_hungarian.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved validate_assign_hungarian.json")
