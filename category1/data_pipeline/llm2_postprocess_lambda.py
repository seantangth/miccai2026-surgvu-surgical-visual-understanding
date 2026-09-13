#!/usr/bin/env python3
# Lambda 版：第二輪 LLM 結果後處理（frames640 池、weak_scale.json、rA 偵測器、cuda）
#!/usr/bin/env python3
"""LLM 標框後處理：results/*.json → (prograsp 吸附偵測器 query 框) → 模板傳播 ±MAXSTEP 秒 → 補同幀其他已安裝類（s1 conf≥0.35）
→ pseudo_llm_v1/{images,labels} ＋ 每類抽驗 montage。
用法：bsenv/bin/python llm_postprocess.py [MAXSTEP=25] [THR=0.6]
"""
import json, glob, sys, os, shutil, random, cv2, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path.home() / "scaleup"
FR = ROOT / "frames640"; RES = ROOT / "llm2_results"
OUT = ROOT / "pseudo_llm_v2"; MONT = ROOT / "llm2_montage"
MAXSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 25; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
CID = {"stapler": 11, "tipup": 4, "prograsp": 10}; IGNORE = {10, 11}
GT2ID = {"needle driver": 0, "monopolar curved scissors": 1, "force bipolar": 2, "clip applier": 3, "tip-up fenestrated grasper": 4,
         "cadiere forceps": 5, "bipolar forceps": 6, "vessel sealer": 7, "suction irrigator": 8, "bipolar dissector": 9,
         "prograsp forceps": 10, "stapler": 11, "permanent cautery hook/spatula": 12, "grasping retractor": 13}
weak = json.load(open(ROOT / "weak_scale.json"))
W, H = 640, 512
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

# 1. 讀 LLM 結果
seeds = []   # (cls_key, vid, sec, [x1,y1,x2,y2], conf)
for f in sorted(glob.glob(str(RES / "*.json"))):
    key = Path(f).stem.rsplit("_", 1)[0]
    try: rows = json.load(open(f))
    except Exception as e: print("BAD JSON", f, e); continue
    key = key.replace("C2", "").replace("2", "").replace("C", "")   # stapler2/tipupC2/prograspC2 → stapler/tipup/prograsp
    for r in rows:
        boxes = r.get("boxes") or [pk["box"] for pk in (r.get("picks") or []) if pk.get("box")]
        thr = 0.5 if r.get("picks") is not None else 0.6   # 選擇題版有 UI 列錨定，門檻放寬到 0.5；自由標框 0.6
        if not r.get("visible") or not boxes or float(r.get("confidence", 0)) < thr: continue   # 低信心種子不用
        p = Path(r["file"]); vid = p.parent.name; sec = int(p.stem.split("_")[1])
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2)); y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
            if x2 - x1 < 8 or y2 - y1 < 8: continue
            seeds.append((key, vid, sec, [x1, y1, x2, y2], float(r.get("confidence", 0.5))))
print("llm seeds:", Counter(s[0] for s in seeds))

# 2. 偵測器（s1）query 框：prograsp 吸附；其他類補框
from ultralytics import RTDETR
model = RTDETR(str(Path.home() / "weights/rA/last.pt"))
cache = {}
def dets(vid, sec):
    k = (vid, sec)
    if k not in cache:
        r = model.predict(str(FR / vid / f"sec_{sec:06d}.jpg"), conf=0.05, imgsz=640, max_det=300, device="cuda", verbose=False)[0]
        cache[k] = [(int(c), float(s), b.tolist()) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    return cache[k]
snapped = 0
for i, (key, vid, sec, box, conf) in enumerate(seeds):
    if key != "prograsp": continue
    best = max(((iou(box, b), b) for c, s, b in dets(vid, sec)), default=(0, None))
    if best[0] >= 0.3: seeds[i] = (key, vid, sec, best[1], conf); snapped += 1
print("prograsp snapped to detector query:", snapped)

# 3. 模板傳播（同影片 1fps 相鄰幀，NCC>=THR，尺寸變化<40%）
def propagate(vid, sec, box):
    out = []; img0 = cv2.imread(str(FR / vid / f"sec_{sec:06d}.jpg"), 0)
    x1, y1, x2, y2 = [int(round(v)) for v in box]; tpl = img0[y1:y2, x1:x2]
    if tpl.size == 0: return out
    for d in (1, -1):
        cur = (x1, y1, x2, y2)
        for step in range(1, MAXSTEP + 1):
            s2 = sec + d * step; f = FR / vid / f"sec_{s2:06d}.jpg"
            if not f.exists(): break
            im = cv2.imread(str(f), 0)
            r = cv2.matchTemplate(im, tpl, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
            if mx < THR: break
            nb = (loc[0], loc[1], loc[0] + tpl.shape[1], loc[1] + tpl.shape[0])
            out.append((s2, [float(v) for v in nb], mx)); cur = nb
    return out
frames = defaultdict(list)   # (vid,sec) -> [(cid, box, src)]
for key, vid, sec, box, conf in seeds:
    frames[(vid, sec)].append((CID[key], box, "llm", 1.0))
    for s2, nb, ncc in propagate(vid, sec, box):
        frames[(vid, s2)].append((CID[key], nb, "prop", float(ncc)))
print("frames after propagation:", len(frames), Counter(x[2] for v in frames.values() for x in v))
# 同幀同類去重：llm 優先、其次 NCC 高者；每類最多 installed 數（未知則 1）；任何同類 IoU>0.3 視為重複
for (vid, sec), items in frames.items():
    installed = Counter(GT2ID[t] for t in (weak.get(vid, {}).get(str(sec)) or []) if t in GT2ID)
    items.sort(key=lambda x: (x[2] != "llm", -x[3]))
    keep = []
    for cid, box, src, q in items:
        cap = max(1, installed.get(cid, 1))
        same = [k for k in keep if k[0] == cid]
        if len(same) >= cap: continue
        if any(iou(box, k[1]) > 0.3 for k in same): continue
        keep.append((cid, box, src))
    frames[(vid, sec)] = keep
print("after per-frame dedupe:", Counter(x[2] for v in frames.values() for x in v))

# 4. 去重（同幀同類 IoU>0.5 留 llm 優先）＋ 補其他已安裝類（s1 conf>=0.35，與目標框 IoU<=0.5）
OUT_I, OUT_L = OUT / "images", OUT / "labels"; shutil.rmtree(OUT, ignore_errors=True); OUT_I.mkdir(parents=True); OUT_L.mkdir(parents=True)
stat = Counter()
for (vid, sec), items in frames.items():
    keep = list(items)
    installed = Counter(GT2ID[t] for t in (weak.get(vid, {}).get(str(sec)) or []) if t in GT2ID)
    tgt = {c for c, _, _ in keep}
    for c, s, b in sorted(dets(vid, sec), key=lambda x: -x[1]):
        if s < 0.35 or c in IGNORE or c in tgt or c not in installed: continue
        if sum(1 for cc, _, _ in keep if cc == c) >= installed[c]: continue
        if any(iou(b, kb) > 0.5 for _, kb, _ in keep): continue
        keep.append((c, b, "s1")); stat["s1_extra"] += 1
    stem = f"{vid}_sec{sec:06d}_llm"
    shutil.copy(FR / vid / f"sec_{sec:06d}.jpg", OUT_I / f"{stem}.jpg")
    (OUT_L / f"{stem}.txt").write_text("\n".join(f"{c} {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}" for c, b, _ in keep))
    for c, _, src in keep: stat[f"{c}:{src}"] += 1
print("STATS:", dict(stat)); json.dump({"stats": dict(stat), "n_frames": len(frames)}, open(OUT / "stats.json", "w"), indent=1)

# 5. 抽驗 montage（每類 30 幀，llm 種子優先）
MONT.mkdir(parents=True, exist_ok=True); random.seed(0)
for key, cid in CID.items():
    picks = [(v, s) for (v, s), items in frames.items() if any(c == cid and src == "llm" for c, _, src in items)]
    random.shuffle(picks); picks = picks[:30]; tiles = []
    for v, s in picks:
        im = cv2.imread(str(FR / v / f"sec_{s:06d}.jpg"))
        for c, b, src in frames[(v, s)]:
            col = (0, 0, 255) if c == cid else (0, 255, 0)
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        cv2.putText(im, f"{v} {s}", (4, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(cv2.resize(im, (320, 256)))
    while len(tiles) % 5: tiles.append(np.zeros((256, 320, 3), np.uint8))
    rows = [np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)]
    if rows: cv2.imwrite(str(MONT / f"{key}_montage.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("montages:", sorted(os.listdir(MONT)))
