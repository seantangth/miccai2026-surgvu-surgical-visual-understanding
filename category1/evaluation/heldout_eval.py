"""heldout_eval.py — 量測器：在「訓練從未用過的時間窗」上，用 tools.csv 弱真值排序候選權重（本機 MPS，不需 GT 框）。

每類 c 回報：
  det_rate_installed  = P(該幀有 conf≥T 的 c 框 | tools.csv 說 c 裝著)     ← 召回代理；工具移出畫面會壓低，但各候選共用同一噪聲
  fire_rate_absent    = P(該幀有 conf≥T 的 c 框 | tools.csv 說 c 沒裝)     ← 誤報代理
  n_inst / n_abs      = 兩種幀數
另回報 set_f1：把每幀預測類別集合（conf≥T）對弱標籤集合算 F1，全幀平均。
用法：bsenv/bin/python heldout_eval.py <heldout_dir> <tag=weights.pt> [tag=weights.pt ...] [--conf 0.25]
  例：heldout_eval.py 1_data/processed/heldout rA2=4_models/v7/rA2/last.pt rB=4_models/v8/rB/last.pt
"""
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch
from ultralytics import RTDETR

GT2ID = {"needle driver": 0, "monopolar curved scissors": 1, "force bipolar": 2, "clip applier": 3,
         "tip-up fenestrated grasper": 4, "cadiere forceps": 5, "bipolar forceps": 6, "vessel sealer": 7,
         "suction irrigator": 8, "bipolar dissector": 9, "prograsp forceps": 10, "stapler": 11,
         "permanent cautery hook/spatula": 12, "grasping retractor": 13}
NAMES = {v: k for k, v in GT2ID.items()}
ap = argparse.ArgumentParser()
ap.add_argument("heldout"); ap.add_argument("models", nargs="+"); ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", default=None)
ap.add_argument("--weak", default=None, help="標籤 JSON {vid:{sec:[tools]}}，預設 heldout_weak.json")
A = ap.parse_args()
H = Path(A.heldout); weak = json.load(open(A.weak or (H / "heldout_weak.json")))
frames = sorted(H.rglob("sec_*.jpg")); frames = [f for f in frames if f.parent.name in weak]
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"heldout frames {len(frames)} | videos {len({f.parent.name for f in frames})} | device {dev} | conf {A.conf}", flush=True)
inst_of = {f: set(GT2ID[t] for t in (weak[f.parent.name].get(str(int(f.stem.split('_')[1]))) or []) if t in GT2ID) for f in frames}
n_inst = Counter(c for s in inst_of.values() for c in s); n_abs = {c: len(frames) - n_inst[c] for c in range(14)}

results = {}
for spec in A.models:
    tag, w = spec.split("=", 1); model = RTDETR(w); t0 = time.time()
    det_inst, fire_abs, f1s = Counter(), Counter(), []
    for i in range(0, len(frames), A.batch):
        batch = frames[i:i + A.batch]
        res = model.predict([str(p) for p in batch], conf=A.conf, imgsz=640, max_det=100, device=dev, half=(dev != "cpu"), verbose=False)
        for f, r in zip(batch, res):
            pred = set(int(c) for c in r.boxes.cls.tolist()); inst = inst_of[f]
            for c in pred:
                if c in inst: det_inst[c] += 1
                else: fire_abs[c] += 1
            tp = len(pred & inst); f1s.append(2 * tp / (len(pred) + len(inst)) if (pred or inst) else 1.0)
    rows = {}
    for c in range(14):
        if n_inst[c] == 0 and fire_abs[c] == 0: continue
        rows[NAMES[c]] = {"n_inst": n_inst[c], "det_rate_installed": det_inst[c] / n_inst[c] if n_inst[c] else None,
                          "n_abs": n_abs[c], "fire_rate_absent": fire_abs[c] / n_abs[c] if n_abs[c] else None}
    results[tag] = {"weights": w, "set_f1": float(np.mean(f1s)), "per_class": rows, "sec": time.time() - t0}
    print(f"\n== {tag}  set_f1 {results[tag]['set_f1']:.4f}  ({time.time()-t0:.0f}s)")
    print(f"{'class':32s} {'n_inst':>6s} {'det|inst':>9s} {'fire|abs':>9s}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["n_inst"]):
        d = f"{r['det_rate_installed']:.3f}" if r["det_rate_installed"] is not None else "   -"
        print(f"{name:32s} {r['n_inst']:6d} {d:>9s} {r['fire_rate_absent']:9.3f}")
out = A.out or str(H / f"heldout_eval_conf{A.conf}{'_' + Path(A.weak).stem if A.weak else ''}.json")
json.dump({"conf": A.conf, "n_frames": len(frames), "results": results}, open(out, "w"), indent=1)
print("saved", out)
