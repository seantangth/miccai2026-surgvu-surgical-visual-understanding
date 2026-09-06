"""ensemble_ap75.py — 多模型 WBF 集成的增益來自「找得到」(AP50) 還是「框得準」(AP75)？

ensemble_eval.py 只報 mean mAP（s1+R1v3 = +0.016）。本腳本拆成 AP50 / AP75 兩半，
用來決定 Final 要不要投集成、以及集成是不是 AP75 的可用槓桿。零成本：只用 raw npz 快取。
用法：bsenv/bin/python ensemble_ap75.py
"""
import json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_eval as OE
from postprocess_patch import wbf_merge_frame
from self_merge_eval import eval_iou, cap, to_eval, CONF, MAXDET

TAGS = ["s1_i640", "v5_R1v3", "v5_R1"]

def load(tag):
    z = np.load(os.path.join(OE.RAWD, f"{tag}.npz"))
    return {v: {"boxes": z[f"b{v}"], "scores": z[f"s{v}"], "ids": z[f"i{v}"]} for v in OE.VIDS}

def dets_per_frame(raw_v):
    d = defaultdict(list)
    for iid, c, s, (x, y, w, h) in OE.decode(raw_v, mode="global_topk", conf=CONF, max_det=MAXDET):
        d[iid].append((c, s, (x, y, x + w, y + h)))
    return d

def as_frames(dfr):
    out = {}
    for iid, dets in dfr.items():
        d = defaultdict(list)
        for c, s, (x1, y1, x2, y2) in dets: d[c].append([x1, y1, x2, y2, s])
        out[iid] = {c: np.array(v) for c, v in d.items()}
    return out

raws = {t: load(t) for t in TAGS}
single = {t: {v: dets_per_frame(raws[t][v]) for v in OE.VIDS} for t in TAGS}
res = {}
print(f"{'combo':32} {'mean':>7} {'AP50':>7} {'AP75':>7}    Dmean    DAP50    DAP75")
base = None
for t in TAGS:
    m, a50, a75 = eval_iou({v: to_eval(cap(as_frames(single[t][v]))) for v in OE.VIDS})
    if base is None: base = (m, a50, a75)
    res[t] = {"mean": m, "AP50": a50, "AP75": a75}
    print(f"{t:32} {m:7.4f} {a50:7.4f} {a75:7.4f}  {m-base[0]:+8.4f} {a50-base[1]:+8.4f} {a75-base[2]:+8.4f}", flush=True)

combos = [("s1_i640", "v5_R1v3"), ("s1_i640", "v5_R1"), ("v5_R1v3", "v5_R1"), tuple(TAGS)]
for combo in combos:
    for ct in ("max", "avg"):
        merged = {}
        for v in OE.VIDS:
            ids = set().union(*(single[t][v].keys() for t in combo)); fr = {}
            for iid in ids:
                fr[iid] = wbf_merge_frame([single[t][v].get(iid, []) for t in combo], iou_thr=0.7, conf_type=ct)
            merged[v] = as_frames(fr)
        m, a50, a75 = eval_iou({v: to_eval(cap(merged[v])) for v in OE.VIDS})
        name = "+".join(c.replace("v5_", "").replace("_i640", "") for c in combo) + f" [{ct}]"
        res[name] = {"mean": m, "AP50": a50, "AP75": a75}
        print(f"{name:32} {m:7.4f} {a50:7.4f} {a75:7.4f}  {m-base[0]:+8.4f} {a50-base[1]:+8.4f} {a75-base[2]:+8.4f}", flush=True)

json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensemble_ap75.json"), "w"), indent=1)
print("saved ensemble_ap75.json")
