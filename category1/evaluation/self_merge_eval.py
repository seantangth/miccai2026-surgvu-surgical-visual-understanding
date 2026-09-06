"""self_merge_eval.py — 單模型自我合併能不能收緊框（AP75）。零成本：只用 raw npz 快取。

動機：RT-DETR 8.4.x 的 postprocess 是 query×class 全域 top-k、不做 NMS，
同一個物體常被多個 query 命中並輸出略微不同的框。目前直接輸出這些重複框。
本實驗把同類、IoU>=T 的框用分數加權平均合併，看 AP75 有沒有改善。

同時輸出逐 IoU 的 AP（0.50 / 0.75 / mean），才能分辨「找得到」與「框得準」。
用法：bsenv/bin/python self_merge_eval.py [tags...]
"""
import json, os, sys
from collections import defaultdict
import numpy as np
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_eval as OE

CONF, MAXDET = 0.05, 100

def iou_mat(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)

def per_frame(raw_v):
    d = defaultdict(lambda: defaultdict(list))
    for iid, c, s, (x, y, w, h) in OE.decode(raw_v, mode="global_topk", conf=CONF, max_det=MAXDET):
        d[iid][c].append([x, y, x + w, y + h, s])
    return {k: {c: np.array(v, np.float64) for c, v in cl.items()} for k, cl in d.items()}

def merge(frames, thr, mode):
    """同類、IoU>=thr 的框合併。mode: 'avg'=分數加權平均座標 / 'top'=只留分數最高者（等同 NMS）"""
    out = {}
    for i, cl in frames.items():
        newcl = {}
        for c, A in cl.items():
            A = A[np.argsort(-A[:, 4])]; used = np.zeros(len(A), bool); rows = []
            for k in range(len(A)):
                if used[k]: continue
                grp = [k]
                if k + 1 < len(A):
                    m = iou_mat(A[k:k+1, :4], A[k+1:, :4])[0]
                    for j, v in enumerate(m):
                        if v >= thr and not used[k+1+j]: grp.append(k+1+j); used[k+1+j] = True
                used[k] = True
                G = A[grp]
                if mode == "avg" and len(grp) > 1:
                    w = G[:, 4]; box = (G[:, :4] * w[:, None]).sum(0) / w.sum()
                else:
                    box = G[0, :4]
                rows.append([*box, G[:, 4].max()])
            newcl[c] = np.array(rows)
        out[i] = newcl
    return out

def cap(frames):
    out = {}
    for i, cl in frames.items():
        rows = [(c, r) for c, A in cl.items() for r in A]
        rows.sort(key=lambda x: -x[1][4]); rows = rows[:MAXDET]
        d = defaultdict(list)
        for c, r in rows: d[c].append(r)
        out[i] = {c: np.array(v) for c, v in d.items()}
    return out

def to_eval(frames):
    return [(int(i), int(c), float(r[4]), [float(r[0]), float(r[1]), float(r[2]-r[0]), float(r[3]-r[1])])
            for i, cl in frames.items() for c, A in cl.items() for r in A]

def eval_iou(dets_by_vid):
    """回傳 (mean_mAP, AP50, AP75)：逐影片 pycocotools，再對影片平均"""
    ms, a50, a75 = [], [], []
    for v in OE.VIDS:
        G = OE.gt(v); cm = OE.cat_map(v)
        res = [{"image_id": d[0], "category_id": cm[d[1]], "bbox": [round(x, 2) for x in d[3]], "score": d[2]}
               for d in dets_by_vid[v]]
        if not res: ms.append(0.0); a50.append(0.0); a75.append(0.0); continue
        E = COCOeval(G, G.loadRes(res), "bbox"); E.params.maxDets = [5, 10, 100]
        E.evaluate(); E.accumulate()
        p = E.eval["precision"]                       # [T,R,K,A,M]
        def ap(ti):
            q = p[ti, :, :, 0, -1] if ti is not None else p[:, :, :, 0, -1]
            return float(q[q > -1].mean()) if (q > -1).any() else 0.0
        ms.append(ap(None)); a50.append(ap(0)); a75.append(ap(5))
    return float(np.mean(ms)), float(np.mean(a50)), float(np.mean(a75))

if __name__ == "__main__":
    tags = sys.argv[1:] or ["s1_i640", "v5_R1v3", "v5_R1"]
    VARIANTS = [("base", None, None)] + [(f"{m}_iou{t}", t, m) for m in ("avg", "top") for t in (0.6, 0.75, 0.9)]
    res = {}
    print(f"{'model':10} {'variant':14} {'mean':>7} {'AP50':>7} {'AP75':>7}   Δmean   ΔAP75")
    for tag in tags:
        z = np.load(os.path.join(OE.RAWD, f"{tag}.npz"))
        raw = {v: {"boxes": z[f"b{v}"], "scores": z[f"s{v}"], "ids": z[f"i{v}"]} for v in OE.VIDS}
        base = {v: per_frame(raw[v]) for v in OE.VIDS}
        b_mean = b75 = None
        for name, thr, mode in VARIANTS:
            fr = base if thr is None else {v: merge(base[v], thr, mode) for v in OE.VIDS}
            m, a50, a75 = eval_iou({v: to_eval(cap(fr[v])) for v in OE.VIDS})
            if b_mean is None: b_mean, b75 = m, a75
            res[f"{tag}/{name}"] = {"mean": m, "AP50": a50, "AP75": a75}
            print(f"{tag:10} {name:14} {m:7.4f} {a50:7.4f} {a75:7.4f}  {m-b_mean:+7.4f} {a75-b75:+7.4f}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_merge_eval.json"), "w"), indent=1)
    print("saved self_merge_eval.json")
