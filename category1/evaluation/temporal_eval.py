"""temporal_eval.py — 時序後處理在官方協定 val(1,7) 上的驗證（用 raw npz 快取，不需 GPU）。

輸入：raw/{tag}.npz（capture_raw 攔截的 decoder 輸出）→ decode 成可部署設定（conf 0.05、每幀 ≤100 框）→ 套時序方法 → coco_eval。
方法（全部逐類、只看相鄰已標註幀 id 差 ≤2）：
  score_smooth(beta, miss)  : s' = (1-beta)*s + beta*mean(前後幀同類 IoU≥0.4 配對框的分數)，沒配到的鄰幀分數視為 miss*s
  box_smooth(iou)           : 座標改為前後幀配對框（同類 IoU≥iou）的分數加權平均（2023 冠軍 Kalman-WBF 的便宜版）
  gap_fill(iou, decay)      : 前後幀都有同類配對框、當幀沒有 → 插入內插框，分數 = decay*min(前後)
用法：python temporal_eval.py [tags...]   預設 s1_i640 v5_R1v3 v5_R1
"""
import itertools, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_eval as OE

CONF, MAXDET = 0.05, 100

def load(tag):
    z = np.load(os.path.join(OE.RAWD, f"{tag}.npz"))
    return {v: {"boxes": z[f"b{v}"], "scores": z[f"s{v}"], "ids": z[f"i{v}"]} for v in OE.VIDS}

def per_frame(raw_v):
    """{frame_id: {cls: [[x1,y1,x2,y2,score], ...]}}"""
    d = defaultdict(lambda: defaultdict(list))
    for iid, c, s, (x, y, w, h) in OE.decode(raw_v, mode="global_topk", conf=CONF, max_det=MAXDET):
        d[iid][c].append([x, y, x + w, y + h, s])
    return {k: {c: np.array(v, np.float64) for c, v in cl.items()} for k, cl in d.items()}

def iou_mat(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)

def match(a, b, thr):
    """貪婪一對一配對，回傳 {i_a: j_b}"""
    if len(a) == 0 or len(b) == 0: return {}
    M = iou_mat(a, b); out = {}; used = set()
    for i, j in sorted(((i, j) for i in range(len(a)) for j in range(len(b))), key=lambda ij: -M[ij]):
        if M[i, j] < thr: break
        if i in out or j in used: continue
        out[i] = j; used.add(j)
    return out

def neighbors(frames, ids, k):
    """第 k 個已標註幀的前後鄰幀（id 差 ≤2），沒有則 None"""
    prev = ids[k - 1] if k > 0 and ids[k] - ids[k - 1] <= 2 else None
    nxt = ids[k + 1] if k + 1 < len(ids) and ids[k + 1] - ids[k] <= 2 else None
    return prev, nxt

def score_smooth(frames, beta=0.3, miss=0.0, iou=0.4):
    ids = sorted(frames); out = {i: {} for i in ids}
    for k, i in enumerate(ids):
        p, n = neighbors(frames, ids, k)
        for c, A in frames[i].items():
            s = A[:, 4].copy(); acc = np.zeros(len(A)); cnt = 0
            for nb in (p, n):
                if nb is None: continue
                cnt += 1; B = frames[nb].get(c)
                m = match(A, B, iou) if B is not None else {}
                sc = np.full(len(A), miss) * s
                for ia, jb in m.items(): sc[ia] = B[jb, 4]
                acc += sc
            new = (1 - beta) * s + beta * (acc / cnt) if cnt else s
            X = A.copy(); X[:, 4] = new; out[i][c] = X
    return out

def box_smooth(frames, iou=0.5):
    ids = sorted(frames); out = {i: {} for i in ids}
    for k, i in enumerate(ids):
        p, n = neighbors(frames, ids, k)
        for c, A in frames[i].items():
            X = A.copy()
            for ia in range(len(A)):
                bs, ws = [A[ia, :4]], [A[ia, 4]]
                for nb in (p, n):
                    if nb is None or frames[nb].get(c) is None: continue
                    m = match(A[ia:ia + 1], frames[nb][c], iou)
                    if 0 in m: bs.append(frames[nb][c][m[0], :4]); ws.append(frames[nb][c][m[0], 4])
                if len(bs) > 1: X[ia, :4] = np.average(np.array(bs), axis=0, weights=np.array(ws))
            out[i][c] = X
    return out

def gap_fill(frames, iou=0.4, decay=0.8):
    ids = sorted(frames); out = {i: {c: A.copy() for c, A in frames[i].items()} for i in ids}
    for k, i in enumerate(ids):
        p, n = neighbors(frames, ids, k)
        if p is None or n is None: continue
        for c in set(frames[p]) & set(frames[n]):
            P, N = frames[p][c], frames[n][c]; m = match(P, N, iou)
            for ip, jn in m.items():
                mid = (P[ip, :4] + N[jn, :4]) / 2; sc = decay * min(P[ip, 4], N[jn, 4])
                cur = out[i].get(c)
                if cur is not None and len(cur) and iou_mat(mid[None], cur[:, :4]).max() >= iou: continue
                row = np.append(mid, sc)[None]
                out[i][c] = row if cur is None or len(cur) == 0 else np.vstack([cur, row])
    return out

def cap100(frames):
    out = {}
    for i, cl in frames.items():
        rows = [(c, r) for c, A in cl.items() for r in A]
        rows.sort(key=lambda x: -x[1][4]); rows = rows[:MAXDET]
        d = defaultdict(list)
        for c, r in rows: d[c].append(r)
        out[i] = {c: np.array(v) for c, v in d.items()}
    return out

def to_eval(frames):
    return [(int(i), int(c), float(r[4]), [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])])
            for i, cl in frames.items() for c, A in cl.items() for r in A]

VARIANTS = {
    "base": lambda f: f,
    "score_b0.3_miss0": lambda f: score_smooth(f, 0.3, 0.0),
    "score_b0.3_miss0.5": lambda f: score_smooth(f, 0.3, 0.5),
    "score_b0.5_miss0.5": lambda f: score_smooth(f, 0.5, 0.5),
    "box_iou0.5": lambda f: box_smooth(f, 0.5),
    "box_iou0.6": lambda f: box_smooth(f, 0.6),
    "gap_d0.8": lambda f: gap_fill(f, 0.4, 0.8),
    "box0.5+score0.3m0.5": lambda f: score_smooth(box_smooth(f, 0.5), 0.3, 0.5),
    "box0.5+gap0.8": lambda f: gap_fill(box_smooth(f, 0.5), 0.4, 0.8),
    "box0.5+gap0.8+score0.3m0.5": lambda f: score_smooth(gap_fill(box_smooth(f, 0.5), 0.4, 0.8), 0.3, 0.5),
}

if __name__ == "__main__":
    tags = sys.argv[1:] or ["s1_i640", "v5_R1v3", "v5_R1"]
    results = {}
    for tag in tags:
        raw = load(tag); base = {v: per_frame(raw[v]) for v in OE.VIDS}
        for name, fn in VARIANTS.items():
            dets = {v: to_eval(cap100(fn(base[v]))) for v in OE.VIDS}
            r = OE.coco_eval(dets, f"{tag}/{name}")
            pc = r["per_class_per_video"]
            nd = pc.get(1, {}).get("needle driver"); cad = pc.get(1, {}).get("cadiere forceps"); clip = pc.get(7, {}).get("clip applier")
            results[f"{tag}/{name}"] = {"mean": r["mean_mAP"], "per_video": r["per_video"], "ND_v1": nd, "cadiere_v1": cad, "clip_v7": clip}
            print(f"{tag:10s} {name:28s} mean {r['mean_mAP']:.4f}  v1 {r['per_video'][1]:.4f} v7 {r['per_video'][7]:.4f}  ND {nd:.3f} cad {cad:.3f} clip {clip:.3f}", flush=True)
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "temporal_eval.json"), "w"), indent=1)
    print("saved temporal_eval.json")
