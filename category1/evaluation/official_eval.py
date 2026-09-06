"""SurgVU Cat1 官方協定評估器 + 原始 decoder 輸出快取。

- capture_raw(cfg): 用 ultralytics predict 跑一遍，hook RTDETRDecoder.postprocess
  攔截 (boxes_norm cxcywh 300x4, scores sigmoid 300x14)，存成 npz。
- decode_*(): 純離線 postprocess 變體，把 raw -> COCO detections。
- coco_eval(): 每支影片各跑 pycocotools（maxDets=[5,10,100]），取 stats[0]，再對影片平均。
"""
import json, time, os, sys
import numpy as np, cv2, torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.environ.get("SURGVU_ROOT", ".")
OUT  = os.path.dirname(os.path.abspath(__file__))
W    = f"{ROOT}/4_models/v3/rtdetr_v3_s1/best.pt"
VIDS = [1, 7]
RAWD = os.path.join(OUT, "raw"); os.makedirs(RAWD, exist_ok=True)

# UI 條帶（由 strip.py 的 per-row temporal std 決定）
UI_TOP = (0, 32)      # y in [0,32)
UI_BOT = (484, 512)   # y in [484,512)

MODEL_NAMES = {0:'needle_driver',1:'monopolar_curved_scissor',2:'force_bipolar',3:'clip_applier',
 4:'tip_up_fenestrated_grasper',5:'cadiere_forceps',6:'bipolar_forceps',7:'vessel_sealer',
 8:'suction_irrigator',9:'bipolar_dissector',10:'prograsp_forceps',11:'stapler',
 12:'permanent_cautery_hook_spatula',13:'grasping_retractor'}

_GT = {}
def gt(v):
    if v not in _GT:
        _GT[v] = COCO(f"{ROOT}/1_data/raw/cat1_val_public/{v}_fps1_coco.json")
    return _GT[v]

def cat_map(v):
    """model class idx -> GT category_id（底線換空白逐字比對）"""
    cats = gt(v).loadCats(gt(v).getCatIds())
    byname = {c['name']: c['id'] for c in cats}
    m = {}
    for i, n in MODEL_NAMES.items():
        gn = n.replace('_', ' ')
        assert gn in byname, f"class name mismatch: {gn}"
        m[i] = byname[gn]
    return m

# ---------------- raw capture ----------------
def blur_ui(f):
    f = f.copy()
    for y0, y1 in (UI_TOP, UI_BOT):
        f[y0:y1] = cv2.GaussianBlur(f[y0:y1], (0, 0), 12)
    return f

def capture_raw(tag, imgsz=640, hflip=False, blur=False, batch=8, force=False):
    """回傳 {vid: {'boxes':(N,300,4) norm cxcywh, 'scores':(N,300,14), 'ids':(N,)}}，並記錄 ms/frame"""
    path = os.path.join(RAWD, f"{tag}.npz")
    if os.path.exists(path) and not force:
        z = np.load(path)
        return {v: {'boxes': z[f'b{v}'], 'scores': z[f's{v}'], 'ids': z[f'i{v}']} for v in VIDS}, float(z['ms'])
    from ultralytics import RTDETR
    model = RTDETR(W)
    model.predict(np.zeros((512,640,3), np.uint8), imgsz=imgsz, conf=0.5, device='mps', verbose=False)  # warmup
    dec = model.model.model[-1]
    RAW = []
    orig = type(dec).postprocess
    def cap(boxes, scores):
        RAW.append((boxes.detach().float().cpu().numpy(), scores.detach().float().cpu().numpy()))
        return orig(dec, boxes, scores)
    dec.postprocess = cap
    store, tot_t, tot_n = {}, 0.0, 0
    for v in VIDS:
        ids_ok = set(gt(v).getImgIds())
        cap_v = cv2.VideoCapture(f"{ROOT}/1_data/raw/cat1_val_public/{v}_fps1.mp4")
        bufs, bids, B, S, IDS = [], [], [], [], []
        idx = 0
        while True:
            ok, fr = cap_v.read()
            if not ok: break
            if idx in ids_ok:
                if blur: fr = blur_ui(fr)
                if hflip: fr = fr[:, ::-1].copy()
                bufs.append(fr); bids.append(idx)
            idx += 1
            if len(bufs) == batch:
                RAW.clear(); t0 = time.perf_counter()
                model.predict(bufs, imgsz=imgsz, conf=0.99, device='mps', verbose=False)
                tot_t += time.perf_counter() - t0; tot_n += len(bufs)
                b = np.concatenate([r[0] for r in RAW]); s = np.concatenate([r[1] for r in RAW])
                B.append(b); S.append(s); IDS += bids; bufs, bids = [], []
        if bufs:
            RAW.clear(); t0 = time.perf_counter()
            model.predict(bufs, imgsz=imgsz, conf=0.99, device='mps', verbose=False)
            tot_t += time.perf_counter() - t0; tot_n += len(bufs)
            b = np.concatenate([r[0] for r in RAW]); s = np.concatenate([r[1] for r in RAW])
            B.append(b); S.append(s); IDS += bids
        cap_v.release()
        store[v] = {'boxes': np.concatenate(B).astype(np.float16),
                    'scores': np.concatenate(S).astype(np.float16),
                    'ids': np.array(IDS, np.int32)}
        assert len(store[v]['ids']) == len(ids_ok), (len(store[v]['ids']), len(ids_ok))
    ms = 1000 * tot_t / tot_n
    np.savez_compressed(path, ms=ms, **{f'b{v}': store[v]['boxes'] for v in VIDS},
                        **{f's{v}': store[v]['scores'] for v in VIDS},
                        **{f'i{v}': store[v]['ids'] for v in VIDS})
    return store, ms

# ---------------- offline decoders ----------------
W_PX, H_PX = 640, 512

def _to_xywh(bx):
    """(...,4) norm cxcywh -> COCO xywh 像素"""
    cx, cy, w, h = bx[..., 0]*W_PX, bx[..., 1]*H_PX, bx[..., 2]*W_PX, bx[..., 3]*H_PX
    return np.stack([cx-w/2, cy-h/2, w, h], -1)

def decode(raw_v, mode='global_topk', conf=0.001, k=1, max_det=300, topk=300, unflip=False):
    """回傳 list of (img_id, cls, score, [x,y,w,h])，score 由高到低"""
    B, S, IDS = raw_v['boxes'].astype(np.float32), raw_v['scores'].astype(np.float32), raw_v['ids']
    out = []
    nq, nc = S.shape[1], S.shape[2]
    for n in range(len(IDS)):
        s, b = S[n], B[n]
        if mode == 'global_topk':
            fl = s.ravel()
            kk = min(topk, fl.size)
            sel = np.argpartition(-fl, kk-1)[:kk]
            sel = sel[np.argsort(-fl[sel])]
            q, c, sc = sel//nc, sel % nc, fl[sel]
        elif mode == 'argmax':
            c = s.argmax(1); q = np.arange(nq); sc = s[q, c]
            o = np.argsort(-sc); q, c, sc = q[o], c[o], sc[o]
        elif mode == 'per_query_topk':
            kk = min(k, nc)
            ci = np.argpartition(-s, kk-1, axis=1)[:, :kk]
            q = np.repeat(np.arange(nq), kk); c = ci.ravel(); sc = s[q, c]
            o = np.argsort(-sc); q, c, sc = q[o], c[o], sc[o]
        elif mode == 'all_pairs':
            fl = s.ravel(); sel = np.nonzero(fl >= conf)[0]
            sel = sel[np.argsort(-fl[sel])]
            q, c, sc = sel//nc, sel % nc, fl[sel]
        else:
            raise ValueError(mode)
        m = sc >= conf
        q, c, sc = q[m][:max_det], c[m][:max_det], sc[m][:max_det]
        if len(q) == 0: continue
        bb = b[q].copy()
        if unflip: bb[:, 0] = 1.0 - bb[:, 0]
        xywh = _to_xywh(bb)
        iid = int(IDS[n])
        for j in range(len(q)):
            out.append((iid, int(c[j]), float(sc[j]), xywh[j].tolist()))
    return out

# ---------------- evaluator ----------------
def coco_eval(dets_by_vid, tag='', extra=None):
    """dets_by_vid: {vid: list of (img_id, cls, score, xywh)}  -> 官方 mean_mAP"""
    per_vid, per_cls_vid = {}, {}
    for v in VIDS:
        G = gt(v); cm = cat_map(v)
        res = [{'image_id': d[0], 'category_id': cm[d[1]], 'bbox': [round(x,2) for x in d[3]],
                'score': d[2]} for d in dets_by_vid[v]]
        if extra: res += extra(v)
        if not res:
            per_vid[v] = 0.0; per_cls_vid[v] = {}; continue
        D = G.loadRes(res)
        E = COCOeval(G, D, 'bbox')
        E.params.maxDets = [5, 10, 100]
        E.evaluate(); E.accumulate()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            E.summarize()
        per_vid[v] = float(E.stats[0])
        # per-class AP@[.5:.95], area=all, maxDets=100
        pc = {}
        cats = G.loadCats(G.getCatIds())
        for ki, cid in enumerate(G.getCatIds()):
            p = E.eval['precision'][:, :, ki, 0, -1]
            if (p > -1).any():
                pc[[c['name'] for c in cats if c['id'] == cid][0]] = float(p[p > -1].mean())
        per_cls_vid[v] = pc
    mean_map = float(np.mean([per_vid[v] for v in VIDS]))
    return {'tag': tag, 'mean_mAP': mean_map, 'per_video': per_vid, 'per_class_per_video': per_cls_vid}
