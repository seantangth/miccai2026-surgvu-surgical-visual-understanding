"""可直接移植到 cat1_container/inference.py 的 RT-DETR postprocess 補丁。

背景（本機實測，agentA/results.json）：
  * ultralytics 8.4.137 的 RTDETRDecoder.postprocess 已經是「(query x class) 全域 top-300」
    的多標籤行為，不是舊版的 per-query argmax。所以 (c) 「top-k per query」在此版本
    其實是**降級**而非升級。
  * 實測 argmax / per-query top2/top3 / all-pairs / max_det 100~1400 的 mean_mAP
    全部落在 baseline ±0.002 內 => 純 postprocess 改動沒有可靠增益。
  * 因此本檔的預設 MODE='global_topk' 等價於現況；提供其他 MODE 只是為了可切換實驗。

用法（在 inference.py 內）：
    from postprocess_patch import install_raw_hook, decode_raw
    model = YOLO(str(WEIGHTS))          # 會自動 dispatch 成 RTDETR/RTDETRPredictor
    model.predict(np.zeros((512,640,3), np.uint8), imgsz=640, conf=0.99, verbose=False)  # 先建立 predictor
    RAW = install_raw_hook(model)
    ...
    RAW.clear()
    model.predict(batch_frames, imgsz=IMG_SZ, conf=0.99, device=device, verbose=False)
    boxes = np.concatenate([r[0] for r in RAW]); scores = np.concatenate([r[1] for r in RAW])
    for i, fid in enumerate(batch_ids):
        for cls, sc, (x1, y1, x2, y2) in decode_raw(boxes[i], scores[i], W=640, H=512,
                                                    mode="global_topk", conf=0.05, max_det=300):
            ...  # 產生 slice_nr_{fid}_{names[cls]}

注意：predict 時把 conf 設成 0.99 只是讓 ultralytics 自己的 postprocess 幾乎不做事，
真正的閾值由 decode_raw 的 conf 決定。
"""
import numpy as np


def install_raw_hook(model):
    """把 RTDETRDecoder.postprocess 換成攔截器，回傳一個 list，
    每次 forward 會 append 一個 (boxes (B,300,4) 正規化 cxcywh, scores (B,300,14) sigmoid)。"""
    dec = model.model.model[-1]
    orig = type(dec).postprocess
    RAW = []

    def hooked(boxes, scores):
        RAW.append((boxes.detach().float().cpu().numpy(),
                    scores.detach().float().cpu().numpy()))
        return orig(dec, boxes, scores)

    dec.postprocess = hooked
    return RAW


def decode_raw(boxes, scores, W=640, H=512, mode="global_topk",
               conf=0.05, k=2, max_det=300, topk=300, hflip=False):
    """boxes (300,4) 正規化 cxcywh；scores (300,14) sigmoid。
    回傳 [(cls, score, (x1,y1,x2,y2) 原生像素), ...]，分數由高到低。

    mode:
      global_topk  — 現行 ultralytics 8.4.x 預設（(query x class) 全域 top-`topk`）
      argmax       — 舊版行為，每個 query 只取最高分類別
      per_query_topk — 每個 query 取前 k 個類別
      all_pairs    — 所有 300*14 組合中 score>=conf 者
    """
    s = np.asarray(scores, np.float32)
    b = np.asarray(boxes, np.float32)
    nq, nc = s.shape

    if mode == "global_topk":
        fl = s.ravel(); kk = min(topk, fl.size)
        sel = np.argpartition(-fl, kk - 1)[:kk]
        sel = sel[np.argsort(-fl[sel])]
        q, c, sc = sel // nc, sel % nc, fl[sel]
    elif mode == "argmax":
        c = s.argmax(1); q = np.arange(nq); sc = s[q, c]
        o = np.argsort(-sc); q, c, sc = q[o], c[o], sc[o]
    elif mode == "per_query_topk":
        kk = min(k, nc)
        ci = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        q = np.repeat(np.arange(nq), kk); c = ci.ravel(); sc = s[q, c]
        o = np.argsort(-sc); q, c, sc = q[o], c[o], sc[o]
    elif mode == "all_pairs":
        fl = s.ravel(); sel = np.nonzero(fl >= conf)[0]
        sel = sel[np.argsort(-fl[sel])]
        q, c, sc = sel // nc, sel % nc, fl[sel]
    else:
        raise ValueError(mode)

    m = sc >= conf
    q, c, sc = q[m][:max_det], c[m][:max_det], sc[m][:max_det]
    bb = b[q].copy()
    if hflip:
        bb[:, 0] = 1.0 - bb[:, 0]          # 反轉 hflip TTA 的 x 座標
    cx, cy, w, h = bb[:, 0] * W, bb[:, 1] * H, bb[:, 2] * W, bb[:, 3] * H
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [(int(c[i]), float(sc[i]), (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])))
            for i in range(len(q))]


# ---------------------------------------------------------------------------
# hflip TTA + WBF（本機實測唯一為正的變體：+0.005，但推論成本 2x，且 2 支影片
# 的 val 變異很大，此增益估計落在雜訊範圍內。若要用，需在容器 requirements
# 加 `ensemble_boxes`（cat1_container/requirements.txt 目前沒有）。）
#
# 用法：
#   raw_n = decode_raw(b0[i], s0[i], mode="global_topk", conf=0.001)
#   raw_f = decode_raw(bf[i], sf[i], mode="global_topk", conf=0.001, hflip=True)
#       其中 bf/sf 來自對 frame[:, ::-1].copy()（BGR 陣列水平翻轉）再 predict
#   fused = wbf_merge_frame([raw_n, raw_f], iou_thr=0.7, conf_type="max")
# ---------------------------------------------------------------------------
def wbf_merge_frame(det_lists, W=640, H=512, iou_thr=0.7, conf_type="max",
                    skip_box_thr=0.001, max_det=300, weights=None):
    """det_lists: 多個 decode_raw() 的輸出（同一幀）。回傳同樣格式的融合結果。"""
    from ensemble_boxes import weighted_boxes_fusion
    import numpy as _np
    BL, SL, LL = [], [], []
    for dets in det_lists:
        BL.append([[x1 / W, y1 / H, x2 / W, y2 / H] for _, _, (x1, y1, x2, y2) in dets])
        SL.append([s for _, s, _ in dets])
        LL.append([c for c, _, _ in dets])
    b, s, l = weighted_boxes_fusion(BL, SL, LL, weights=weights, iou_thr=iou_thr,
                                    skip_box_thr=skip_box_thr, conf_type=conf_type)
    o = _np.argsort(-s)[:max_det]
    return [(int(l[j]), float(s[j]),
             (float(b[j][0] * W), float(b[j][1] * H), float(b[j][2] * W), float(b[j][3] * H)))
            for j in o]
