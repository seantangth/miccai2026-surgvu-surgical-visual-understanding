"""SurgVU 2026 Category 1 — v7 容器（2026-09-05）：按類別路由的多模型混合。
model tarball（/opt/ml/model）內容：
  *.pt            成員權重（RT-DETR / YOLO，類別表必須一致）
  routing.json    可選。{"groups": [{"weights": ["a_rA2.pt"], "classes": ["needle_driver", ...]},
                                    {"weights": ["c_rC.pt"], "classes": ["prograsp_forceps", ...]}]}
                  每個 group 只輸出自己 classes 內的框；group 內多個 weights 時做 WBF（iou 0.7、conf_type max）。
                  可選 "imgsz": 800 指定該 group 的推論解析度（預設 IMG_SZ=640）；單一權重可用 "b_rA2_800.pt@800"
                  覆寫自己的解析度（800 訓練的權重必須用 800 推論；同 group 內不同解析度的成員照常 WBF）。
                  沒有 routing.json 時退回 v6 行為：全部 .pt 一個 group、全部類別。
  沒有任何 .pt 時退回 /opt/app/resources/best.pt（s1）。
時間預算守衛：推估總時間超過 BUDGET_S 時，每個 group 只保留第一個 weight（主模型由 routing 明確指定，不靠檔名順序）。
輸出：每幀最多 MAX_DET 框（跨 group 依分數截斷）、conf ≥ CONF_MIN、原生像素座標。
"""
from pathlib import Path
import json, os, time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

INPUT_PATH = Path(os.environ.get("SURGVU_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("SURGVU_OUTPUT", "/output"))
MODEL_DIR = Path(os.environ.get("SURGVU_MODEL_DIR", "/opt/ml/model"))
FALLBACK_WEIGHTS = Path("/opt/app/resources/best.pt")
CONF_MIN = 0.05       # P7 教訓：conf 0.001×300 框撐爆官方評分器；0.05 對 0.001 只差 +0.0014
IMG_SZ = 640
MAX_DET = 100         # 官方 maxDets 上限 100
WBF_IOU = 0.7
BUDGET_S = 480        # 10 分鐘上限留餘裕
BATCH = 16


def resolve_groups():
    """回傳 [(weights_paths, class_name_set_or_None)]；None 表示全部類別。"""
    cands = sorted(MODEL_DIR.rglob("*.pt")) if MODEL_DIR.exists() else []
    cands = [c for c in cands if c.is_file() and c.stat().st_size > 1_000_000]
    if not cands:
        print("weights fallback:", FALLBACK_WEIGHTS)
        return [([(FALLBACK_WEIGHTS, IMG_SZ)], None)]
    by_name = {c.name: c for c in cands}
    rj = next(iter(MODEL_DIR.rglob("routing.json")), None)
    if rj is None:
        print("no routing.json → single group, all classes:", [c.name for c in cands])
        return [([(c, IMG_SZ) for c in cands], None)]
    spec = json.loads(rj.read_text())
    groups = []
    for g in spec["groups"]:
        ws = []
        for w in g["weights"]:                              # "name.pt" 或 "name.pt@800"
            name, _, sz = w.partition("@")
            ws.append((by_name[name], int(sz) if sz else int(g.get("imgsz", IMG_SZ))))   # KeyError = tarball 與 routing 不一致，寧可早死
        groups.append((ws, set(g["classes"])))
    print("routing:", [([f"{w.name}@{sz}" for w, sz in ws], sorted(cs)) for ws, cs in groups])
    return groups


def wbf(det_lists, W, H):
    from ensemble_boxes import weighted_boxes_fusion
    BL, SL, LL = [], [], []
    for dets in det_lists:
        BL.append([[max(0, x1 / W), max(0, y1 / H), min(1, x2 / W), min(1, y2 / H)] for _, _, (x1, y1, x2, y2) in dets])
        SL.append([s for _, s, _ in dets]); LL.append([c for c, _, _ in dets])
    if all(len(b) == 0 for b in BL):
        return []
    b, s, l = weighted_boxes_fusion(BL, SL, LL, iou_thr=WBF_IOU, skip_box_thr=CONF_MIN, conf_type="max")
    return [(int(l[j]), float(s[j]), (float(b[j][0] * W), float(b[j][1] * H), float(b[j][2] * W), float(b[j][3] * H))) for j in range(len(s))]


def run():
    inputs = json.loads((INPUT_PATH / "inputs.json").read_text())
    def slug_of(sv):
        meta = sv.get("socket") or sv.get("interface") or {}
        return meta.get("slug", "")
    print("sockets:", sorted(slug_of(sv) for sv in inputs))
    device = 0 if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == 0 else "")

    groups = resolve_groups()
    cache = {}
    for ws, _ in groups:
        for w, _sz in ws:
            if w not in cache:
                cache[w] = YOLO(str(w))
    names = next(iter(cache.values())).names
    for m in cache.values():
        assert list(m.names.values()) == list(names.values()), "member class names differ"
    name2id = {v: k for k, v in names.items()}
    # group 的類別集合換成 id；驗證每個類別名都存在於模型
    active = []
    for ws, cs in groups:
        ids = None if cs is None else {name2id[c] for c in cs}
        active.append([list(ws), ids])
    print("members:", len(cache), "| names:", list(names.values()))

    video_path = INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_boxes = []; slice_nr = 0; batch_frames, batch_ids = [], []
    t0 = time.time(); degraded = False

    def predict(key, frames):
        w, sz = key
        res = cache[w].predict(frames, conf=CONF_MIN, imgsz=sz, max_det=MAX_DET, device=device,
                               half=(device == 0), verbose=False)
        return [[(int(b.cls), float(b.conf), tuple(b.xyxy[0].tolist())) for b in r.boxes] for r in res]

    def flush():
        nonlocal batch_frames, batch_ids, degraded
        if not batch_frames:
            return
        H, W = batch_frames[0].shape[:2]
        # 每個權重只前向一次（同一權重可能出現在多個 group）
        needed = {k for ws, _ in active for k in ws}     # key = (weight, imgsz)，同權重不同解析度各前向一次
        per_w = {k: predict(k, batch_frames) for k in needed}
        for i, fid in enumerate(batch_ids):
            dets = []
            for ws, ids in active:
                member = [[d for d in per_w[k][i] if ids is None or d[0] in ids] for k in ws]
                dets += member[0] if len(ws) == 1 else wbf(member, W, H)
            dets.sort(key=lambda d: -d[1]); dets = dets[:MAX_DET]
            for c, s, (x1, y1, x2, y2) in dets:
                all_boxes.append({"name": f"slice_nr_{fid}_{names[c]}",
                                  "corners": [[x1, y1, 0.5], [x2, y1, 0.5], [x2, y2, 0.5], [x1, y2, 0.5]],
                                  "probability": round(s, 5)})
        batch_frames, batch_ids = [], []
        el = time.time() - t0; done = slice_nr
        if not degraded and any(len(ws) > 1 for ws, _ in active) and done >= 32 and n_total > 0 and el / done * n_total > BUDGET_S:
            print(f"budget guard: est {el/done*n_total:.0f}s > {BUDGET_S}s → first weight per group")
            for g in active: g[0] = g[0][:1]
            degraded = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch_frames.append(frame); batch_ids.append(slice_nr); slice_nr += 1
        if len(batch_frames) == BATCH:
            flush()
    flush(); cap.release()
    print(f"processed {slice_nr} frames, {len(all_boxes)} boxes, {time.time()-t0:.1f}s, degraded={degraded}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "surgical-tools.json").write_text(json.dumps({"type": "Multiple 2D bounding boxes", "boxes": all_boxes, "version": {"major": 1, "minor": 0}}))
    print("json file generated by the submission container")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
