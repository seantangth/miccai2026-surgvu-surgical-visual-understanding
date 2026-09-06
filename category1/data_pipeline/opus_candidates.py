#!/usr/bin/env python3
"""prograsp／tip-up 候選框版：s1 類別無關候選框（conf>=0.12，類別無關 NMS 0.6，最多 6 個）畫成 A-F 標在圖上，
輸出 annotated 圖 + 批次檔（每批 10 幀），讓 Opus 做選擇題。用法：bsenv/bin/python opus_candidates.py"""
import json, os, cv2, numpy as np
from pathlib import Path
from ultralytics import RTDETR
ROOT = Path(os.environ.get("SURGVU_ROOT", "."))
sel = json.load(open(ROOT / "1_data/processed/opus_bbox/selection_v1.json"))
ANN = ROOT / "1_data/processed/opus_bbox/annotated"; ANN.mkdir(exist_ok=True)
BAT = ROOT / "1_data/processed/opus_bbox/batches"
model = RTDETR(str(ROOT / "4_models/v3/rtdetr_v3_s1/best.pt"))
NAMES = ['needle_driver','monopolar_curved_scissor','force_bipolar','clip_applier','tip_up_fenestrated_grasper','cadiere_forceps','bipolar_forceps','vessel_sealer','suction_irrigator','bipolar_dissector','prograsp_forceps','stapler','permanent_cautery_hook_spatula','grasping_retractor']
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)
CONV = {
 'tipup': "TIP-UP FENESTRATED GRASPER: a grasper whose lower jaw is fixed and the upper jaw tips upward; both jaws have fenestrations (holes/slots). BOX RULE: wrist (clevis) plus BOTH jaws INCLUDING the tips; no shaft.",
 'prograsp': "PROGRASP FORCEPS: a robust grasper with slotted (fenestrated) jaws and a ratchet, similar to Cadiere forceps. BOX RULE (clevis convention): the wrist joint region where the jaws pivot on the shaft plus the proximal jaws; no shaft.",
}
letters = "ABCDEF"
for key in ("tipup", "prograsp"):
    items = sel[key]; batches = []
    for it in items:
        p = ROOT / it["path"]; r = model.predict(str(p), conf=0.12, imgsz=640, max_det=300, device="mps", verbose=False)[0]
        cands = sorted([(float(s), b.tolist(), int(c)) for s, b, c in zip(r.boxes.conf, r.boxes.xyxy, r.boxes.cls)], key=lambda x: -x[0])
        keep = []
        for s, b, c in cands:
            if any(iou(b, kb) > 0.6 for _, kb, _ in keep): continue
            keep.append((s, b, c))
            if len(keep) >= 6: break
        im = cv2.imread(str(p))
        for L, (s, b, c) in zip(letters, keep):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(im, L, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out = ANN / f"{it['vid']}_sec{it['sec']:06d}.jpg"; cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        batches.append({"file": str(p), "annotated": str(out), "installed": it["installed"],
                        "expected_count": it["installed"].count({'tipup': 'tip-up fenestrated grasper', 'prograsp': 'prograsp forceps'}[key]),
                        "candidates": {L: {"box": [int(round(v)) for v in b], "detector_guess": NAMES[c], "score": round(s, 2)} for L, (s, b, c) in zip(letters, keep)}})
    for i in range(0, len(batches), 10):
        bid = f"{key}C_{i//10:03d}"
        json.dump({"batch": bid, "cls": key, "convention": CONV[key], "frames": batches[i:i+10]}, open(BAT / f"{bid}.json", "w"), indent=0)
    print(key, "frames", len(batches), "batches", (len(batches) + 9) // 10)
