#!/usr/bin/env python3
"""第二輪 LLM 批次（新影片池）：stapler 自由標框；tipup/prograsp 候選框選擇題。輸出到 1_data/processed/llm_bbox2/{batches,annotated}"""
import json, os, cv2
from pathlib import Path
from ultralytics import RTDETR
ROOT = Path(os.environ.get("SURGVU_ROOT", ".")); B2 = ROOT / "1_data/processed/llm_bbox2"
sel = json.load(open(B2 / "llm2_selection.json")); (B2 / "batches").mkdir(exist_ok=True); (B2 / "annotated").mkdir(exist_ok=True); (B2 / "results").mkdir(exist_ok=True)
NAMES = ['needle_driver','monopolar_curved_scissor','force_bipolar','clip_applier','tip_up_fenestrated_grasper','cadiere_forceps','bipolar_forceps','vessel_sealer','suction_irrigator','bipolar_dissector','prograsp_forceps','stapler','permanent_cautery_hook_spatula','grasping_retractor']
CONV = {
 'stapler': "STAPLER (da Vinci SureForm/EndoWrist stapler): a large, elongated silver/grey jaw assembly (cartridge + anvil) attached to a thick shaft via a wrist hinge; the jaw is much longer than any grasper jaw and may show a staple line. BOX RULE: the box must cover the wrist/hinge PLUS THE ENTIRE JAW ASSEMBLY all the way to the distal tip of the anvil. Do NOT include the long shaft behind the wrist.",
 'tipup': "TIP-UP FENESTRATED GRASPER: a grasper whose lower jaw is fixed and the upper jaw tips upward; both jaws have fenestrations (holes/slots). BOX RULE: wrist (clevis) plus BOTH jaws INCLUDING the tips; no shaft.",
 'prograsp': "PROGRASP FORCEPS: a robust grasper with slotted (fenestrated) jaws and a ratchet, similar to Cadiere forceps. BOX RULE (clevis convention): the wrist joint region where the jaws pivot on the shaft plus the proximal jaws; no shaft.",
}
GTN = {'stapler': 'stapler', 'tipup': 'tip-up fenestrated grasper', 'prograsp': 'prograsp forceps'}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)
model = RTDETR(str(ROOT / "4_models/v7/rA/last.pt"))   # 候選框用 Round A 模型（比 s1 強）
letters = "ABCDEF"; total = 0
for key, items in sel.items():
    frames = []
    for it in items:
        p = B2 / it["path"]; entry = {"file": str(p), "installed": it["installed"], "expected_count": it["installed"].count(GTN[key])}
        if key != "stapler":
            r = model.predict(str(p), conf=0.12, imgsz=640, max_det=300, device="mps", verbose=False)[0]
            cands = sorted([(float(s), b.tolist(), int(c)) for s, b, c in zip(r.boxes.conf, r.boxes.xyxy, r.boxes.cls)], key=lambda x: -x[0]); keep = []
            for s, b, c in cands:
                if any(iou(b, kb) > 0.6 for _, kb, _ in keep): continue
                keep.append((s, b, c))
                if len(keep) >= 6: break
            im = cv2.imread(str(p))
            for L, (s, b, c) in zip(letters, keep):
                x1, y1, x2, y2 = [int(round(v)) for v in b]; cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2); cv2.putText(im, L, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            out = B2 / "annotated" / f"{it['vid']}_sec{it['sec']:06d}.jpg"; cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            entry["annotated"] = str(out); entry["candidates"] = {L: {"box": [int(round(v)) for v in b], "detector_guess": NAMES[c], "score": round(s, 2)} for L, (s, b, c) in zip(letters, keep)}
        frames.append(entry)
    for i in range(0, len(frames), 10):
        bid = f"{key}{'C' if key != 'stapler' else ''}2_{i//10:03d}"
        json.dump({"batch": bid, "cls": key, "convention": CONV[key], "frames": frames[i:i+10]}, open(B2 / "batches" / f"{bid}.json", "w"), indent=0); total += 1
    print(key, len(frames), "frames")
print("batches", total)
