#!/usr/bin/env python3
"""retractor_batches.py — 把 llm_retractor 的候選幀切成每批 10 幀的 LLM 標註任務（自由標框）。
grasping retractor 沒有候選框可選（偵測器對它全盲），所以只能自由標框，做法比照第二輪的 stapler。
輸出：1_data/processed/llm_retractor/batches/retr_NNN.json
用法：bsenv/bin/python retractor_batches.py
"""
import os
import json
from pathlib import Path

ROOT = Path(os.environ.get("SURGVU_ROOT", "."))
B = ROOT / "1_data/processed/llm_retractor"
sel = json.load(open(B / "retractor_sel.json"))["retractor"]
(B / "batches").mkdir(exist_ok=True); (B / "results").mkdir(exist_ok=True)

CONV = (
 "GRASPING RETRACTOR (da Vinci Small Grasping Retractor / Tenaculum-style retractor): a robust two-jaw "
 "instrument used to hold and retract tissue. Its jaws are BLUNT, THICK and often SERRATED or ridged on the "
 "inner surface, noticeably heavier-looking than a Cadiere or Prograsp forceps, and it is frequently seen "
 "clamped onto tissue and left stationary for long stretches while other instruments work. It usually enters "
 "from the side of the frame and may be partly out of view.\n"
 "BOX RULE: the bounding box must cover the WRIST/CLEVIS (where the jaws pivot on the shaft) PLUS THE ENTIRE "
 "JAWS ALL THE WAY TO THE TIPS. This is one of the tools where the official annotation includes the tool tip. "
 "Do NOT include the long straight shaft behind the wrist. If the jaws grip tissue, still box only the metal "
 "instrument, not the tissue."
)

frames = [{"file": str(B / it["path"]), "vid": it["vid"], "sec": it["sec"],
           "installed": it["installed"] or [], "expected_count": (it["installed"] or []).count("grasping retractor")}
          for it in sel]
n = 0
for i in range(0, len(frames), 10):
    bid = f"retr_{i//10:03d}"
    json.dump({"batch": bid, "cls": "retractor", "convention": CONV, "frames": frames[i:i+10]},
              open(B / "batches" / f"{bid}.json", "w"), indent=0)
    n += 1
print(f"frames {len(frames)} → batches {n}  ({B/'batches'})")
