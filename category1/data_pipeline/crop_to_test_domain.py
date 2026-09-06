"""V3-1 域對齊管線：720p 訓練幀 → 官方測試域幾何（裁黑邊 → 640×512）
依據 2026-09-01 黑邊分析：全部 20 case 內容區一致 = x∈[192,1088]、全高 720（896×720）。
官方驗證集 = 640×512 且 UI 保留（僅去黑邊），故此變換即 crop[192:1088]×[0:720] → resize(640,512)。
輸出：1_data/processed/rare_frames_640/{case}_p{part}/sec_*.jpg（鏡射結構，JPEG q=95）
　　　1_data/processed/crop_params.json（agy 框重投影等下游座標轉換用）
用法: bsenv/bin/python 3_src/crop_to_test_domain.py [--workers 8]
"""
import os
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(os.environ.get("SURGVU_ROOT", "."))
SRC = ROOT / "1_data/processed/rare_frames"
DST = ROOT / "1_data/processed/rare_frames_640"
X0, X1, Y0, Y1 = 192, 1088, 0, 720   # 20 case 實測一致的內容區
OW, OH = 640, 512

def process_one(args):
    src, dst = args
    img = cv2.imread(str(src))
    if img is None:
        return f"READ_FAIL {src}"
    if img.shape != (720, 1280, 3):
        return f"BAD_SHAPE {src} {img.shape}"
    crop = img[Y0:Y1, X0:X1]
    out = cv2.resize(crop, (OW, OH), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None

def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    jobs = []
    for d in sorted(SRC.iterdir()):
        if not d.is_dir():
            continue
        (DST / d.name).mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.jpg")):
            dst = DST / d.name / f.name
            if not dst.exists():
                jobs.append((f, dst))
    print(f"to process: {len(jobs)}", flush=True)
    t0 = time.time()
    errs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(process_one, jobs, chunksize=64)):
            if r:
                errs.append(r)
            if (i + 1) % 2000 == 0:
                print(f"{i+1}/{len(jobs)} ({(i+1)/(time.time()-t0):.0f} img/s)", flush=True)
    params = {"note": "720p→test-domain 變換（全 case 一致）",
              "crop": {"x0": X0, "x1": X1, "y0": Y0, "y1": Y1},
              "resize": {"w": OW, "h": OH},
              "scale_x": OW / (X1 - X0), "scale_y": OH / (Y1 - Y0)}
    json.dump(params, open(ROOT / "1_data/processed/crop_params.json", "w"), indent=1)
    n_out = sum(1 for _ in DST.rglob("*.jpg"))
    print(f"DONE. out frames: {n_out} | errors: {len(errs)}", flush=True)
    for e in errs[:20]:
        print(" ", e, flush=True)

if __name__ == "__main__":
    main()
