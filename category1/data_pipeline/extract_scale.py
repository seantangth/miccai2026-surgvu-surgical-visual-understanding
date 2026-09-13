#!/usr/bin/env python3
"""大規模抽幀（Lambda 用）：manifest 中尚未抽過的影片 → 分片 → 逐支處理：
  remotezip 下載單支 → ffprobe 時長 → 視窗（均勻 STEP 秒一個 WIN 秒視窗 ＋ 弱類安裝時段加密）
  → ffmpeg 1fps → 立刻裁到測試域（x∈[192,1088] → 640×512）→ 刪 720p 幀與影片。
輸出：<out>/{case}_p{part}/sec_{絕對秒:06d}.jpg（640×512），命名與 rare_frames_640 相容。
用法：python3 extract_scale.py --shard 0 --nshards 2 --out ~/scaleup/frames640 --labels ~/labels \
      --manifest ~/videos_zip_manifest.json --done ~/rare_frames_done_videos.json --deadline-min 120
"""
import argparse, csv, json, os, shutil, subprocess, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import cv2
from remotezip import RemoteZip

URL = "https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip"
X0, X1, Y0, Y1 = 192, 1088, 0, 720      # crop_params.json（720p → 內容區 896×720）
OUT_W, OUT_H = 640, 512
BOOST = ["tip-up fenestrated grasper", "stapler", "prograsp forceps", "permanent cautery hook/spatula", "vessel sealer",
         "clip applier", "grasping retractor", "force bipolar", "cadiere forceps"]  # stapler/prograsp 供 LLM 標記用

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--manifest", required=True); ap.add_argument("--done", required=True)
ap.add_argument("--tmp", default=os.path.expanduser("~/scaleup/tmp"))
ap.add_argument("--deadline-min", type=float, default=120, help="超過此分鐘數不再開始新下載")
ap.add_argument("--max-videos", type=int, default=10000)
ap.add_argument("--step", type=int, default=180); ap.add_argument("--win", type=int, default=15)
ap.add_argument("--boost-step", type=int, default=60); ap.add_argument("--boost-cap", type=int, default=60)
ap.add_argument("--max-windows", type=int, default=150)
ap.add_argument("--dl-workers", type=int, default=2); ap.add_argument("--ff-workers", type=int, default=16)
A = ap.parse_args()

OUT = Path(A.out); TMP = Path(A.tmp); OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
LABELS = Path(A.labels)
done = set(tuple(x) for x in json.load(open(A.done)))
man = json.load(open(A.manifest))
vids = []
for m in man:
    n = m["name"]                      # surgvu24/case_081/case_081_video_part_001.mp4
    if not n.endswith(".mp4"): continue
    case = n.split("/")[1]; part = int(n.rsplit("_part_", 1)[1][:3])
    if (case, part) in done: continue
    vids.append((case, part, n, m["size"]))
vids.sort()
mine = [v for i, v in enumerate(vids) if i % A.nshards == A.shard][:A.max_videos]
print(f"candidates {len(vids)} | shard {A.shard}/{A.nshards} -> {len(mine)} videos | {sum(v[3] for v in mine)/2**30:.1f} GiB", flush=True)

def t2s(t):
    try: h, m, s = t.split(":"); return float(h)*3600 + float(m)*60 + float(s)
    except Exception: return None

def boost_intervals(case, part):
    """該 part 內 BOOST 類的安裝區間 [(cls, a, b)]（跨 part 以 0/inf 處理）"""
    f = LABELS / case / "tools.csv"; out = []
    if not f.exists(): return out
    for r in csv.DictReader(open(f)):
        gt = r["groundtruth_toolname"].strip().lower()
        if gt not in BOOST: continue
        try: pa, pb = int(float(r["install_case_part"])), int(float(r["uninstall_case_part"]))
        except Exception: continue
        a, b = t2s(r["install_case_time"]), t2s(r["uninstall_case_time"])
        if a is None or b is None: continue
        if pa == pb == part: out.append((gt, a, b))
        elif pa == part < pb: out.append((gt, a, 1e9))
        elif pa < part == pb: out.append((gt, 0, b))
        elif pa < part < pb: out.append((gt, 0, 1e9))
    return out

def windows(case, part, D):
    W = A.win
    ws = [(s, min(s + W, D)) for s in range(30, int(D) - W, A.step)]
    # 弱類加密：稀有優先，全程列（超長區間）排後
    iv = boost_intervals(case, part)
    iv.sort(key=lambda x: (BOOST.index(x[0]), (x[2] - x[1])))
    nb = 0
    for cls, a, b in iv:
        a, b = max(0, a), min(D, b)
        if b - a > 3600 * 2: continue          # 全程垃圾列跳過
        for s in range(int(a) + 5, int(b) - W, A.boost_step):
            ws.append((s, s + W)); nb += 1
            if nb >= A.boost_cap: break
        if nb >= A.boost_cap: break
    # 合併重疊
    ws.sort(); merged = []
    for a, b in ws:
        if merged and a <= merged[-1][1] + 2: merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else: merged.append((a, b))
    return merged[:A.max_windows]

def download(v):
    case, part, name, size = v
    local = TMP / f"{case}_p{part}.mp4"
    t0 = time.time()
    try:
        with RemoteZip(URL) as z:
            z.extract(name, path=str(TMP / f"dl_{case}_p{part}"))
        (TMP / f"dl_{case}_p{part}" / name).rename(local)
        shutil.rmtree(TMP / f"dl_{case}_p{part}", ignore_errors=True)
        return v, local, time.time() - t0, None
    except Exception as e:
        shutil.rmtree(TMP / f"dl_{case}_p{part}", ignore_errors=True)
        return v, None, time.time() - t0, str(e)[:200]

def ffwin(local, a, b, raw_dir):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(a), "-i", str(local),
           "-t", str(b - a), "-vf", "fps=1", "-q:v", "2", "-start_number", str(a), str(raw_dir / "sec_%06d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stderr[:120]

def crop_one(src, dst):
    im = cv2.imread(str(src))
    if im is None or im.shape[0] < Y1 or im.shape[1] < X1: return False
    im = cv2.resize(im[Y0:Y1, X0:X1], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95]); return True

def process(v, local):
    case, part, name, size = v
    t0 = time.time()
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(local)], capture_output=True, text=True)
    try: D = float(p.stdout.strip())
    except Exception: return f"{case}_p{part}: ffprobe failed"
    ws = windows(case, part, D)
    raw_dir = TMP / f"raw_{case}_p{part}"; raw_dir.mkdir(exist_ok=True)
    frame_dir = OUT / f"{case}_p{part}"; frame_dir.mkdir(exist_ok=True)
    bad = 0
    with ThreadPoolExecutor(A.ff_workers) as ex:
        for rc, err in ex.map(lambda w: ffwin(local, w[0], w[1], raw_dir), ws):
            if rc != 0: bad += 1
    raws = sorted(raw_dir.glob("*.jpg"))
    with ThreadPoolExecutor(A.ff_workers) as ex:
        oks = list(ex.map(lambda f: crop_one(f, frame_dir / f.name), raws))
    shutil.rmtree(raw_dir, ignore_errors=True); local.unlink(missing_ok=True)
    n = sum(oks)
    return f"{case}_p{part}: dur {D/60:.0f} min | windows {len(ws)} (ff err {bad}) | frames {n} | {time.time()-t0:.0f}s"

T0 = time.time(); total = 0; ndone = 0
with ThreadPoolExecutor(A.dl_workers) as dl:
    queue = list(mine); futs = []
    def submit_next():
        if queue and (time.time() - T0) / 60 < A.deadline_min:
            futs.append(dl.submit(download, queue.pop(0))); return True
        return False
    for _ in range(A.dl_workers): submit_next()
    while futs:
        f = next(as_completed(futs)); futs.remove(f)
        v, local, dt, err = f.result()
        submit_next()
        if err: print(f"[{time.strftime('%H:%M:%S')}] DOWNLOAD FAILED {v[0]}_p{v[1]}: {err}", flush=True); continue
        print(f"[{time.strftime('%H:%M:%S')}] downloaded {v[0]}_p{v[1]} {v[3]/2**30:.2f} GiB in {dt/60:.1f} min ({v[3]/2**20/max(dt,1):.0f} MB/s)", flush=True)
        msg = process(v, local); ndone += 1
        n = len(list((OUT / f"{v[0]}_p{v[1]}").glob("*.jpg"))); total += n
        print(f"[{time.strftime('%H:%M:%S')}] {msg} | videos done {ndone} | total frames {total} | elapsed {(time.time()-T0)/60:.0f} min", flush=True)
print(f"EXTRACT DONE videos={ndone} frames={total} elapsed={(time.time()-T0)/60:.0f} min", flush=True)
