"""V3-2 weak_labels 重生成：per-arm 解析 ＋ 保留同名重複 ＋ 跨 part 正確展開
tools.csv 三種髒污（2026-09-01 深夜實證）：
  1. 完全重複列（同 arm 同工具同區間 ×N）→ 去重
  2. 「全程垃圾列」[00:00:00 p1 → case 結尾]（無真實安裝事件）→ 同 arm 衝突時取最短區間，
     全程列自動讓位；該 arm 無其他資訊時才 fallback 用它
  3. 跨 part 安裝（install p1 → uninstall p2）→ 正確展開（舊版 skip 導致漏標）
每 arm 每秒最多 1 支 ⇒ 每幀 multiset ≤4 支（相機臂排除）。真雙裝（兩臂同名工具）保留重複。
輸出：1_data/processed/rare_frames_weak_labels_v3.json  {case_pN: {sec: [tool,...]}}
用法: python3 3_src/build_weak_labels_v3.py
"""
import os
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("SURGVU_ROOT", "."))
FRAMES = ROOT / "1_data/processed/rare_frames_640"
LABELS = ROOT / "1_data/raw/labels/labels"
OUT = ROOT / "1_data/processed/rare_frames_weak_labels_v3.json"

def t2s(t):
    try:
        h, m, s = t.split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return None

def load_intervals(case):
    """每列 = (arm, gt, pa, a, pb, b)，完全重複列去重，排除相機/空白。"""
    f = LABELS / case / "tools.csv"
    seen, rows = set(), []
    if not f.exists():
        return rows
    for r in csv.DictReader(open(f)):
        gt = r["groundtruth_toolname"].strip().lower()
        if not gt or gt.startswith("nan"):
            continue
        a, b = t2s(r["install_case_time"]), t2s(r["uninstall_case_time"])
        try:
            pa, pb = int(float(r["install_case_part"])), int(float(r["uninstall_case_part"]))
        except (ValueError, TypeError):
            continue
        if a is None or b is None:
            continue
        key = (r["arm"], gt, pa, a, pb, b)
        if key in seen:
            continue
        seen.add(key)
        rows.append((r["arm"], gt, pa, a, pb, b))
    return rows

def covers(row, part, sec):
    _, _, pa, a, pb, b = row
    if pa == pb:
        return part == pa and a <= sec <= b
    return (part == pa and sec >= a) or (pa < part < pb) or (part == pb and sec <= b)

def span(row):
    """區間長度（跨 part 以大數近似 part 長度 6h，讓全程列穩定墊底）"""
    _, _, pa, a, pb, b = row
    return (b - a) if pa == pb else (pb - pa) * 21600 + b - a

def installed_at(rows, part, sec):
    """per-arm：覆蓋該秒的列中取最短區間者 → multiset（保留兩臂同名重複）。"""
    by_arm = defaultdict(list)
    for row in rows:
        if covers(row, part, sec):
            by_arm[row[0]].append(row)
    return sorted(min(cand, key=span)[1] for cand in by_arm.values())

weak = {}
dup_frames = 0
fallback_only = 0  # 幀含「僅全程列支撐」的 arm（低信度標記統計用）
for d in sorted(FRAMES.iterdir()):
    if not d.is_dir():
        continue
    case, part = d.name.rsplit("_p", 1)
    part = int(part)
    rows = load_intervals(case)
    per = {}
    for f in sorted(d.glob("sec_*.jpg")):
        sec = int(f.stem.split("_")[1])
        tools = installed_at(rows, part, sec)
        per[str(sec)] = tools
        if len(tools) > len(set(tools)):
            dup_frames += 1
    weak[d.name] = per

json.dump(weak, open(OUT, "w"))
n_frames = sum(len(v) for v in weak.values())
print(f"videos: {len(weak)} | frames: {n_frames}")
print(f"幀含同名雙裝（真雙裝）: {dup_frames}（{dup_frames/max(1,n_frames)*100:.1f}%）")
sizes = Counter(len(v) for per in weak.values() for v in per.values())
print("每幀安裝數分佈:", dict(sorted(sizes.items())))
tool_freq = Counter(t for per in weak.values() for v in per.values() for t in v)
print("工具幀數 top:", dict(tool_freq.most_common(10)))
