#!/usr/bin/env python3
"""弱標籤（per-arm multiset）— build_weak_labels_v3.py 的可攜版（Lambda 用）。
用法：python3 weak_labels_scale.py <frames_dir> <labels_dir> <out.json>
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
FRAMES, LABELS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

def t2s(t):
    try: h, m, s = t.split(":"); return float(h)*3600 + float(m)*60 + float(s)
    except (ValueError, AttributeError): return None

def load_intervals(case):
    f = LABELS / case / "tools.csv"; seen, rows = set(), []
    if not f.exists(): return rows
    for r in csv.DictReader(open(f)):
        gt = r["groundtruth_toolname"].strip().lower()
        if not gt or gt.startswith("nan"): continue
        a, b = t2s(r["install_case_time"]), t2s(r["uninstall_case_time"])
        try: pa, pb = int(float(r["install_case_part"])), int(float(r["uninstall_case_part"]))
        except (ValueError, TypeError): continue
        if a is None or b is None: continue
        key = (r["arm"], gt, pa, a, pb, b)
        if key in seen: continue
        seen.add(key); rows.append((r["arm"], gt, pa, a, pb, b))
    return rows

def covers(row, part, sec):
    _, _, pa, a, pb, b = row
    if pa == pb: return part == pa and a <= sec <= b
    return (part == pa and sec >= a) or (pa < part < pb) or (part == pb and sec <= b)

def span(row):
    _, _, pa, a, pb, b = row
    return (b - a) if pa == pb else (pb - pa) * 21600 + b - a

def installed_at(rows, part, sec):
    by_arm = defaultdict(list)
    for row in rows:
        if covers(row, part, sec): by_arm[row[0]].append(row)
    return sorted(min(cand, key=span)[1] for cand in by_arm.values())

weak = {}; dup = 0
for d in sorted(FRAMES.iterdir()):
    if not d.is_dir(): continue
    case, part = d.name.rsplit("_p", 1); part = int(part)
    rows = load_intervals(case); per = {}
    for f in sorted(d.glob("sec_*.jpg")):
        sec = int(f.stem.split("_")[1]); tools = installed_at(rows, part, sec); per[str(sec)] = tools
        if len(tools) > len(set(tools)): dup += 1
    weak[d.name] = per
json.dump(weak, open(OUT, "w"))
n = sum(len(v) for v in weak.values())
print(f"videos: {len(weak)} | frames: {n} | 真雙裝幀 {dup} ({dup/max(1,n)*100:.1f}%)")
print("每幀安裝數分佈:", dict(sorted(Counter(len(v) for per in weak.values() for v in per.values()).items())))
print("工具幀數:", dict(Counter(t for per in weak.values() for v in per.values() for t in v).most_common(14)))
