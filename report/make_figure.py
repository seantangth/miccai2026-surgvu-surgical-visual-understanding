"""產生 APC_TDC 報告的 pipeline 圖（Category 1）。
輸出：TeamDocs2026/seantangth/figure1.png（300 dpi，白底）
設計原則：兩排由左至右，連線只走方塊之間的空白通道，不穿越任何方塊。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "TeamDocs2026" / "APC_TDC" / "figure1.png"

INK = "#1a1a1a"
GREY = "#6b6b6b"
DATA, MODEL, GATE = "#2f6f9f", "#b4541e", "#43763a"
FILL = {"data": "#e9f1f8", "model": "#fbefe7", "gate": "#ebf3e9"}

fig, ax = plt.subplots(figsize=(13.4, 5.6))
ax.set_xlim(0, 134)
ax.set_ylim(0, 60)
ax.axis("off")

TOP_Y, BOT_Y, H = 32.0, 6.0, 17.0


def box(x, w, y, title, sub, kind, fs=9.0):
    edge = {"data": DATA, "model": MODEL, "gate": GATE}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.6, edgecolor=edge, facecolor=FILL[kind]))
    ax.text(x + w / 2, y + H - 2.6, title, ha="center", va="top", fontsize=fs,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + H - 7.0, sub, ha="center", va="top", fontsize=fs - 1.5,
            color="#3d3d3d", linespacing=1.5)


def arrow(p1, p2, color=GREY, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                                 color=color, linestyle=ls, shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def line(p1, p2, color=GREY, lw=1.4):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, solid_capstyle="round")


# --- 上排：偽標籤生成 -------------------------------------------------------
box(1, 22, TOP_Y, "280 training videos", "60 fps, 720p, 840 h\n155 sessions\ntool presence only", "data")
box(26, 23, TOP_Y, "Domain-aligned frames", "1 fps sampling\ncrop $x\\in[192,1088]$\n$\\rightarrow$ 640$\\times$512", "data")
box(52, 21, TOP_Y, "Teacher ensemble", "2 $\\times$ RT-DETR-L\nWBF, IoU 0.7\nconf_type max", "model")
box(76, 27, TOP_Y, "Strict gate (tools.csv)", "per-class box count $=$\nper-arm installed count,\nall boxes conf $\\geq$ 0.35", "gate")
box(106, 21, TOP_Y, "Pseudo-labels", "82,152 frames\n26% pass rate\nno size filtering", "data")

for x1, x2 in ((23, 26), (49, 52), (73, 76), (103, 106)):
    arrow((x1, TOP_Y + H / 2), (x2, TOP_Y + H / 2))

# --- 迭代迴圈（正交繞行於上排之上，保證不穿越方塊）--------------------------
LOOP_Y = 53.0
line((116.5, TOP_Y + H), (116.5, LOOP_Y), color=MODEL, lw=1.5)
line((116.5, LOOP_Y), (62.5, LOOP_Y), color=MODEL, lw=1.5)
arrow((62.5, LOOP_Y), (62.5, TOP_Y + H), color=MODEL, lw=1.5)
ax.text(89.5, 56.4, "iterate: the retrained student becomes the next teacher   (pass rate 26% $\\rightarrow$ 41%)",
        ha="center", va="center", fontsize=8.4, color=MODEL, style="italic")

# --- 上排 → 下排（走兩排之間的通道）----------------------------------------
line((116, TOP_Y), (116, 27.5))
line((116, 27.5), (37.5, 27.5))
arrow((37.5, 27.5), (37.5, BOT_Y + H))

# --- 下排：訓練與提交 -------------------------------------------------------
box(1, 22, BOT_Y, "Model-assisted boxes", "718 LLM-annotated frames\n$\\rightarrow$ 2,096, plus 202\nseeded retractor frames", "model")
box(26, 23, BOT_Y, "Training set", "93,660 frames\nofficial GT merged in\n(ALLGT)", "data")
box(52, 21, BOT_Y, "Student training", "RT-DETR-L\n640 px and 800 px\nA100 40 GB", "model")
box(76, 27, BOT_Y, "Submitted algorithm", "WBF of the 640 and 800\nmodels, $\\leq$100 boxes/frame\nat conf $\\geq$ 0.05", "model")

for x1, x2 in ((23, 26), (49, 52), (73, 76)):
    arrow((x1, BOT_Y + H / 2), (x2, BOT_Y + H / 2))

ax.text(67, 2.0,
        "final training set  =  5,178 official GT frames  $+$  82,152 pseudo-labelled  $+$  2,298 model-assisted  $+$  4,032 first-round frames",
        ha="center", va="center", fontsize=8.0, color="#555555")

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", OUT, OUT.stat().st_size, "bytes")
