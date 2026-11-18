"""2주차 산출물 curve.png: results.csv에서 분할 손실과 mask mAP를 epoch별로 그린다."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/week2/train/results.csv")
ap.add_argument("--out", default="results/week2/curve.png")
ap.add_argument("--best-epoch", type=int, help="지정하지 않으면 box·mask mAP50-95 합이 가장 큰 epoch")
a = ap.parse_args()

FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_manager.fontManager.addfont(FONT)

rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(a.csv))]
col = lambda k: [r[k] for r in rows]
ep = col("epoch")
if a.best_epoch is None:
    a.best_epoch = int(max(rows, key=lambda r: r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(M)"])["epoch"])

plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(), "axes.unicode_minus": False, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
panels = [
    (axes[0], "분할 손실 (seg loss)", [("train/seg_loss", "train", BLUE), ("val/seg_loss", "val", ORANGE)]),
    (axes[1], "검증셋 mask mAP", [("metrics/mAP50(M)", "mAP50", BLUE), ("metrics/mAP50-95(M)", "mAP50-95", ORANGE)]),
]
for ax, title, series in panels:
    ax.set_facecolor(SURFACE)
    for k, label, c in series:
        ax.plot(ep, col(k), color=c, lw=2, label=label)
    ax.axvline(a.best_epoch, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_xlim(1, ep[-1])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="upper right", ncol=2, labelcolor=INK)
    ax.annotate(f"best.pt (epoch {a.best_epoch})", xy=(a.best_epoch, 0), xytext=(4, 6),
                textcoords="offset points", color=MUTED, fontsize=9)
axes[0].set_ylim(0, 5)
axes[1].set_ylim(0, 0.9)
fig.tight_layout()
fig.savefig(a.out, dpi=150, facecolor=SURFACE)

best = next(r for r in rows if int(r["epoch"]) == a.best_epoch)
print(a.out, "epochs", len(rows), "best", {k: round(best[k], 4) for k in
      ("metrics/mAP50(M)", "metrics/mAP50-95(M)", "val/seg_loss", "train/seg_loss")})
print("val seg_loss min", min((r["val/seg_loss"], int(r["epoch"])) for r in rows))
print("max mAP50(M)", max((r["metrics/mAP50(M)"], int(r["epoch"])) for r in rows))
print("box+mask mAP50-95 top3", sorted(((round(r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(M)"], 4), int(r["epoch"])) for r in rows), reverse=True)[:3])
