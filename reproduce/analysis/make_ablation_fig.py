import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
OUT = os.path.join(HERE, "figures", "fig_ablation.png")

C_ACC, C_REC, C_BASE = "#31688E", "#35B779", "#440154"
plt.rcParams.update({"font.size": 12, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

D = json.load(open(os.path.join(HERE, "out", "ablation_final.json")))
ORDER = ["full", "-EATR", "-DATP", "-CPKB", "-AHKR", "EATR only"]
LABELS = ["DELTA-RAG\n(full)", "$-$EATR", "$-$DATP", "$-$CPKB", "$-$AHKR", "EATR\nonly"]
PANELS = [("complextr", "(a) ComplexTR ($N{=}329$)"),
          ("tempreason_l3", "(b) TempReason-L3 ($N{=}500$)")]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


VCOL = ["#31688E", "#35B779", "#B5DE2B", "#8DB3CF", "#9ADBB8", "#DCEF95"]
VLAB = ["DELTA-RAG (full)", "$-$EATR", "$-$DATP", "$-$CPKB", "$-$AHKR", "EATR only"]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
x = np.arange(len(PANELS)); w = 0.135
for j, (k, lab, c) in enumerate(zip(ORDER, VLAB, VCOL)):
    v = [D[ds][k]["acc"] for ds, _ in PANELS]
    xs = x + (j - 2.5) * w
    bars = ax.bar(xs, v, w, label=lab, color=c)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
    for i, (ds, _) in enumerate(PANELS):
        st = stars(D[ds][k]["p"]) if k != "full" else ""
        if st:
            ax.annotate(st, (xs[i], v[i] + 4.5), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#C0392B")
for i, (ds, _) in enumerate(PANELS):
    dyg = D[ds]["DyG-RAG"]["acc"]
    ax.hlines(dyg, x[i] - 3.2 * w, x[i] + 3.2 * w, colors="#7F7F7F", linestyles="--", lw=1.2,
              label="DyG-RAG accuracy" if i == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(["ComplexTR ($N{=}329$)", "TempReason-L3 ($N{=}500$)"], fontsize=11)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Leave-one-module-out ablation", fontsize=12)
fig.legend(frameon=False, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.2, columnspacing=0.9)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT, dpi=300)
print("saved", OUT)
