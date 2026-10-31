import os, sys, json
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reproduce", "baselines"))
os.chdir(ROOT)
from baselines_common import metrics

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
C_ACC, C_REC, C_STAR = "#31688E", "#35B779", "#C0392B"
G_DARK, G_MID = "#35B779", "#B5DE2B"
T_BLUE, T_GREEN, T_YG = "#8DB3CF", "#9ADBB8", "#DCEF95"
V_GREY = "#D4EFDF"
plt.rcParams.update({"font.size": 12, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def load(path, arm):
    return {str(r["question_id"]): r for r in json.load(open(path))["arms"][arm]["results"]}


def acc(r):
    return int(metrics([r])["accuracy"])


def exact_mcnemar(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def label_bars(ax, bars, fs=9):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", fontsize=fs, xytext=(0, 2), textcoords="offset points")


DS = {"ComplexTR": ("reproduce/fair_eval_329_baseline.json", "complextr"),
      "TimeQA": ("reproduce/fair_eval_timeqa_baseline.json", "timeqa"),
      "TempReason-L3": ("reproduce/fair_eval_tempreason_l3_500.json", "tempreason_l3")}
R = {}
for name, (base, key) in DS.items():
    R[name] = {"DyG-RAG": load(base, "baseline"),
               "NaiveRAG": load(f"reproduce/baselines/results/naiverag_{key}.json", "naiverag"),
               "DELTA-RAG": load(f"reproduce/results/{key}__delta+eatr.json", "delta+eatr")}
R["ComplexTR"]["GraphRAG"] = load("reproduce/baselines/results/graphrag-local_complextr.json", "graphrag-local")
R["ComplexTR"]["LightRAG"] = load("reproduce/baselines/results/lightrag-hybrid_complextr.json", "lightrag-hybrid")


def score(M):
    return 100 * np.mean([acc(r) for r in M.values()]), 100 * np.mean([metrics([r])["recall"] for r in M.values()])


def p_vs_delta(name, sysname):
    D, B = R[name]["DELTA-RAG"], R[name][sysname]
    w = sum(acc(D[k]) > acc(B[k]) for k in D); l = sum(acc(D[k]) < acc(B[k]) for k in D)
    return exact_mcnemar(w, l)


rows = []
for name in DS:
    B, D = R[name]["DyG-RAG"], R[name]["DELTA-RAG"]
    both = bo = oo = ne = 0
    for k in D:
        b, d = acc(B[k]), acc(D[k])
        both += b and d; bo += b and not d; oo += d and not b; ne += (not b) and (not d)
    rows.append(dict(benchmark=name, n=both + bo + oo + ne, both=both, ours_only=oo, base_only=bo, neither=ne,
                     net=oo - bo, discordant=oo + bo, p_mcnemar=exact_mcnemar(oo, bo)))
json.dump(rows, open(os.path.join(HERE, "out", "significance_breakdowns.json"), "w"), indent=2)


def fmt_p(p):
    if p >= 0.001:
        return "$p \\approx %.3f$" % p
    exp, m = 0, p
    while m < 1:
        m *= 10; exp += 1
    return "$p \\approx %.0f \\times 10^{-%d}$" % (m, exp)


cats = ["Both correct", "Rescued (DELTA-RAG only)", "Lost (DyG-RAG only)", "Neither"]
keys = ["both", "ours_only", "base_only", "neither"]
cols = [C_ACC, G_DARK, G_MID, V_GREY]
fig, ax = plt.subplots(figsize=(6.6, 3.7))
x = np.arange(len(rows)); w = 0.2
for j, (c_, k, col) in enumerate(zip(cats, keys, cols)):
    v = [100 * r[k] / r["n"] for r in rows]
    bars = ax.bar(x + (j - 1.5) * w, v, w, label=c_, color=col)
    label_bars(ax, bars, fs=8)
for i, r in enumerate(rows):
    ax.text(x[i], 58, "net $%+d$ of %d discordant\n%s" % (r["net"], r["discordant"], fmt_p(r["p_mcnemar"])),
            ha="center", va="bottom", fontsize=8.5, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#95A5A6", alpha=0.9))
ax.set_xticks(x)
ax.set_xticklabels(["%s\n($N{=}%s$)" % (r["benchmark"], format(r["n"], ",").replace(",", "{,}")) for r in rows], fontsize=11)
ax.set_ylabel("Share of questions (%)"); ax.set_ylim(0, 72)
ax.set_title("Per-question outcomes against DyG-RAG")
fig.legend(frameon=False, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.2, columnspacing=0.9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(OUT, "fig_outcome_breakdown_all.png"), dpi=300); plt.close(fig)

order = [("DELTA-RAG (ours)", "DELTA-RAG", C_ACC), ("DyG-RAG", "DyG-RAG", G_DARK), ("NaiveRAG", "NaiveRAG", G_MID),
         ("GraphRAG (local)", "GraphRAG", T_BLUE), ("LightRAG (hybrid)", "LightRAG", T_GREEN)]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
x = np.arange(2); w = 0.16
for j, (lab, k, c) in enumerate(order):
    a_, r_ = score(R["ComplexTR"][k])
    xs = x + (j - 2) * w
    bars = ax.bar(xs, [a_, r_], w, label=lab, color=c)
    label_bars(ax, bars, fs=8)
    if k != "DELTA-RAG":
        st = stars(p_vs_delta("ComplexTR", k))
        if st:
            ax.annotate(st, (xs[0], a_ + 5.0), ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_STAR)
ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Recall"], fontsize=11)
ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
ax.set_title("ComplexTR ($N{=}329$)")
fig.legend(frameon=False, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.2, columnspacing=0.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_main_compare.png"), dpi=300); plt.close(fig)

names = list(DS)
sysl = [("DELTA-RAG (ours)", "DELTA-RAG", C_ACC), ("DyG-RAG", "DyG-RAG", G_DARK), ("NaiveRAG", "NaiveRAG", G_MID)]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
x = np.arange(3); w = 0.26
for j, (lab, k, c) in enumerate(sysl):
    v = [score(R[n][k])[0] for n in names]
    bars = ax.bar(x + (j - 1) * w, v, w, label=lab, color=c)
    label_bars(ax, bars)
    if k != "DELTA-RAG":
        for i, n in enumerate(names):
            st = stars(p_vs_delta(n, k))
            if st:
                ax.annotate(st, (x[i] + (j - 1) * w, v[i] + 5.5), ha="center", va="bottom", fontsize=11,
                            fontweight="bold", color=C_STAR)
ax.set_xticks(x); ax.set_xticklabels(["ComplexTR", "TimeQA", "TempReason-L3"], fontsize=11)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
ax.set_title("Accuracy across benchmarks")
fig.legend(frameon=False, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_three_benchmarks.png"), dpi=300); plt.close(fig)
print("saved to", OUT)
