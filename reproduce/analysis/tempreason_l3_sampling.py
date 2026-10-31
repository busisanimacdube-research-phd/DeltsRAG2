import os, sys, json, re
from collections import Counter
import numpy as np
from scipy.stats import chisquare

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reproduce", "baselines"))
from baselines_common import metrics

REL_NAMES = {"P108": "employer", "P102": "political party", "P54": "sports team",
             "P69": "educated at", "P39": "position held", "P463": "member of",
             "P286": "head coach", "P488": "chairperson", "P6": "head of government",
             "P127": "owned by", "P35": "head of state", "P26": "spouse"}

full = [json.loads(l) for l in open("datasets/TempReasonL3/test_l3.json")]
sample = json.load(open("datasets/TempReason_L3_sample500/Question.json"))


def rel(qid):
    return qid.split("_")[2]


def subj(qid):
    return qid.split("_")[1]


def direction(q):
    ql = q.lower()
    return "before" if " before " in ql else ("after" if " after " in ql else "other")


def decade(d):
    m = re.search(r"(\d{4})", d or "")
    return f"{int(m.group(1)) // 10 * 10}s" if m else "none"


def gof(name, full_keys, samp_keys, min_exp=5):
    fc, sc = Counter(full_keys), Counter(samp_keys)
    n = len(samp_keys)
    cats = sorted(fc)
    exp = np.array([fc[c] / len(full_keys) * n for c in cats])
    obs = np.array([sc.get(c, 0) for c in cats], float)
    big = exp >= min_exp
    if (~big).any():
        exp = np.append(exp[big], exp[~big].sum()); obs = np.append(obs[big], obs[~big].sum())
    stat, p = chisquare(obs, exp)
    return {"variable": name, "categories": len(cats), "df": len(exp) - 1,
            "chi2": float(stat), "p": float(p),
            "full_share": {c: fc[c] / len(full_keys) for c in cats},
            "sample_share": {c: sc.get(c, 0) / n for c in cats}}


out = {"n_full": len(full), "n_sample": len(sample),
       "distinct_subjects_full": len({subj(q["id"]) for q in full}),
       "distinct_subjects_sample": len({subj(q["id"]) for q in sample}),
       "tests": [
           gof("relation type", [rel(q["id"]) for q in full], [rel(q["id"]) for q in sample]),
           gof("direction", [direction(q["question"]) for q in full], [direction(q["question"]) for q in sample]),
           gof("query decade", [decade(q["date"]) for q in full], [decade(q["date"]) for q in sample]),
       ]}

res_path = sys.argv[1] if len(sys.argv) > 1 else "reproduce/fair_eval_tempreason_l3_500.json"
res = json.load(open(res_path))
full_rel = Counter(rel(q["id"]) for q in full)
rng = np.random.default_rng(0)
out["systems"] = {}
for arm, v in res["arms"].items():
    per_q = []
    for r in v["results"]:
        per_q.append((rel(r["question_id"]), metrics([r])["accuracy"]))
    acc = np.array([a for _, a in per_q])
    boots = [acc[rng.integers(0, len(acc), len(acc))].mean() for _ in range(10000)]
    by_rel = {}
    for rl in sorted(full_rel):
        a = [x for r_, x in per_q if r_ == rl]
        if a:
            by_rel[rl] = {"name": REL_NAMES.get(rl, rl), "n": len(a), "acc": float(np.mean(a))}
    post = sum(full_rel[r] / len(full) * by_rel[r]["acc"] for r in by_rel) / \
        sum(full_rel[r] / len(full) for r in by_rel)
    out["systems"][arm] = {"acc": float(acc.mean()),
                           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           "post_stratified_acc": float(post), "by_relation": by_rel}

arms = list(res["arms"])
if len(arms) >= 2:
    a = np.array([metrics([r])["accuracy"] for r in res["arms"][arms[0]]["results"]])
    b = np.array([metrics([r])["accuracy"] for r in res["arms"][arms[1]]["results"]])
    idx = [rng.integers(0, len(a), len(a)) for _ in range(10000)]
    d = [b[i].mean() - a[i].mean() for i in idx]
    out["difference"] = {"arms": f"{arms[1]} - {arms[0]}", "diff": float(b.mean() - a.mean()),
                         "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

os.makedirs("reproduce/analysis/out", exist_ok=True)
json.dump(out, open("reproduce/analysis/out/tempreason_l3_sampling.json", "w"), indent=2)
for t in out["tests"]:
    print(f"{t['variable']:14s} chi2={t['chi2']:.2f} df={t['df']} p={t['p']:.3f}")
print("subjects full/sample:", out["distinct_subjects_full"], out["distinct_subjects_sample"])
for arm, s in out["systems"].items():
    print(arm, f"acc={s['acc']:.3f} CI={s['ci95'][0]:.3f}-{s['ci95'][1]:.3f} post-strat={s['post_stratified_acc']:.3f}")
    for r_, x in s["by_relation"].items():
        print(f"   {r_:5s} {x['name']:18s} n={x['n']:3d} acc={x['acc']:.3f}")
print(out.get("difference"))
