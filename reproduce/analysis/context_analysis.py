import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines"))
sys.path.insert(0, ".")
from baselines_common import metrics, DATASETS
from graphrag._utils import encode_string_by_tiktoken

ARMS = {"DELTA-RAG": "delta+eatr", "DELTA-RAG - EATR": "delta", "EATR only": "base+eatr"}
BINS = [(1, 5), (6, 10), (11, 15), (16, 20)]


def load(ds, arm):
    p = f"reproduce/results/{ds}__{arm}.json"
    return json.load(open(p))["arms"][arm]["results"] if os.path.exists(p) else None


def naive_tokens(ds):
    p = f"reproduce/baselines/results/naiverag_{ds}.json"
    if not os.path.exists(p):
        return None
    from naive_rag import NAIVE_PROMPT, RESPONSE_TYPE
    ch = json.load(open(os.path.join(DATASETS[ds]["dir"], "kv_store_text_chunks.json")))
    out = []
    for r in json.load(open(p))["arms"]["naiverag"]["results"]:
        ids = r["retrieved_chunk_ids"][: r["n_ctx_chunks"]]
        sec = "--New Chunk--\n".join(ch[i]["content"] for i in ids)
        out.append(len(encode_string_by_tiktoken(NAIVE_PROMPT.format(content_data=sec, response_type=RESPONSE_TYPE)
                                                  + r["question"])))
    return out


def pct(a, q):
    return float(np.percentile(a, q))


def main(dss):
    res = {}
    for ds in dss:
        R = {}
        for name, arm in ARMS.items():
            rows = load(ds, arm)
            if not rows:
                continue
            toks = [r["context"]["prompt_tokens"] for r in rows if r.get("context")]
            R[name] = {"n": len(rows), "tokens_median": pct(toks, 50), "tokens_p95": pct(toks, 95),
                       "tokens_max": max(toks),
                       "events_median": pct([r["context"]["n_events"] for r in rows], 50),
                       "chunks_median": pct([r["context"]["n_chunks"] for r in rows], 50)}
        nt = naive_tokens(ds)
        if nt:
            R["NaiveRAG"] = {"n": len(nt), "tokens_median": pct(nt, 50), "tokens_p95": pct(nt, 95), "tokens_max": max(nt)}
        full = load(ds, "delta+eatr")
        acc = np.array([metrics([r])["accuracy"] for r in full])
        first = [min(r["context"]["gold_chunk_positions"]) if r["context"]["gold_chunk_positions"] else None for r in full]
        pos = {}
        for lo, hi in BINS:
            idx = [i for i, f in enumerate(first) if f is not None and lo <= f <= hi]
            pos[f"{lo}-{hi}"] = {"n": len(idx), "acc": float(acc[idx].mean()) if idx else None}
        idx = [i for i, f in enumerate(first) if f is None]
        pos["not in chunks"] = {"n": len(idx), "acc": float(acc[idx].mean()) if idx else None}
        toks = np.array([r["context"]["prompt_tokens"] for r in full])
        qs = np.percentile(toks, [25, 50, 75])
        quart = {}
        for k, (lo, hi) in enumerate([(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)], 1):
            idx = np.where((toks > lo) & (toks <= hi))[0]
            quart[f"Q{k}"] = {"n": int(len(idx)), "range": [int(toks[idx].min()), int(toks[idx].max())],
                              "acc": float(acc[idx].mean())}
        noe = {str(r["question_id"]): r["context"]["prompt_tokens"] for r in (load(ds, "delta") or [])}
        diff = [r["context"]["prompt_tokens"] - noe[str(r["question_id"])] for r in full if str(r["question_id"]) in noe]
        res[ds] = {"systems": R, "position": pos, "length_quartiles": quart,
                   "eatr_added_tokens_median": pct(diff, 50) if diff else None,
                   "gold_in_context": float(np.mean([r["context"]["gold_anywhere"] for r in full]))}
    os.makedirs("reproduce/analysis/out", exist_ok=True)
    out = "reproduce/analysis/out/context_analysis.json"
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res); json.dump(old, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["complextr", "tempreason_l3"])
