import os, sys, json
from collections import defaultdict
import numpy as np
import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from graphrag.delta_rag import entity_matching as em
from graphrag.delta_rag.temporal_index import day_index
from graphrag._utils import compute_args_hash
from graphrag.evaluate import Evaluator

SETS = {
    "complextr": ("datasets/ComplexTR/Question.json", "reproduce/delta_complextr_fair"),
    "tempreason_l3": ("datasets/TempReason_L3_sample500/Question.json", "reproduce/tempreason_full_delta"),
    "timeqa": ("datasets/TimeQA/Question.json", "reproduce/delta_timeqa_fair"),
}
M = int(os.getenv("EATR_M", 20))
ev = Evaluator()


def run(ds, qf, d, mode):
    norm = em.normalize_v1 if mode == "v1" else em.normalize_strict
    match = em.match_v1 if mode == "v1" else em.match_strict
    cache = json.load(open(os.path.join(d, "kv_store_llm_response_cache.json")))
    g = nx.read_graphml(os.path.join(d, "graph_event_dynamic_graph.graphml"))
    chunks = json.load(open(os.path.join(d, "kv_store_text_chunks.json")))
    index = defaultdict(list)
    for nid, data in g.nodes(data=True):
        try:
            ents = json.loads(data.get("entities_involved", "[]") or "[]")
        except Exception:
            ents = [data.get("entities_involved")]
        for e in ents or []:
            k = norm(e)
            if k and (mode == "v1" or em.valid_index_key(k)):
                index[k].append((nid, data))
    keys = list(index)
    rows = []
    for q in json.load(open(qf)):
        c = cache.get(compute_args_hash("dynamic", q["question"], "time_entities"))
        if not c or c.get("cache_type") != "time_entities":
            continue
        p = json.loads(c["content"])
        tc, ents = p.get("time_constraints") or {}, p.get("entities") or []
        if mode != "v1":
            ents = em.filter_query_entities(ents, q["question"])
        qn = [norm(e) for e in ents if norm(e)]
        mk = [k for k in keys if any(match(x, k) for x in qn)]
        seen, matched = set(), []
        for k in mk:
            for nid, data in index[k]:
                if nid not in seen:
                    seen.add(nid); matched.append((nid, data))
        anchor = day_index(tc.get("start_time") or tc.get("end_time"))
        matched.sort(key=lambda it: (1, 0) if anchor is None or day_index(it[1].get("timestamp")) is None
                     else (0, abs(day_index(it[1].get("timestamp")) - anchor)))
        top = matched[:M]
        srcs = {s for _, dd in top for s in str(dd.get("source_id", "")).split("<SEP>") if s}
        text = " ".join(chunks.get(s, {}).get("content", "") for s in srcs)
        gold = [a for a in ev.parse_multiple_answers(str(q.get("answer", ""))) if a.strip()]
        hit = bool(gold) and all(ev.eval_accuracy(text, x.strip()) == 1 for x in gold)
        rows.append({"q": q["question"], "ents": ents, "keys": len(mk), "events": len(matched),
                     "anchored": len(top) > 0, "hit": hit, "n_src": len(srcs), "sample_keys": mk[:8]})
    return rows


def main():
    out = {}
    for ds, (qf, d) in SETS.items():
        out[ds] = {}
        for mode in ("v1", "strict"):
            r = run(ds, qf, d, mode)
            out[ds][mode] = {
                "n": len(r),
                "anchored_rate": float(np.mean([x["anchored"] for x in r])),
                "keys_median": float(np.median([x["keys"] for x in r])),
                "events_median": float(np.median([x["events"] for x in r])),
                "anchor_recall": float(np.mean([x["hit"] for x in r])),
                "source_chunks_median": float(np.median([x["n_src"] for x in r])),
                "missed_examples": [x for x in r if not x["anchored"]][:5],
            }
            s = out[ds][mode]
            print(f"{ds:14s} {mode:6s} n={s['n']:4d} anchored={s['anchored_rate']:.3f} "
                  f"keys_med={s['keys_median']:.0f} events_med={s['events_median']:.0f} "
                  f"src_med={s['source_chunks_median']:.0f} anchor_recall={s['anchor_recall']:.3f}", flush=True)
    os.makedirs("reproduce/analysis/out", exist_ok=True)
    json.dump(out, open("reproduce/analysis/out/eatr_matching_compare.json", "w"), indent=2)


if __name__ == "__main__":
    main()
