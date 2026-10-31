import json, os, sys
from collections import defaultdict
import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from graphrag.delta_rag import entity_matching as em
from graphrag.delta_rag.temporal_index import day_index_exact, day_index_approx
from graphrag._utils import compute_args_hash

SETS = [("complextr", "datasets/ComplexTR/Question.json", "reproduce/delta_complextr_fair"),
        ("tempreason_l3", "datasets/TempReason_L3_sample500/Question.json", "reproduce/tempreason_full_delta"),
        ("timeqa", "datasets/TimeQA/Question.json", "reproduce/delta_timeqa_fair")]
M = 20


def main():
    out = {}
    for ds, qf, d in SETS:
        cache = json.load(open(os.path.join(d, "kv_store_llm_response_cache.json")))
        g = nx.read_graphml(os.path.join(d, "graph_event_dynamic_graph.graphml"))
        index = defaultdict(list)
        for nid, data in g.nodes(data=True):
            for e in json.loads(data.get("entities_involved", "[]") or "[]"):
                k = em.normalize_strict(e)
                if k and em.valid_index_key(k):
                    index[k].append((nid, data))
        keys = list(index)
        n = same = 0
        overlap = []
        for q in json.load(open(qf)):
            c = cache.get(compute_args_hash("dynamic", q["question"], "time_entities"))
            if not c:
                continue
            p = json.loads(c["content"])
            tc = p.get("time_constraints") or {}
            qn = [em.normalize_strict(e) for e in em.filter_query_entities(p.get("entities") or [], q["question"])]
            seen, matched = set(), []
            for k in keys:
                if any(em.match_strict(x, k) for x in qn):
                    for nid, data in index[k]:
                        if nid not in seen:
                            seen.add(nid); matched.append((nid, data))
            if not matched:
                continue
            anchor = tc.get("start_time") or tc.get("end_time")

            def top(fn):
                ad = fn(anchor)
                key = lambda it: (1, 0) if ad is None or fn(it[1].get("timestamp")) is None \
                    else (0, abs(fn(it[1].get("timestamp")) - ad))
                return [x[0] for x in sorted(matched, key=key)[:M]]
            a, b = top(day_index_approx), top(day_index_exact)
            n += 1
            same += set(a) == set(b)
            overlap.append(len(set(a) & set(b)) / max(len(a), 1))
        out[ds] = {"anchored_queries": n, "identical_top20": same,
                   "mean_overlap": sum(overlap) / max(len(overlap), 1)}
        print(ds, out[ds], flush=True)
    os.makedirs("reproduce/analysis/out", exist_ok=True)
    json.dump(out, open("reproduce/analysis/out/date_index_effect.json", "w"), indent=2)


if __name__ == "__main__":
    main()
