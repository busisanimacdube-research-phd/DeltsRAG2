import os, sys, json, re
from collections import Counter, defaultdict
import numpy as np
import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from graphrag.delta_rag.ahkr import AdaptiveHierarchicalRetrieval, AHKRConfig
from graphrag.delta_rag.temporal_index import day_index
from graphrag._utils import compute_args_hash
from graphrag.evaluate import Evaluator

SETS = {
    "complextr": ("datasets/ComplexTR/Question.json", "reproduce/delta_complextr_fair"),
    "tempreason_l3": ("datasets/TempReason_L3_sample500/Question.json", "reproduce/tempreason_full_delta"),
    "timeqa": ("datasets/TimeQA/Question.json", "reproduce/delta_timeqa_fair"),
}
M = 20


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def match_kind(q, e):
    if not q or not e:
        return None
    if q == e:
        return "exact"
    if len(q) >= 4 and (q in e or e in q):
        return "containment"
    qw, ew = set(q.split()), set(e.split())
    inter = len(qw & ew)
    if qw and ew and inter and inter / len(qw | ew) >= 0.6 and any(len(t) >= 4 for t in (qw & ew)):
        return "jaccard"
    return None


def load_cache(d):
    caches = {}
    for f in (os.path.join(d, "kv_store_llm_response_cache.json"),):
        if os.path.exists(f):
            caches.update(json.load(open(f)))
    return caches


def main():
    ev = Evaluator()
    ahkr = AdaptiveHierarchicalRetrieval(AHKRConfig())
    report = {}
    for ds, (qf, d) in SETS.items():
        qs = json.load(open(qf))
        cache = load_cache(d)
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
                if k:
                    index[k].append((nid, data))
        keys = list(index)

        labels, ts_, es_ = Counter(), [], []
        n_parsed = n_ent = n_anchor = 0
        kinds = Counter()
        n_keys, n_events, n_injected = [], [], []
        gold_in_anchor = gold_in_anchor_den = 0
        multi_referent = 0
        for q in qs:
            h = compute_args_hash("dynamic", q["question"], "time_entities")
            c = cache.get(h)
            if not c or c.get("cache_type") != "time_entities":
                continue
            p = json.loads(c["content"])
            tc, ents = p.get("time_constraints") or {}, p.get("entities") or []
            n_parsed += 1
            info = ahkr._classify_query(q["question"], tc, ents)
            labels[info["label"]] += 1
            ts_.append(info["temporal_strength"]); es_.append(info["entity_strength"])
            qn = [norm(e) for e in ents if norm(e)]
            if not qn:
                continue
            n_ent += 1
            matched_keys, seen, matched = [], set(), []
            for k in keys:
                kk = [match_kind(x, k) for x in qn]
                kk = [x for x in kk if x]
                if kk:
                    matched_keys.append(k)
                    kinds["exact" if "exact" in kk else kk[0]] += 1
                    for nid, data in index[k]:
                        if nid not in seen:
                            seen.add(nid); matched.append((nid, data))
            if not matched:
                continue
            n_anchor += 1
            n_keys.append(len(matched_keys)); n_events.append(len(matched))
            roots = []
            for k in sorted(matched_keys, key=len):
                if not any(r in k for r in roots):
                    roots.append(k)
            if len(roots) > len(qn):
                multi_referent += 1
            anchor = day_index(tc.get("start_time") or tc.get("end_time"))
            matched.sort(key=lambda it: (1, 0) if anchor is None or day_index(it[1].get("timestamp")) is None
                         else (0, abs(day_index(it[1].get("timestamp")) - anchor)))
            top = matched[:M]
            n_injected.append(len(top))
            gold = [a for a in ev.parse_multiple_answers(str(q.get("answer", ""))) if a.strip()]
            if gold:
                gold_in_anchor_den += 1
                srcs = {s for _, dd in top for s in str(dd.get("source_id", "")).split("<SEP>") if s}
                text = " ".join(chunks.get(s, {}).get("content", "") for s in srcs)
                if all(ev.eval_accuracy(text, x.strip()) == 1 for x in gold):
                    gold_in_anchor += 1
        n = max(n_parsed, 1)
        report[ds] = {
            "questions": len(qs), "parsed_from_cache": n_parsed,
            "query_types": {k: v / n for k, v in labels.items()},
            "temporal_strength_mean": float(np.mean(ts_)) if ts_ else None,
            "entity_strength_mean": float(np.mean(es_)) if es_ else None,
            "queries_with_entities": n_ent / n,
            "anchor_found_rate": n_anchor / n,
            "match_rule_share": {k: v / max(sum(kinds.values()), 1) for k, v in kinds.items()},
            "matched_keys_median": float(np.median(n_keys)) if n_keys else None,
            "matched_events_median": float(np.median(n_events)) if n_events else None,
            "matched_events_p90": float(np.percentile(n_events, 90)) if n_events else None,
            "injected_events_mean": float(np.mean(n_injected)) if n_injected else None,
            "queries_matching_extra_referents": multi_referent / max(n_anchor, 1),
            "gold_answer_in_anchored_source_chunks": gold_in_anchor / max(gold_in_anchor_den, 1),
        }
        print(ds, json.dumps(report[ds], indent=1))
    os.makedirs("reproduce/analysis/out", exist_ok=True)
    json.dump(report, open("reproduce/analysis/out/query_and_anchor_analysis.json", "w"), indent=2)


if __name__ == "__main__":
    main()
