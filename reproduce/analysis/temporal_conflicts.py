import os, sys, json, random
import networkx as nx
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); os.chdir(ROOT); sys.path.insert(0, ROOT)
from graphrag.delta_rag.conflicts import ConflictIndex, conflicting, self_consistent

WD = {"ComplexTR": "reproduce/delta_complextr_fair", "TempReason-L3": "reproduce/tempreason_full_delta",
      "TimeQA": "reproduce/delta_timeqa_fair"}


def main():
    out = {}
    for name, wd in WD.items():
        g = nx.read_graphml(f"{wd}/graph_event_dynamic_graph.graphml")
        chunks = json.load(open(f"{wd}/kv_store_text_chunks.json"))
        idx = ConflictIndex(dict(g.nodes(data=True)), {c: v.get("full_doc_id", c) for c, v in chunks.items()})
        pairs = lambda v: [(a, b) for a in v for b in v if a < b and conflicting(a, b)]
        conf = [k for k, v in idx.support.items() if pairs(v)]
        cross = [k for k in conf if any(not (idx.support[k][a] & idx.support[k][b]) for a, b in pairs(idx.support[k]))]
        recur = [k for k in conf if k not in cross]
        extr = [k for k in conf if any(not self_consistent(idx.rep[k][t]) for t in idx.support[k])]
        cross_clean = [k for k in cross if all(self_consistent(idx.rep[k][t]) for t in idx.support[k])]
        dated = sum(1 for _, d in g.nodes(data=True) if str(d.get("timestamp", ""))[:4].isdigit())
        ex = [{"entities": list(k[0]), "reports": {t: {"docs": len(idx.support[k][t]), "sentence": idx.rep[k][t]["sentence"]}
                                                   for t in idx.support[k]}} for k in random.Random(0).sample(conf, min(8, len(conf)))]
        out[name] = {"events": g.number_of_nodes(), "dated_events": dated, "event_keys": len(idx.support),
                     "same_event_different_dates": len(conf), "of_which_extraction_error": len(extr),
                     "of_which_recurring_same_document": len(recur), "conflicting_reports_disjoint_documents": len(cross),
                     "conflicting_reports_disjoint_documents_no_extraction_error": len(cross_clean), "examples": ex,
                     "cross_examples": [{t: (len(idx.support[k][t]), idx.rep[k][t]["sentence"][:120]) for t in idx.support[k]} for k in cross_clean[:10]]}
        print(name, {k: v for k, v in out[name].items() if k not in ("examples", "cross_examples")})
    json.dump(out, open("reproduce/analysis/out/temporal_conflicts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
