import os, sys, json, shutil, argparse, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx

from graphrag.delta_rag import (
    DensityAdaptiveTemporalPartitioning, DATPConfig,
    CrossPartitionKGBridging, CPKBConfig,
)
from graphrag.delta_rag.temporal_index import day_index

GRAPH = "graph_event_dynamic_graph.graphml"


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def parse_entities(raw):
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [s.strip() for s in str(raw).split(",") if s.strip()]


def span_years(span):
    a, b = span
    if not a or not b:
        return None
    da, db = day_index(a), day_index(b)
    return None if da is None or db is None else (db - da) / 365.2425


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with the pristine DyG-RAG graph")
    ap.add_argument("--dst", required=True, help="DELTA working dir to (re)build")
    ap.add_argument("--temporal-weight", type=float, default=1.0)
    ap.add_argument("--no-core-span", action="store_true")
    ap.add_argument("--decay", type=float, default=0.005)
    ap.add_argument("--stats-only", action="store_true",
                    help="compute DATP/CPKB statistics without writing state")
    args = ap.parse_args()

    state = os.path.join(args.dst, "delta_rag_state")
    if not args.stats_only:
        if os.path.isdir(state) and not os.path.isdir(state + ".v1_submitted"):
            shutil.copytree(state, state + ".v1_submitted")
        g_dst = os.path.join(args.dst, GRAPH)
        if os.path.exists(g_dst) and not os.path.exists(g_dst + ".v1_submitted"):
            shutil.copy2(g_dst, g_dst + ".v1_submitted")
        os.makedirs(state, exist_ok=True)

    g = nx.read_graphml(os.path.join(args.src, GRAPH))
    assert not any(d.get("relation_type") == "semantic_bridge" for _, _, d in g.edges(data=True))
    events = [{
        "event_id": nid,
        "timestamp": d.get("timestamp", "static") or "static",
        "sentence": d.get("sentence", "") or "",
        "context": d.get("context", "") or "",
        "entities_involved": parse_entities(d.get("entities_involved", "")),
        "source_id": d.get("source_id", "") or "",
    } for nid, d in g.nodes(data=True)]
    print(f"{len(events)} events, {g.number_of_edges()} edges from {args.src}")

    emb = None
    for cand in (state + ".v1_submitted", state):
        ep, evp = os.path.join(cand, "embeddings.npy"), os.path.join(cand, "events.json")
        if os.path.exists(ep) and os.path.exists(evp):
            old = json.load(open(evp))
            if [e["event_id"] for e in old] == [e["event_id"] for e in events]:
                emb = np.load(ep); print(f"reusing embeddings from {cand}"); break
    if emb is None:
        import torch
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3"),
                                 device="mps" if torch.backends.mps.is_available() else "cpu")
        texts = [e["sentence"] + (f" (Time: {e['timestamp']})" if e["timestamp"] != "static" else "")
                 for e in events]
        emb = st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    cfg = DATPConfig(temporal_weight=args.temporal_weight,
                     span_from_core_only=not args.no_core_span)
    datp = DensityAdaptiveTemporalPartitioning(cfg)
    tkps = datp.fit(events, emb)
    noise = datp.noise_mask_ if datp.noise_mask_ is not None else np.zeros(len(events), bool)

    infl = []
    for t in tkps:
        core = [events[i]["timestamp"] for i in t.event_indices
                if not noise[i] and day_index(events[i]["timestamp"]) is not None]
        allts = [events[i]["timestamp"] for i in t.event_indices
                 if day_index(events[i]["timestamp"]) is not None]
        if core and allts:
            c = sorted(core, key=day_index); a = sorted(allts, key=day_index)
            infl.append((span_years((a[0], a[-1])) or 0) - (span_years((c[0], c[-1])) or 0))
    spans = [s for s in (span_years(t.temporal_span) for t in tkps) if s is not None]
    sizes = [len(t.event_indices) for t in tkps]

    cpkb = CrossPartitionKGBridging(CPKBConfig(temporal_decay_rate=args.decay))
    bridges = cpkb.compute_bridges(tkps, events, emb) if len(tkps) >= 2 else []

    coh = []
    for t in tkps:
        c = emb[t.event_indices].mean(axis=0); c = c / max(np.linalg.norm(c), 1e-12)
        coh.append(float(np.mean(emb[t.event_indices] @ c)))
    gaps = []
    for b in bridges:
        da = day_index(events[b["source_event_idx"]]["timestamp"])
        db = day_index(events[b["target_event_idx"]]["timestamp"])
        if da is not None and db is not None:
            gaps.append(abs(da - db) / 365.2425)

    stats = {
        "n_events": len(events), "n_dated": sum(day_index(e["timestamp"]) is not None for e in events),
        "temporal_weight": cfg.temporal_weight, "span_from_core_only": cfg.span_from_core_only,
        "dbcv_by_min_cluster_size": datp.tuning_scores_, "min_cluster_size": datp._best_min_cluster_size,
        "n_tkps": len(tkps), "tkp_size_median": float(np.median(sizes)) if sizes else 0,
        "n_noise": int(noise.sum()), "noise_frac": float(noise.mean()) if len(noise) else 0,
        "span_years_median": float(np.median(spans)) if spans else None,
        "span_years_p90": float(np.percentile(spans, 90)) if spans else None,
        "noise_span_inflation_years_mean": float(np.mean(infl)) if infl else 0,
        "noise_span_inflation_years_max": float(np.max(infl)) if infl else 0,
        "tkps_inflated_by_noise": int(sum(1 for x in infl if x > 0)),
        "semantic_coherence_mean": float(np.mean(coh)) if coh else None,
        "cpkb": cpkb.get_statistics(),
        "cpkb_candidates_above_threshold": getattr(cpkb, "n_candidates", None),
        "bridge_gap_years_median": float(np.median(gaps)) if gaps else None,
        "bridge_gap_years_max": float(np.max(gaps)) if gaps else None,
        "bridges_between_dated": len(gaps),
    }
    print(json.dumps(stats, indent=2, cls=NpEnc))
    if args.stats_only:
        out = os.getenv("STATS_OUT")
        if out:
            json.dump(stats, open(out, "w"), indent=2, cls=NpEnc)
        return stats

    np.save(os.path.join(state, "embeddings.npy"), emb)
    json.dump(events, open(os.path.join(state, "events.json"), "w"))
    json.dump([{
        "partition_id": t.partition_id, "event_indices": t.event_indices, "event_ids": t.event_ids,
        "confidence_scores": {str(k): v for k, v in t.confidence_scores.items()},
        "temporal_span": list(t.temporal_span), "avg_confidence": t.avg_confidence,
    } for t in tkps], open(os.path.join(state, "tkps.json"), "w"), cls=NpEnc)
    json.dump(bridges, open(os.path.join(state, "bridges.json"), "w"), cls=NpEnc)
    json.dump(stats, open(os.path.join(state, "datp_cpkb_stats.json"), "w"), indent=2, cls=NpEnc)
    mp = os.path.join(state, "mstc_model.pt")
    if os.path.exists(mp):
        os.remove(mp)

    n_new = 0
    for b in bridges:
        if g.has_edge(b["source_event_id"], b["target_event_id"]):
            continue
        n_new += 1
        g.add_edge(b["source_event_id"], b["target_event_id"],
                   relation_type="semantic_bridge", weight=float(b["bridge_weight"]),
                   time_distance=0, shared_entities=",".join(b.get("shared_entities", [])[:20]),
                   description=f"CPKB bridge TKP-{b['source_partition']}<->TKP-{b['target_partition']}",
                   source_id="", is_undirected=True)
    nx.write_graphml(g, os.path.join(args.dst, GRAPH))
    stats["bridge_edges_added_to_graph"] = n_new
    json.dump(stats, open(os.path.join(state, "datp_cpkb_stats.json"), "w"), indent=2, cls=NpEnc)
    print(f"wrote state; {len(bridges)} bridges, {n_new} new graph edges -> {args.dst}")


if __name__ == "__main__":
    main()
