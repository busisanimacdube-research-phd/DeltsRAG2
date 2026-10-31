import os, sys, re, json, time, random, asyncio, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_experiments as RE
from graphrag import GraphRAG, QueryParam
from graphrag.evaluate import Evaluator
from graphrag._utils import compute_args_hash
from baselines_common import metrics

EV = Evaluator()
YEAR = re.compile(r"\b(1[0-9]\d\d|20\d\d)\b")
FAKE = "chunk-injected"
N_SAMPLE, SEED = 150, 0


def golds(a):
    return [EV.normalize_answer(g) for g in EV.parse_multiple_answers(str(a or "")) if g.strip()]


class NotInjected(Exception):
    pass


def make_hook(gold, info, strict=False):
    def hook(events, query):
        for i, ev in enumerate(events):
            text = EV.normalize_answer(f"{ev.get('sentence', '')} {ev.get('context', '')}")
            ys = YEAR.findall(str(ev.get("timestamp", ""))[:4])
            if ys and any(g and g in text for g in gold):
                y = int(ys[0]); off = 8 if y + 8 <= 2025 else -8
                fake = dict(ev)
                fake["id"] = f"event-injected-{ev.get('id', '')}"
                fake["timestamp"] = str(y + off) + str(ev["timestamp"])[4:]
                fake["sentence"] = YEAR.sub(lambda m: str(int(m.group()) + off), ev.get("sentence", ""))
                fake["context"] = YEAR.sub(lambda m: str(int(m.group()) + off), ev.get("context", "") or "")
                fake["source_id"] = FAKE
                info.update(injected=True, true_event=ev.get("sentence", ""), true_ts=ev.get("timestamp"),
                            fake_event=fake["sentence"], fake_ts=fake["timestamp"], offset=off)
                info["fake_chunk"] = f"{fake['sentence']} {fake['context']}".strip()
                return events[: i + 1] + [fake] + events[i + 1:]
        info["injected"] = False
        if strict:
            raise NotInjected()
        return events
    return hook


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["scan", "answer", "sample", "prepass"])
    ap.add_argument("--dataset", required=True, choices=list(RE.DATASETS))
    ap.add_argument("--arm", default="")
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    questions = json.load(open(RE.DATASETS[a.dataset]["questions"]))
    qid = lambda i, q: str(q.get("id", f"q{i}"))
    evald = {str(r["question_id"]): r for r in json.load(open(
        f"reproduce/results/{a.dataset}__delta+eatr.json"))["arms"]["delta+eatr"]["results"]}

    if a.mode == "sample":
        pool = [qid(i, q) for i, q in enumerate(questions)
                if eval(str(evald.get(qid(i, q), {}).get("context") or "{}")).get("gold_event_positions")]
        ids = random.Random(SEED).sample(pool, min(3 * N_SAMPLE, len(pool)))
        p = os.path.join(HERE, "out", f"conflict_candidates_{a.dataset}.json")
        json.dump({"seed": SEED, "pool": len(pool), "ids": ids}, open(p, "w"), indent=1)
        print(f"{a.dataset}: {len(pool)} questions with the answer in a retrieved event; sampled {len(ids)} -> {p}")
        return

    state = {}
    if a.mode in ("scan", "prepass"):
        async def bmf(prompt, system_prompt=None, history_messages=None, hashing_kv=None, **kw):
            if "# TEXT CHUNKS" in prompt or "## Chunk " in prompt:
                return ""
            raise RuntimeError("cache miss")
        model_func = bmf
    else:
        model_func = RE.bmf
    wd, delta, extra = RE.arm_def(a.dataset, "delta+eatr")
    import torch
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3"),
                             device="mps" if torch.backends.mps.is_available() else "cpu", trust_remote_code=True)
    ef = RE.EmbeddingFunc(embedding_dim=st.get_sentence_embedding_dimension(), max_token_size=8192, model=st)
    fusion = a.mode == "scan" or a.arm in ("natural", "inject+fusion", "inject+fusion_eq")
    gr = GraphRAG(working_dir=wd, embedding_func=ef, best_model_func=model_func, cheap_model_func=model_func,
                  enable_llm_cache=True, best_model_max_token_size=16384, cheap_model_max_token_size=16384,
                  model_path="./models", ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
                  ner_model_name="dslim_bert_base_ner", ner_device="cpu",
                  enable_cast_rag=False, enable_delta_rag=delta, enable_conflict_fusion=fusion,
                  source_credibility={FAKE: 0.5} if a.arm == "inject+fusion" else {}, **extra)
    if a.mode in ("scan", "prepass"):
        async def noop(*x, **k):
            return None
        for kv in [gr.llm_response_cache, gr.full_docs, gr.text_chunks]:
            kv.upsert = noop; kv.index_done_callback = noop
        gr._query_done = noop
        if a.mode == "prepass":
            cands = json.load(open(os.path.join(HERE, "out", f"conflict_candidates_{a.dataset}.json")))
            byid = {qid(i, q): q for i, q in enumerate(questions)}
            keep, tried = [], 0
            for q_id in cands["ids"]:
                if len(keep) >= N_SAMPLE:
                    break
                info = {}; tried += 1
                gr.event_hook = make_hook(golds(byid[q_id].get("answer")), info)
                try:
                    await gr.aquery(byid[q_id]["question"], param=QueryParam(top_k=20, mode="dynamic",
                                                                              response_type="Short Answer or Direct Response"))
                except Exception:
                    continue
                if info.get("injected"):
                    keep.append(q_id)
            p = os.path.join(HERE, "out", f"conflict_sample_{a.dataset}.json")
            json.dump({"seed": SEED, "pool": cands["pool"], "candidates_tried": tried, "ids": keep}, open(p, "w"), indent=1)
            print(f"{a.dataset}: injection possible for {len(keep)} of {tried} candidates -> {p}")
            return
        hits, n = [], 0
        idx = await gr._build_conflict_index()
        for i, q in enumerate(questions):
            try:
                await gr.aquery(q["question"], param=QueryParam(top_k=20, mode="dynamic",
                                                                 response_type="Short Answer or Direct Response"))
            except Exception:
                continue
            n += 1
            if gr.last_conflict_stats.get("conflicts"):
                hits.append({"question_id": qid(i, q), **gr.last_conflict_stats})
        p = os.path.join(HERE, "out", f"conflict_scan_{a.dataset}.json")
        json.dump({"n_scanned": n, "with_conflict": len(hits), "index": idx.stats(), "questions": hits}, open(p, "w"), indent=1)
        print(f"{a.dataset}: scanned {n}, natural conflict in the retrieved events of {len(hits)} -> {p}")
        return

    ids = [x["question_id"] if isinstance(x, dict) else x for x in
           (lambda d: d.get("ids") or d.get("questions"))(json.load(open(a.ids)))]
    arm = f"delta+eatr+{a.arm}"
    out = f"reproduce/results/{a.dataset}__{arm}.json"
    partial = out + ".partial.jsonl"
    done = {json.loads(l)["question_id"]: json.loads(l) for l in open(partial)} if os.path.exists(partial) else {}
    done = {k: v for k, v in done.items() if v.get("status") == "success"}
    get_chunks = gr.text_chunks.get_by_ids
    async def chunks_with_fake(cids, *x, **k):
        r = await get_chunks(cids, *x, **k)
        return [({"content": state.get("fake_chunk", "")} if c == FAKE else v) for c, v in zip(cids, r)]
    gr.text_chunks.get_by_ids = chunks_with_fake
    byid = {qid(i, q): q for i, q in enumerate(questions)}
    results = []
    for k_, q_id in enumerate(ids):
        if q_id in done:
            results.append(done[q_id]); continue
        q = byid[q_id]; state.clear(); RE._last_prompt["text"] = None
        gr.event_hook = make_hook(golds(q.get("answer")), state, strict=True) if a.arm.startswith("inject") else None
        t0 = time.time()
        try:
            ans = await gr.aquery(q["question"], param=QueryParam(top_k=20, mode="dynamic",
                                                                   response_type="Short Answer or Direct Response"))
        except NotInjected:
            rec = {"question_id": q_id, "question": q["question"], "status": "not_injected"}
            with open(partial, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  [{arm}/{a.dataset}] {k_ + 1}/{len(ids)} not injected (skipped, no LLM call)", flush=True)
            continue
        rec = {"question_id": q_id, "question": q["question"], "output": ans, "answer": q.get("answer", ""),
               "status": "success", "context": RE.prompt_stats(RE._last_prompt["text"], q.get("answer", "")),
               "injection": {k: v for k, v in state.items() if k != "fake_chunk"},
               "conflict_stats": gr.last_conflict_stats,
               "fake_in_prompt": bool(state.get("fake_event")) and state["fake_event"][:60] in (RE._last_prompt["text"] or "")}
        results.append(rec)
        with open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [{arm}/{a.dataset}] {len(results)}/{len(ids)} ({time.time()-t0:.0f}s) "
              f"injected={state.get('injected')} fusion={gr.last_conflict_stats}", flush=True)
    m = metrics(results)
    json.dump({"model": RE.MODEL, "n": len(results), "arms": {arm: {"config": {"arm": a.arm, "ids": a.ids}, "metrics": m,
                                                                     "results": results}}}, open(out, "w"), indent=2)
    print(f"{a.dataset} / {arm}: {m}")


if __name__ == "__main__":
    asyncio.run(main())
