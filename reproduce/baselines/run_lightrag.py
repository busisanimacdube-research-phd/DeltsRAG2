from __future__ import annotations
import os, sys, json, time, asyncio, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(ROOT)

import numpy as np
from openai import AsyncOpenAI
from baselines_common import DATASETS, metrics, MODEL, BASE_URL, chat_with_retry

RESPONSE_TYPE = "Short Answer or Direct Response"
TOP_K = 20
MAX_TOTAL_TOKENS = 16384


def workdir(ds):
    prefix = "smoke_" if os.getenv("SMOKE_DOCS") else ""
    return f"reproduce/baselines/work/{prefix}lightrag_{ds}"


_client = None


async def llm_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kw):
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=1800, max_retries=0)
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + list(history_messages or []) + [{"role": "user", "content": prompt}]
    extra = {"max_tokens": 4096}
    if keyword_extraction:
        extra["response_format"] = {"type": "json_object"}
    return await chat_with_retry(_client, msgs, **extra)


_st = None
_embed_lock = None


async def embed(texts):
    global _st
    if _st is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3"),
                                  device="mps" if torch.backends.mps.is_available() else "cpu")
    global _embed_lock
    if _embed_lock is None:
        _embed_lock = asyncio.Lock()
    async with _embed_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _st.encode(
            list(texts), batch_size=16, normalize_embeddings=True, show_progress_bar=False))


async def make_rag(ds):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    os.makedirs(workdir(ds), exist_ok=True)
    rag = LightRAG(
        working_dir=workdir(ds),
        llm_model_func=llm_func, llm_model_name=MODEL, llm_model_max_async=4,
        default_llm_timeout=1800,
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed),
        embedding_func_max_async=4,
        chunk_token_size=1200, chunk_overlap_token_size=64, tiktoken_model_name="gpt-4o",
        enable_llm_cache=True, max_parallel_insert=int(os.getenv("LIGHTRAG_PARALLEL_INSERT", "1")),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def index(ds):
    rag = await make_rag(ds)
    corpus = json.load(open(DATASETS[ds]["corpus"]))
    docs = [f"Title: {o['title']}\nDocument ID: {i}\n\n{o['context']}" for i, o in enumerate(corpus)]
    if os.getenv("SMOKE_DOCS"):
        docs = docs[: int(os.getenv("SMOKE_DOCS"))]
    ids = [f"doc-{i}" for i in range(len(docs))]
    t0 = time.time()
    for s in range(0, len(docs), 10):
        await rag.ainsert(docs[s:s + 10], ids=ids[s:s + 10])
        print(f"  [lightrag-index/{ds}] {min(s+10, len(docs))}/{len(docs)} docs ({time.time()-t0:.0f}s)", flush=True)
    status_file = os.path.join(workdir(ds), "kv_store_doc_status.json")
    for attempt in range(3):
        st = json.load(open(status_file)) if os.path.exists(status_file) else {}
        bad = [k for k in ids if st.get(k, {}).get("status") != "processed"]
        if not bad:
            break
        print(f"  retrying {len(bad)} unprocessed docs (attempt {attempt+1})", flush=True)
        for k in bad:
            await rag.adelete_by_doc_id(k)
        await rag.ainsert([docs[ids.index(k)] for k in bad], ids=bad)
    await rag.finalize_storages()
    st = json.load(open(status_file))
    bad = [k for k in ids if st.get(k, {}).get("status") != "processed"]
    if bad:
        raise SystemExit(f"{len(bad)} documents not processed: {bad[:10]}")
    open(os.path.join(workdir(ds), "INDEX_DONE"), "w").write(time.ctime())


async def query(ds, mode, n):
    from lightrag import QueryParam
    assert os.path.exists(os.path.join(workdir(ds), "INDEX_DONE")), "index first"
    rag = await make_rag(ds)
    qs = json.load(open(DATASETS[ds]["questions"]))
    if n:
        qs = qs[:n]
    out = f"reproduce/baselines/results/lightrag-{mode}_{ds}.json"
    partial = out + ".partial.jsonl"
    done = {}
    if os.path.exists(partial):
        for line in open(partial):
            r = json.loads(line); done[str(r["question_id"])] = r
    sem = asyncio.Semaphore(int(os.getenv("QUERY_CONCURRENCY", "1")))
    n_done = sum(1 for i, q in enumerate(qs) if str(q.get("id", f"q{i}")) in done)

    async def answer(i, q):
        nonlocal n_done
        qid = str(q.get("id", f"q{i}"))
        if qid in done:
            return done[qid]
        async with sem:
            t0 = time.time()
            p = QueryParam(mode=mode, top_k=TOP_K, chunk_top_k=TOP_K, response_type=RESPONSE_TYPE,
                           max_total_tokens=MAX_TOTAL_TOKENS, max_entity_tokens=4000, max_relation_tokens=4000,
                           enable_rerank=False)
            ans = await rag.aquery(q["question"], param=p)
        rec = {"question_id": qid, "question": q["question"], "output": str(ans),
               "answer": q.get("answer", ""), "status": "success"}
        with open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_done += 1
        print(f"  [lightrag-{mode}/{ds}] {n_done}/{len(qs)} ({time.time()-t0:.0f}s)", flush=True)
        return rec

    results = list(await asyncio.gather(*(answer(i, q) for i, q in enumerate(qs))))
    m = metrics(results)
    json.dump({"model": MODEL, "n": len(results), "arms": {f"lightrag-{mode}": {
        "config": {"mode": mode, "top_k": TOP_K, "max_total_tokens": MAX_TOTAL_TOKENS, "version": "lightrag-hku 1.5.7"},
        "metrics": m, "results": results}}}, open(out, "w"), indent=2)
    print(f"LightRAG-{mode} {ds}: {m}\nsaved -> {out}")
    await rag.finalize_storages()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--stage", required=True, choices=["index", "query"])
    ap.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "mix", "naive"])
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(index(a.dataset) if a.stage == "index" else query(a.dataset, a.mode, a.n))
