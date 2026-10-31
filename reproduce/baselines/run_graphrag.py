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


def workdir(ds):
    prefix = "smoke_" if os.getenv("SMOKE_DOCS") else ""
    return f"reproduce/baselines/work/{prefix}graphrag_{ds}"


_client = None


async def llm(prompt, system_prompt=None, history_messages=None, hashing_kv=None, **kw):
    global _client
    from nano_graphrag._utils import compute_args_hash
    if _client is None:
        _client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=1800, max_retries=0)
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + list(history_messages or []) + [{"role": "user", "content": prompt}]
    if hashing_kv is not None:
        h = compute_args_hash(MODEL, msgs)
        c = await hashing_kv.get_by_id(h)
        if c is not None:
            return c["return"]
    extra = {"response_format": kw["response_format"]} if "response_format" in kw else {}
    extra["max_tokens"] = 4096
    ans = await chat_with_retry(_client, msgs, **extra)
    if hashing_kv is not None:
        await hashing_kv.upsert({h: {"return": ans, "model": MODEL}})
        await hashing_kv.index_done_callback()
    return ans


_st = None
_embed_lock = None


def make_embed():
    from nano_graphrag._utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
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
    return embed


def make_rag(ds):
    from nano_graphrag import GraphRAG
    os.makedirs(workdir(ds), exist_ok=True)
    return GraphRAG(
        working_dir=workdir(ds),
        best_model_func=llm, cheap_model_func=llm,
        best_model_max_async=4, cheap_model_max_async=4,
        best_model_max_token_size=16384, cheap_model_max_token_size=16384,
        embedding_func=make_embed(), embedding_func_max_async=4,
        chunk_token_size=1200, chunk_overlap_token_size=64, tiktoken_model_name="gpt-4o",
        enable_llm_cache=True, enable_local=True, enable_naive_rag=False,
    )


def index(ds):
    wd = workdir(ds)
    if os.path.isdir(wd) and not os.path.exists(os.path.join(wd, "INDEX_DONE")):
        for f in os.listdir(wd):
            if f != "kv_store_llm_response_cache.json":
                p = os.path.join(wd, f)
                os.remove(p) if os.path.isfile(p) else None
    rag = make_rag(ds)
    corpus = json.load(open(DATASETS[ds]["corpus"]))
    docs = [f"Title: {o['title']}\nDocument ID: {i}\n\n{o['context']}" for i, o in enumerate(corpus)]
    if os.getenv("SMOKE_DOCS"):
        docs = docs[: int(os.getenv("SMOKE_DOCS"))]
    t0 = time.time()
    rag.insert(docs)
    open(os.path.join(workdir(ds), "INDEX_DONE"), "w").write(time.ctime())
    print(f"graphrag index {ds} done in {time.time()-t0:.0f}s")


def query(ds, n):
    from nano_graphrag import QueryParam
    assert os.path.exists(os.path.join(workdir(ds), "INDEX_DONE")), "index first"
    rag = make_rag(ds)
    qs = json.load(open(DATASETS[ds]["questions"]))
    if n:
        qs = qs[:n]
    out = f"reproduce/baselines/results/graphrag-local_{ds}.json"
    partial = out + ".partial.jsonl"
    done = {}
    if os.path.exists(partial):
        for line in open(partial):
            r = json.loads(line); done[str(r["question_id"])] = r
    results = []
    for i, q in enumerate(qs):
        qid = str(q.get("id", f"q{i}"))
        if qid in done:
            results.append(done[qid]); continue
        t0 = time.time()
        ans = rag.query(q["question"], param=QueryParam(mode="local", top_k=TOP_K, response_type=RESPONSE_TYPE))
        rec = {"question_id": qid, "question": q["question"], "output": str(ans),
               "answer": q.get("answer", ""), "status": "success"}
        results.append(rec)
        with open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [graphrag-local/{ds}] {len(results)}/{len(qs)} ({time.time()-t0:.0f}s)", flush=True)
    m = metrics(results)
    json.dump({"model": MODEL, "n": len(results), "arms": {"graphrag-local": {
        "config": {"mode": "local", "top_k": TOP_K, "version": "nano-graphrag 0.0.8.2"},
        "metrics": m, "results": results}}}, open(out, "w"), indent=2)
    print(f"GraphRAG-local {ds}: {m}\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--stage", required=True, choices=["index", "query"])
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    index(a.dataset) if a.stage == "index" else query(a.dataset, a.n)
