from __future__ import annotations
import os, sys, json, time, asyncio, argparse, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from graphrag._utils import truncate_list_by_token_size
from baselines_common import DATASETS, metrics, LLMCache, MODEL, BASE_URL, chat_with_retry

NAIVE_PROMPT = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""
RESPONSE_TYPE = "Short Answer or Direct Response"
TOP_K = 20
MAX_CTX_TOKENS = 12000


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=0, help="0 = all questions")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    out_path = args.out or f"reproduce/baselines/results/naiverag_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = out_path + ".partial.jsonl"

    questions = json.load(open(cfg["questions"]))
    if args.n:
        questions = questions[: args.n]
    chunks = json.load(open(os.path.join(cfg["dir"], "kv_store_text_chunks.json")))
    ids = list(chunks)
    texts = [chunks[i]["content"] for i in ids]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st = SentenceTransformer(os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3"), device=device)
    emb_cache = f"reproduce/baselines/cache/naive_chunks_{args.dataset}.npy"
    os.makedirs(os.path.dirname(emb_cache), exist_ok=True)
    if os.path.exists(emb_cache):
        E = np.load(emb_cache)
    else:
        E = st.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.save(emb_cache, E)
    assert E.shape[0] == len(ids)

    done = {}
    if os.path.exists(partial):
        for line in open(partial):
            r = json.loads(line); done[str(r["question_id"])] = r
    cache = LLMCache(f"reproduce/baselines/cache/llm_naiverag_{args.dataset}.json")
    client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=900, max_retries=0)

    results = []
    for i, q in enumerate(questions):
        qid = str(q.get("id", f"q{i}"))
        if qid in done:
            results.append(done[qid]); continue
        t0 = time.time()
        qv = st.encode([q["question"]], normalize_embeddings=True)[0]
        top = np.argsort(-(E @ qv))[:TOP_K]
        ctx_chunks = truncate_list_by_token_size(
            [chunks[ids[j]] for j in top], key=lambda x: x["content"], max_token_size=MAX_CTX_TOKENS)
        section = "--New Chunk--\n".join(c["content"] for c in ctx_chunks)
        msgs = [{"role": "system", "content": NAIVE_PROMPT.format(content_data=section, response_type=RESPONSE_TYPE)},
                {"role": "user", "content": q["question"]}]
        ans = cache.get(msgs)
        if ans is None:
            ans = await chat_with_retry(client, msgs)
            cache.put(msgs, ans)
        st_ = "success"
        rec = {"question_id": qid, "question": q["question"], "output": ans,
               "answer": q.get("answer", ""), "status": st_,
               "n_ctx_chunks": len(ctx_chunks), "retrieved_chunk_ids": [ids[j] for j in top]}
        results.append(rec)
        with open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [naiverag/{args.dataset}] {len(results)}/{len(questions)} ({time.time()-t0:.0f}s) {st_}", flush=True)

    m = metrics(results)
    json.dump({"model": MODEL, "n": len(results), "arms": {"naiverag": {
        "config": {"top_k": TOP_K, "max_ctx_tokens": MAX_CTX_TOKENS, "chunks_from": cfg["dir"]},
        "metrics": m, "results": results}}}, open(out_path, "w"), indent=2)
    print(f"NaiveRAG {args.dataset}: {m}\nsaved -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
