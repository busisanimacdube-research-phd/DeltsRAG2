from __future__ import annotations
import os, sys, json, time, asyncio, argparse, logging
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import torch

from graphrag import GraphRAG, QueryParam
from graphrag._utils import compute_args_hash
sys.path.insert(0, os.path.join(ROOT, "reproduce", "baselines"))
from baselines_common import metrics

logging.basicConfig(level=logging.WARNING)

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("QWEN_BEST", "qwen2.5:14b")

DATASETS = {
    "complextr": {"questions": "datasets/ComplexTR/Question.json",
                  "base": "reproduce/baseline_complextr_dir",
                  "delta": "reproduce/delta_complextr_fair"},
    "tempreason_l3": {"questions": "datasets/TempReason_L3_sample500/Question.json",
                      "base": "reproduce/tempreason_full_dir",
                      "delta": "reproduce/tempreason_full_delta"},
    "timeqa": {"questions": "datasets/TimeQA/Question.json",
               "base": "reproduce/timeqa_shared_dir",
               "delta": "reproduce/delta_timeqa_fair"},
}


def arm_def(ds, arm):
    d = DATASETS[ds]
    abl = f"reproduce/ablation_dirs/{ds}"
    table = {
        "baseline":        (d["base"],  False, {}),
        "base+eatr":       (d["base"],  False, {"enable_entity_anchor": True}),
        "delta":           (d["delta"], True,  {}),
        "delta+eatr":      (d["delta"], True,  {"enable_entity_anchor": True}),
        "delta+eatr-datp": (f"{abl}_noDATP", True, {"enable_entity_anchor": True}),
        "delta+eatr-cpkb": (f"{abl}_noCPKB", True, {"enable_entity_anchor": True}),
        "delta+eatr-ahkr": (d["delta"], True, {"enable_entity_anchor": True, "enable_ahkr": False}),
        "delta+eatr_commit": (d["delta"], True, {"enable_entity_anchor": True, "answer_commit": True}),
        "delta+eatr_v1match": (d["delta"], True, {"enable_entity_anchor": True,
                                                   "entity_anchor_matching": "v1"}),
    }
    if arm.startswith("delta+eatr_M"):
        m = int(arm.split("_M")[1])
        return d["delta"], True, {"enable_entity_anchor": True, "entity_anchor_top_m": m}
    return table[arm]


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model: SentenceTransformer

    async def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        enc = lambda: self.model.encode(texts, batch_size=32, convert_to_numpy=True,
                                        show_progress_bar=False, normalize_embeddings=True)
        return await loop.run_in_executor(None, enc)


_client = None
_last_prompt = {"text": None}


def prompt_stats(prompt, gold):
    import re as _re
    from graphrag._utils import encode_string_by_tiktoken
    from graphrag.evaluate import Evaluator
    if not prompt:
        return {}
    ev = Evaluator()
    golds = [g for g in ev.parse_multiple_answers(str(gold or "")) if g.strip()]
    parts = _re.split(r"\n## Chunk (\d+)\n", prompt)
    chunks = [parts[i + 1] for i in range(1, len(parts) - 1, 2)]
    events_part = prompt.split("# TEXT CHUNKS")[0] if "# TEXT CHUNKS" in prompt else ""
    if "\n# EVENTS" in events_part:
        events_part = events_part.split("\n# EVENTS", 1)[1]
    ev_lines = _re.findall(r"Event #\d+[^\n]*", events_part)
    def has(t):
        return bool(golds) and any(ev.eval_accuracy(t, g.strip()) == 1 for g in golds)
    return {
        "prompt_tokens": len(encode_string_by_tiktoken(prompt)),
        "n_chunks": len(chunks),
        "n_events": len(ev_lines),
        "gold_chunk_positions": [i + 1 for i, c in enumerate(chunks) if has(c)],
        "gold_event_positions": [i + 1 for i, e in enumerate(ev_lines) if has(e)],
        "gold_anywhere": has(prompt),
    }


async def _llm(prompt, *, system_prompt=None, history_messages=None, hashing_kv=None, **kw):
    global _client
    hm = history_messages or []
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + hm + [{"role": "user", "content": prompt}]
    _last_prompt["text"] = prompt
    if hashing_kv:
        h = compute_args_hash(MODEL, msgs)
        c = await hashing_kv.get_by_id(h)
        if c:
            return c["return"]
    if _client is None:
        _client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=900, max_retries=0)
    kw.pop("response_format", None)
    for k in range(60):
        try:
            r = await _client.chat.completions.create(model=MODEL, messages=msgs, **kw)
            a = r.choices[0].message.content
            break
        except Exception as e:
            wait = min(30 * (k + 1), 300)
            print(f"    LLM call failed ({type(e).__name__}); retry {k+1} in {wait}s", flush=True)
            await asyncio.sleep(wait)
    else:
        raise RuntimeError("LLM unavailable")
    if hashing_kv:
        await hashing_kv.upsert({h: {"return": a, "model": MODEL}})
        await hashing_kv.index_done_callback()
    return a


async def bmf(prompt, system_prompt=None, history_messages=None, **kw):
    return await _llm(prompt, system_prompt=system_prompt, history_messages=history_messages, **kw)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wd, delta, extra = arm_def(args.dataset, args.arm)
    out = args.out or f"reproduce/results/{args.dataset}__{args.arm}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    partial = out + ".partial.jsonl"

    questions = json.load(open(DATASETS[args.dataset]["questions"]))
    if args.n:
        questions = questions[: args.n]
    done = {}
    if os.path.exists(partial):
        for line in open(partial):
            r = json.loads(line)
            if r.get("status") == "success":
                done[str(r["question_id"])] = r

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st = SentenceTransformer(os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3"), device=device, trust_remote_code=True)
    ef = EmbeddingFunc(embedding_dim=st.get_sentence_embedding_dimension(), max_token_size=8192, model=st)
    gr = GraphRAG(
        working_dir=wd, embedding_func=ef,
        best_model_func=bmf, cheap_model_func=bmf, enable_llm_cache=True,
        best_model_max_token_size=16384, cheap_model_max_token_size=16384,
        model_path="./models", ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        ner_model_name="dslim_bert_base_ner", ner_device="cpu",
        enable_cast_rag=False, enable_delta_rag=delta, **extra,
    )
    print(f"{args.dataset} / {args.arm}: dir={wd} delta={delta} extra={extra} "
          f"n={len(questions)} resumed={len(done)}", flush=True)

    results = []
    for i, q in enumerate(questions):
        qid = str(q.get("id", f"q{i}"))
        if qid in done:
            results.append(done[qid]); continue
        param = QueryParam(top_k=20, mode="dynamic", response_type="Short Answer or Direct Response")
        t0 = time.time()
        ans = None
        _last_prompt["text"] = None
        for attempt in range(5):
            try:
                ans = await gr.aquery(q["question"], param=param)
                break
            except Exception as e:
                print(f"    query error ({type(e).__name__}: {e}); retry {attempt+1}", flush=True)
                await asyncio.sleep(60)
        status = "success" if ans is not None else "error"
        rec = {"question_id": qid, "question": q["question"], "output": ans if ans is not None else "",
               "answer": q.get("answer", ""), "status": status,
               "context": prompt_stats(_last_prompt["text"], q.get("answer", ""))}
        results.append(rec)
        with open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [{args.arm}/{args.dataset}] {len(results)}/{len(questions)} ({time.time()-t0:.0f}s) {status}", flush=True)

    m = metrics(results)
    json.dump({"model": MODEL, "n": len(results), "arms": {args.arm: {
        "config": {"dir": wd, "delta": delta, **extra}, "metrics": m, "results": results}}},
        open(out, "w"), indent=2)
    print(f"{args.dataset} / {args.arm}: {m}\nsaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
