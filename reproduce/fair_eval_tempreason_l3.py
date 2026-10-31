from __future__ import annotations
import os, sys, json, time, asyncio, argparse, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import torch

from graphrag import GraphRAG, QueryParam
from graphrag._utils import compute_args_hash
from graphrag.evaluate import Evaluator

logging.basicConfig(level=logging.WARNING)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("QWEN_BEST", "qwen2.5:14b")
LOCAL_BGE_PATH = os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3")
QUESTIONS_FILE = "datasets/TempReason_L3_sample500/Question.json"


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

    def __getstate__(self):
        s = self.__dict__.copy(); s['model'] = None; return s
    def __setstate__(self, s):
        self.__dict__.update(s)


async def _llm(prompt, *, system_prompt=None, history_messages=None, hashing_kv=None, **kw):
    hm = history_messages or []
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + hm + [{"role": "user", "content": prompt}]
    if hashing_kv:
        h = compute_args_hash(MODEL, msgs)
        c = await hashing_kv.get_by_id(h)
        if c:
            return c["return"]
    cl = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)
    kw.pop("response_format", None)
    r = await cl.chat.completions.create(model=MODEL, messages=msgs, **kw)
    a = r.choices[0].message.content
    if hashing_kv:
        await hashing_kv.upsert({h: {"return": a, "model": MODEL}})
        await hashing_kv.index_done_callback()
    return a


async def bmf(prompt, system_prompt=None, history_messages=None, **kw):
    return await _llm(prompt, system_prompt=system_prompt, history_messages=history_messages, **kw)


def make_gr(working_dir, ef, delta, eatr=False):
    return GraphRAG(
        working_dir=working_dir, embedding_func=ef,
        best_model_func=bmf, cheap_model_func=bmf, enable_llm_cache=True,
        best_model_max_token_size=16384, cheap_model_max_token_size=16384,
        model_path="./models", ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        ner_model_name="dslim_bert_base_ner", ner_device="cpu",
        enable_cast_rag=False, enable_delta_rag=delta,
        enable_entity_anchor=eatr,
    )


async def run_system(name, gr, questions, concurrency, checkpoint_path=None):
    done_results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done_results[r["question_id"]] = r
                except json.JSONDecodeError:
                    continue
        print(f"  [{name}] resuming: {len(done_results)} already done", flush=True)

    todo = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        if qid not in done_results:
            todo.append((q, i, qid))

    total = len(questions)
    done = [len(done_results)]
    sem = asyncio.Semaphore(concurrency)
    ckpt_lock = asyncio.Lock()
    ckpt_f = open(checkpoint_path, "a") if checkpoint_path else None

    async def one(q, i, qid):
        async with sem:
            param = QueryParam(top_k=20, mode="dynamic", response_type="Short Answer or Direct Response")
            t0 = time.time()
            try:
                ans = await gr.aquery(q["question"], param=param)
                st = "success"
            except Exception as e:
                ans, st = f"[ERROR {e}]", "error"
            rec = {"question_id": qid, "question": q["question"],
                   "output": ans, "answer": q.get("answer", ""), "status": st}
            done[0] += 1
            print(f"  [{name}] {done[0]}/{total} ({time.time()-t0:.0f}s) {st}", flush=True)
            if ckpt_f:
                async with ckpt_lock:
                    ckpt_f.write(json.dumps(rec) + "\n")
                    ckpt_f.flush()
            return rec

    try:
        new_results = await asyncio.gather(*[one(q, i, qid) for (q, i, qid) in todo])
    finally:
        if ckpt_f:
            ckpt_f.close()

    all_results = dict(done_results)
    for r in new_results:
        all_results[r["question_id"]] = r
    ordered = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        if qid in all_results:
            ordered.append(all_results[qid])
    return ordered


def metrics(results):
    ev = Evaluator()
    accs, recs, f1s = [], [], []
    for r in results:
        pred = str(r["output"] or "").replace("|", "\n")
        pred = " ".join(pred.split("\n"))
        ans_list = [a for a in ev.parse_multiple_answers(str(r["answer"] or "")) if a.strip()]
        if len(ans_list) <= 1:
            g = ans_list[0].strip() if ans_list else ""
            acc = ev.eval_accuracy(pred, g)
            f1, _, rec = ev.f1_score(pred, g)
        else:
            cov = sum(1 for g in ans_list if ev.eval_accuracy(pred, g.strip()) == 1)
            acc = 1 if cov == len(ans_list) else 0
            rec = cov / len(ans_list); f1 = rec
        accs.append(acc); recs.append(rec); f1s.append(f1)
    n = max(len(accs), 1)
    return {"n": len(accs), "accuracy": sum(accs)/n, "recall": sum(recs)/n, "f1": sum(f1s)/n}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--qids", type=str, default="", help="comma-separated question ids to restrict to")
    ap.add_argument("--arms", type=str, default="baseline,delta+eatr",
                    help="comma list from: baseline, base+eatr, delta, delta+eatr")
    ap.add_argument("--out", type=str, default="fair_eval_tempreason_l3_500.json")
    args = ap.parse_args()

    ARM_DEFS = {
        "baseline":   ("tempreason_full_dir",   False, False),
        "base+eatr":  ("tempreason_full_dir",   False, True),
        "delta":      ("tempreason_full_delta", True,  False),
        "delta+eatr": ("tempreason_full_delta", True,  True),
    }
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    questions = json.load(open(QUESTIONS_FILE))
    if args.qids:
        want = set(args.qids.split(","))
        questions = [q for q in questions if str(q.get("id")) in want]
    else:
        questions = questions[: args.n]
    print(f"Loaded {len(questions)} questions (same set for all arms). Model={MODEL}\nArms: {arms}\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st = SentenceTransformer(LOCAL_BGE_PATH, device=device, trust_remote_code=True)
    ef = EmbeddingFunc(embedding_dim=st.get_sentence_embedding_dimension(), max_token_size=8192, model=st)

    out = {"model": MODEL, "n": len(questions), "arms": {}}
    arm_metrics = {}
    for arm in arms:
        wd, delta, eatr = ARM_DEFS[arm]
        print(f"\n=== {arm}  (dir={wd}, delta={delta}, eatr={eatr}) ===", flush=True)
        gr = make_gr(wd, ef, delta=delta, eatr=eatr)
        ckpt = f"{args.out}.{arm}.partial.jsonl"
        res = await run_system(arm, gr, questions, args.concurrency, checkpoint_path=ckpt)
        out["arms"][arm] = {"config": {"dir": wd, "delta": delta, "eatr": eatr},
                            "metrics": metrics(res), "results": res}
        arm_metrics[arm] = metrics(res)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    ref = arms[0]
    print("\n" + "=" * 64)
    print(f"COMPARISON on {len(questions)} TempReason-L3 questions (model={MODEL})")
    print("=" * 64)
    hdr = f"{'metric':<12}" + "".join(f"{a:>14}" for a in arms)
    print(hdr)
    for k in ["accuracy", "recall", "f1"]:
        row = f"{k:<12}" + "".join(f"{arm_metrics[a][k]*100:>13.1f}%" for a in arms)
        print(row)
    print("-" * 64)
    print(f"(Δ vs {ref}):")
    for a in arms[1:]:
        ds = "  ".join(f"{k} {(arm_metrics[a][k]-arm_metrics[ref][k])*100:+.1f}%" for k in ["accuracy","recall"])
        print(f"  {a}: {ds}")
    print("=" * 64)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
