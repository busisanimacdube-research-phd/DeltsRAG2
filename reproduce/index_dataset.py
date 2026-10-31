from __future__ import annotations
import os, sys, json, time, asyncio, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import torch
from graphrag import GraphRAG, QueryParam
from graphrag._utils import compute_args_hash, logger

logging.basicConfig(level=logging.INFO)

DATASET = sys.argv[1] if len(sys.argv) > 1 else "TimeQA"
WORK_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else f"{DATASET.lower()}_shared_dir")
WORK_DIR.mkdir(exist_ok=True)
CORPUS_FILE = Path(f"datasets/{DATASET}/Corpus.json")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("QWEN_BEST", "qwen2.5:14b")
LOCAL_BGE_PATH = os.getenv("LOCAL_BGE_PATH", "BAAI/bge-m3")


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model: SentenceTransformer
    async def __call__(self, texts):
        if isinstance(texts, str): texts = [texts]
        loop = asyncio.get_event_loop()
        enc = lambda: self.model.encode(texts, batch_size=32, convert_to_numpy=True,
                                        show_progress_bar=False, normalize_embeddings=True)
        return await loop.run_in_executor(None, enc) if loop.is_running() else enc()
    def __getstate__(self):
        s = self.__dict__.copy(); s['model'] = None; return s
    def __setstate__(self, s): self.__dict__.update(s)


async def _llm(prompt, *, model, system_prompt=None, history_messages=None, hashing_kv=None, **kw):
    hm = history_messages or []
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + hm + [{"role": "user", "content": prompt}]
    if hashing_kv is not None:
        h = compute_args_hash(model, msgs)
        c = await hashing_kv.get_by_id(h)
        if c is not None: return c["return"]
    cl = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)
    kw.pop("response_format", None)
    r = await cl.chat.completions.create(model=model, messages=msgs, **kw)
    a = r.choices[0].message.content
    if hashing_kv is not None:
        await hashing_kv.upsert({h: {"return": a, "model": model}})
        await hashing_kv.index_done_callback()
    return a


async def bmf(prompt, system_prompt=None, history_messages=None, **kw):
    return await _llm(prompt, model=MODEL, system_prompt=system_prompt, history_messages=history_messages, **kw)


def load_json(fp):
    with open(fp) as f:
        try: return json.load(f)
        except json.JSONDecodeError:
            f.seek(0); return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.device_count() > 0 else "cpu")
    st = SentenceTransformer(LOCAL_BGE_PATH, device=device, trust_remote_code=True)
    ef = EmbeddingFunc(embedding_dim=st.get_sentence_embedding_dimension(), max_token_size=8192, model=st)
    mr = ef.model; ef.model = None
    gr = GraphRAG(
        working_dir=str(WORK_DIR), embedding_func=ef, best_model_func=bmf, cheap_model_func=bmf,
        enable_llm_cache=True, best_model_max_token_size=16384, cheap_model_max_token_size=16384,
        model_path="./models", ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        ner_model_name="dslim_bert_base_ner", ner_device="cpu",
        enable_cast_rag=False, enable_delta_rag=False,
    )
    ef.model = mr

    corpus = load_json(CORPUS_FILE)
    docs = [f"Title: {o['title']}\nDocument ID: {i}\n\n{o['context']}" for i, o in enumerate(corpus)]
    logger.info(f"[{DATASET}] Indexing {len(docs)} documents into {WORK_DIR} ...")
    t0 = time.time()
    gr.insert(docs)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(gr._insert_done())
    logger.info(f"[{DATASET}] Indexing complete in {(time.time()-t0)/60:.1f} min -> {WORK_DIR}")
