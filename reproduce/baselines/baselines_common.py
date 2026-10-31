from __future__ import annotations
import os, json, hashlib, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL = os.getenv("QWEN_BEST", "qwen2.5:14b")
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:11434/v1")

DATASETS = {
    "complextr": {"questions": "datasets/ComplexTR/Question.json",
                  "corpus": "datasets/ComplexTR/Corpus.json",
                  "dir": "reproduce/baseline_complextr_dir"},
    "tempreason_l3": {"questions": "datasets/TempReason_L3_sample500/Question.json",
                      "corpus": "datasets/TempReason/Corpus.json",
                      "dir": "reproduce/tempreason_full_dir"},
    "timeqa": {"questions": "datasets/TimeQA/Question.json",
               "corpus": "datasets/TimeQA/Corpus.json",
               "dir": "reproduce/timeqa_shared_dir"},
}


def metrics(results):
    from graphrag.evaluate import Evaluator
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
    return {"n": len(accs), "accuracy": sum(accs) / n, "recall": sum(recs) / n, "f1": sum(f1s) / n}


async def chat_with_retry(client, msgs, attempts=50, **kw):
    import asyncio
    for k in range(attempts):
        try:
            r = await client.chat.completions.create(model=MODEL, messages=msgs, **kw)
            return r.choices[0].message.content
        except Exception as e:
            wait = min(30 * (k + 1), 300)
            print(f"    LLM call failed ({type(e).__name__}: {e}); retry {k+1}/{attempts} in {wait}s", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError("LLM unavailable after retries")


class LLMCache:

    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    @staticmethod
    def _key(msgs):
        return hashlib.md5((MODEL + json.dumps(msgs, sort_keys=True)).encode()).hexdigest()

    def get(self, msgs):
        return self.d.get(self._key(msgs))

    def put(self, msgs, ans):
        self.d[self._key(msgs)] = ans
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"))
        os.replace(tmp, self.path)
