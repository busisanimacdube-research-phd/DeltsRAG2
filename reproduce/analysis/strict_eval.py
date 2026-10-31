from __future__ import annotations
import os, sys, json, re, hashlib, asyncio, string, collections
from openai import AsyncOpenAI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reproduce", "baselines"))
from baselines_common import metrics, chat_with_retry
import baselines_common

JUDGE = os.getenv("JUDGE_MODEL", "qwen2.5:32b")
baselines_common.MODEL = JUDGE
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:11434/v1")
CACHE = "reproduce/analysis/out/judge_cache.json"

PROMPT = """You are grading an answer to a temporal question-answering task.

Question: {question}
Gold answer(s): {gold}
Model response:
<<<
{response}
>>>

Do two things.
1. final_answer: copy, verbatim and as briefly as possible, the answer that the response finally COMMITS to (a name, entity, date or short phrase). If the response gives no answer or says the information is not available, write "NONE".
2. verdict: "CORRECT" if the committed answer is semantically equivalent to the gold answer (for several gold answers, all of them must be given). Mark "INCORRECT" if the committed answer differs, if the response hedges between several candidate answers, lists several mutually exclusive answers, or only mentions the gold answer while committing to something else.

Reply with JSON only: {{"final_answer": "...", "verdict": "CORRECT" or "INCORRECT"}}"""


def norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def golds_of(ans):
    from graphrag.evaluate import Evaluator
    return [g.strip() for g in Evaluator().parse_multiple_answers(str(ans or "")) if g.strip()]


async def main(paths):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=3600, max_retries=0)
    summary = {}
    for spec in paths:
        path, _, only = spec.partition("::")
        data = json.load(open(path))
        for arm, v in data["arms"].items():
            if only and arm not in only.split(","):
                continue
            rows = []
            for r in v["results"]:
                golds = golds_of(r["answer"])
                prompt = PROMPT.format(question=r["question"], gold=" | ".join(golds), response=str(r["output"])[:12000])
                key = hashlib.md5((JUDGE + prompt).encode()).hexdigest()
                if key not in cache:
                    raw = await chat_with_retry(client, [{"role": "user", "content": prompt}],
                                                temperature=0, response_format={"type": "json_object"})
                    cache[key] = raw
                    if len(cache) % 20 == 0:
                        json.dump(cache, open(CACHE, "w"))
                    print(f"  judged {arm} {len(rows)+1}/{len(v['results'])}", flush=True)
                try:
                    j = json.loads(re.search(r"\{.*\}", cache[key], re.S).group(0))
                except Exception:
                    j = {"final_answer": "NONE", "verdict": "INCORRECT"}
                fa = str(j.get("final_answer", "NONE"))
                parts = [p for p in re.split(r"\s*(?:,|;| and |\|)\s*", fa) if p]
                em = int(bool(golds) and all(any(norm(p) == norm(g) for p in parts + [fa]) for g in golds))
                judge = int(str(j.get("verdict", "")).upper().startswith("CORRECT"))
                rows.append({"question_id": r["question_id"], "final_answer": fa, "em": em, "judge": judge,
                             "inclusion": metrics([r])["accuracy"]})
            n = max(len(rows), 1)
            summary[f"{os.path.basename(path)}::{arm}"] = {
                "n": len(rows),
                "inclusion_acc": sum(x["inclusion"] for x in rows) / n,
                "exact_match": sum(x["em"] for x in rows) / n,
                "judge_acc": sum(x["judge"] for x in rows) / n,
                "inclusion_but_judge_incorrect": sum(1 for x in rows if x["inclusion"] and not x["judge"]) / n,
                "rows": rows,
            }
            s = summary[f"{os.path.basename(path)}::{arm}"]
            print(f"{path} {arm}: incl={s['inclusion_acc']:.3f} EM={s['exact_match']:.3f} judge={s['judge_acc']:.3f} "
                  f"incl-but-judge-wrong={s['inclusion_but_judge_incorrect']:.3f}", flush=True)
        json.dump(cache, open(CACHE, "w"))
    os.makedirs("reproduce/analysis/out", exist_ok=True)
    json.dump(summary, open("reproduce/analysis/out/strict_eval.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
