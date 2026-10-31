import os, re, sys, json, random, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE))
import strict_eval as SE
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

OUTDIR = os.path.join(ROOT, "reproduce", "analysis", "annotation")
SHEET = os.path.join(OUTDIR, "judge_validation_sheet.xlsx")
KEY = os.path.join(OUTDIR, "judge_validation_key.json")

FILES = {
    "ComplexTR": ["reproduce/results/complextr__delta+eatr.json", "reproduce/baselines/results/naiverag_complextr.json",
                  "reproduce/baselines/results/lightrag-hybrid_complextr.json", "reproduce/baselines/results/graphrag-local_complextr.json",
                  "reproduce/fair_eval_329_baseline.json::baseline"],
    "TempReason-L3": ["reproduce/results/tempreason_l3__delta+eatr.json", "reproduce/baselines/results/naiverag_tempreason_l3.json",
                      "reproduce/fair_eval_tempreason_l3_500.json::baseline"],
    "TimeQA": ["reproduce/results/timeqa__delta+eatr.json", "reproduce/baselines/results/naiverag_timeqa.json"],
}


def items():
    from baselines_common import metrics
    cache = json.load(open(SE.CACHE))
    for ds, specs in FILES.items():
        for spec in specs:
            path, _, only = spec.partition("::")
            for arm, v in json.load(open(path))["arms"].items():
                if only and arm != only:
                    continue
                for r in v["results"]:
                    golds = SE.golds_of(r["answer"])
                    prompt = SE.PROMPT.format(question=r["question"], gold=" | ".join(golds), response=str(r["output"])[:12000])
                    key = hashlib.md5((SE.JUDGE + prompt).encode()).hexdigest()
                    if key not in cache:
                        continue
                    try:
                        j = json.loads(re.search(r"\{.*\}", cache[key], re.S).group(0))
                    except Exception:
                        j = {"final_answer": "NONE", "verdict": "INCORRECT"}
                    yield dict(dataset=ds, system=arm, qid=str(r["question_id"]), question=r["question"], gold=" | ".join(golds),
                               response=str(r["output"]), judge=int(str(j.get("verdict", "")).upper().startswith("CORRECT")),
                               judge_final=str(j.get("final_answer", "")), inclusion=int(metrics([r])["accuracy"]))


def main():
    if os.path.exists(SHEET):
        print("sheet exists - not overwritten:", SHEET); return
    allit = list(items()); rng = random.Random(2026)
    disputed = []
    for ds in FILES:
        pool = [x for x in allit if x["dataset"] == ds and x["system"] == "delta+eatr" and x["inclusion"] == 1 and x["judge"] == 0]
        disputed += rng.sample(pool, min(len(pool), {"ComplexTR": 14, "TempReason-L3": 13, "TimeQA": 13}[ds]))
    rest = [x for x in allit if x not in disputed]
    rnd = rng.sample(rest, 40)
    sample = [dict(x, group="disputed") for x in disputed] + [dict(x, group="random") for x in rnd]
    rng.shuffle(sample)
    for n, x in enumerate(sample, 1):
        x["item"] = n
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(sample, open(KEY, "w"), indent=1)

    wb = Workbook(); ins = wb.active; ins.title = "Instructions"
    text = """JUDGE CHECK - about 1 minute per item (80 items). Your answers are compared with an automatic judge.
For each row read the QUESTION, the GOLD ANSWER and the RESPONSE of a system (you are not told which system).
In column F choose:
  Correct   - the response clearly COMMITS to an answer that means the same as the gold answer
              (for several gold answers, all of them must be given).
  Incorrect - the response commits to a different answer, OR hedges between several candidates
              (e.g. "it could be 1994 or 1996"), OR only mentions the gold answer while concluding something else,
              OR says the information is not available.
Optional column G: a few words if the item is unclear or if you think the rule is too strict for it.
Judge only by the gold answer and the response - not by your own knowledge."""
    for i, line in enumerate(text.splitlines(), 1):
        ins.cell(row=i, column=1, value=line)
    ins.column_dimensions["A"].width = 130
    ws = wb.create_sheet("Check"); wb.active = 1
    cols = [("Item", 6), ("Dataset", 13), ("Question", 45), ("Gold answer(s)", 22), ("Response", 90),
            ("YOUR VERDICT (Correct / Incorrect)", 18), ("Comment (optional)", 30)]
    for j, (h, w) in enumerate(cols, 1):
        c = ws.cell(row=1, column=j, value=h); c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="31688E" if j < 6 else "C0392B")
        c.alignment = Alignment(wrap_text=True, vertical="top"); ws.column_dimensions[c.column_letter].width = w
    for r, x in enumerate(sample, 2):
        for j, v in enumerate([x["item"], x["dataset"], x["question"], x["gold"], x["response"][:6000]], 1):
            ws.cell(row=r, column=j, value=v).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 220
    dv = DataValidation(type="list", formula1='"Correct,Incorrect"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"F2:F{len(sample)+1}")
    ws.freeze_panes = "C2"; wb.save(SHEET)
    print("wrote", SHEET, "| key", KEY, "| disputed", len(disputed), "random", len(rnd), "| pool sizes:",
          {ds: sum(1 for x in allit if x['dataset'] == ds and x['system'] == 'delta+eatr' and x['inclusion'] and not x['judge']) for ds in FILES})


if __name__ == "__main__":
    main()
