import os, json, random
import networkx as nx
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reproduce", "analysis", "annotation")
SETS = {"ComplexTR": "reproduce/baseline_complextr_dir",
        "TempReason-L3": "reproduce/tempreason_full_dir",
        "TimeQA": "reproduce/timeqa_shared_dir"}
PER_SET, SEED = 50, 2026

COLS = [("Item", 6), ("Dataset", 13), ("Event ID", 16), ("Source chunk (what the LLM read)", 90),
        ("Extracted event sentence", 45), ("Extracted context", 30), ("Extracted timestamp", 13),
        ("Extracted entities (NER)", 30),
        ("Q1 Event faithful?", 16), ("Q2 Timestamp", 20), ("Q3 Time expression type", 16),
        ("Q4 # correct entities", 10), ("Q5 # wrong/fragment entities", 10), ("Q6 # missed entities", 10),
        ("Notes / error description", 40)]
CHOICES = {
    "I": '"Correct,Partially correct,Not supported (hallucinated)"',
    "J": '"Correct,Wrong date,Too coarse,Wrongly static,Should be static"',
    "K": '"Explicit,Relative,Implicit,None"',
}

INSTRUCTIONS = """DEU MANUAL VERIFICATION - INSTRUCTIONS

For each row, read the SOURCE CHUNK, then judge what the pipeline extracted from it.

Q1 Event faithful?  Is the extracted event sentence supported by the chunk?
   Correct = fully supported; Partially correct = main fact right but a detail wrong/merged;
   Not supported = the fact is not in the chunk (hallucination).
Q2 Timestamp: compare the extracted timestamp with the time the chunk gives for this event.
   Correct = right date at the granularity the text supports (e.g. text says 1994 -> "1994");
   Wrong date = a different date; Too coarse = right but less precise than the text allows
   (text says "March 1994", extracted "1994"); Wrongly static = the text dates the event but
   it was extracted as "static"; Should be static = a date was assigned to a timeless fact.
Q3 Time expression type in the chunk for this event: Explicit ("in 1994"), Relative
   ("two years later", "the following spring"), Implicit (inferred from context), None.
Q4 Number of extracted entities that are correct and complete names.
Q5 Number of extracted entities that are wrong, truncated fragments (e.g. "ri adams"), or not entities.
Q6 Number of people/organisations/places mentioned in the EVENT SENTENCE that were not extracted.
Notes: short description of any error (used for the typical-error analysis).

Judge only what is extracted, not whether it is useful for any question.
Annotator 2 (supervisor) labels the 50 items of the second file independently, without
seeing annotator 1's answers; we report agreement (Cohen's kappa) on those 50 items.
"""


def main():
    rng = random.Random(SEED)
    items = []
    for name, d in SETS.items():
        g = nx.read_graphml(os.path.join(ROOT, d, "graph_event_dynamic_graph.graphml"))
        chunks = json.load(open(os.path.join(ROOT, d, "kv_store_text_chunks.json")))
        nodes = sorted(g.nodes())
        for nid in rng.sample(nodes, PER_SET):
            data = g.nodes[nid]
            src = str(data.get("source_id", "")).split("<SEP>")[0]
            items.append({"ds": name, "id": nid, "chunk": chunks.get(src, {}).get("content", ""),
                          "sent": data.get("sentence", ""), "ctx": data.get("context", ""),
                          "ts": data.get("timestamp", ""), "ents": data.get("entities_involved", "")})
    by = {k: [x for x in items if x["ds"] == k] for k in SETS}
    inter = [by[k][i] for i in range(PER_SET) for k in SETS]
    os.makedirs(OUT, exist_ok=True)
    json.dump(inter, open(os.path.join(OUT, "deu_sample_key.json"), "w"), indent=1)
    for fname, rows in (("DEU_annotation_annotator1.xlsx", inter),
                        ("DEU_annotation_annotator2_overlap50.xlsx", inter[:50])):
        wb = Workbook()
        ins = wb.active; ins.title = "Instructions"
        for i, line in enumerate(INSTRUCTIONS.splitlines(), 1):
            ins.cell(row=i, column=1, value=line)
        ins.column_dimensions["A"].width = 120
        ws = wb.create_sheet("Annotate")
        for j, (h, w) in enumerate(COLS, 1):
            c = ws.cell(row=1, column=j, value=h)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F4E79" if j <= 8 else "2E7D32")
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions[c.column_letter].width = w
        for i, x in enumerate(rows, 2):
            vals = [i - 1, x["ds"], x["id"], x["chunk"], x["sent"], x["ctx"], x["ts"], x["ents"]]
            for j, v in enumerate(vals, 1):
                ws.cell(row=i, column=j, value=v).alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[i].height = 180
        for col, formula in CHOICES.items():
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            ws.add_data_validation(dv); dv.add(f"{col}2:{col}{len(rows)+1}")
        for col in "LMN":
            dv = DataValidation(type="whole", operator="between", formula1="0", formula2="50", allow_blank=True)
            ws.add_data_validation(dv); dv.add(f"{col}2:{col}{len(rows)+1}")
        ws.freeze_panes = "E2"
        wb.save(os.path.join(OUT, fname))
        print("wrote", os.path.join(OUT, fname), len(rows), "items")


if __name__ == "__main__":
    main()
