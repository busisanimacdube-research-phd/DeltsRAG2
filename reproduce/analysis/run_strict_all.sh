#!/bin/bash
cd "$(dirname "$0")/../.."
FILES=()
for f in reproduce/results/*.json reproduce/baselines/results/*.json; do
  [[ "$f" == *partial* ]] && continue; FILES+=("$f")
done
FILES+=("reproduce/fair_eval_329_baseline.json::baseline"
        "reproduce/fair_eval_tempreason_l3_500.json::baseline"
        "reproduce/fair_eval_timeqa_baseline.json::baseline")
exec /opt/miniconda3/bin/python reproduce/analysis/strict_eval.py "${FILES[@]}"
