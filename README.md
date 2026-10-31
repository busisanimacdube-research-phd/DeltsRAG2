# DELTA-RAG

Code and results for DELTA-RAG, a temporal knowledge-graph RAG method built on DyG-RAG and nano-graphrag.

## Contents

- `graphrag/` - implementation (DELTA-RAG modules in `graphrag/delta_rag/`)
- `datasets/` - ComplexTR, TimeQA, TempReason and the 500-question TempReason-L3 sample
- `reproduce/run_experiments.py` - DELTA-RAG runs and ablations
- `reproduce/fair_eval*.py` - DyG-RAG baseline runs
- `reproduce/baselines/` - NaiveRAG, GraphRAG and LightRAG baselines
- `reproduce/analysis/` - metrics, significance tests and figures
- `reproduce/results/`, `reproduce/baselines/results/`, `reproduce/analysis/out/` - result files

## Setup

```
pip install -r requirements.txt
```

Reader: Qwen2.5-14B-Instruct served through an OpenAI-compatible endpoint. Embeddings: BGE-M3.

## Example

```
python reproduce/run_experiments.py --dataset complextr --arm delta+eatr
```

## License

MIT. See `LICENSE`.
