# Thesis Blueprint (Start to Submission)

## Thesis Focus
Efficient chunking and embedding strategies for structured maritime CSV data (PRJ896 + PRJ912) in a RAG-based incident-aware decision support workflow.

## Locked Technical Decisions
- OS: macOS
- Python: 3.11/3.12 (`.venv`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: Chroma persistent local storage
- RAG generation: evidence-first deterministic formatter (optional LLM layer can be added later)

## Reproducible Pipeline
1. Data cleaning + feature engineering
2. Chunk strategy generation (A/B/C)
3. Embedding + indexing per strategy
4. Retrieval evaluation and ablation
5. Streamlit demo with visible evidence

## Commands
```bash
python -m src.thesis.data_pipeline \
  --prj912 data/PRJ912.csv \
  --prj896 data/PRJ896.csv \
  --out_dir data/thesis_processed

python -m src.thesis.chunking \
  --processed_dir data/thesis_processed \
  --out_dir data/thesis_chunks \
  --strategy all

python -m src.thesis.embed_index \
  --chunks_dir data/thesis_chunks \
  --persist_dir data/thesis_chroma \
  --strategy all \
  --rebuild

python -m src.thesis.evaluate \
  --questions evaluation/thesis/questions.jsonl \
  --persist_dir data/thesis_chroma \
  --strategies A,B,C \
  --top_k 5 \
  --out_dir evaluation/thesis/results

streamlit run src/thesis/rag_app.py
```

## Core Outputs
- `data/thesis_processed/*.parquet`
- `data/thesis_chunks/strategy_*.jsonl`
- `data/thesis_chroma/manifest_*.json`
- `evaluation/thesis/results/summary.json`
- `evaluation/thesis/results/per_query_results.csv`
- `evaluation/thesis/results/manual_relevance_template.csv`
- `evaluation/thesis/results/hit_at_k.png`
- `evaluation/thesis/results/latency_ms.png`

## Research Questions
- RQ1: Which chunking strategy yields highest Hit@k under maritime port/date filters?
- RQ2: What is the latency/index-size tradeoff across strategies?
- RQ3: Which strategy provides better grounding quality for incident-aware summaries?

## Evaluation Metrics
- Retrieval: Hit@k (k=3/5)
- Relevance: manual 0-2 scoring (`manual_relevance_template.csv`)
- Grounding: hallucination rate proxy from answer/evidence lexical checks
- Efficiency: mean latency, chunk count, index size

## Recommended Thesis Timeline
- Week 1-2: data preparation + feature engineering
- Week 3: chunking strategies A/B/C
- Week 4: embedding/indexing + baseline retrieval
- Week 5: evaluation scripts + first ablation
- Week 6: incident-aware answer templates + demo hardening
- Week 7: full experiments and figures
- Week 8: writing and revision

## Scope Boundaries (must refuse)
Unsupported without external terminal ops data:
- Berth-level congestion
- Crane utilization
- TEU throughput
- Gate queue length

## Notes for Writing
- Report congestion as a proxy (arrivals + dwell), not direct berth occupancy truth.
- State all assumptions explicitly.
- Keep evidence IDs visible in all demo answers.
