# Thesis Execution Guide (Mac)

This guide is for the thesis-only pipeline:
- structured maritime data processing
- chunking strategy comparison (A/B/C)
- local embedding index with Chroma
- retrieval evaluation and evidence-first demo

## 1) Setup

```bash
cd "/Users/praharshchintu/Documents/New project"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Put PRJ896 + PRJ912 data in `data/`

If CSV files already exist:
- `/Users/praharshchintu/Documents/New project/data/PRJ896.csv`
- `/Users/praharshchintu/Documents/New project/data/PRJ912.csv`

skip this section.

If you only have ZIPs in Downloads:

```bash
mkdir -p "/Users/praharshchintu/Documents/New project/data"
unzip -j "/Users/praharshchintu/Downloads/PRJ912.csv - Challenge4 (Risk Assessment).zip" "*.csv" -d "/Users/praharshchintu/Documents/New project/data"
unzip -j "/Users/praharshchintu/Downloads/PRJ896.csv - Challenge 5. Data mining excellence.zip" "*.csv" -d "/Users/praharshchintu/Documents/New project/data"
```

Verify:

```bash
ls -lh "/Users/praharshchintu/Documents/New project/data/PRJ896.csv" "/Users/praharshchintu/Documents/New project/data/PRJ912.csv"
```

## 3) Run end-to-end thesis pipeline

Default (uses `data/PRJ912.csv` + `data/PRJ896.csv`):

```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
./run_thesis_pipeline.sh
```

If you want to pass explicit paths:

```bash
./run_thesis_pipeline.sh \
  "/Users/praharshchintu/Documents/New project/data/PRJ912.csv" \
  "/Users/praharshchintu/Documents/New project/data/PRJ896.csv"
```

Fast dev mode:

```bash
LIMIT_ROWS=20000 ./run_thesis_pipeline.sh
```

## 4) Launch thesis demo UI

```bash
cd "/Users/praharshchintu/Documents/New project"
source .venv/bin/activate
./run_streamlit.sh
```

## 5) Core outputs you will cite in thesis

- `data/thesis_processed/dataset_profile.json`
- `data/thesis_processed/thesis_context.json`
- `data/thesis_chunks/chunk_stats.json`
- `data/thesis_chroma/manifest_a.json`
- `data/thesis_chroma/manifest_b.json`
- `data/thesis_chroma/manifest_c.json`
- `evaluation/thesis/results/summary.json`
- `evaluation/thesis/results/per_query_results.csv`
- `evaluation/thesis/results/manual_relevance_template.csv`
- `evaluation/thesis/results/hit_at_k.png`
- `evaluation/thesis/results/latency_ms.png`

## 6) Strategy definitions

- **A (event chunks):** one chunk per AIS/port-call event
- **B (port-day chunks):** one chunk per port/day aggregation
- **C (hybrid windows):** grouped temporal windows of events

## 7) Optional PDFs (not required for core thesis experiments)

Your current thesis contribution is CSV chunking/embedding/retrieval. Regulatory PDFs are optional background evidence.

Safe PDFs to include if you extend to docs RAG:
- NIS2 official EUR-Lex PDF: `CELEX_32022L2555_EN_TXT.pdf`
- IMO/ILO publicly sharable guidance document you already have: `ILOIMOCodeOfPracticeEnglish.pdf`

Recommended storage path:

```bash
mkdir -p "/Users/praharshchintu/Documents/New project/data/pdfs"
cp "/Users/praharshchintu/Documents/New project/data/CELEX_32022L2555_EN_TXT.pdf" "/Users/praharshchintu/Documents/New project/data/pdfs/"
cp "/Users/praharshchintu/Documents/New project/data/ILOIMOCodeOfPracticeEnglish.pdf" "/Users/praharshchintu/Documents/New project/data/pdfs/"
```

## 8) What this pipeline proves

- retrieval quality comparison by chunk strategy (`Hit@k`)
- efficiency comparison (`latency`, `index size`, `chunk count`)
- grounding/traceability via visible evidence IDs and metadata filters
