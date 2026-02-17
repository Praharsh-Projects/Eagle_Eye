#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.

PRJ912="${1:-data/PRJ912.csv}"
PRJ896="${2:-data/PRJ896.csv}"
PROCESSED_DIR="${PROCESSED_DIR:-data/thesis_processed}"
CHUNKS_DIR="${CHUNKS_DIR:-data/thesis_chunks}"
PERSIST_DIR="${PERSIST_DIR:-data/thesis_chroma}"
EVAL_DIR="${EVAL_DIR:-evaluation/thesis/results}"
MODEL_NAME="${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

if [[ ! -f "$PRJ912" ]]; then
  echo "ERROR: Missing $PRJ912"
  exit 2
fi
if [[ ! -f "$PRJ896" ]]; then
  echo "ERROR: Missing $PRJ896"
  exit 2
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT_ROWS:-}" ]]; then
  LIMIT_ARGS+=(--limit_rows "$LIMIT_ROWS")
fi

echo "[1/4] Building thesis processed datasets..."
python -m src.thesis.data_pipeline \
  --prj912 "$PRJ912" \
  --prj896 "$PRJ896" \
  --out_dir "$PROCESSED_DIR" \
  "${LIMIT_ARGS[@]}"

echo "[2/4] Building chunking strategies A/B/C..."
python -m src.thesis.chunking \
  --processed_dir "$PROCESSED_DIR" \
  --out_dir "$CHUNKS_DIR" \
  --strategy all

echo "[3/4] Embedding and indexing strategies..."
python -m src.thesis.embed_index \
  --chunks_dir "$CHUNKS_DIR" \
  --persist_dir "$PERSIST_DIR" \
  --strategy all \
  --embedding_model "$MODEL_NAME" \
  --rebuild

echo "[4/4] Running retrieval evaluation..."
python -m src.thesis.evaluate \
  --questions evaluation/thesis/questions.jsonl \
  --persist_dir "$PERSIST_DIR" \
  --strategies A,B,C \
  --embedding_model "$MODEL_NAME" \
  --top_k 5 \
  --out_dir "$EVAL_DIR"

echo "Done. Launch demo with: streamlit run src/thesis/rag_app.py"
