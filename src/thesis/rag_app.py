"""Thesis Streamlit demo: strategy-aware retrieval + incident-aware response."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.thesis.retrieve import QueryFilters, ThesisRetriever, format_incident_aware_answer


@st.cache_resource
def _load_retriever(persist_dir: str, strategy: str, embedding_model: str) -> ThesisRetriever:
    return ThesisRetriever(persist_dir=persist_dir, strategy=strategy, embedding_model=embedding_model)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Thesis Maritime RAG", layout="wide")
    st.title("Maritime Traffic RAG Thesis Demo")
    st.caption("Chunking strategy comparison with visible evidence and metadata filters")

    with st.sidebar:
        st.subheader("Runtime")
        persist_dir = st.text_input("Chroma persist dir", value="data/thesis_chroma")
        embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
        strategy = st.selectbox("Chunking strategy", options=["A", "B", "C"], index=0)
        top_k = st.slider("Top k", min_value=1, max_value=10, value=5)

        st.markdown("---")
        st.markdown("Build pipeline:")
        st.code("python -m src.thesis.data_pipeline")
        st.code("python -m src.thesis.chunking --strategy all")
        st.code("python -m src.thesis.embed_index --strategy all --rebuild")

    manifest = _read_json(Path(persist_dir) / f"manifest_{strategy.lower()}.json")
    with st.expander("Index Diagnostics", expanded=True):
        st.write(f"Strategy: `{strategy}`")
        if manifest:
            st.json(manifest)
        else:
            st.warning("No manifest found for this strategy. Run embed_index first.")

    question = st.text_area(
        "Question",
        value="Was there congestion at SEGOT on 2021-02-03 and what should operations check first?",
        height=90,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        port = st.text_input("Port / LOCODE", value="")
    with c2:
        date_from = st.text_input("Date from", value="")
    with c3:
        date_to = st.text_input("Date to", value="")
    with c4:
        vessel_type = st.text_input("Vessel type", value="")
    with c5:
        anomaly_opt = st.selectbox("Anomaly flag", options=["any", "true", "false"], index=0)

    ask = st.button("Ask", type="primary")
    if not ask:
        return

    anomaly = None
    if anomaly_opt == "true":
        anomaly = True
    elif anomaly_opt == "false":
        anomaly = False

    try:
        retriever = _load_retriever(
            persist_dir=persist_dir,
            strategy=strategy,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        st.error(f"Retriever init failed: {exc}")
        return

    filters = QueryFilters(
        port=port or None,
        date_from=date_from or None,
        date_to=date_to or None,
        vessel_type=vessel_type or None,
        anomaly=anomaly,
    )

    retrieved = retriever.retrieve(query=question, top_k=top_k, filters=filters)
    response = format_incident_aware_answer(query=question, retrieved=retrieved)

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Interpretation")
    st.write(response["interpretation"])

    st.subheader("Suggested Check")
    st.write(response["suggested_check"])

    st.subheader("Evidence Summary")
    if response["evidence"]:
        for line in response["evidence"]:
            st.markdown(f"- {line}")
    else:
        st.info("No evidence retrieved.")

    st.subheader("Retrieved Chunks")
    rows = []
    for chunk in retrieved.chunks:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "distance": chunk.distance,
                "port": chunk.metadata.get("port"),
                "date": chunk.metadata.get("date"),
                "vessel_type": chunk.metadata.get("vessel_type"),
                "anomaly": chunk.metadata.get("anomaly"),
                "text": chunk.text,
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df[["chunk_id", "distance", "port", "date", "vessel_type", "anomaly"]])
        with st.expander("Chunk texts"):
            for row in rows:
                st.markdown(f"**{row['chunk_id']}**")
                st.write(row["text"])
    else:
        st.warning("No chunks matched the query/filters.")


if __name__ == "__main__":
    main()
