from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from src.thesis.evaluate import LexicalBaselineIndex, _lexical_retrieve
from src.thesis.retrieve import QueryFilters


def test_lexical_retrieve_respects_filters_and_returns_relevant_chunk() -> None:
    documents = [
        "Vessel ALPHA arrived at SEGOT on 2022-03-01 from LTKLJ.",
        "Vessel BETA arrived at LVVNT on 2022-03-01 from SEGOT.",
        "Vessel GAMMA departed PLGDN on 2022-03-05 to LVRIX.",
    ]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    index = LexicalBaselineIndex(
        strategy="A",
        ids=["c1", "c2", "c3"],
        documents=documents,
        metadatas=[
            {"port": "SEGOT", "locode": "SEGOT", "date": "2022-03-01", "vessel_type": "cargo", "anomaly": False},
            {"port": "LVVNT", "locode": "LVVNT", "date": "2022-03-01", "vessel_type": "cargo", "anomaly": False},
            {"port": "PLGDN", "locode": "PLGDN", "date": "2022-03-05", "vessel_type": "cargo", "anomaly": False},
        ],
        tfidf_matrix=matrix,
        vectorizer=vectorizer,
        label="TF-IDF lexical baseline",
    )

    retrieved = _lexical_retrieve(
        index=index,
        query="How many vessel arrivals were recorded at SEGOT on 2022-03-01?",
        filters=QueryFilters(port="SEGOT", date_from="2022-03-01", date_to="2022-03-01"),
        top_k=2,
    )

    assert retrieved.chunks
    assert retrieved.chunks[0].chunk_id == "c1"
    assert retrieved.chunks[0].metadata["port"] == "SEGOT"
    assert retrieved.latency_ms >= 0.0
