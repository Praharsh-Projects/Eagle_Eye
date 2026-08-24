from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import AnswerState, QueryRequest
from src.query.service import QueryService
from src.rag.retriever import (
    EvidenceItem as RetrievedItem,
    QueryFilters,
    RAGRetriever,
    RetrievalResult,
)
from src.rag.synthesis import (
    OllamaSynthesizer,
    SynthesisResult,
    normalize_evidence_citations,
)


class _FakeRetriever:
    retrieval_backend = "test_persist:local_lexical"

    def __init__(self) -> None:
        self.traffic_calls = 0
        self.docs_calls = 0

    def query_traffic(self, *, question: str, filters: QueryFilters, top_k: int):
        self.traffic_calls += 1
        return RetrievalResult(
            mode="traffic",
            backend=self.retrieval_backend,
            where_filter={"locode": filters.locode},
            evidence=[
                RetrievedItem(
                    id="traffic_1",
                    text="Vessel call at SEGOT on 2022-03-01.",
                    metadata={"locode_norm": "SEGOT", "timestamp_date": "2022-03-01"},
                    source_kind="traffic",
                    distance=0.1,
                )
            ][:top_k],
        )

    def query_docs(self, *, question: str, top_k: int):
        self.docs_calls += 1
        return RetrievalResult(
            mode="docs",
            backend=self.retrieval_backend,
            where_filter=None,
            evidence=[
                RetrievedItem(
                    id="solas_local_1",
                    text="SOLAS Chapter V contains navigational safety requirements.",
                    metadata={"title": "SOLAS local excerpt", "source_file": "solas.pdf"},
                    source_kind="docs",
                    distance=0.05,
                )
            ][:top_k],
        )


class _FakeSynthesizer:
    provider = "ollama"
    model = "qwen-test"

    def synthesize_research(self, question, evidence):
        assert evidence and evidence[0].id == "solas_local_1"
        return SynthesisResult(
            text="SOLAS Chapter V addresses navigational safety [solas_local_1].",
            provider=self.provider,
            model=self.model,
        )

    def answer_general(self, question):
        return SynthesisResult(
            text="A lighthouse joke with no invented analytics.",
            provider=self.provider,
            model=self.model,
        )


def _service(tmp_path: Path, *, retriever=None, synthesizer=None) -> QueryService:
    processed = "data/processed"
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "rag.sqlite3"),
        retriever=retriever,
        retriever_reason="test retriever",
        local_synthesizer=synthesizer,
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def test_analytics_retrieval_adds_lineage_without_rewriting_facts(tmp_path: Path) -> None:
    retriever = _FakeRetriever()
    service = _service(tmp_path, retriever=retriever)
    prompt = "How many arrivals at Gothenburg in March 2022?"

    without_retrieval = service.query(QueryRequest(question=prompt, top_k_evidence=0))
    with_retrieval = service.query(QueryRequest(question=prompt, top_k_evidence=3))

    assert without_retrieval.answer == with_retrieval.answer
    assert without_retrieval.facts == with_retrieval.facts
    assert without_retrieval.datasets == with_retrieval.datasets
    assert without_retrieval.evidence == []
    assert with_retrieval.evidence[0].source_type == "traffic_event"
    assert with_retrieval.trace.sources == ["structured_datasets", "traffic_event"]
    assert with_retrieval.trace.retrieval_mode == "traffic"
    assert with_retrieval.trace.retrieval_backend == retriever.retrieval_backend
    assert with_retrieval.trace.retrieval_status == "ok"
    assert with_retrieval.trace.retrieval_top_k == 3
    assert with_retrieval.trace.model == "deterministic"
    assert retriever.traffic_calls == 1


def test_local_qwen_route_is_grounded_for_research_but_never_used_for_analytics(
    tmp_path: Path,
) -> None:
    retriever = _FakeRetriever()
    service = _service(tmp_path, retriever=retriever, synthesizer=_FakeSynthesizer())

    research = service.query(QueryRequest(question="What does SOLAS require?", top_k_evidence=3))
    analytics = service.query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=3,
        )
    )

    assert research.state == AnswerState.RETRIEVED
    assert research.answer.endswith("[solas_local_1].")
    assert research.trace.model == "ollama/qwen-test"
    assert research.trace.retrieval_mode == "documents"
    assert not any(fact.name.startswith("answer_number_") for fact in research.facts)
    assert analytics.trace.model == "deterministic"
    assert "488 vessel arrivals" in analytics.answer


def test_local_general_synthesis_is_noncurrent_only(tmp_path: Path) -> None:
    service = _service(tmp_path, synthesizer=_FakeSynthesizer())

    ordinary = service.query(QueryRequest(question="Tell me a joke about lighthouses."))
    current = service.query(QueryRequest(question="Who is the current prime minister of Sweden?"))

    assert ordinary.state == AnswerState.GENERAL
    assert ordinary.trace.model == "ollama/qwen-test"
    assert ordinary.trace.retrieval_status == "not_required"
    assert current.state == AnswerState.NO_CURRENT_DATA
    assert current.trace.retrieval_mode == "web"
    assert current.trace.retrieval_status == "unavailable"


def test_identifier_anchor_blocks_unrelated_regulatory_chunks() -> None:
    retriever = object.__new__(RAGRetriever)
    evidence = retriever._lexical_rank(
        question="What does SOLAS require?",
        ids=["nis2", "solas"],
        documents=[
            "NIS2 requires cyber risk controls.",
            "The SOLAS convention includes navigation requirements.",
        ],
        metadatas=[
            {"source_file": "NIS2-directive.pdf"},
            {"source_file": "SOLAS-guide.pdf"},
        ],
        source_kind="docs",
        top_k=5,
    )

    assert [item.id for item in evidence] == ["solas"]
    assert all("nis2" not in item.id for item in evidence)


def test_demo_metadata_without_serialized_text_uses_faithful_local_excerpt(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EAGLE_EYE_RETRIEVAL_PROVIDER", "lexical")
    retriever = RAGRetriever("demo_data/chroma")

    started = time.perf_counter()
    result = retriever.query_traffic(
        "Show events at SEGOT in March 2022",
        QueryFilters(locode="SEGOT", date_from="2022-03-01", date_to="2022-03-31"),
        top_k=3,
    )
    latency = time.perf_counter() - started

    assert len(result.evidence) == 3
    assert all(item.text.startswith("AIS event |") for item in result.evidence)
    assert all(item.metadata.get("locode_norm") == "SEGOT" for item in result.evidence)
    assert latency < 2.0


def test_local_docs_retrieval_never_counts_the_full_vector_store() -> None:
    class _DocsCollection:
        def count(self):
            raise AssertionError("local document retrieval must not count the full Chroma store")

        def get(self, *, where, limit, offset, include):
            if offset:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": ["isps_1"],
                "documents": ["The ISPS Code establishes a maritime security framework."],
                "metadatas": [{"source_file": "isps.html", "title": "ISPS Code"}],
            }

    retriever = object.__new__(RAGRetriever)
    retriever.openai = None
    retriever.top_k = 5
    retriever.local_scan_limit = 100
    retriever.docs_collection = _DocsCollection()
    retriever._docs_lexical_cache = None
    retriever.vector_backend = "test"
    retriever.query_backend = "local_lexical"

    result = retriever.query_docs("What does the ISPS Code establish?", top_k=3)

    assert [item.id for item in result.evidence] == ["isps_1"]


def test_metadata_preload_prefers_bounded_excerpt_without_native_arrow_strings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic_metadata_index.csv"
    pd.DataFrame(
        [
            {
                "stable_id": "event_1",
                "serialized_text": "full event text that must not be retained",
                "serialized_excerpt": "bounded event excerpt",
                "mmsi": "123456789",
                "locode_norm": "SEGOT",
                "timestamp_date": "2022-03-01",
                "latitude": 57.7,
                "longitude": 11.9,
            }
        ]
    ).to_csv(path, index=False)
    retriever = object.__new__(RAGRetriever)
    retriever.metadata_index_path = path
    retriever._metadata_df = None

    loaded = retriever._load_metadata_df()

    assert loaded is not None
    assert "serialized_excerpt" in loaded.columns
    assert "serialized_text" not in loaded.columns
    assert loaded["stable_id"].dtype == object


def test_ollama_research_requires_valid_citations(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [{
                                    "text": "The supplied excerpt supports this",
                                    "evidence_ids": ["E1"],
                                }],
                                "boundary": None,
                            }
                        )
                    }
                }
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("src.rag.synthesis.urlopen", fake_urlopen)
    synthesizer = OllamaSynthesizer(model="qwen-test", timeout_seconds=2)
    evidence = [
        type(
            "Evidence",
            (),
            {"id": "doc_1", "title": "Local document", "excerpt": "Verified text."},
        )()
    ]

    result = synthesizer.synthesize_research("What is verified?", evidence)

    assert result.text == "The supplied excerpt supports this [doc_1]."
    assert captured["payload"]["options"]["temperature"] == 0
    assert captured["payload"]["options"]["seed"] == 20260808
    assert captured["payload"]["format"]["type"] == "object"
    assert captured["payload"]["format"]["required"] == ["claims", "boundary"]
    assert captured["payload"]["format"]["additionalProperties"] is False
    assert captured["timeout"] == 2
    system_prompt = captured["payload"]["messages"][0]["content"]
    assert "exactly one JSON object with keys claims and boundary" in system_prompt
    assert "evidence_ids" in system_prompt


def test_grouped_research_citations_expand_to_immutable_evidence_ids() -> None:
    normalized = normalize_evidence_citations(
        "AIS supplements other information [E1, E2].",
        {"E1": "ais_limits_1", "E2": "ais_limits_2"},
    )

    assert normalized == "AIS supplements other information [ais_limits_1] [ais_limits_2]."


def test_structured_partial_research_answer_renders_claim_and_boundary() -> None:
    rendered = OllamaSynthesizer._render_research_claims(
        json.dumps({
            "claims": [{
                "text": "AIS carries voyage-related information",
                "evidence_ids": ["E1"],
            }],
            "boundary": {
                "text": "The supplied excerpts do not establish a complete official arrival schedule",
                "evidence_ids": ["E1", "E2"],
            },
        }),
        {"E1": "ais_data_1", "E2": "ais_limits_2"},
    )

    assert rendered == (
        "AIS carries voyage-related information [ais_data_1]. "
        "The supplied excerpts do not establish a complete official arrival schedule "
        "[ais_data_1] [ais_limits_2]."
    )


@pytest.mark.parametrize(
    "text",
    (
        "AIS is supported [E1] but this marker is malformed E2].",
        "AIS is supported [E1 E2].",
        "AIS is supported [E3].",
    ),
)
def test_research_citation_normalization_rejects_malformed_or_unknown_markers(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="evidence citation|evidence marker"):
        normalize_evidence_citations(text, {"E1": "doc_1", "E2": "doc_2"})
