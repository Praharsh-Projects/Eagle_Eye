from __future__ import annotations

import ast
from pathlib import Path

from src.app.streamlit_query_bridge import (
    _evidence,
    _legacy_result,
    canonical_presentation,
    distinct_visual_summary,
    run_canonical_query,
)
from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import (
    AnswerState,
    AppliedScope,
    AssuranceAssessment,
    AvailabilityInfo,
    ETAWatchIntent,
    FreshnessInfo,
    QueryOperation,
    QueryRequest,
)
from src.query.service import QueryService


STREAMLIT_SOURCE_PATH = Path("src/app/streamlit_app.py")


def _function_calls(source: str, function_name: str) -> set[str]:
    module = ast.parse(source)
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    calls: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _service(tmp_path: Path) -> QueryService:
    processed = Path("data/processed")
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "streamlit_bridge.sqlite3"),
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def test_old_chat_and_category_submit_paths_use_only_canonical_bridge() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    for function_name in ("_render_page_chat", "_render_query_category_page"):
        calls = _function_calls(source, function_name)
        assert "run_canonical_query" in calls
        assert "_handle_ask_question" not in calls
        assert "classify_question" not in calls
        assert "run_turn" not in calls

    bridge_source = Path("src/app/streamlit_query_bridge.py").read_text(encoding="utf-8")
    assert "service.query(" in bridge_source


def test_bridge_preserves_zero_top_k_and_canonical_answer(tmp_path: Path) -> None:
    service = _service(tmp_path)
    captured = []
    original_query = service.query

    def recording_query(request):
        captured.append(request)
        return original_query(request)

    service.query = recording_query  # type: ignore[method-assign]
    bridged = run_canonical_query(
        service,
        question="Plot daily arrivals at Gothenburg for March 2022.",
        conversation_id="streamlit_test_conversation",
        top_k_evidence=0,
        user_filters={},
    )

    assert captured[0].top_k_evidence == 0
    assert captured[0].conversation_id == "streamlit_test_conversation"
    assert bridged.result.answer == bridged.envelope.answer
    assert bridged.envelope.visualizations[0].kind == "cartesian"


def test_zero_top_k_skips_analytics_and_research_retrieval(tmp_path: Path) -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.traffic_calls = 0
            self.docs_calls = 0
            self.retrieval_backend = "recording"

        def query_traffic(self, *args, **kwargs):
            self.traffic_calls += 1
            raise AssertionError("traffic retrieval must not run when top_k_evidence is zero")

        def query_docs(self, *args, **kwargs):
            self.docs_calls += 1
            raise AssertionError("document retrieval must not run when top_k_evidence is zero")

    service = _service(tmp_path)
    retriever = RecordingRetriever()
    service.retriever = retriever  # type: ignore[assignment]

    analytics = service.query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    research = service.query(
        QueryRequest(question="What does SOLAS require?", top_k_evidence=0)
    )

    assert analytics.evidence == []
    assert research.evidence == []
    assert research.state == AnswerState.NO_DATA
    assert "Evidence top K is set to 0" in research.answer
    assert retriever.traffic_calls == 0
    assert retriever.docs_calls == 0


def test_canonical_presentation_never_labels_unsupported_as_computed(tmp_path: Path) -> None:
    computed = _service(tmp_path).query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    unsupported = computed.model_copy(
        update={
            "state": AnswerState.UNSUPPORTED,
            "answer": "This request is unsupported.",
            "confidence": "not_applicable",
        }
    )

    view = canonical_presentation(unsupported)

    assert view.source_label == "Unsupported request"
    assert view.state_label == "Unsupported"
    assert view.confidence_label == "Not applicable"
    assert all("ran the planned deterministic" not in line.lower() for line in view.method_steps)
    assert "Ports: SEGOT." in view.applied_scope
    assert "Requested dates: 2022-03-01 to 2022-03-31." in view.applied_scope
    assert any(line.startswith("Global dataset coverage:") for line in view.freshness)


def test_legacy_assurance_unavailable_is_presented_neutrally_by_the_bridge(
    tmp_path: Path,
) -> None:
    computed = _service(tmp_path).query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    unavailable = computed.model_copy(
        update={
            "state": AnswerState.ASSURANCE_UNAVAILABLE,
            "answer": "This result did not pass the current assurance gate.",
            "confidence": "not_applicable",
            "assurance": AssuranceAssessment(
                status="unavailable",
                level="not_applicable",
                basis="direct_computation",
                reason="A proxy caveat prevents high-assurance publication.",
                checks=["result_state=COMPUTED", "disqualifying_caveat=proxy"],
            ),
            "availability": AvailabilityInfo(
                code="coverage_unavailable",
                provider="structured_datasets",
                retryable=False,
            ),
        }
    )

    view = canonical_presentation(unavailable)
    legacy_result = _legacy_result(unavailable)

    assert view.source_label == "Source-grounded answer unavailable"
    assert view.confidence_label == "Not applicable"
    assert view.state_label == "Assurance Unavailable"
    assert "attached sources" in view.source_detail
    assert "Availability: coverage unavailable." in view.source_detail
    assert view.method_steps == []
    assert legacy_result.status == "ASSURANCE_UNAVAILABLE"
    assert legacy_result.answer == unavailable.answer
    assert legacy_result.caveats == []


def test_legacy_low_result_is_not_given_policy_copy_by_streamlit_bridge(
    tmp_path: Path,
) -> None:
    canonical = _service(tmp_path).query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    stored_answer = "Stored legacy answer remains byte-for-byte unchanged."
    legacy = canonical.model_copy(
        update={
            "answer": stored_answer,
            "confidence": "low",
            "assurance": None,
            "availability": None,
        }
    )

    view = canonical_presentation(legacy)
    legacy_result = _legacy_result(legacy)

    assert view.answer == stored_answer
    assert view.confidence_label == "Available"
    assert view.state_label == "Analysis result"
    assert view.method_steps == []
    assert legacy_result.answer == stored_answer
    assert legacy_result.caveats == []


def test_eta_watch_scope_and_vessel_broadcast_assurance_are_presented(
    tmp_path: Path,
) -> None:
    canonical = _service(tmp_path).query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    live = canonical.model_copy(
        update={
            "plan": canonical.plan.model_copy(
                update={
                    "operation": QueryOperation.VESSEL_ETA,
                    "eta_watch_intent": ETAWatchIntent.VESSEL_STATUS,
                }
            ),
            "applied_scope": AppliedScope(
                ports=["SESTO"],
                date_from="2026-07-27T00:00:00Z",
                date_to="2026-07-28T00:00:00Z",
                vessel_name="TEST VESSEL",
                mmsi="230123456",
                imo="9876543",
                horizon_hours=24,
                source_scope="aisstream_baltic_broadcast",
            ),
            "freshness": FreshnessInfo(
                data_from="2026-07-27T00:00:00Z",
                data_to="2026-07-28T00:00:00Z",
                as_of="2026-07-27T14:00:00Z",
                historical=False,
                message="AISStream vessel-broadcast snapshot validated at 2026-07-27T14:00:00Z.",
            ),
            "assurance": AssuranceAssessment(
                status="verified",
                level="high",
                basis="official_live_source",
                reason=(
                    "Fresh AIS identity, destination, ETA, position, and observation-time "
                    "checks passed for this vessel-broadcast observation."
                ),
                checks=["ais_broadcast_observation_gates=passed"],
            ),
            "availability": AvailabilityInfo(
                code="available",
                provider="aisstream",
                retryable=False,
            ),
        }
    )

    view = canonical_presentation(live)

    assert view.source_label == "Official live source"
    assert view.confidence_label == "Available"
    assert view.state_label == "Analysis result"
    assert "Vessel: TEST VESSEL." in view.applied_scope
    assert "MMSI: 230123456." in view.applied_scope
    assert "IMO: 9876543." in view.applied_scope
    assert "Live horizon: 24 hours." in view.applied_scope
    assert "Source scope: aisstream_baltic_broadcast." in view.applied_scope
    assert "Source snapshot as of: 2026-07-27T14:00:00Z." in view.freshness
    assert view.method_steps == []
    assert all("official live source" not in line.lower() for line in view.method_steps)
    assert all("fintraffic" not in line.lower() for line in view.method_steps)


def test_bridge_forwards_optional_live_vessel_filters(tmp_path: Path) -> None:
    canonical = _service(tmp_path).query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    captured = []

    class RecordingService:
        def query(self, request):
            captured.append(request)
            return canonical

    run_canonical_query(
        RecordingService(),  # type: ignore[arg-type]
        question="Where is Test Vessel, and what ETA is it transmitting?",
        conversation_id="live_filter_test",
        top_k_evidence=5,
        user_filters={
            "port": "SESTO",
            "vessel_name": "Test Vessel",
            "mmsi": "230123456",
            "imo": "9876543",
        },
    )

    assert captured[0].filters.port == "SESTO"
    assert captured[0].filters.vessel_name == "Test Vessel"
    assert captured[0].filters.mmsi == "230123456"
    assert captured[0].filters.imo == "9876543"


def test_bridge_evidence_preserves_canonical_retrieval_and_assurance_metadata(
    tmp_path: Path,
) -> None:
    canonical = _service(tmp_path).query(
        QueryRequest(
            question="How many arrivals at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    canonical = canonical.model_copy(
        update={
            "trace": canonical.trace.model_copy(
                update={"retrieval_status": "unavailable"}
            ),
            "availability": AvailabilityInfo(
                code="source_unavailable",
                provider="aisstream",
                retryable=True,
            ),
        }
    )

    bridged = _evidence(canonical)

    assert bridged.trace["retrieval_status"] == "unavailable"
    assert bridged.trace["assurance"]["status"] == "verified"
    assert bridged.trace["availability"] == {
        "code": "source_unavailable",
        "provider": "aisstream",
        "retryable": True,
    }


def test_duplicate_omitted_visual_copy_is_suppressed() -> None:
    reason = "No validated rows are available for a meaningful graph."
    assert distinct_visual_summary(reason, reason) is None
    assert distinct_visual_summary(reason, f"  {reason}  ") is None
    assert distinct_visual_summary(reason, "A table fallback is still available.") == (
        "A table fallback is still available."
    )


def test_previous_navigation_wording_remains_exact_and_has_no_advanced_mode() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    analyst_pages = None
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "analyst_pages" for target in node.targets):
            analyst_pages = ast.literal_eval(node.value)
            break

    assert analyst_pages == [
        "Overview",
        "Chat Assistant",
        "Traffic Monitoring",
        "Vessel Investigation",
        "ETA & Delay",
        "Port Pressure",
        "Carbon Emissions",
    ]
    assert "Advanced" not in analyst_pages
    assert "Voyage-grade Emissions Workspace" not in analyst_pages


def test_chat_and_each_category_use_separate_stable_session_keys() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    assert 'st.session_state["chat_conversation_id"]' in source
    assert 'conversation_key = f"canonical_conversation_{key}"' in source
    assert 'st.session_state[conversation_key] = conversation_id' in source


def test_product_default_enables_evidence_and_zero_remains_an_explicit_opt_out() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    slider = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "slider"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Evidence top K"
    )
    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in slider.keywords}

    assert options["min_value"] == 0
    assert options["value"] == 5


def test_missing_openai_key_still_initializes_and_attaches_local_rag() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    ensure_retriever = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_retriever"
    )
    ensure_source = ast.get_source_segment(source, ensure_retriever)
    assert ensure_source is not None

    assert "if not api_key" not in ensure_source
    assert "retriever, retriever_reason = _ensure_retriever()" in source
    assert "query_service.retriever = retriever" in source
    assert "Local RAG evidence active" in source
    assert "Optional external-source retrieval is off" not in source
    assert 'st.warning("Set `OPENAI_API_KEY`' not in source
    assert "OpenAI is used for planning and explanation" not in source
