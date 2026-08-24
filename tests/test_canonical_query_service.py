from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import (
    AnswerState,
    DateScope,
    QueryMode,
    QueryOperation,
    QueryPlan,
    QueryRequest,
)
from src.query.planner import QueryPlanner
from src.query.service import QueryService


def _service(tmp_path, *, database_name: str = "conversations.sqlite3") -> QueryService:
    processed = "data/processed"
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / database_name),
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def test_default_launcher_serves_react_and_preserves_streamlit_qa_interface() -> None:
    wrapper = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    launcher = Path("run_eagle_eye.sh").read_text(encoding="utf-8")
    streamlit_launcher = Path("run_streamlit.sh").read_text(encoding="utf-8")
    previous_ui = Path("src/app/streamlit_app.py").read_text(encoding="utf-8")
    assert "from src.app.streamlit_app import main" in wrapper
    assert 'exec "$EAGLE_EYE_ROOT/run_fastapi.sh"' in launcher
    assert '"$VENV_DIR/bin/streamlit" run app/streamlit_app.py' in streamlit_launcher
    assert '"Carbon Emissions"' in previous_ui
    assert '"Advanced"' not in previous_ui[previous_ui.index("analyst_pages = ["):previous_ui.index("page_options = analyst_pages")]
    assert '"Voyage-grade Emissions Workspace"' not in previous_ui[previous_ui.index("analyst_pages = ["):previous_ui.index("page_options = analyst_pages")]


def test_canonical_qa_and_fastapi_react_paths_remain_available() -> None:
    qa_surface = Path("src/app/streamlit_canonical.py").read_text(encoding="utf-8")
    api_launcher = Path("run_fastapi.sh").read_text(encoding="utf-8")
    assert "service.query(" in qa_surface
    assert "classify_question" not in qa_surface
    assert "_handle_ask_question" not in qa_surface
    assert "src.api.server:app" in api_launcher


@pytest.mark.parametrize(
    ("question", "mode", "operation"),
    [
        ("Will you explain this app?", QueryMode.APP_HELP, QueryOperation.HELP),
        ("What does SOLAS Chapter V require?", QueryMode.MARITIME_RESEARCH, QueryOperation.RESEARCH),
        ("Tell me a joke about lighthouses.", QueryMode.GENERAL_CHAT, QueryOperation.GENERAL_RESPONSE),
        (
            "What is crane utilization at berth 3 in SEGOT today?",
            QueryMode.UNSUPPORTED,
            QueryOperation.UNSUPPORTED,
        ),
        (
            "What arrivals should I expect at SEGOT in March 2022?",
            QueryMode.ANALYTICS,
            QueryOperation.ARRIVALS,
        ),
    ],
)
def test_planner_never_defaults_unknown_prompts_to_arrivals(question, mode, operation) -> None:
    plan = QueryPlanner().plan(question)
    assert plan.mode == mode
    assert plan.operation == operation


def test_answer_and_visualization_contracts_are_intent_aware(tmp_path) -> None:
    service = _service(tmp_path)

    scalar = service.query(
        QueryRequest(question="How many ship calls did Gothenburg receive during March of 2022?")
    )
    assert scalar.state == AnswerState.COMPUTED
    assert scalar.plan.operation == QueryOperation.ARRIVALS
    assert scalar.answer == "Matched 488 vessel arrivals across 31 day buckets for SEGOT."
    assert scalar.visualizations[0].kind == "kpi"

    trend = service.query(QueryRequest(question="Plot daily arrivals at Gothenburg for March 2022."))
    assert trend.visualizations[0].kind == "cartesian"
    assert trend.visualizations[0].chart_type == "line"

    weekdays = service.query(
        QueryRequest(question="Is Monday busier than Friday at Gothenburg in March 2022?")
    )
    assert weekdays.plan.operation == QueryOperation.WEEKDAY_COMPARISON
    assert weekdays.visualizations[0].kind == "cartesian"
    assert weekdays.visualizations[0].chart_type == "grouped_bar"

    composition = service.query(
        QueryRequest(question="Show share of arrivals by vessel type at Gothenburg in March 2022.")
    )
    assert composition.plan.operation == QueryOperation.VESSEL_TYPE_COMPOSITION
    assert composition.visualizations[0].kind == "cartesian"
    assert composition.visualizations[0].chart_type == "stacked_bar"

    distribution = service.query(
        QueryRequest(question="Show distribution of vessel dwell times at Gothenburg in March 2022.")
    )
    assert distribution.plan.operation == QueryOperation.DWELL_DISTRIBUTION
    assert distribution.visualizations[0].kind == "distribution"
    assert distribution.visualizations[0].chart_type == "histogram"
    assert distribution.visualizations[0].count_field == "calls"

    # Strict JSON serialization must reject NaN/Infinity before a response is emitted.
    json.dumps(distribution.model_dump(mode="json"), allow_nan=False)


def test_current_historical_query_is_explicitly_stale(tmp_path) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(question="How many arrivals are there at Gothenburg today?")
    )
    assert envelope.plan.operation == QueryOperation.CURRENT_ARRIVALS
    assert envelope.applied_scope.ports == ["SEGOT"]
    assert envelope.state == AnswerState.NO_CURRENT_DATA
    assert envelope.visualizations[0].kind == "omitted"
    assert envelope.visualizations[0].reason_code == "stale_data"
    assert envelope.freshness.historical is True


def test_sqlite_context_survives_service_recreation(tmp_path) -> None:
    first_service = _service(tmp_path)
    first = first_service.query(
        QueryRequest(question="How many arrivals at Gothenburg in March 2022?")
    )

    restarted_service = _service(tmp_path)
    follow_up = restarted_service.query(
        QueryRequest(question="What about Karlshamn?", conversation_id=first.conversation_id)
    )
    assert follow_up.plan.operation == QueryOperation.ARRIVALS
    assert follow_up.applied_scope.ports == ["SEKAN"]
    assert follow_up.applied_scope.date_from == "2022-03-01"
    assert follow_up.applied_scope.date_to == "2022-03-31"
    assert follow_up.state == AnswerState.COMPUTED
    assert follow_up.confidence == "high"
    assert follow_up.assurance is not None
    assert follow_up.assurance.status == "verified"
    assert follow_up.datasets
    assert all(visual.kind != "omitted" for visual in follow_up.visualizations)
    assert "Matched 72 vessel arrivals" in follow_up.answer

    explanation = restarted_service.query(
        QueryRequest(
            question="Explain that result in simpler terms.",
            conversation_id=first.conversation_id,
        )
    )
    assert explanation.mode == QueryMode.ANALYTICS
    assert explanation.plan.operation == QueryOperation.EXPLAIN_PREVIOUS
    assert "previous_answer" in explanation.plan.context_inherited
    assert explanation.applied_scope.ports == ["SEKAN"]
    assert explanation.visualizations[0].kind == "omitted"
    assert explanation.state == AnswerState.COMPUTED
    assert explanation.confidence == "high"
    assert explanation.assurance is not None
    assert explanation.assurance.status == "verified"
    assert explanation.answer == (
        "In plain language: Matched 72 vessel arrivals across 30 day buckets for SEKAN."
    )
    assert explanation.datasets == []
    assert explanation.visualizations[0].reason_code == "not_requested"


def test_contextual_follow_ups_replace_explicit_dates_and_never_leak_stale_scope(tmp_path) -> None:
    service = _service(tmp_path)
    first = service.query(
        QueryRequest(question="How many arrivals were recorded at Gothenburg in March 2022?")
    )
    conversation_id = first.conversation_id

    april = service.query(
        QueryRequest(question="What about April?", conversation_id=conversation_id)
    )
    assert april.plan.operation == QueryOperation.ARRIVALS
    assert april.applied_scope.ports == ["SEGOT"]
    assert april.applied_scope.date_from == "2022-04-01"
    assert april.applied_scope.date_to == "2022-04-30"
    assert april.plan.date_scope.is_current is False

    karlshamn_april = service.query(
        QueryRequest(
            question="What about Karlshamn in April 2022?",
            conversation_id=conversation_id,
        )
    )
    assert karlshamn_april.plan.operation == QueryOperation.ARRIVALS
    assert karlshamn_april.applied_scope.ports == ["SEKAN"]
    assert karlshamn_april.applied_scope.date_from == "2022-04-01"
    assert karlshamn_april.applied_scope.date_to == "2022-04-30"

    current = service.query(
        QueryRequest(
            question="What about Karlshamn today?",
            conversation_id=conversation_id,
        )
    )
    today = datetime.now(timezone.utc).date().isoformat()
    assert current.plan.operation == QueryOperation.ARRIVALS
    assert current.applied_scope.ports == ["SEKAN"]
    assert current.applied_scope.date_from == today
    assert current.applied_scope.date_to == today
    assert current.plan.date_scope.is_current is True
    assert current.state == AnswerState.NO_CURRENT_DATA
    assert "Historical validated data" not in current.answer
    assert "75 vessel arrivals" not in current.answer


def test_contextual_follow_ups_merge_vessel_metric_route_and_reject_unsupported_slots(tmp_path) -> None:
    service = _service(tmp_path)
    arrivals = service.query(
        QueryRequest(question="How many arrivals at Gothenburg in March 2022?")
    )

    tanker = service.query(
        QueryRequest(question="And tanker vessels?", conversation_id=arrivals.conversation_id)
    )
    assert tanker.plan.operation == QueryOperation.ARRIVALS
    assert tanker.plan.vessel_type == "tanker"
    assert tanker.applied_scope.ports == ["SEGOT"]
    assert tanker.applied_scope.date_from == "2022-03-01"
    assert tanker.applied_scope.date_to == "2022-03-31"

    dwell = service.query(
        QueryRequest(question="What about dwell time?", conversation_id=arrivals.conversation_id)
    )
    assert dwell.plan.operation == QueryOperation.DWELL_SUMMARY
    assert dwell.plan.metric == "dwell_minutes"
    assert dwell.plan.vessel_type == "tanker"
    assert dwell.applied_scope.ports == ["SEGOT"]
    assert dwell.applied_scope.date_from == "2022-03-01"
    assert dwell.applied_scope.date_to == "2022-03-31"

    unsupported = service.query(
        QueryRequest(question="And crane utilization today?", conversation_id=arrivals.conversation_id)
    )
    assert unsupported.mode == QueryMode.UNSUPPORTED
    assert unsupported.plan.operation == QueryOperation.UNSUPPORTED
    assert unsupported.plan.date_scope.is_current is True
    assert unsupported.state == AnswerState.UNSUPPORTED

    route = service.query(
        QueryRequest(
            question="Show route travel time from Gothenburg to Karlshamn in March 2022."
        )
    )
    changed_route = service.query(
        QueryRequest(
            question="What about the route from Gdansk to Gdynia in April 2022?",
            conversation_id=route.conversation_id,
        )
    )
    assert changed_route.plan.operation == QueryOperation.ROUTE_TRAVEL_TIME
    assert changed_route.applied_scope.origin_port == "PLGDN"
    assert changed_route.applied_scope.destination_port == "PLGDY"
    assert changed_route.applied_scope.date_from == "2022-04-01"
    assert changed_route.applied_scope.date_to == "2022-04-30"


def test_optional_structured_planner_is_never_required_for_fallback() -> None:
    class _Response:
        model = "gpt-5.6-terra-test"
        output_parsed = QueryPlan(
            mode=QueryMode.GENERAL_CHAT,
            operation=QueryOperation.GENERAL_RESPONSE,
            reason="Structured model classification.",
        )

    class _Responses:
        def __init__(self) -> None:
            self.kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return _Response()

    class _Client:
        def __init__(self) -> None:
            self.responses = _Responses()

    client = _Client()
    planner = QueryPlanner(openai_client=client, enable_openai=True)  # type: ignore[arg-type]
    plan = planner.plan("Tell me a joke about lighthouses.")
    assert plan.planner_source == "openai_structured"
    assert plan.planner_model == "gpt-5.6-terra-test"
    assert client.responses.kwargs["reasoning"] == {"effort": "medium"}
    assert client.responses.kwargs["text_format"] is QueryPlan


def test_model_planner_cannot_widen_deterministic_unsupported_or_current_boundaries() -> None:
    class _Response:
        model = "gpt-5.6-terra-test"

        def __init__(self, output_parsed: QueryPlan) -> None:
            self.output_parsed = output_parsed

    class _Responses:
        def __init__(self, output_parsed: QueryPlan) -> None:
            self.output_parsed = output_parsed
            self.calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            return _Response(self.output_parsed)

    class _Client:
        def __init__(self, output_parsed: QueryPlan) -> None:
            self.responses = _Responses(output_parsed)

    adversarial_analytics = QueryPlan(
        mode=QueryMode.ANALYTICS,
        operation=QueryOperation.ARRIVALS,
        metric="arrivals_vessels",
        ports=["Gothenburg"],
        date_scope=DateScope(is_current=False),
        reason="Unsafe model reclassification.",
    )
    adversarial_client = _Client(adversarial_analytics)
    planner = QueryPlanner(openai_client=adversarial_client, enable_openai=True)  # type: ignore[arg-type]

    unsupported = planner.plan("What is crane utilization at berth 3 in SEGOT today?")
    assert unsupported.mode == QueryMode.UNSUPPORTED
    assert unsupported.operation == QueryOperation.UNSUPPORTED
    assert unsupported.date_scope.is_current is True
    assert adversarial_client.responses.calls == 0

    current_general = planner.plan("What is the weather at Gothenburg today?")
    assert current_general.mode == QueryMode.GENERAL_CHAT
    assert current_general.operation == QueryOperation.GENERAL_RESPONSE
    assert current_general.date_scope.is_current is True
    assert adversarial_client.responses.calls == 1

    stale_general = QueryPlan(
        mode=QueryMode.GENERAL_CHAT,
        operation=QueryOperation.GENERAL_RESPONSE,
        date_scope=DateScope(is_current=False),
        reason="Model omitted the current-data marker.",
    )
    general_client = _Client(stale_general)
    general_planner = QueryPlanner(openai_client=general_client, enable_openai=True)  # type: ignore[arg-type]
    preserved = general_planner.plan("What is the weather at Gothenburg today?")
    assert preserved.mode == QueryMode.GENERAL_CHAT
    assert preserved.operation == QueryOperation.GENERAL_RESPONSE
    assert preserved.date_scope.is_current is True
    assert preserved.planner_source == "openai_structured"
