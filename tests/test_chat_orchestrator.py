from __future__ import annotations

from src.carbon.query import (
    CARBON_STATE_COMPUTED,
    CARBON_STATE_NOT_COMPUTABLE,
    CARBON_STATE_RETRIEVAL_ONLY,
    CarbonResult,
)
from src.chat.orchestrator import (
    CHAT_STATE_NO_DATA,
    CHAT_STATE_RETRIEVAL_ONLY,
    ChatOrchestrator,
    derive_chat_contract,
)
from src.kpi.query import AnalyticsResult
from src.forecast.forecast import ForecastResult
from src.qa.intent import classify_question


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text})()


class _FakeOpenAI:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


def test_derive_chat_contract_for_computed_analytics() -> None:
    result = AnalyticsResult(
        status="ok",
        answer="Deterministic answer",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=[],
    )
    state, source = derive_chat_contract(result, evidence_rows=[])
    assert state == "COMPUTED"
    assert source == "Computed"


def test_derive_chat_contract_for_no_data_with_retrieval_rows() -> None:
    result = AnalyticsResult(
        status="no_data",
        answer="No deterministic rows",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=[],
    )
    state, source = derive_chat_contract(result, evidence_rows=[{"vector_id": "x"}])
    assert state == CHAT_STATE_RETRIEVAL_ONLY
    assert source == "Retrieved"


def test_derive_chat_contract_for_carbon_states() -> None:
    result = CarbonResult(
        status="no_data",
        answer="missing",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=[],
        boundary="TTW",
        pollutants=["CO2e"],
        source_label="Computed from deterministic inventory",
        confidence_label="low",
        confidence_reason="missing",
        uncertainty_interval={},
        params_version="v1",
        evidence_ids=[],
        segment_ids=[],
        result_state=CARBON_STATE_NOT_COMPUTABLE,
        diagnostics={},
    )
    state, source = derive_chat_contract(result, evidence_rows=[])
    assert state == CHAT_STATE_NO_DATA
    assert source == "Retrieved"

    result.result_state = CARBON_STATE_RETRIEVAL_ONLY
    state, source = derive_chat_contract(result, evidence_rows=[{"vector_id": "a"}])
    assert state == CHAT_STATE_RETRIEVAL_ONLY
    assert source == "Retrieved"

    result.result_state = CARBON_STATE_COMPUTED
    state, source = derive_chat_contract(result, evidence_rows=[])
    assert state == CARBON_STATE_COMPUTED
    assert source == "Computed"


def test_derive_chat_contract_labels_forecasts_as_forecast_only() -> None:
    result = ForecastResult(
        status="ok",
        answer="Forecasted port pressure is 2.10.",
        history=None,
        forecast=None,
        coverage_notes=[],
        caveats=[],
    )

    state, source = derive_chat_contract(result, evidence_rows=[])

    assert state == "FORECAST_ONLY"
    assert source == "Estimated"


def test_run_turn_without_openai_uses_deterministic_answer() -> None:
    orchestrator = ChatOrchestrator(model="gpt-4o-mini", max_history_turns=4, openai_client=None)
    result = AnalyticsResult(
        status="ok",
        answer="Matched 10 arrivals.",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=["Assumption: deterministic subset."],
    )
    intent = classify_question("How many arrivals at SEGOT in March 2022?")
    turn = orchestrator.run_turn(
        question="How many arrivals at SEGOT in March 2022?",
        intent=intent,
        result=result,
        evidence_lines=["date=2022-03-01 | arrivals=4"],
        evidence_rows=[],
        tool_trace={"retrieval_status": "computed_only"},
        conversation_id="chat_test",
        history=[],
    )
    assert turn.conversation_id == "chat_test"
    assert turn.answer == "Matched 10 arrivals."
    assert turn.result_state == "COMPUTED"
    assert turn.source_type == "Computed"
    assert turn.evidence_lines


def test_responses_synthesis_returns_one_valid_conversational_answer() -> None:
    client = _FakeOpenAI("Gothenburg recorded 473 vessel arrivals across 31 daily buckets in March 2022.")
    orchestrator = ChatOrchestrator(model="gpt-test", openai_client=client)
    result = AnalyticsResult(
        status="ok",
        answer="Matched 473 vessel arrivals across 31 day buckets for SEGOT.",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=[],
    )
    turn = orchestrator.run_turn(
        question="How many arrivals were recorded at Gothenburg in March 2022?",
        intent=classify_question("How many arrivals were recorded at Gothenburg in March 2022?"),
        result=result,
        evidence_lines=[],
        evidence_rows=[],
        tool_trace={},
    )
    assert turn.answer == client.responses.output_text
    assert len(client.responses.calls) == 1
    assert "chat.completions" not in str(client.responses.calls[0])


def test_responses_synthesis_may_omit_secondary_diagnostic_number() -> None:
    client = _FakeOpenAI("Gothenburg recorded 473 vessel arrivals in March 2022.")
    orchestrator = ChatOrchestrator(model="gpt-test", openai_client=client)
    result = AnalyticsResult(
        status="ok",
        answer="Matched 473 vessel arrivals across 31 day buckets for SEGOT.",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=[],
    )
    question = "How many arrivals were recorded at Gothenburg in March 2022?"
    turn = orchestrator.run_turn(
        question=question,
        intent=classify_question(question),
        result=result,
        evidence_lines=[],
        evidence_rows=[],
        tool_trace={},
    )
    assert turn.answer == client.responses.output_text


def test_responses_synthesis_rejects_unsupported_numbers() -> None:
    client = _FakeOpenAI("Gothenburg recorded 999 vessel arrivals across 31 daily buckets in March 2022.")
    orchestrator = ChatOrchestrator(model="gpt-test", openai_client=client)
    result = AnalyticsResult(
        status="ok",
        answer="Matched 473 vessel arrivals across 31 day buckets for SEGOT.",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=[],
    )
    turn = orchestrator.run_turn(
        question="How many arrivals were recorded at Gothenburg in March 2022?",
        intent=classify_question("How many arrivals were recorded at Gothenburg in March 2022?"),
        result=result,
        evidence_lines=[],
        evidence_rows=[],
        tool_trace={},
    )
    assert turn.answer == result.answer
    assert "999" not in turn.answer


def test_unsupported_fallback_explains_boundary_naturally() -> None:
    orchestrator = ChatOrchestrator(model="gpt-test", openai_client=None)
    result = AnalyticsResult(
        status="unsupported",
        answer="I don't have evidence in the dataset to answer that.",
        table=None,
        chart=None,
        coverage_notes=[],
        caveats=["Crane utilization requires terminal operating-system data."],
    )
    turn = orchestrator.run_turn(
        question="What is crane utilization now?",
        intent=classify_question("What is crane utilization now?"),
        result=result,
        evidence_lines=[],
        evidence_rows=[],
        tool_trace={},
    )
    assert "can't answer" in turn.answer
    assert "terminal operating-system data" in turn.answer
