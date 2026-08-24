from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import AnswerEnvelope, QueryRequest
from src.query.service import QueryService


GOLD_PATH = Path("evaluation/gold/query_regressions_v4.json")


@pytest.fixture(scope="module")
def gold() -> dict[str, Any]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> QueryService:
    runtime = tmp_path_factory.mktemp("query-regressions-v4")
    processed = Path("data/processed")
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(runtime / "conversations.sqlite3"),
        processed_dir=processed,
        export_dir=runtime / "exports",
    )


def _scope_ports(envelope: AnswerEnvelope) -> list[str]:
    if envelope.applied_scope.ports:
        return envelope.applied_scope.ports
    return [
        value
        for value in (envelope.applied_scope.origin_port, envelope.applied_scope.destination_port)
        if value
    ]


def _assert_required_fact(envelope: AnswerEnvelope, name: str, expected: Any) -> None:
    if name == "winner":
        assert str(expected).lower() in envelope.answer.lower()
        return
    if name == "day_buckets":
        table = next(dataset for dataset in envelope.datasets if dataset.id == "table")
        assert table.row_count == expected
        return

    # Aggregate fields such as arrival_count are authoritative table values.
    table = next((dataset for dataset in envelope.datasets if dataset.id == "table"), None)
    if table is not None and any(column.field == name for column in table.columns):
        values = [row.get(name) for row in table.rows]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if numeric and math.isclose(sum(numeric), float(expected), rel_tol=0, abs_tol=1e-9):
            return

    # Entity-labelled comparison/composition facts must exist in one row, not
    # merely somewhere in prose.
    entity = str(name).lower()
    for dataset in envelope.datasets:
        for row in dataset.rows:
            labels = {str(value).lower() for value in row.values() if isinstance(value, str)}
            numbers = [float(value) for value in row.values() if isinstance(value, (int, float))]
            if entity in labels and any(
                math.isclose(value, float(expected), rel_tol=0, abs_tol=1e-9) for value in numbers
            ):
                return
    pytest.fail(f"Missing immutable oracle {name}={expected!r} in {envelope.turn_id}")


def _assert_case(envelope: AnswerEnvelope, expected: dict[str, Any]) -> None:
    assert envelope.mode.value == expected["mode"]
    assert envelope.plan.operation.value == expected["operation"]
    if "ports" in expected:
        assert _scope_ports(envelope) == expected["ports"]
    if "mmsi" in expected:
        assert envelope.applied_scope.mmsi == expected["mmsi"]
    if "date_from" in expected:
        assert envelope.applied_scope.date_from == expected["date_from"]
    if "date_to" in expected:
        assert envelope.applied_scope.date_to == expected["date_to"]
    if "result_state" in expected:
        assert envelope.state.value == expected["result_state"]

    visual_expected = expected["visualization"]
    assert envelope.visualizations, "Every response needs a visual or explicit omission."
    visual = envelope.visualizations[0]
    assert visual.kind == visual_expected["kind"]
    if "chart_type" in visual_expected:
        assert getattr(visual, "chart_type", None) == visual_expected["chart_type"]
    if "reason" in visual_expected:
        assert getattr(visual, "reason_code", None) == visual_expected["reason"]

    for name, value in expected.get("required_facts", {}).items():
        _assert_required_fact(envelope, name, value)
    for term in expected.get("answer_terms", []):
        assert str(term).lower() in envelope.answer.lower()
    caveat_text = " ".join(envelope.caveats).lower()
    for term in expected.get("caveat_terms", []):
        assert str(term).lower() in caveat_text

    # The final public object must always be standards-compliant JSON.
    json.dumps(envelope.model_dump(mode="json"), allow_nan=False)


def test_frozen_human_authored_query_catalog(
    service: QueryService,
    gold: dict[str, Any],
) -> None:
    assert gold["authorship"].startswith("Human-authored expectations")
    for case in gold["cases"]:
        prompts = [case["prompt"], *case.get("paraphrases", [])]
        for prompt_index, prompt in enumerate(prompts):
            envelope = service.query(
                QueryRequest(
                    question=prompt,
                    conversation_id=f"gold_{case['id']}_{prompt_index}",
                )
            )
            _assert_case(envelope, case["expected"])


def test_frozen_context_follow_ups(
    service: QueryService,
    gold: dict[str, Any],
) -> None:
    for conversation in gold["conversation_cases"]:
        conversation_id = f"gold_conversation_{conversation['id']}"
        for turn in conversation["turns"]:
            envelope = service.query(
                QueryRequest(question=turn["prompt"], conversation_id=conversation_id)
            )
            expected = turn["expected"]
            assert envelope.mode.value == expected["mode"]
            assert envelope.plan.operation.value == expected["operation"]
            if "ports" in expected:
                assert _scope_ports(envelope) == expected["ports"]
            if "date_from" in expected:
                assert envelope.applied_scope.date_from == expected["date_from"]
            if "date_to" in expected:
                assert envelope.applied_scope.date_to == expected["date_to"]


def test_deterministic_queries_do_not_create_exports(service: QueryService) -> None:
    assert not service.export_dir.exists()
    prompts = [
        "How many arrivals at Gothenburg in March 2022?",
        "Show the distribution of vessel dwell times at Gothenburg in March 2022.",
        "Estimate Carbon Emissions for Gothenburg in March 2022.",
    ]
    for index, prompt in enumerate(prompts):
        service.query(QueryRequest(question=prompt, conversation_id=f"read_only_{index}"))
    assert not service.export_dir.exists(), "Queries may not write export artifacts."


def test_immutable_fact_slots_preserve_scope_state_units_and_winner(service: QueryService) -> None:
    envelope = service.query(
        QueryRequest(
            question="Is Monday busier than Friday at Gothenburg in March 2022?",
            conversation_id="immutable_fact_slots",
        )
    )
    by_name = {fact.name: fact for fact in envelope.facts}
    assert by_name["result_state"].value == "COMPUTED"
    assert by_name["operation"].value == "weekday_comparison"
    assert by_name["ports_1"].value == "SEGOT"
    assert by_name["date_from"].value == "2022-03-01"
    assert by_name["date_to"].value == "2022-03-31"
    assert str(by_name["comparison_winner"].value).lower() == "friday"
    assert by_name["comparison_polarity"].value == "higher"
    assert all(fact.immutable for fact in envelope.facts)
    assert any(fact.value == 56 and fact.unit == "arrivals" for fact in envelope.facts)
    assert envelope.trace.model == "deterministic"
    assert envelope.trace.sources == ["structured_datasets"]
    assert envelope.trace.result_state == "COMPUTED"
    assert envelope.trace.failure_state is None
    assert len(envelope.trace.result_hash) == 64


def test_deterministic_local_p95_is_at_most_two_seconds(service: QueryService) -> None:
    prompts = [
        "How many arrivals at Gothenburg in March 2022?",
        "Plot daily arrivals at Gothenburg for March 2022.",
        "Is Monday busier than Friday at Gothenburg in March 2022?",
        "Which port had the most arrivals, Gothenburg or Karlshamn, in March 2022?",
        "Show the share of arrivals by vessel type at Gothenburg in March 2022.",
        "Show the distribution of vessel dwell times at Gothenburg in March 2022.",
    ]
    latencies: list[float] = []
    for index, prompt in enumerate(prompts):
        started = time.perf_counter()
        service.query(QueryRequest(question=prompt, conversation_id=f"latency_{index}"))
        latencies.append(time.perf_counter() - started)
    p95 = statistics.quantiles(latencies, n=100, method="inclusive")[94]
    assert p95 <= 2.0, f"local deterministic p95 was {p95:.3f}s"
