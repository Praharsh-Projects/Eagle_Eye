from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine, _with_arrival_semantics
from src.query.context import ConversationStore
from src.query.models import AnswerState, QueryMode, QueryOperation, QueryRequest
from src.query.planner import QueryPlanner
from src.query.service import QueryService


def _service(tmp_path: Path) -> QueryService:
    processed = "data/processed"
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "measurement-contract.sqlite3"),
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def _fact(envelope, name: str):
    return next(fact.value for fact in envelope.facts if fact.name == name)


def test_same_mmsi_two_calls_on_one_day_counts_two_arrival_events() -> None:
    engine = KPIQueryEngine("data/processed")
    engine._arrivals_daily = _with_arrival_semantics(
        pd.DataFrame(
            [
                {
                    "source_kind": "port_call",
                    "port_key": "SEGOT",
                    "port_label": "Göteborg (SEGOT)",
                    "locode_norm": "SEGOT",
                    "port_name_norm": "Göteborg",
                    "date": pd.Timestamp("2022-03-01", tz="UTC"),
                    "vessel_type_norm": "cargo ship",
                    "arrivals_vessels": 1,
                    "arrivals_events": 2,
                }
            ]
        )
    )

    result = engine.get_arrivals("SEGOT", "2022-03-01", "2022-03-01")

    assert result.status == "ok"
    assert result.table is not None
    assert int(result.table["arrival_count"].sum()) == 2
    assert int(result.table["daily_distinct_vessels"].sum()) == 1
    assert "Matched 2 vessel arrivals" in result.answer


def test_development_structured_contract_values_and_semantic_facts(tmp_path: Path) -> None:
    service = _service(tmp_path)

    arrivals = service.query(
        QueryRequest(
            question="How many vessel arrivals were recorded at Gothenburg (SEGOT) in March 2022?",
            top_k_evidence=0,
        )
    )
    assert arrivals.state == AnswerState.COMPUTED
    assert arrivals.plan.operation == QueryOperation.ARRIVALS
    assert arrivals.applied_scope.ports == ["SEGOT"]
    assert _fact(arrivals, "arrival_count") == 488
    assert arrivals.visualizations[0].kind == "kpi"
    assert arrivals.visualizations[0].value_field == "arrival_count"

    comparison = service.query(
        QueryRequest(
            question="Compare arrivals at Gothenburg and Södertälje during March 2022.",
            top_k_evidence=0,
        )
    )
    assert _fact(comparison, "port_arrival_counts") == [
        {"port_locode": "SEGOT", "arrival_count": 488},
        {"port_locode": "SESOE", "arrival_count": 176},
    ]
    assert _fact(comparison, "winner_port_locodes") == ["SEGOT"]
    assert _fact(comparison, "absolute_margin") == 312


def test_country_ranking_composition_and_dwell_contract(tmp_path: Path) -> None:
    service = _service(tmp_path)

    ranking = service.query(
        QueryRequest(
            question="Which five Swedish ports recorded the most arrivals in March 2022?",
            top_k_evidence=0,
        )
    )
    assert ranking.plan.operation == QueryOperation.TOP_PORTS
    assert ranking.applied_scope.country_codes == ["SE"]
    assert _fact(ranking, "port_ranking") == [
        {"port_locode": "SEGOT", "port_name": "Göteborg", "arrival_count": 488, "rank": 1},
        {"port_locode": "SESOE", "port_name": "Södertälje", "arrival_count": 176, "rank": 2},
        {"port_locode": "SEBRO", "port_name": "Brofjorden", "arrival_count": 95, "rank": 3},
        {"port_locode": "SEHEL", "port_name": "Helsingborg", "arrival_count": 90, "rank": 4},
        {"port_locode": "SENRK", "port_name": "Norrköping", "arrival_count": 79, "rank": 5},
    ]

    composition = service.query(
        QueryRequest(
            question="How were Gothenburg arrivals divided between cargo ships and tankers in March 2022?",
            top_k_evidence=0,
        )
    )
    assert composition.applied_scope.ports == ["SEGOT"]
    assert _fact(composition, "vessel_type_arrival_counts") == {"cargo": 253, "tanker": 235}
    assert composition.visualizations[0].y_fields == ["arrival_count"]

    mean_dwell = service.query(
        QueryRequest(
            question="What was the mean completed dwell time at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    assert _fact(mean_dwell, "complete_dwell_count") == 485
    assert abs(_fact(mean_dwell, "mean_dwell_hours") - 14.975558419243987) < 1e-12
    assert mean_dwell.visualizations[0].kind == "kpi"

    median_dwell = service.query(
        QueryRequest(
            question="What was the median completed dwell time at Gothenburg in March 2022?",
            top_k_evidence=0,
        )
    )
    assert abs(_fact(median_dwell, "median_dwell_hours") - 11.138611111111112) < 1e-12
    assert median_dwell.visualizations[0].kind == "distribution"


def test_replacement_development_cases_use_supported_arrival_events(tmp_path: Path) -> None:
    service = _service(tmp_path)

    tanker_total = service.query(
        QueryRequest(
            question=(
                "How many recorded arrival events involved tankers at Gothenburg "
                "(SEGOT) in March 2022?"
            ),
            top_k_evidence=0,
        )
    )
    assert tanker_total.plan.operation == QueryOperation.ARRIVALS
    assert _fact(tanker_total, "arrival_count") == 235
    assert tanker_total.visualizations[0].kind == "kpi"

    peak = service.query(
        QueryRequest(
            question=(
                "Which date had the highest recorded arrival-event count at "
                "Gothenburg (SEGOT) in March 2022?"
            ),
            top_k_evidence=0,
        )
    )
    assert peak.plan.operation == QueryOperation.PEAK_ARRIVAL_DAY
    assert _fact(peak, "peak_arrival_count") == 22
    assert _fact(peak, "peak_dates_utc") == ["2022-03-29"]
    assert _fact(peak, "peak_arrival_date") == "2022-03-29"
    assert peak.visualizations[0].kind == "cartesian"
    assert peak.visualizations[0].y_fields == ["arrival_count"]


def test_ambiguity_current_and_documentary_routes(tmp_path: Path) -> None:
    service = _service(tmp_path)

    missing_geo = service.query(
        QueryRequest(question="How many arrivals were there in March 2022?", top_k_evidence=0)
    )
    missing_dwell_scope = service.query(
        QueryRequest(question="What was the dwell time at Gothenburg?", top_k_evidence=0)
    )
    current_arrivals = service.query(
        QueryRequest(question="How many vessels are arriving at Gothenburg today?", top_k_evidence=0)
    )
    current_positions = service.query(
        QueryRequest(question="Where are cargo vessels located right now?", top_k_evidence=0)
    )

    assert missing_geo.state == AnswerState.CLARIFICATION_REQUIRED
    assert missing_dwell_scope.state == AnswerState.CLARIFICATION_REQUIRED
    assert current_arrivals.state == AnswerState.NO_CURRENT_DATA
    assert current_arrivals.plan.operation == QueryOperation.CURRENT_ARRIVALS
    assert current_positions.state == AnswerState.NO_CURRENT_DATA
    assert current_positions.plan.operation == QueryOperation.CURRENT_POSITIONS
    assert all(visual.kind == "omitted" for visual in current_positions.visualizations)

    planner = QueryPlanner()
    for question in (
        "What safety purpose does IMO assign to automatic identification systems?",
        "Why must AIS information not replace prudent navigation or other available information?",
        "Does an AIS feed provide a complete official port-arrival board and authoritative scheduled ETAs?",
    ):
        plan = planner.plan(question)
        assert plan.mode == QueryMode.MARITIME_RESEARCH
        assert plan.operation == QueryOperation.RESEARCH
