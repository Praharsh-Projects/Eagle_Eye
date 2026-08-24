from __future__ import annotations

from pathlib import Path
import re

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import AnswerState, QueryOperation, QueryRequest
from src.query.service import QueryService


def _service(tmp_path: Path) -> QueryService:
    processed = Path("data/processed")
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "assurance.sqlite3"),
        events_path=processed / "events.parquet",
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


PROHIBITED_PUBLIC_METHOD_LABELS = re.compile(
    r"\b(?:assurance|confidence|caveat|proxy|heuristic|reconstruct(?:ed|ion)?|"
    r"estimated|fallback|publication[- ]gate|partial result)\b",
    re.IGNORECASE,
)


def _public_result_text(envelope) -> str:
    """Collect presentation copy while deliberately excluding the submitted question."""

    parts = [envelope.answer, *envelope.caveats]
    parts.extend(insight.statement for insight in envelope.chart_insights)
    for visual in envelope.visualizations:
        for field in ("title", "accessible_summary", "reason", "omission_reason"):
            value = getattr(visual, field, None)
            if value:
                parts.append(str(value))
    for dataset in envelope.datasets:
        for column in dataset.columns:
            parts.extend((column.field, column.label))
    return "\n".join(str(part) for part in parts if part)


def _assert_direct_publication(envelope) -> None:
    assert envelope.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}
    assert envelope.confidence == "high"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert envelope.assurance.level == "high"
    assert envelope.availability is not None
    assert envelope.availability.code == "available"
    assert envelope.datasets
    assert all(dataset.row_count > 0 for dataset in envelope.datasets)
    assert envelope.visualizations
    assert all(visual.kind != "omitted" for visual in envelope.visualizations)
    assert envelope.caveats == []
    assert PROHIBITED_PUBLIC_METHOD_LABELS.search(_public_result_text(envelope)) is None


def test_direct_historical_rows_are_verified_independently_of_legacy_label(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question="Show distribution of vessel dwell times at Gothenburg in March 2022.",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert envelope.assurance.level == "high"
    assert "structured_rows=true" in envelope.assurance.checks
    _assert_direct_publication(envelope)


def test_ranked_historical_rows_are_published_directly(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question="Rank the top 5 ports by arrivals in March 2022.",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.plan.operation == QueryOperation.TOP_PORTS
    assert envelope.answer == "Top 5 ports by arrivals were computed for the selected filters."
    _assert_direct_publication(envelope)


def test_finite_forecast_is_published_with_its_available_interval(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question="Forecast vessel arrivals at SEGOT for the next four weeks.",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert envelope.assurance.basis == "validated_model"
    assert "forecast_rows_and_finite_values=passed" in envelope.assurance.checks
    assert any(dataset.id == "table" and dataset.row_count == 28 for dataset in envelope.datasets)
    _assert_direct_publication(envelope)


def test_carbon_finite_central_values_publish_directly(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question="Compare TTW versus WTW CO2e totals at SETRG for March 2022.",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.plan.operation == QueryOperation.CARBON
    assert "87.57 tCO2e" in envelope.answer
    assert "106.29 tCO2e" in envelope.answer
    _assert_direct_publication(envelope)


def test_route_distribution_release_regression_is_fully_published(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question=(
                "What is median and p90 route travel time from PLSZZ to PLSWI in "
                "2021-02?"
            ),
            top_k_evidence=0,
        )
    )

    assert envelope.plan.operation == QueryOperation.ROUTE_TRAVEL_TIME
    assert envelope.answer == (
        "Route travel time for PLSZZ→PLSWI: median=2.76 h, p90=4.35 h over "
        "56 voyage(s)."
    )
    observations = next(
        dataset for dataset in envelope.datasets if dataset.id == "distribution_observations"
    )
    summary = next(dataset for dataset in envelope.datasets if dataset.id == "distribution_summary")
    assert observations.row_count == 56
    assert round(summary.rows[0]["median"], 2) == 2.76
    assert round(summary.rows[0]["p90"], 2) == 4.35
    assert summary.rows[0]["count"] == 56
    assert [visual.kind for visual in envelope.visualizations] == ["distribution"]
    _assert_direct_publication(envelope)


def test_pressure_result_and_all_valid_charts_publish(tmp_path: Path) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question="Is pressure at SEGVX above baseline on 2022-03-10?",
            top_k_evidence=0,
        )
    )

    assert envelope.plan.operation == QueryOperation.CONGESTION
    assert envelope.answer == (
        "The port pressure index at SEGVX on 2022-03-10 is 0.99, which is below "
        "the 1.00 historical baseline."
    )
    assert [visual.kind for visual in envelope.visualizations] == ["kpi", "cartesian"]
    _assert_direct_publication(envelope)


def test_mixed_result_publishes_available_components_without_policy_copy(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question=(
                "Compare arrivals at PLSZZ and PLSWI and route durations from PLSZZ "
                "to PLSWI and PLSWI to SEMMA in 2021-02."
            ),
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.PARTIAL
    assert envelope.plan.operation == QueryOperation.MIXED_PORT_ROUTE_COMPARISON
    assert envelope.answer == (
        "Computed 2 requested port scope(s) and 1 requested route scope(s). "
        "No matching rows were available for routes: PLSWI->SEMMA."
    )
    assert {dataset.id for dataset in envelope.datasets} >= {
            "metric_arrival_count",
        "metric_route_duration_median_h",
        "metric_route_duration_p90_h",
    }
    assert len(envelope.visualizations) == 3
    _assert_direct_publication(envelope)


def test_genuine_no_data_remains_unavailable_and_graph_free(tmp_path: Path) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question="Show suspicious AIS jumps for MMSI 246521000 on 2022-03-10.",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.NO_DATA
    assert envelope.confidence == "not_applicable"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "unavailable"
    assert envelope.datasets == []
    assert [visual.kind for visual in envelope.visualizations] == ["omitted"]


def test_help_and_current_unavailable_states_are_not_low_confidence(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    help_result = service.query(
        QueryRequest(question="Will you explain this app?", top_k_evidence=0)
    )
    current_result = service.query(
        QueryRequest(
            question="What is the weather in Gothenburg today?",
            top_k_evidence=0,
        )
    )

    assert help_result.confidence == "not_applicable"
    assert help_result.assurance is not None
    assert help_result.assurance.status == "not_applicable"
    assert current_result.state == AnswerState.NO_CURRENT_DATA
    assert current_result.confidence == "not_applicable"
    assert current_result.assurance is not None
    assert current_result.assurance.status == "unavailable"
    assert current_result.availability is not None
    assert current_result.availability.code in {"source_stale", "source_unavailable"}
