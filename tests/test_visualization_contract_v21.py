from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.chart_analytics import build_chart_insights, enrich_chart_datasets
from src.query.context import ConversationStore
from src.query.models import (
    AnswerState,
    CartesianVisualization,
    QueryMode,
    QueryOperation,
    QueryPlan,
    QueryRequest,
)
from src.query.serialization import dataframe_to_dataset
from src.query.service import QueryService
from src.query.visuals import build_visualizations


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> QueryService:
    root = tmp_path_factory.mktemp("visualization-v21")
    return QueryService(
        kpi=KPIQueryEngine("data/processed"),
        forecaster=ForecastEngine("data/processed"),
        carbon=CarbonQueryEngine("data/processed", auto_build=False),
        conversation_store=ConversationStore(root / "conversation.sqlite3"),
        processed_dir="data/processed",
        export_dir=root / "exports",
    )


def _query(service: QueryService, question: str):
    return service.query(QueryRequest(question=question, top_k_evidence=0))


def _assert_insight_references_reconcile(envelope) -> None:
    fact_names = {fact.name for fact in envelope.facts}
    evidence_ids = {item.id for item in envelope.evidence}
    assert len(envelope.chart_insights) <= 3
    for insight in envelope.chart_insights:
        assert set(insight.fact_names).issubset(fact_names)
        assert set(insight.evidence_ids).issubset(evidence_ids)
        assert any(visual.id == insight.visualization_id for visual in envelope.visualizations)


def test_arrivals_keep_answer_and_add_stable_rows_peak_median_and_rolling_series(
    service: QueryService,
) -> None:
    question = "Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28."
    first = _query(service, question)
    second = _query(service, question)

    assert first.answer == "Matched 58 vessel arrivals across 25 day buckets for LVVNT."
    assert first.answer == second.answer
    assert first.visualization_contract_version == "2.1"
    chart = next(dataset for dataset in first.datasets if dataset.id == "chart")
    repeat = next(dataset for dataset in second.datasets if dataset.id == "chart")
    assert [row["row_id"] for row in chart.rows] == [row["row_id"] for row in repeat.rows]
    assert len(set(row["row_id"] for row in chart.rows)) == chart.row_count
    visual = next(
        item for item in first.visualizations if isinstance(item, CartesianVisualization)
    )
    assert visual.row_id_field == "row_id"
    assert visual.fitted_series[0].method == "rolling_median"
    assert visual.fitted_series[0].y_field == "rolling_median_7"
    assert [item.insight_type for item in first.chart_insights] == ["peak", "trend"]
    _assert_insight_references_reconcile(first)


def test_pressure_has_structured_100_baseline_and_neutral_status_insight(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Show port pressure trend at LVVNT between 2022-02-01 and 2022-02-28.",
    )

    assert (
        envelope.answer
        == "Average port pressure index is 1.33; the highest-pressure day is 2022-02-15 at 2.90."
    )
    visual = next(
        item for item in envelope.visualizations if isinstance(item, CartesianVisualization)
    )
    baseline = visual.reference_lines[0]
    assert baseline.id == "pressure_baseline"
    assert baseline.value == 1.0
    assert baseline.unit == "index"
    assert envelope.chart_insights[0].insight_type == "baseline_deviation"
    assert envelope.chart_insights[0].statement == (
        "Peak port pressure index was 2.90—1.90 above the 1.00 baseline."
    )
    assert next(
        fact.value
        for fact in envelope.facts
        if fact.name == f"chart.{visual.id}.baseline"
    ) == 1.0
    _assert_insight_references_reconcile(envelope)


def test_carbon_central_values_and_valid_uncertainty_publish_directly(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Compare TTW versus WTW CO2e totals at SETRG for March 2022.",
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert "87.57 tCO2e" in envelope.answer
    assert "106.29 tCO2e" in envelope.answer
    assert envelope.datasets
    assert [visual.kind for visual in envelope.visualizations] == ["cartesian"]
    visual = envelope.visualizations[0]
    assert {band.point_field for band in visual.interval_bands} == {
        "ttw_co2e_t",
        "wtw_co2e_t",
    }
    assert [insight.insight_type for insight in envelope.chart_insights] == [
        "boundary_delta",
        "peak",
    ]


def _synthetic_carbon_result(rows: list[dict[str, object]]):
    dataset = dataframe_to_dataset(
        pd.DataFrame(rows),
        dataset_id="table",
        unit_overrides={
            "ttw_co2e_t": "tCO2e",
            "wtw_co2e_t": "tCO2e",
            "ttw_co2e_t_lower": "tCO2e",
            "ttw_co2e_t_upper": "tCO2e",
            "wtw_co2e_t_lower": "tCO2e",
            "wtw_co2e_t_upper": "tCO2e",
        },
    )
    assert dataset is not None
    plan = QueryPlan(
        mode=QueryMode.ANALYTICS,
        operation=QueryOperation.CARBON,
        reason="synthetic carbon contract",
        carbon_boundary="TTW_WTW",
        pollutants=["CO2e"],
        dimensions=["date"],
    )
    datasets = enrich_chart_datasets(plan, [dataset])
    visualizations = build_visualizations(plan, datasets)
    insights, facts = build_chart_insights(
        plan=plan,
        datasets=datasets,
        visualizations=visualizations,
    )
    return datasets, visualizations, insights, facts


def test_carbon_interval_rejects_only_bounds_without_a_finite_central_point() -> None:
    _, visualizations, insights, facts = _synthetic_carbon_result(
        [
            {
                "date": pd.Timestamp("2022-03-01T00:00:00Z"),
                "ttw_co2e_t": None,
                "wtw_co2e_t": 12.0,
                "ttw_co2e_t_lower": 9.0,
                "ttw_co2e_t_upper": 11.0,
                "wtw_co2e_t_lower": 11.0,
                "wtw_co2e_t_upper": 13.0,
            }
        ]
    )

    assert [visual.kind for visual in visualizations] == ["cartesian"]
    visual = visualizations[0]
    assert visual.y_fields == ["ttw_co2e_t", "wtw_co2e_t"]
    assert [band.point_field for band in visual.interval_bands] == ["wtw_co2e_t"]
    assert all(band.point_field != "ttw_co2e_t" for band in visual.interval_bands)
    assert insights == []
    assert facts == []


def test_carbon_boundary_delta_requires_complete_ttw_wtw_row_coverage() -> None:
    _, visualizations, insights, facts = _synthetic_carbon_result(
        [
            {
                "date": pd.Timestamp("2022-03-01T00:00:00Z"),
                "ttw_co2e_t": 10.0,
                "wtw_co2e_t": 12.0,
            },
            {
                "date": pd.Timestamp("2022-03-02T00:00:00Z"),
                "ttw_co2e_t": None,
                "wtw_co2e_t": 15.0,
            },
        ]
    )

    assert [visual.kind for visual in visualizations] == ["cartesian"]
    assert all(insight.insight_type != "boundary_delta" for insight in insights)
    assert all(not fact.name.endswith("boundary_delta") for fact in facts)
    assert all(not fact.name.endswith("ttw_total") for fact in facts)
    assert all(not fact.name.endswith("wtw_total") for fact in facts)


@pytest.mark.parametrize(
    "question",
    [
        (
            "What are call-level emissions for MMSI 209468000 and call_id "
            "209468000_2021-01-06T10-17-56_SETRG?"
        ),
        "Estimate carbon emissions for a tanker in manoeuvring mode for 2 hours at 6 knots.",
        "Show WTW CO2e at SEGVX between 2022-03-01 and 2022-03-31.",
    ],
)
def test_finite_carbon_results_are_stably_published(
    service: QueryService,
    question: str,
) -> None:
    first = _query(service, question)
    repeat = _query(service, question)

    assert first.answer == repeat.answer
    assert first.state == AnswerState.COMPUTED
    assert repeat.state == AnswerState.COMPUTED
    assert first.confidence == "high"
    assert first.assurance is not None
    assert first.assurance.status == "verified"
    assert first.datasets
    assert first.visualizations
    assert all(visual.kind != "omitted" for visual in first.visualizations)
    assert first.caveats == []


def test_daily_carbon_values_publish_without_internal_method_columns(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Show WTW CO2e at SEGVX between 2022-03-01 and 2022-03-31.",
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert envelope.caveats == []
    assert envelope.datasets
    prohibited_fields = {
        "confidence_label",
        "confidence_reason",
        "fallback_usage_ratio",
        "proxy_class",
        "source_kind",
    }
    assert all(
        prohibited_fields.isdisjoint(column.field for column in dataset.columns)
        for dataset in envelope.datasets
    )
    assert all(visual.kind != "omitted" for visual in envelope.visualizations)


def test_distribution_exposes_exact_bins_quartiles_whiskers_and_outliers(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Show the distribution of dwell times at Gothenburg in March 2022.",
    )

    assert (
        envelope.answer
        == "Dwell distribution uses 485 port calls at SEGOT: median 668.3 minutes, "
        "middle 50% 286.1-1220.9 minutes, p90 1969.9 minutes."
    )
    visual = next(item for item in envelope.visualizations if item.kind == "distribution")
    assert visual.bin_lower_field == "bin_start_minutes"
    assert visual.bin_upper_field == "bin_end_minutes"
    assert visual.summary_dataset_id == "distribution_summary"
    assert visual.outlier_dataset_id == "distribution_outliers"
    summary = visual.five_number_summary
    assert summary is not None
    assert summary.count == 485
    assert summary.minimum <= summary.lower_whisker <= summary.q1
    assert summary.q1 <= summary.median <= summary.q3
    assert summary.q3 <= summary.upper_whisker <= summary.maximum
    assert summary.median == pytest.approx(668.3166666666667)
    assert summary.p90 == pytest.approx(1969.8833333333346)
    outliers = next(
        dataset for dataset in envelope.datasets if dataset.id == "distribution_outliers"
    )
    assert outliers.row_count == 26
    assert all(row["is_outlier"] is True for row in outliers.rows)
    _assert_insight_references_reconcile(envelope)


def test_explicit_boxplot_uses_exact_observations_when_they_are_available(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Show a box plot of dwell times at Gothenburg in March 2022.",
    )

    visual = envelope.visualizations[0]
    assert visual.kind == "distribution"
    assert visual.chart_type == "boxplot"
    assert visual.dataset_id == "distribution_observations"
    assert visual.five_number_summary is not None
    assert visual.outlier_dataset_id == "distribution_outliers"
    _assert_insight_references_reconcile(envelope)


def test_correlation_binds_backend_ols_values_and_labels_association(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Is there a correlation between arrivals and dwell time at Gothenburg in March 2022?",
    )

    assert envelope.answer.endswith("This is an association, not evidence of causation.")
    visual = next(
        item for item in envelope.visualizations if isinstance(item, CartesianVisualization)
    )
    fit = visual.fitted_series[0]
    assert fit.method == "ols"
    assert fit.association_only is True
    assert fit.y_field == "ols_fitted_median_dwell_minutes"
    assert fit.slope == pytest.approx(-29.70045312203454)
    assert fit.r_squared == pytest.approx(0.1884837426151822)
    dataset = next(item for item in envelope.datasets if item.id == visual.dataset_id)
    for row in dataset.rows:
        assert row[fit.y_field] == pytest.approx(
            fit.intercept + fit.slope * row[fit.x_field]
        )
    assert "association, not causation" in envelope.chart_insights[0].statement
    _assert_insight_references_reconcile(envelope)


def test_forecast_boundary_and_quality_only_use_passing_metadata(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Forecast vessel arrivals at SEGOT for the next four weeks.",
    )

    assert (
        envelope.answer
        == "Historical-data forecast anchored after 2022-04-30: mean arrivals are "
        "16.64 vessels/day over the following 4 week(s)."
    )
    visual = next(item for item in envelope.visualizations if item.kind == "forecast")
    assert visual.forecast_boundary == "2022-05-01T00:00:00Z"
    assert visual.quality_metrics is not None
    assert visual.quality_metrics.mase == pytest.approx(0.920)
    assert visual.quality_metrics.interval_coverage == pytest.approx(0.857)
    assert visual.quality_metrics.gate_passed is True
    assert envelope.chart_insights[0].insight_type == "forecast_quality"
    _assert_insight_references_reconcile(envelope)


def test_arrival_anomaly_rows_and_threshold_view_publish_directly(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Show arrival anomalies at Gothenburg in March 2022.",
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert envelope.datasets
    assert [visual.kind for visual in envelope.visualizations] == ["cartesian"]
    assert envelope.chart_insights[0].insight_type == "threshold_exceedance"
    assert "2 flagged event(s)" in envelope.chart_insights[0].statement


def test_ranking_and_composition_both_publish_their_validated_insights(
    service: QueryService,
) -> None:
    ranking = _query(service, "Rank the top 5 ports by arrivals in March 2022.")
    composition = _query(
        service,
        "What share of arrivals by vessel type at SEGOT in March 2022?",
    )

    assert ranking.state == AnswerState.COMPUTED
    assert ranking.datasets
    assert [visual.kind for visual in ranking.visualizations] == ["cartesian"]
    assert ranking.chart_insights[0].insight_type == "ranking_margin"
    composition_visual = next(
        item for item in composition.visualizations if isinstance(item, CartesianVisualization)
    )
    assert composition_visual.chart_type == "stacked_bar"
    assert composition_visual.y_fields == ["arrival_count"]
    assert composition_visual.y_unit == "count"
    assert composition.chart_insights[0].insight_type == "dominant_share"
    assert "51.84 percent" in composition.chart_insights[0].statement
    _assert_insight_references_reconcile(composition)


def test_ais_map_uses_only_explicit_provenance_backed_segments() -> None:
    dataset = dataframe_to_dataset(
        pd.DataFrame(
            {
                "stable_id": ["event-1"],
                "mmsi": ["265650000"],
                "timestamp_full": [pd.Timestamp("2022-03-01T10:00:00Z")],
                "prev_latitude": [57.6],
                "prev_longitude": [11.8],
                "latitude": [57.8],
                "longitude": [12.1],
                "distance_km": [28.0],
            }
        ),
        dataset_id="table",
    )
    assert dataset is not None
    plan = QueryPlan(
        mode=QueryMode.ANALYTICS,
        operation=QueryOperation.AIS_JUMP,
        reason="segment contract",
    )
    datasets = enrich_chart_datasets(plan, [dataset])
    visual = build_visualizations(plan, datasets)[0]

    assert visual.kind == "map"
    assert visual.geometry_mode == "segments"
    assert visual.start_latitude_field == "prev_latitude"
    assert visual.start_longitude_field == "prev_longitude"
    assert visual.end_latitude_field == "latitude"
    assert visual.end_longitude_field == "longitude"
    assert visual.timestamp_field == "timestamp_full"


def test_trace_records_v21_chart_profile_without_client_only_metrics(
    service: QueryService,
) -> None:
    envelope = _query(
        service,
        "Show port pressure trend at LVVNT between 2022-02-01 and 2022-02-28.",
    )

    assert envelope.trace.visualization_contract_version == "2.1"
    assert envelope.trace.chart_profile == ["cartesian:line"]
    assert envelope.trace.visualization_dataset_ids == ["chart"]
    assert envelope.trace.visualization_fallback_reasons == []
    _assert_insight_references_reconcile(envelope)
