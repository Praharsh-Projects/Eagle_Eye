from __future__ import annotations

import pandas as pd
import pytest

from src.carbon.query import CARBON_STATE_COMPUTED, CarbonQueryEngine, CarbonResult
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import (
    AnswerState,
    QueryRequest,
    QueryMode,
    QueryOperation,
    QueryPlan,
    VisualizationIntent,
)
from src.query.planner import QueryPlanner
from src.query.serialization import dataframe_to_dataset
from src.query.service import QueryService
from src.query.visuals import build_visualizations


def _plan(
    operation: QueryOperation,
    *,
    requested_visual: VisualizationIntent = VisualizationIntent.AUTO,
    dimensions: list[str] | None = None,
    pollutants: list[str] | None = None,
    carbon_boundary: str = "TTW",
) -> QueryPlan:
    return QueryPlan(
        mode=QueryMode.ANALYTICS,
        operation=operation,
        requested_visual=requested_visual,
        dimensions=dimensions or [],
        pollutants=pollutants or [],
        carbon_boundary=carbon_boundary,
        reason="visualization contract regression",
    )


def _service_shell() -> QueryService:
    # _datasets is a deterministic serializer and does not depend on initialized
    # engines or stores. Keeping this unit-level avoids filesystem side effects.
    return object.__new__(QueryService)


def _canonical_service(tmp_path) -> QueryService:
    processed = "data/processed"
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "visualization-contract.sqlite3"),
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def _carbon_result(table: pd.DataFrame, chart: pd.DataFrame, pollutants: list[str]) -> CarbonResult:
    intervals = {
        pollutant: {"point": 3.0, "lower": 2.4, "upper": 3.6}
        for pollutant in pollutants
    }
    return CarbonResult(
        status="ok",
        answer="Computed carbon emissions.",
        table=table,
        chart=chart,
        coverage_notes=[],
        caveats=[],
        boundary="TTW",
        pollutants=pollutants,
        source_label="test inventory",
        confidence_label="high",
        confidence_reason="contract fixture",
        uncertainty_interval=intervals,
        params_version="test",
        evidence_ids=[],
        segment_ids=[],
        result_state=CARBON_STATE_COMPUTED,
    )


def test_explicit_boxplot_is_not_silently_rendered_as_histogram() -> None:
    plan = QueryPlanner().plan("Show a box plot of dwell times at Gothenburg in March 2022.")
    assert plan.operation == QueryOperation.DWELL_DISTRIBUTION
    assert plan.requested_visual == VisualizationIntent.BOXPLOT

    chart = dataframe_to_dataset(
        pd.DataFrame({"bin_midpoint_minutes": [10.0, 20.0], "calls": [4, 7]}),
        dataset_id="chart",
    )
    assert chart is not None
    visual = build_visualizations(plan, [chart])[0]
    assert visual.kind == "omitted"
    assert visual.reason_code == "unsupported_visual"
    assert "raw observations or exact quartiles" in visual.reason


def test_carbon_trend_uses_dated_metric_and_carbon_units() -> None:
    table = pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-02-01", tz="UTC")],
            "ttw_co2e_t": [1.0, 2.0],
        }
    )
    result = _carbon_result(table, table.set_index("date")[["ttw_co2e_t"]], ["CO2e"])
    plan = _plan(
        QueryOperation.CARBON,
        requested_visual=VisualizationIntent.LINE,
        dimensions=["date"],
        pollutants=["CO2e"],
    )
    datasets = _service_shell()._datasets(plan, result)
    visual = build_visualizations(plan, datasets)[0]

    assert visual.kind == "cartesian"
    assert visual.chart_type == "line"
    assert visual.x_field == "date"
    assert visual.y_fields == ["ttw_co2e_t"]
    assert visual.y_unit == "tCO2e"
    summary = next(dataset for dataset in datasets if dataset.id == "summary")
    assert next(column for column in summary.columns if column.field == "value").unit == "tCO2e"


def test_monthly_carbon_prompt_plans_a_time_dimension() -> None:
    plan = QueryPlanner().plan("Show the monthly carbon emissions trend for Gothenburg in 2022.")
    assert plan.operation == QueryOperation.CARBON
    assert plan.requested_visual == VisualizationIntent.LINE
    assert plan.dimensions == ["date"]


def test_exact_daily_arrival_prompt_plans_and_renders_a_time_series(tmp_path) -> None:
    envelope = _canonical_service(tmp_path).query(
        QueryRequest(
            question="Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28."
        )
    )
    assert envelope.plan.operation == QueryOperation.ARRIVALS
    assert envelope.plan.dimensions == ["date"]
    assert len(envelope.visualizations) == 1
    visual = envelope.visualizations[0]
    assert visual.kind == "cartesian"
    assert visual.chart_type in {"line", "area"}
    assert visual.x_field == "date"
    assert visual.y_fields == ["arrival_count"]
    assert visual.y_unit == "count"


def test_exact_pressure_trend_prompt_renders_a_line_not_a_table(tmp_path) -> None:
    envelope = _canonical_service(tmp_path).query(
        QueryRequest(
            question="Show port pressure trend at LVVNT between 2022-02-01 and 2022-02-28."
        )
    )
    assert envelope.plan.operation == QueryOperation.CONGESTION
    assert envelope.plan.dimensions == ["date"]
    assert len(envelope.visualizations) == 1
    visual = envelope.visualizations[0]
    assert visual.kind == "cartesian"
    assert visual.chart_type == "line"
    assert visual.x_field == "date"
    assert visual.y_fields == ["congestion_index"]
    assert visual.y_unit == "index"


@pytest.mark.parametrize(
    ("question", "operation", "chart_type", "x_field", "y_field", "keeps_kpi"),
    [
        (
            "How many vessel arrivals were recorded at SEGOT in March 2022?",
            QueryOperation.ARRIVALS,
            "line",
            "date",
            "arrival_count",
            True,
        ),
        (
            "What was the peak arrival day at SEGOT in March 2022?",
            QueryOperation.PEAK_ARRIVAL_DAY,
            "line",
            "date",
            "arrival_count",
            False,
        ),
        (
            "For MMSI 245286000, how long was the vessel in port on 2021-01-01?",
            QueryOperation.MMSI_PORT_STAYS,
            "bar",
            "arrival_time",
            "dwell_minutes",
            True,
        ),
        (
            "For MMSI 304833000, show port-stay duration evidence during 2022-03.",
            QueryOperation.MMSI_PORT_STAYS,
            "bar",
            "arrival_time",
            "dwell_minutes",
            True,
        ),
        (
            "Is pressure at SEGVX above baseline on 2022-03-10?",
            QueryOperation.CONGESTION,
            "bar",
            "date",
            "congestion_index",
            True,
        ),
        (
            "Show WTW CO2e emissions at LVVNT between 2022-02-01 and 2022-02-28.",
            QueryOperation.CARBON,
            "line",
            "date",
            "wtw_co2e_t",
            True,
        ),
    ],
)
def test_data_bearing_scalar_samples_keep_answer_and_add_truthful_graph(
    tmp_path,
    question: str,
    operation: QueryOperation,
    chart_type: str,
    x_field: str,
    y_field: str,
    keeps_kpi: bool,
) -> None:
    envelope = _canonical_service(tmp_path).query(QueryRequest(question=question))

    assert envelope.plan.operation == operation
    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    graph = next(visual for visual in envelope.visualizations if visual.kind == "cartesian")
    assert graph.chart_type == chart_type
    assert graph.x_field == x_field
    assert graph.y_fields == [y_field]
    assert graph.table_fallback_dataset_id == "table"
    graph_dataset = next(dataset for dataset in envelope.datasets if dataset.id == graph.dataset_id)
    assert graph_dataset.row_count > 0
    assert {column.field for column in graph_dataset.columns} >= {x_field, y_field}
    assert any(visual.kind == "kpi" for visual in envelope.visualizations) is keeps_kpi


def test_exact_multi_pollutant_carbon_prompt_keeps_every_requested_series(tmp_path) -> None:
    envelope = _canonical_service(tmp_path).query(
        QueryRequest(
            question="What are TTW emissions at SEGOT in March 2022 for CO2e, NOx, SOx, and PM?"
        )
    )
    assert envelope.plan.operation == QueryOperation.CARBON
    assert envelope.plan.pollutants == ["CO2e", "NOx", "SOx", "PM"]
    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert envelope.datasets
    visuals = [visual for visual in envelope.visualizations if visual.kind == "cartesian"]
    assert len(visuals) == 2
    assert {visual.y_unit for visual in visuals} == {"tCO2e", "kg"}
    assert {field for visual in visuals for field in visual.y_fields} == {
        "ttw_co2e_t",
        "nox_kg",
        "sox_kg",
        "pm_kg",
    }


def test_exact_ttw_wtw_comparison_plots_both_compatible_co2e_series(tmp_path) -> None:
    envelope = _canonical_service(tmp_path).query(
        QueryRequest(
            question="Compare TTW versus WTW CO2e totals at SETRG for March 2022."
        )
    )
    assert envelope.plan.operation == QueryOperation.CARBON
    assert envelope.plan.carbon_boundary == "TTW_WTW"
    assert envelope.state == AnswerState.COMPUTED
    assert len(envelope.visualizations) == 1
    visual = envelope.visualizations[0]
    assert visual.kind == "cartesian"
    assert visual.chart_type == "line"
    assert visual.y_fields == ["ttw_co2e_t", "wtw_co2e_t"]
    assert visual.y_unit == "tCO2e"
    assert {band.point_field for band in visual.interval_bands} == {
        "ttw_co2e_t",
        "wtw_co2e_t",
    }


def test_successful_target_date_forecast_comparison_renders_sorted_bar(tmp_path) -> None:
    envelope = _canonical_service(tmp_path).query(
        QueryRequest(
            question="Compare forecast congestion on 2022-03-06 for LVVNT and SEGOT."
        )
    )
    assert envelope.plan.operation == QueryOperation.FORECAST_COMPARISON
    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert len(envelope.visualizations) == 1
    visual = envelope.visualizations[0]
    assert envelope.datasets
    assert visual.kind == "cartesian"
    assert visual.chart_type == "bar"
    assert visual.x_field == "port"
    assert visual.y_fields == ["predicted"]
    assert visual.sort == "descending"


def test_carbon_composition_separates_incompatible_unit_families() -> None:
    table = pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-03-01", tz="UTC"), pd.Timestamp("2022-03-02", tz="UTC")],
            "ttw_co2e_t": [1.0, 2.0],
            "nox_kg": [10.0, 11.0],
            "sox_kg": [2.0, 3.0],
            "pm_kg": [0.5, 0.7],
        }
    )
    pollutants = ["CO2e", "NOx", "SOx", "PM"]
    result = _carbon_result(table, table.set_index("date")[["ttw_co2e_t"]], pollutants)
    plan = _plan(
        QueryOperation.CARBON,
        requested_visual=VisualizationIntent.STACKED_BAR,
        pollutants=pollutants,
    )
    datasets = _service_shell()._datasets(plan, result)
    visuals = build_visualizations(plan, datasets)

    assert {visual.y_unit for visual in visuals} == {"tCO2e", "kg"}
    assert all(visual.kind == "cartesian" and visual.chart_type == "stacked_bar" for visual in visuals)
    assert all(len({visual.y_unit}) == 1 for visual in visuals)
    kg_visual = next(visual for visual in visuals if visual.y_unit == "kg")
    assert kg_visual.y_fields == ["nox_kg", "sox_kg", "pm_kg"]

    planned = QueryPlanner().plan(
        "Show an emissions breakdown by pollutant at Gothenburg in March 2022."
    )
    assert planned.operation == QueryOperation.CARBON
    assert planned.requested_visual == VisualizationIntent.STACKED_BAR


@pytest.mark.parametrize(
    ("operation", "expected_unit"),
    [
        (QueryOperation.FORECAST_ARRIVALS, "vessels"),
        (QueryOperation.FORECAST_CONGESTION, "index"),
    ],
)
def test_forecast_dataset_and_visual_units_follow_operation(
    operation: QueryOperation,
    expected_unit: str,
) -> None:
    forecast = pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-04-01", tz="UTC")],
            "predicted": [4.0],
            "lower": [3.0],
            "upper": [5.0],
        }
    )
    result = ForecastResult(
        status="ok",
        answer="Forecast computed.",
        history=None,
        forecast=forecast,
        coverage_notes=[],
        caveats=[],
    )
    plan = _plan(operation)
    datasets = _service_shell()._datasets(plan, result)
    chart = next(dataset for dataset in datasets if dataset.id == "chart")
    by_field = {column.field: column.unit for column in chart.columns}
    assert {by_field[field] for field in ("predicted", "lower", "upper")} == {expected_unit}
    visual = build_visualizations(plan, datasets)[0]
    assert visual.kind == "forecast"
    assert visual.unit == expected_unit


def test_mixed_comparison_datasets_and_visuals_preserve_each_metric_unit() -> None:
    table = pd.DataFrame(
        [
            {"scope_label": "SEGOT", "metric": "arrivals_vessels", "value": 12.0},
            {"scope_label": "SEGOT", "metric": "median_dwell_minutes", "value": 90.0},
            {"scope_label": "SEGOT", "metric": "congestion_index", "value": 1.4},
        ]
    )
    result = AnalyticsResult(
        status="ok",
        answer="Mixed comparison computed.",
        table=table,
        chart=table,
        coverage_notes=[],
        caveats=[],
    )
    plan = _plan(QueryOperation.MIXED_PORT_ROUTE_COMPARISON)
    datasets = _service_shell()._datasets(plan, result)
    visuals = build_visualizations(plan, datasets)
    units = {visual.dataset_id: visual.y_unit for visual in visuals}
    assert units == {
        "metric_arrivals_vessels": "vessels",
        "metric_congestion_index": "index",
        "metric_median_dwell_minutes": "minutes",
    }


@pytest.mark.parametrize(
    ("operation", "field", "values", "expected"),
    [
        (QueryOperation.BUSIEST_HOUR, "hour", [18, 0, 7], [0, 7, 18]),
        (
            QueryOperation.BUSIEST_WEEKDAY,
            "day_of_week",
            ["Sunday", "Monday", "Friday"],
            ["Monday", "Friday", "Sunday"],
        ),
    ],
)
def test_calendar_visual_datasets_are_emitted_in_calendar_order(
    operation: QueryOperation,
    field: str,
    values: list[object],
    expected: list[object],
) -> None:
    table = pd.DataFrame({field: values, "arrivals_vessels": [1, 3, 2]})
    original_chart = table.set_index(field)[["arrivals_vessels"]]
    result = AnalyticsResult(
        status="ok",
        answer="Calendar comparison computed.",
        table=table,
        chart=original_chart,
        coverage_notes=[],
        caveats=[],
    )
    plan = _plan(operation)
    datasets = _service_shell()._datasets(plan, result)
    chart = next(dataset for dataset in datasets if dataset.id == "chart")
    assert [row[field] for row in chart.rows] == expected
    visual = build_visualizations(plan, datasets)[0]
    assert visual.kind == "cartesian"
    assert visual.sort == "calendar"
    assert original_chart.index.tolist() == values


def test_ais_jump_map_adds_chronological_unit_separated_motion_plots() -> None:
    table = dataframe_to_dataset(
        pd.DataFrame(
            {
                "mmsi": ["111111111", "111111111"],
                "latitude": [57.7, 57.8],
                "longitude": [11.9, 12.0],
                "distance_km": [12.0, 25.0],
                "implied_speed_kn": [42.0, 55.0],
            }
        ),
        dataset_id="table",
    )
    chart = dataframe_to_dataset(
        pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2022-03-01T10:00:00Z"),
                    pd.Timestamp("2022-03-01T11:00:00Z"),
                ],
                "distance_km": [12.0, 25.0],
                "implied_speed_kn": [42.0, 55.0],
            }
        ).set_index("timestamp"),
        dataset_id="chart",
    )
    assert table is not None and chart is not None
    visuals = build_visualizations(_plan(QueryOperation.AIS_JUMP), [table, chart])

    assert [visual.kind for visual in visuals] == ["map", "cartesian", "cartesian"]
    assert [visual.y_fields for visual in visuals[1:]] == [["distance_km"], ["implied_speed_kn"]]
    assert [visual.y_unit for visual in visuals[1:]] == ["km", "knots"]
    assert all(visual.table_fallback_dataset_id == "table" for visual in visuals)


def test_ais_jump_without_motion_series_retains_explicit_table_fallback() -> None:
    table = dataframe_to_dataset(
        pd.DataFrame({"mmsi": ["111111111"], "latitude": [57.7], "longitude": [11.9]}),
        dataset_id="table",
    )
    assert table is not None
    visuals = build_visualizations(_plan(QueryOperation.AIS_JUMP), [table])
    assert len(visuals) == 1
    assert visuals[0].kind == "map"
    assert visuals[0].table_fallback_dataset_id == "table"
    assert "data-table fallback" in visuals[0].accessible_summary


def test_arrival_anomaly_exposes_typed_highlight_events() -> None:
    engine = KPIQueryEngine("unused")
    engine._arrivals_daily = pd.DataFrame(
        {
            "date": pd.date_range("2022-03-01", periods=10, freq="D", tz="UTC"),
            "arrivals_vessels": [10, 10, 10, 10, 10, 10, 10, 50, 10, 10],
        }
    )
    engine._caps = {"date_max": "2022-03-10"}
    result = engine.detect_arrival_spikes(port=None, start=None, end=None)
    assert result.chart is not None
    assert "is_anomaly" in result.chart.columns
    assert bool(result.chart["is_anomaly"].any()) is True

    plan = _plan(QueryOperation.ARRIVAL_ANOMALY)
    datasets = _service_shell()._datasets(plan, result)
    chart = next(dataset for dataset in datasets if dataset.id == "chart")
    assert any(row["is_anomaly"] is True for row in chart.rows)
    visual = build_visualizations(plan, datasets)[0]
    assert visual.kind == "cartesian"
    assert visual.y_fields == ["arrival_count", "threshold"]
    assert visual.highlight is not None
    assert visual.highlight.condition_field == "is_anomaly"
    assert visual.highlight.value_field == "arrival_count"
    assert visual.highlight.label == "Detected arrival spike"
