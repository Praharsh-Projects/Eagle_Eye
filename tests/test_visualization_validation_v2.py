from __future__ import annotations

from src.query.models import (
    CartesianVisualization,
    ColumnSpec,
    DatasetSpec,
    ForecastVisualization,
    MapVisualization,
    QueryMode,
    QueryOperation,
    QueryPlan,
    TimelineVisualization,
)
from src.query.visuals import _validated


def _plan(operation: QueryOperation = QueryOperation.ARRIVALS) -> QueryPlan:
    return QueryPlan(mode=QueryMode.ANALYTICS, operation=operation, reason="contract test")


def test_mixed_unit_axis_is_rejected() -> None:
    dataset = DatasetSpec(
        id="chart",
        columns=[
            ColumnSpec(field="date", label="Date", data_type="datetime"),
            ColumnSpec(field="arrivals", label="Arrivals", data_type="integer", unit="vessels"),
            ColumnSpec(field="duration", label="Duration", data_type="number", unit="hours"),
        ],
        rows=[{"date": "2022-01-01T00:00:00Z", "arrivals": 2, "duration": 4.5}],
        row_count=1,
    )
    visual = CartesianVisualization(
        id="mixed",
        title="Mixed",
        dataset_id="chart",
        accessible_summary="Mixed units",
        chart_type="line",
        x_field="date",
        y_fields=["arrivals", "duration"],
    )
    validated = _validated(visual, {"chart": dataset}, _plan())
    assert validated.kind == "omitted"
    assert validated.reason_code == "validation_failed"


def test_unordered_temporal_rows_are_rejected() -> None:
    dataset = DatasetSpec(
        id="chart",
        columns=[
            ColumnSpec(field="date", label="Date", data_type="datetime"),
            ColumnSpec(field="value", label="Value", data_type="integer", unit="vessels"),
        ],
        rows=[
            {"date": "2022-01-02T00:00:00Z", "value": 2},
            {"date": "2022-01-01T00:00:00Z", "value": 1},
        ],
        row_count=2,
    )
    visual = CartesianVisualization(
        id="trend",
        title="Trend",
        dataset_id="chart",
        accessible_summary="Trend",
        chart_type="line",
        x_field="date",
        y_fields=["value"],
        y_unit="vessels",
    )
    validated = _validated(visual, {"chart": dataset}, _plan())
    assert validated.kind == "omitted"
    assert "chronological" in validated.reason


def test_forecast_interval_must_contain_prediction() -> None:
    dataset = DatasetSpec(
        id="chart",
        columns=[
            ColumnSpec(field="date", label="Date", data_type="datetime"),
            ColumnSpec(field="predicted", label="Predicted", data_type="number", unit="index"),
            ColumnSpec(field="lower", label="Lower", data_type="number", unit="index"),
            ColumnSpec(field="upper", label="Upper", data_type="number", unit="index"),
        ],
        rows=[{"date": "2022-01-01T00:00:00Z", "predicted": 3.0, "lower": 1.0, "upper": 2.0}],
        row_count=1,
    )
    visual = ForecastVisualization(
        id="forecast",
        title="Forecast",
        dataset_id="chart",
        accessible_summary="Forecast",
        date_field="date",
        predicted_field="predicted",
        lower_field="lower",
        upper_field="upper",
    )
    validated = _validated(visual, {"chart": dataset}, _plan(QueryOperation.FORECAST_CONGESTION))
    assert validated.kind == "omitted"
    assert "does not contain" in validated.reason


def test_invalid_map_coordinates_are_rejected() -> None:
    dataset = DatasetSpec(
        id="table",
        columns=[
            ColumnSpec(field="latitude", label="Latitude", data_type="number", unit="degrees"),
            ColumnSpec(field="longitude", label="Longitude", data_type="number", unit="degrees"),
        ],
        rows=[{"latitude": 95.0, "longitude": 11.0}],
        row_count=1,
    )
    visual = MapVisualization(
        id="map",
        title="Map",
        dataset_id="table",
        accessible_summary="Coordinates",
        latitude_field="latitude",
        longitude_field="longitude",
    )
    validated = _validated(visual, {"table": dataset}, _plan(QueryOperation.AIS_JUMP))
    assert validated.kind == "omitted"
    assert "outside valid" in validated.reason


def test_timeline_rejects_mixed_valid_and_missing_timestamps() -> None:
    dataset = DatasetSpec(
        id="eta_timeline",
        columns=[
            ColumnSpec(field="reported_eta_utc", label="Reported ETA", data_type="datetime"),
            ColumnSpec(field="vessel_label", label="Vessel", data_type="string"),
        ],
        rows=[
            {
                "reported_eta_utc": "2026-07-28T13:30:00Z",
                "vessel_label": "VALID ETA",
            },
            {
                "reported_eta_utc": None,
                "vessel_label": "MISSING ETA",
            },
        ],
        row_count=2,
    )
    visual = TimelineVisualization(
        id="eta_watch_timeline",
        title="ETA watch",
        dataset_id="eta_timeline",
        accessible_summary="Valid vessel-reported ETAs.",
        time_field="reported_eta_utc",
        label_field="vessel_label",
    )

    validated = _validated(
        visual,
        {"eta_timeline": dataset},
        _plan(QueryOperation.VESSEL_ETA),
    )

    assert validated.kind == "omitted"
    assert "missing" in validated.reason


def test_timeline_rejects_invalid_timestamp_text() -> None:
    dataset = DatasetSpec(
        id="eta_timeline",
        columns=[
            ColumnSpec(field="reported_eta_utc", label="Reported ETA", data_type="datetime"),
            ColumnSpec(field="vessel_label", label="Vessel", data_type="string"),
        ],
        rows=[
            {
                "reported_eta_utc": "not-a-timestamp",
                "vessel_label": "INVALID ETA",
            },
        ],
        row_count=1,
    )
    visual = TimelineVisualization(
        id="eta_watch_timeline",
        title="ETA watch",
        dataset_id="eta_timeline",
        accessible_summary="Valid vessel-reported ETAs.",
        time_field="reported_eta_utc",
        label_field="vessel_label",
    )

    validated = _validated(
        visual,
        {"eta_timeline": dataset},
        _plan(QueryOperation.VESSEL_ETA),
    )

    assert validated.kind == "omitted"
    assert "invalid timestamp" in validated.reason
