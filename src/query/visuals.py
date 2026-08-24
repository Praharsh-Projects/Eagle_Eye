"""Intent-aware visualization selection and field validation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .chart_analytics import finite_number, forecast_quality, ols_metadata
from .models import (
    CartesianHighlight,
    CartesianVisualization,
    DatasetSpec,
    DistributionVisualization,
    ETAWatchIntent,
    FiveNumberSummary,
    FittedSeriesBinding,
    FlaggedPointAnnotation,
    ForecastVisualization,
    ForecastQualitySpec,
    HeatmapVisualization,
    IntervalBandSpec,
    KPIVisualization,
    KPIThresholdSpec,
    MapVisualization,
    OmittedVisualization,
    QueryOperation,
    QueryPlan,
    ReferenceLineSpec,
    TableVisualization,
    TimelineVisualization,
    VisualizationIntent,
    VisualizationSpec,
)
from .serialization import dataset_fields


def _lookup(datasets: Iterable[DatasetSpec]) -> Dict[str, DatasetSpec]:
    return {dataset.id: dataset for dataset in datasets}


def _field(dataset: DatasetSpec, candidates: Iterable[str]) -> Optional[str]:
    fields = dataset_fields(dataset)
    return next((candidate for candidate in candidates if candidate in fields), None)


def _numeric_fields(dataset: DatasetSpec) -> List[str]:
    return [column.field for column in dataset.columns if column.data_type in {"number", "integer"}]


def _datetime_field(dataset: DatasetSpec) -> Optional[str]:
    return next((column.field for column in dataset.columns if column.data_type == "datetime"), None)


def _unit(dataset: DatasetSpec, field: str) -> Optional[str]:
    return next((column.unit for column in dataset.columns if column.field == field), None)


def _has_finite_values(dataset: DatasetSpec, field: str) -> bool:
    for row in dataset.rows:
        value = row.get(field)
        if isinstance(value, bool):
            continue
        try:
            if math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _has_values(dataset: DatasetSpec, field: str) -> bool:
    for row in dataset.rows:
        value = row.get(field)
        if value is not None and value != "":
            return True
    return False


def _usable_interval(
    dataset: DatasetSpec,
    *,
    point_field: str,
    lower_field: str,
    upper_field: str,
) -> bool:
    """Return true only when at least one complete contained interval exists."""

    complete = 0
    for row in dataset.rows:
        point = finite_number(row.get(point_field))
        lower = finite_number(row.get(lower_field))
        upper = finite_number(row.get(upper_field))
        if lower is None and upper is None:
            continue
        if point is None or lower is None or upper is None or not lower <= point <= upper:
            return False
        complete += 1
    return complete > 0


def _carbon_metric_fields(plan: QueryPlan, dataset: DatasetSpec) -> List[str]:
    boundary = plan.carbon_boundary.strip().upper()
    co2e_field = "wtw_co2e_t" if boundary == "WTW" else "ttw_co2e_t"
    pollutant_fields = {
        "CO2E": ["ttw_co2e_t", "wtw_co2e_t"] if boundary == "TTW_WTW" else [co2e_field],
        "CO2": ["co2_t"],
        "NOX": ["nox_kg"],
        "SOX": ["sox_kg"],
        "PM": ["pm_kg"],
    }
    requested = plan.pollutants or ["CO2e", "NOx", "SOx", "PM"]
    available = dataset_fields(dataset)
    fields: List[str] = []
    for pollutant in requested:
        for field in pollutant_fields.get(str(pollutant).strip().upper(), []):
            if field in available and field not in fields:
                fields.append(field)
    return fields


def _omitted(reason_code: str, reason: str) -> OmittedVisualization:
    return OmittedVisualization(
        id="visualization_omitted",
        title="Visualization unavailable",
        accessible_summary=reason,
        reason_code=reason_code,  # type: ignore[arg-type]
        reason=reason,
    )


def _with_v21_bindings(
    visualization: VisualizationSpec,
    datasets: Dict[str, DatasetSpec],
    plan: QueryPlan,
    caveats: Iterable[str],
) -> VisualizationSpec:
    """Attach optional 2.1 encodings without changing a legacy chart's meaning."""

    dataset = datasets.get(visualization.dataset_id or "")
    updates: Dict[str, object] = {}
    if dataset is not None and "row_id" in dataset_fields(dataset):
        updates["row_id_field"] = "row_id"

    if isinstance(visualization, KPIVisualization):
        if plan.operation == QueryOperation.CONGESTION:
            updates.update(
                {
                    "baseline_value": 1.0,
                    "thresholds": [
                        KPIThresholdSpec(
                            id="pressure_baseline",
                            label="Historical baseline",
                            value=1.0,
                            unit="index",
                        )
                    ],
                }
            )

    elif isinstance(visualization, CartesianVisualization) and dataset is not None:
        reference_lines = list(visualization.reference_lines)
        interval_bands = list(visualization.interval_bands)
        annotations = list(visualization.annotations)
        fitted_series = list(visualization.fitted_series)
        fields = dataset_fields(dataset)

        if (
            plan.operation
            in {
                QueryOperation.CONGESTION,
                QueryOperation.PRESSURE_BY_VESSEL_TYPE,
                QueryOperation.PEAK_CONGESTION_DAY,
                QueryOperation.CONGESTION_WEEKDAY_COMPARISON,
                QueryOperation.FORECAST_CONGESTION,
            }
            and any(field in {"congestion_index", "pressure_index", "predicted"} for field in visualization.y_fields)
        ):
            reference_lines.append(
                ReferenceLineSpec(
                    id="pressure_baseline",
                    label="pressure_v2 baseline",
                    axis="y",
                    value=1.0,
                    unit="index",
                    line_style="dashed",
                )
            )

        if (
            plan.operation in {QueryOperation.ARRIVALS, QueryOperation.PEAK_ARRIVAL_DAY}
            and "rolling_median_7" in fields
        ):
            fitted_series.append(
                FittedSeriesBinding(
                    id="arrivals_rolling_median_7",
                    label="7-day rolling median",
                    x_field=visualization.x_field,
                    y_field="rolling_median_7",
                    method="rolling_median",
                    association_only=False,
                )
            )

        if plan.operation == QueryOperation.CARBON:
            for field in visualization.y_fields:
                lower_field = f"{field}_lower"
                upper_field = f"{field}_upper"
                if {lower_field, upper_field}.issubset(fields) and _usable_interval(
                    dataset,
                    point_field=field,
                    lower_field=lower_field,
                    upper_field=upper_field,
                ):
                    interval_bands.append(
                        IntervalBandSpec(
                            id=f"{field}_uncertainty",
                            label=f"{field.replace('_', ' ').title()} uncertainty",
                            lower_field=lower_field,
                            upper_field=upper_field,
                            unit=_unit(dataset, field),
                            point_field=field,
                            display="whisker" if dataset.row_count <= 5 else "band",
                        )
                    )
            selected_fields = _carbon_metric_fields(plan, dataset)
            if (
                "is_peak" in fields
                and selected_fields
                and selected_fields[0] in visualization.y_fields
            ):
                annotations.append(
                    FlaggedPointAnnotation(
                        id=f"{visualization.id}_carbon_peak",
                        label="Peak selected carbon metric",
                        condition_field="is_peak",
                        x_field=visualization.x_field,
                        y_field=selected_fields[0],
                    )
                )

        if (
            plan.operation == QueryOperation.FORECAST_COMPARISON
            and {"lower", "upper"}.issubset(fields)
            and "predicted" in visualization.y_fields
            and _usable_interval(
                dataset,
                point_field="predicted",
                lower_field="lower",
                upper_field="upper",
            )
        ):
            interval_bands.append(
                IntervalBandSpec(
                    id="forecast_comparison_interval",
                    label="80% forecast interval",
                    lower_field="lower",
                    upper_field="upper",
                    unit=_unit(dataset, "predicted"),
                )
            )

        if plan.operation == QueryOperation.CORRELATION:
            fit_field = "ols_fitted_median_dwell_minutes"
            if fit_field in fields and visualization.y_fields:
                fit = ols_metadata(dataset, visualization.x_field, visualization.y_fields[0])
                if fit is not None:
                    fitted_series.append(
                        FittedSeriesBinding(
                            id="ols_association_fit",
                            label="OLS association fit",
                            x_field=visualization.x_field,
                            y_field=fit_field,
                            method="ols",
                            association_only=True,
                            slope=fit["slope"],
                            intercept=fit["intercept"],
                            r_squared=fit["r_squared"],
                        )
                    )

        if visualization.highlight:
            annotations.append(
                FlaggedPointAnnotation(
                    id=f"{visualization.id}_flagged_points",
                    label=visualization.highlight.label,
                    condition_field=visualization.highlight.condition_field,
                    x_field=visualization.x_field,
                    y_field=visualization.highlight.value_field,
                )
            )

        if reference_lines:
            updates["reference_lines"] = reference_lines
        if interval_bands:
            updates["interval_bands"] = interval_bands
        if annotations:
            updates["annotations"] = annotations
        if fitted_series:
            updates["fitted_series"] = fitted_series

    elif isinstance(visualization, ForecastVisualization) and dataset is not None:
        boundary = next(
            (
                str(row.get(visualization.date_field))
                for row in dataset.rows
                if finite_number(row.get(visualization.predicted_field)) is not None
                and (
                    not visualization.actual_field
                    or finite_number(row.get(visualization.actual_field)) is None
                )
            ),
            None,
        )
        quality = forecast_quality(caveats)
        if boundary:
            updates["forecast_boundary"] = boundary
        if quality is not None:
            updates["quality_metrics"] = ForecastQualitySpec(
                mase=quality["mase"],
                interval_coverage=quality["interval_coverage"],
                interval_level=quality["interval_level"],
            )

    elif isinstance(visualization, DistributionVisualization):
        summary_dataset = datasets.get("distribution_summary")
        outlier_dataset = datasets.get("distribution_outliers")
        source_table = datasets.get("table")
        if (
            visualization.chart_type == "histogram"
            and source_table is not None
            and {"bin_start_minutes", "bin_end_minutes", visualization.value_field}.issubset(
                dataset_fields(source_table)
            )
        ):
            dataset = source_table
            updates.update(
                {
                    "dataset_id": source_table.id,
                    "row_id_field": "row_id" if "row_id" in dataset_fields(source_table) else None,
                    "bin_lower_field": "bin_start_minutes",
                    "bin_upper_field": "bin_end_minutes",
                }
            )
        if summary_dataset and summary_dataset.rows:
            row = summary_dataset.rows[0]
            keys = {
                "minimum",
                "q1",
                "median",
                "q3",
                "maximum",
                "lower_whisker",
                "upper_whisker",
                "count",
            }
            if keys.issubset(row) and all(finite_number(row.get(key)) is not None for key in keys):
                updates["five_number_summary"] = FiveNumberSummary(
                    minimum=float(row["minimum"]),
                    q1=float(row["q1"]),
                    median=float(row["median"]),
                    q3=float(row["q3"]),
                    maximum=float(row["maximum"]),
                    lower_whisker=float(row["lower_whisker"]),
                    upper_whisker=float(row["upper_whisker"]),
                    p90=finite_number(row.get("p90")),
                    count=int(row["count"]),
                )
                updates["summary_dataset_id"] = summary_dataset.id
        observations = datasets.get("distribution_observations")
        if observations is not None and "is_outlier" in dataset_fields(observations):
            updates["outlier_dataset_id"] = (
                outlier_dataset.id if outlier_dataset is not None else observations.id
            )
            updates["outlier_condition_field"] = "is_outlier"
            updates["outlier_value_field"] = _field(observations, ["dwell_minutes", "duration_h"])

    elif isinstance(visualization, MapVisualization) and dataset is not None:
        fields = dataset_fields(dataset)
        if {"prev_latitude", "prev_longitude", "latitude", "longitude"}.issubset(fields):
            updates.update(
                {
                    "geometry_mode": "segments",
                    "start_latitude_field": "prev_latitude",
                    "start_longitude_field": "prev_longitude",
                    "end_latitude_field": "latitude",
                    "end_longitude_field": "longitude",
                    "timestamp_field": _field(dataset, ["timestamp_full", "timestamp"]),
                }
            )
        else:
            timestamp = _field(dataset, ["timestamp_full", "timestamp"])
            if timestamp:
                updates["timestamp_field"] = timestamp

    elif isinstance(visualization, TimelineVisualization) and dataset is not None:
        end_time = _field(dataset, ["departure_time", "end_time"])
        lane = _field(dataset, ["port_key", "vessel_type_norm", "event_kind"])
        if end_time and end_time != visualization.time_field:
            updates["end_time_field"] = end_time
        if lane and not visualization.lane_field:
            updates["lane_field"] = lane

    return visualization.model_copy(update=updates) if updates else visualization


def _validated(
    visualization: VisualizationSpec,
    datasets: Dict[str, DatasetSpec],
    plan: QueryPlan,
) -> VisualizationSpec:
    dataset_id = visualization.dataset_id
    if not dataset_id or dataset_id not in datasets:
        return _omitted("validation_failed", "The selected visualization dataset is unavailable.")
    fields = dataset_fields(datasets[dataset_id])
    required: List[str] = []
    if visualization.row_id_field:
        required.append(visualization.row_id_field)
    if isinstance(visualization, KPIVisualization):
        required.append(visualization.value_field)
        if visualization.trend_field:
            required.append(visualization.trend_field)
        if visualization.comparison_field:
            required.append(visualization.comparison_field)
    elif isinstance(visualization, CartesianVisualization):
        required.extend([visualization.x_field, *visualization.y_fields])
        if visualization.series_field:
            required.append(visualization.series_field)
        if visualization.highlight:
            required.extend(
                [visualization.highlight.condition_field, visualization.highlight.value_field]
            )
        for interval in visualization.interval_bands:
            required.extend([interval.lower_field, interval.upper_field])
            if interval.point_field:
                required.append(interval.point_field)
        for annotation in visualization.annotations:
            required.extend(
                [
                    annotation.condition_field,
                    annotation.x_field,
                    annotation.y_field,
                ]
            )
        for fitted in visualization.fitted_series:
            required.extend([fitted.x_field, fitted.y_field])
    elif isinstance(visualization, ForecastVisualization):
        required.extend(
            [
                visualization.date_field,
                visualization.predicted_field,
                visualization.lower_field,
                visualization.upper_field,
            ]
        )
        if visualization.actual_field:
            required.append(visualization.actual_field)
    elif isinstance(visualization, DistributionVisualization):
        required.append(visualization.value_field)
        if visualization.count_field:
            required.append(visualization.count_field)
        if visualization.category_field:
            required.append(visualization.category_field)
        if visualization.bin_lower_field:
            required.append(visualization.bin_lower_field)
        if visualization.bin_upper_field:
            required.append(visualization.bin_upper_field)
    elif isinstance(visualization, HeatmapVisualization):
        required.extend([visualization.x_field, visualization.y_field, visualization.value_field])
    elif isinstance(visualization, MapVisualization):
        required.extend([visualization.latitude_field, visualization.longitude_field])
        if visualization.label_field:
            required.append(visualization.label_field)
        for field in (
            visualization.start_latitude_field,
            visualization.start_longitude_field,
            visualization.end_latitude_field,
            visualization.end_longitude_field,
            visualization.path_field,
            visualization.sequence_field,
            visualization.timestamp_field,
        ):
            if field:
                required.append(field)
    elif isinstance(visualization, TimelineVisualization):
        required.extend(
            [visualization.time_field, visualization.label_field, *visualization.detail_fields]
        )
        if visualization.end_time_field:
            required.append(visualization.end_time_field)
        if visualization.lane_field:
            required.append(visualization.lane_field)
    elif isinstance(visualization, TableVisualization):
        required.extend(visualization.visible_fields)
    missing = [field for field in required if field not in fields]
    if missing:
        return _omitted(
            "validation_failed",
            f"Visualization omitted because required field(s) are missing: {', '.join(missing)}.",
        )

    dataset = datasets[dataset_id]
    units = {column.field: column.unit for column in dataset.columns}

    def finite_number(value: object) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def ordered(field: str) -> bool:
        values = [row.get(field) for row in dataset.rows if row.get(field) is not None]
        if len(values) < 2:
            return True
        parsed: List[datetime] = []
        for value in values:
            try:
                parsed.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
            except ValueError:
                try:
                    parsed.append(datetime.strptime(str(value), "%Y-%m"))
                except ValueError:
                    return False
        return parsed == sorted(parsed)

    def calendar_ordered(field: str) -> bool:
        values = [row.get(field) for row in dataset.rows if row.get(field) is not None]
        if len(values) < 2:
            return True
        if field == "hour":
            try:
                ranks = [int(value) for value in values]
            except (TypeError, ValueError):
                return False
            return all(0 <= value <= 23 for value in ranks) and ranks == sorted(ranks)
        if field == "day_of_week":
            weekday_order = {
                day: index
                for index, day in enumerate(
                    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                )
            }
            try:
                ranks = [weekday_order[str(value).title()] for value in values]
            except KeyError:
                return False
            return ranks == sorted(ranks)
        return False

    if isinstance(visualization, KPIVisualization):
        if not dataset.rows or finite_number(dataset.rows[0].get(visualization.value_field)) is None:
            return _omitted("validation_failed", "The KPI value is missing or non-finite.")
        if visualization.baseline_value is not None and finite_number(visualization.baseline_value) is None:
            return _omitted("validation_failed", "The KPI baseline is non-finite.")
        if any(finite_number(threshold.value) is None for threshold in visualization.thresholds):
            return _omitted("validation_failed", "A KPI threshold is non-finite.")
        if plan.operation == QueryOperation.ARRIVALS and "table" in datasets:
            table = datasets["table"]
            values = [
                finite_number(row.get("arrival_count", row.get("arrivals_vessels")))
                for row in table.rows
            ]
            values = [value for value in values if value is not None]
            displayed = finite_number(dataset.rows[0].get(visualization.value_field))
            if values and displayed is not None and not math.isclose(displayed, sum(values), abs_tol=1e-9):
                return _omitted("validation_failed", "The KPI does not reconcile with its supporting arrival rows.")
    elif isinstance(visualization, CartesianVisualization):
        unit_families = {units.get(field) for field in visualization.y_fields if units.get(field)}
        if len(unit_families) > 1:
            return _omitted("validation_failed", "Series with different unit families cannot share one axis.")
        if visualization.y_unit and unit_families and visualization.y_unit not in unit_families:
            return _omitted("validation_failed", "The declared axis unit does not match the dataset fields.")
        if visualization.chart_type in {"line", "area"} and not ordered(visualization.x_field):
            return _omitted("validation_failed", "Temporal chart rows are not in chronological order.")
        if visualization.sort == "calendar" and not calendar_ordered(visualization.x_field):
            return _omitted("validation_failed", "Calendar chart rows are not in calendar order.")
        if visualization.highlight:
            condition_field = visualization.highlight.condition_field
            value_field = visualization.highlight.value_field
            for row in dataset.rows:
                condition = row.get(condition_field)
                if condition not in {True, False, 0, 1}:
                    return _omitted(
                        "validation_failed",
                        "Highlight conditions must be boolean values.",
                    )
                if bool(condition) and finite_number(row.get(value_field)) is None:
                    return _omitted(
                        "validation_failed",
                        "A highlighted event has no finite marker value.",
                    )
        for reference in visualization.reference_lines:
            if reference.axis == "y":
                if finite_number(reference.value) is None:
                    return _omitted("validation_failed", "A y-axis reference line is non-finite.")
                if reference.unit and visualization.y_unit and reference.unit != visualization.y_unit:
                    return _omitted(
                        "validation_failed",
                        "A reference-line unit does not match the chart axis.",
                    )
        for interval in visualization.interval_bands:
            if interval.unit and visualization.y_unit and interval.unit != visualization.y_unit:
                return _omitted(
                    "validation_failed",
                    "An interval-band unit does not match the chart axis.",
                )
            contained_rows = 0
            point_field = interval.point_field
            if point_field and point_field not in visualization.y_fields:
                return _omitted(
                    "validation_failed",
                    "An interval point binding does not reference a displayed series.",
                )
            if point_field is None:
                point_field = next(
                    (
                        field
                        for field in visualization.y_fields
                        if interval.lower_field == f"{field}_lower"
                        and interval.upper_field == f"{field}_upper"
                    ),
                    None,
                )
            if (
                point_field is None
                and interval.lower_field == "lower"
                and interval.upper_field == "upper"
                and len(visualization.y_fields) == 1
            ):
                point_field = visualization.y_fields[0]
            for row in dataset.rows:
                lower = finite_number(row.get(interval.lower_field))
                upper = finite_number(row.get(interval.upper_field))
                point = finite_number(row.get(point_field)) if point_field else None
                if lower is None and upper is None:
                    continue
                if lower is None or upper is None or lower > upper:
                    return _omitted("validation_failed", "An interval band contains invalid bounds.")
                if point_field and point is None:
                    return _omitted(
                        "validation_failed",
                        "An interval band has bounds without a finite bound series value.",
                    )
                if point is not None and not lower <= point <= upper:
                    return _omitted(
                        "validation_failed",
                        "An interval band does not contain its bound series value.",
                    )
                contained_rows += 1
            if contained_rows == 0:
                return _omitted("validation_failed", "An interval band has no complete rows.")
        for annotation in visualization.annotations:
            for row in dataset.rows:
                condition = row.get(annotation.condition_field)
                if condition not in {True, False, 0, 1}:
                    return _omitted(
                        "validation_failed",
                        "Annotation conditions must be boolean values.",
                    )
                if bool(condition) and finite_number(row.get(annotation.y_field)) is None:
                    return _omitted(
                        "validation_failed",
                        "A flagged annotation has no finite y value.",
                    )
        for fitted in visualization.fitted_series:
            if fitted.method == "ols":
                if (
                    not fitted.association_only
                    or finite_number(fitted.slope) is None
                    or finite_number(fitted.intercept) is None
                    or finite_number(fitted.r_squared) is None
                ):
                    return _omitted(
                        "validation_failed",
                        "An OLS fit is missing validated association metadata.",
                    )
            if not any(
                finite_number(row.get(fitted.y_field)) is not None
                for row in dataset.rows
            ):
                return _omitted("validation_failed", "A fitted series has no finite values.")
        if not any(
            finite_number(row.get(field)) is not None
            for row in dataset.rows
            for field in visualization.y_fields
        ):
            return _omitted("validation_failed", "The chart has no finite numeric values.")
    elif isinstance(visualization, ForecastVisualization):
        forecast_fields = [
            visualization.predicted_field,
            visualization.lower_field,
            visualization.upper_field,
            *([visualization.actual_field] if visualization.actual_field else []),
        ]
        forecast_units = {units.get(field) for field in forecast_fields if units.get(field)}
        if len(forecast_units) > 1:
            return _omitted("validation_failed", "Forecast series with different units cannot share one axis.")
        if visualization.unit and forecast_units and visualization.unit not in forecast_units:
            return _omitted("validation_failed", "The declared forecast unit does not match the dataset fields.")
        if not ordered(visualization.date_field):
            return _omitted("validation_failed", "Forecast rows are not in chronological order.")
        interval_rows = 0
        for row in dataset.rows:
            predicted = finite_number(row.get(visualization.predicted_field))
            lower = finite_number(row.get(visualization.lower_field))
            upper = finite_number(row.get(visualization.upper_field))
            if predicted is None and lower is None and upper is None:
                continue
            if predicted is None or lower is None or upper is None or not lower <= predicted <= upper:
                return _omitted("validation_failed", "A forecast interval does not contain its prediction.")
            interval_rows += 1
        if interval_rows == 0:
            return _omitted("validation_failed", "No complete forecast interval rows are available.")
        if visualization.forecast_boundary and not any(
            str(row.get(visualization.date_field)) == visualization.forecast_boundary
            for row in dataset.rows
        ):
            return _omitted(
                "validation_failed",
                "The forecast boundary is not present in the forecast dataset.",
            )
        if visualization.quality_metrics:
            quality = visualization.quality_metrics
            if not (
                quality.gate_passed
                and quality.mase < 1.0
                and 0.70 <= quality.interval_coverage <= 0.90
                and math.isclose(quality.interval_level, 0.8, abs_tol=1e-9)
            ):
                return _omitted(
                    "validation_failed",
                    "Forecast quality annotations did not pass the configured gate.",
                )
    elif isinstance(visualization, DistributionVisualization):
        if not any(finite_number(row.get(visualization.value_field)) is not None for row in dataset.rows):
            return _omitted("validation_failed", "The distribution has no finite numeric values.")
        if visualization.count_field and any(
            (value := finite_number(row.get(visualization.count_field))) is None or value < 0
            for row in dataset.rows
        ):
            return _omitted("validation_failed", "Distribution counts must be finite and non-negative.")
        if bool(visualization.bin_lower_field) != bool(visualization.bin_upper_field):
            return _omitted(
                "validation_failed",
                "Histogram bin boundaries must include both lower and upper fields.",
            )
        if visualization.bin_lower_field and visualization.bin_upper_field:
            for row in dataset.rows:
                lower = finite_number(row.get(visualization.bin_lower_field))
                upper = finite_number(row.get(visualization.bin_upper_field))
                value = finite_number(row.get(visualization.value_field))
                if (
                    lower is None
                    or upper is None
                    or value is None
                    or not lower <= value <= upper
                ):
                    return _omitted(
                        "validation_failed",
                        "A histogram bin has invalid boundaries or midpoint.",
                    )
        if visualization.five_number_summary:
            summary = visualization.five_number_summary
            if not (
                summary.minimum
                <= summary.lower_whisker
                <= summary.q1
                <= summary.median
                <= summary.q3
                <= summary.upper_whisker
                <= summary.maximum
            ):
                return _omitted(
                    "validation_failed",
                    "The five-number summary or Tukey whiskers are not ordered.",
                )
            if visualization.summary_dataset_id not in datasets:
                return _omitted(
                    "validation_failed",
                    "The distribution summary dataset is unavailable.",
                )
        if visualization.outlier_dataset_id:
            outliers = datasets.get(visualization.outlier_dataset_id)
            if outliers is None:
                return _omitted("validation_failed", "The outlier dataset is unavailable.")
            outlier_fields = dataset_fields(outliers)
            if (
                not visualization.outlier_condition_field
                or not visualization.outlier_value_field
                or visualization.outlier_condition_field not in outlier_fields
                or visualization.outlier_value_field not in outlier_fields
            ):
                return _omitted(
                    "validation_failed",
                    "The outlier field bindings are unavailable.",
                )
            if any(
                row.get(visualization.outlier_condition_field) is not True
                or finite_number(row.get(visualization.outlier_value_field)) is None
                for row in outliers.rows
            ):
                return _omitted(
                    "validation_failed",
                    "An outlier binding contains an invalid row.",
                )
    elif isinstance(visualization, HeatmapVisualization):
        if not any(finite_number(row.get(visualization.value_field)) is not None for row in dataset.rows):
            return _omitted("validation_failed", "The heatmap has no finite numeric values.")
    elif isinstance(visualization, MapVisualization):
        coordinate_pairs = [
            (visualization.latitude_field, visualization.longitude_field),
        ]
        if visualization.geometry_mode == "segments":
            if not all(
                (
                    visualization.start_latitude_field,
                    visualization.start_longitude_field,
                    visualization.end_latitude_field,
                    visualization.end_longitude_field,
                )
            ):
                return _omitted(
                    "validation_failed",
                    "Segment geometry requires explicit start and end coordinate fields.",
                )
            coordinate_pairs = [
                (
                    visualization.start_latitude_field or "",
                    visualization.start_longitude_field or "",
                ),
                (
                    visualization.end_latitude_field or "",
                    visualization.end_longitude_field or "",
                ),
            ]
        elif visualization.geometry_mode == "ordered_path" and not all(
            (visualization.path_field, visualization.sequence_field, visualization.timestamp_field)
        ):
            return _omitted(
                "validation_failed",
                "Ordered-path geometry requires path, sequence, and timestamp fields.",
            )
        valid_coordinates = 0
        for row in dataset.rows:
            row_valid = True
            for latitude_field, longitude_field in coordinate_pairs:
                latitude = finite_number(row.get(latitude_field))
                longitude = finite_number(row.get(longitude_field))
                if latitude is None or longitude is None:
                    row_valid = False
                    continue
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    return _omitted(
                        "validation_failed",
                        "A map coordinate is outside valid latitude/longitude bounds.",
                    )
            if row_valid:
                valid_coordinates += 1
        if valid_coordinates == 0:
            return _omitted("validation_failed", "No valid map coordinates are available.")
    elif isinstance(visualization, TimelineVisualization):
        for row in dataset.rows:
            timestamp = row.get(visualization.time_field)
            if timestamp is None or not str(timestamp).strip():
                return _omitted(
                    "validation_failed",
                    "A timeline row is missing its required timestamp.",
                )
            try:
                datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return _omitted(
                    "validation_failed",
                    "A timeline row has an invalid timestamp.",
                )
        if not ordered(visualization.time_field):
            return _omitted("validation_failed", "Timeline rows are not in chronological order.")
        if visualization.end_time_field:
            for row in dataset.rows:
                start = row.get(visualization.time_field)
                end = row.get(visualization.end_time_field)
                if start is None or end is None:
                    continue
                try:
                    start_value = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
                    end_value = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
                except ValueError:
                    return _omitted(
                        "validation_failed",
                        "A timeline interval has an invalid timestamp.",
                    )
                if end_value < start_value:
                    return _omitted(
                        "validation_failed",
                        "A timeline interval ends before it starts.",
                    )
    return visualization


def build_visualizations(
    plan: QueryPlan,
    datasets: List[DatasetSpec],
    *,
    caveats: Iterable[str] = (),
) -> List[VisualizationSpec]:
    if plan.requested_visual == VisualizationIntent.NONE:
        return [_omitted("not_requested", "The user requested a text-only answer.")]
    by_id = _lookup(datasets)
    table = by_id.get("table")
    chart = by_id.get("chart") or table
    summary = by_id.get("summary")
    if chart is None and summary is None:
        return [_omitted("insufficient_data", "No validated rows are available for a meaningful graph.")]

    operation = plan.operation
    visuals: List[VisualizationSpec] = []

    if plan.requested_visual == VisualizationIntent.TABLE:
        if table is None:
            return [_omitted("insufficient_data", "No validated table rows are available.")]
        return [
            _validated(
                _with_v21_bindings(
                    TableVisualization(
                        id="requested_table",
                        title="Result data",
                        dataset_id=table.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary="Validated result rows requested as a data table.",
                        visible_fields=[column.field for column in table.columns],
                    ),
                    by_id,
                    plan,
                    caveats,
                ),
                by_id,
                plan,
            )
        ]

    intent_compatibility = {
        VisualizationIntent.MAP: {
            QueryOperation.AIS_JUMP,
            QueryOperation.VESSEL_ETA,
        },
        VisualizationIntent.HEATMAP: {QueryOperation.ARRIVAL_PATTERN},
        VisualizationIntent.TIMELINE: {
            QueryOperation.FIRST_ARRIVAL,
            QueryOperation.LAST_ARRIVAL,
            QueryOperation.FIRST_DEPARTURE,
            QueryOperation.FIRST_ROUTE_VESSEL,
            QueryOperation.LIVE_PORT_ARRIVALS,
            QueryOperation.VESSEL_ETA,
            QueryOperation.VESSEL_DELAY,
            QueryOperation.ETA_COMPARISON,
        },
        VisualizationIntent.DISTRIBUTION: {
            QueryOperation.DWELL_SUMMARY,
            QueryOperation.DWELL_DISTRIBUTION,
            QueryOperation.ROUTE_TRAVEL_TIME,
        },
        VisualizationIntent.BOXPLOT: {
            QueryOperation.DWELL_SUMMARY,
            QueryOperation.DWELL_DISTRIBUTION,
            QueryOperation.ROUTE_TRAVEL_TIME,
        },
        VisualizationIntent.STACKED_BAR: {
            QueryOperation.VESSEL_TYPE_COMPOSITION,
            QueryOperation.CARBON,
        },
    }
    compatible_operations = intent_compatibility.get(plan.requested_visual)
    if compatible_operations is not None and operation not in compatible_operations:
        return [
            _omitted(
                "unsupported_visual",
                f"The requested {plan.requested_visual.value} would not represent this result faithfully.",
            )
        ]

    if (
        plan.requested_visual == VisualizationIntent.BOXPLOT
        and "distribution_observations" in by_id
        and "distribution_summary" in by_id
    ):
        observations = by_id["distribution_observations"]
        value_field = _field(observations, ["dwell_minutes", "duration_h"])
        if value_field:
            visual = DistributionVisualization(
                id="distribution_boxplot",
                title="Duration box plot",
                dataset_id=observations.id,
                table_fallback_dataset_id=table.id if table else observations.id,
                accessible_summary=(
                    "Exact quartiles, Tukey whiskers, and provenance-backed outliers "
                    "computed from the returned duration observations."
                ),
                chart_type="boxplot",
                value_field=value_field,
                unit=_unit(observations, value_field),
            )
            return [
                _validated(
                    _with_v21_bindings(visual, by_id, plan, caveats),
                    by_id,
                    plan,
                )
            ]
    if plan.requested_visual == VisualizationIntent.BOXPLOT:
        return [
            _omitted(
                "unsupported_visual",
                "A truthful box plot requires raw observations or exact quartiles; this result exposes only aggregate histogram or percentile rows.",
            )
        ]

    carbon_dataset = (table or chart) if operation == QueryOperation.CARBON else None
    carbon_fields = _carbon_metric_fields(plan, carbon_dataset) if carbon_dataset is not None else []
    carbon_multiseries_intents = {
        VisualizationIntent.AUTO,
        VisualizationIntent.LINE,
        VisualizationIntent.AREA,
        VisualizationIntent.BAR,
        VisualizationIntent.STACKED_BAR,
    }
    if (
        operation == QueryOperation.CARBON
        and carbon_dataset is not None
        and plan.requested_visual in carbon_multiseries_intents
        and (len(carbon_fields) > 1 or plan.requested_visual == VisualizationIntent.STACKED_BAR)
    ):
        x = _field(carbon_dataset, ["date", "scenario", "call_id", "port_key", "port_label", "mmsi"])
        if not x or len(carbon_fields) < 2:
            return [
                _omitted(
                    "unsupported_visual",
                    "A carbon composition chart requires a category or date plus at least two requested pollutant series.",
                )
            ]
        fields_by_unit: Dict[str, List[str]] = {}
        for field in carbon_fields:
            unit = _unit(carbon_dataset, field)
            if not unit:
                return [_omitted("validation_failed", f"Carbon field {field} has no declared unit.")]
            fields_by_unit.setdefault(unit, []).append(field)
        for index, (unit, fields) in enumerate(fields_by_unit.items(), start=1):
            if plan.requested_visual == VisualizationIntent.STACKED_BAR:
                chart_type = "stacked_bar"
            elif plan.requested_visual == VisualizationIntent.BAR:
                chart_type = "grouped_bar" if len(fields) > 1 else "bar"
            elif plan.requested_visual == VisualizationIntent.AREA:
                chart_type = "area"
            elif x == "date" and carbon_dataset.row_count > 1:
                chart_type = "line"
            else:
                chart_type = "grouped_bar" if len(fields) > 1 else "bar"
            visuals.append(
                CartesianVisualization(
                    id=f"carbon_series_{index}",
                    title=(
                        f"Carbon emissions over time ({unit})"
                        if chart_type in {"line", "area"}
                        else f"Carbon emissions comparison ({unit})"
                    ),
                    dataset_id=carbon_dataset.id,
                    table_fallback_dataset_id=table.id if table else carbon_dataset.id,
                    accessible_summary=(
                        "All available requested pollutant series are shown, with incompatible emissions units separated into small multiples."
                    ),
                    chart_type=chart_type,
                    x_field=x,
                    y_fields=fields,
                    y_unit=unit,
                    stacked=chart_type == "stacked_bar",
                )
            )

    elif operation in {QueryOperation.ARRIVALS, QueryOperation.PEAK_ARRIVAL_DAY, QueryOperation.CARBON}:
        time_series_dataset = (
            carbon_dataset
            if operation == QueryOperation.CARBON and carbon_dataset is not None
            else chart
        )
        dated_chart = bool(
            time_series_dataset
            and (_datetime_field(time_series_dataset) or _field(time_series_dataset, ["date"]))
        )
        explicit_time_series = plan.requested_visual in {
            VisualizationIntent.LINE,
            VisualizationIntent.AREA,
            VisualizationIntent.BAR,
        }
        planned_time_series = plan.requested_visual == VisualizationIntent.AUTO and "date" in plan.dimensions
        auto_companion = bool(
            plan.requested_visual == VisualizationIntent.AUTO
            and time_series_dataset is not None
            and (
                time_series_dataset.row_count > 0
                if operation == QueryOperation.CARBON
                else time_series_dataset.row_count > 1
            )
            and dated_chart
        )
        wants_time_series = explicit_time_series or planned_time_series or auto_companion

        # Preserve the scalar answer as the first presentation while exposing
        # the validated dated rows that support it as a companion graph.
        if (
            auto_companion
            and "date" not in plan.dimensions
            and operation in {QueryOperation.ARRIVALS, QueryOperation.CARBON}
            and summary is not None
        ):
            value = _field(summary, ["arrival_count", "value", "arrivals_vessels", "total"])
            if value:
                visuals.append(
                    KPIVisualization(
                        id="primary_kpi",
                        title="Computed result",
                        dataset_id=summary.id,
                        table_fallback_dataset_id=table.id if table else summary.id,
                        accessible_summary="The principal computed value for the selected scope.",
                        value_field=value,
                        label=plan.metric or "Result",
                        unit=_unit(summary, value),
                    )
                )

        if wants_time_series and time_series_dataset is not None and dated_chart:
            date_field = _datetime_field(time_series_dataset) or "date"
            y_candidates = (
                [
                    *_carbon_metric_fields(plan, time_series_dataset),
                    "ttw_co2e_t",
                    "wtw_co2e_t",
                    "co2_t",
                    "nox_kg",
                    "sox_kg",
                    "pm_kg",
                ]
                if operation == QueryOperation.CARBON
                else ["arrival_count", "arrivals_vessels", "value"]
            )
            y = _field(time_series_dataset, y_candidates)
            if y:
                visuals.append(
                    CartesianVisualization(
                        id="primary_chart",
                        title=(
                            "Carbon emissions over time"
                            if operation == QueryOperation.CARBON
                            else "Daily arrivals supporting the peak"
                            if operation == QueryOperation.PEAK_ARRIVAL_DAY
                            else "Arrivals over time"
                        ),
                        dataset_id=time_series_dataset.id,
                        table_fallback_dataset_id=(
                            table.id if table else time_series_dataset.id
                        ),
                        accessible_summary=(
                            "The computed peak is shown in the context of the full chronological daily arrival series."
                            if operation == QueryOperation.PEAK_ARRIVAL_DAY
                            else "A time-series view of the computed values in chronological order."
                        ),
                        chart_type=(
                            "bar"
                            if plan.requested_visual == VisualizationIntent.BAR
                            else "area"
                            if plan.requested_visual == VisualizationIntent.AREA
                            else "line"
                        ),
                        x_field=date_field,
                        y_fields=[y],
                        y_unit=_unit(time_series_dataset, y),
                    )
                )
        if (
            not any(isinstance(visual, CartesianVisualization) for visual in visuals)
            and wants_time_series
            and plan.requested_visual in {VisualizationIntent.LINE, VisualizationIntent.AREA, VisualizationIntent.BAR}
        ):
            return [
                _omitted(
                    "unsupported_visual",
                    "The requested time-series chart needs dated numeric rows, which are unavailable for this result.",
                )
            ]
        if not visuals and summary is not None:
            value = _field(summary, ["arrival_count", "value", "arrivals_vessels", "total"])
            if value:
                visuals.append(
                    KPIVisualization(
                        id="primary_kpi",
                        title="Computed result",
                        dataset_id=summary.id,
                        table_fallback_dataset_id=table.id if table else summary.id,
                        accessible_summary="The principal computed value for the selected scope.",
                        value_field=value,
                        label=plan.metric or "Result",
                        unit=next((c.unit for c in summary.columns if c.field == value), None),
                    )
                )

    elif operation in {
        QueryOperation.TOP_PORTS,
        QueryOperation.ARRIVALS_MULTI,
        QueryOperation.PORT_COMPARISON,
        QueryOperation.PRESSURE_BY_VESSEL_TYPE,
        QueryOperation.DIAGNOSTIC,
        QueryOperation.BUSIEST_WEEKDAY,
        QueryOperation.BUSIEST_HOUR,
        QueryOperation.WEEKDAY_COMPARISON,
        QueryOperation.CONGESTION_WEEKDAY_COMPARISON,
        QueryOperation.PEAK_CONGESTION_DAY,
    } and chart is not None:
        x = _field(
            chart,
            [
                "port",
                "port_key",
                "port_label",
                "vessel_type_norm",
                "day_of_week",
                "hour",
                "date",
                "scope_label",
            ],
        )
        y = _field(
            chart,
            [
                "value",
                "arrival_count",
                "arrivals_vessels",
                "congestion_index",
                "pressure_index",
                "mean_congestion_index",
                "median_dwell_minutes",
                "calls",
            ],
        )
        if x and y:
            calendar = x in {"day_of_week", "hour"}
            comparison_operation = operation in {
                QueryOperation.WEEKDAY_COMPARISON,
                QueryOperation.CONGESTION_WEEKDAY_COMPARISON,
            }
            visuals.append(
                CartesianVisualization(
                    id="primary_comparison",
                    title="Comparison",
                    dataset_id=chart.id,
                    table_fallback_dataset_id=table.id if table else chart.id,
                    accessible_summary="A comparison of the computed categories for the selected scope.",
                    chart_type="grouped_bar" if comparison_operation else "bar",
                    x_field=x,
                    y_fields=[y],
                    orientation="horizontal" if not calendar else "vertical",
                    sort="calendar" if calendar else "descending",
                    y_unit=next((c.unit for c in chart.columns if c.field == y), None),
                )
            )

    elif operation == QueryOperation.CONGESTION:
        explicit_pressure_chart = plan.requested_visual in {
            VisualizationIntent.LINE,
            VisualizationIntent.AREA,
            VisualizationIntent.BAR,
        }
        auto_pressure_chart = bool(
            plan.requested_visual == VisualizationIntent.AUTO
            and chart is not None
            and chart.row_count > 0
            and (_datetime_field(chart) or _field(chart, ["date"]))
        )
        wants_pressure_chart = explicit_pressure_chart or auto_pressure_chart

        # A one-day pressure answer remains a KPI, with the observed dated
        # index added as a compact bar rather than implying a multi-day trend.
        if auto_pressure_chart and chart is not None and chart.row_count == 1 and summary is not None:
            value_field = _field(summary, ["congestion_index"])
            if value_field:
                visuals.append(
                    KPIVisualization(
                        id="pressure_kpi",
                        title="Port pressure",
                        dataset_id=summary.id,
                        table_fallback_dataset_id=table.id if table else summary.id,
                        accessible_summary="The validated pressure_v2 value; 1.00 is the matching historical baseline.",
                        value_field=value_field,
                        label="Pressure vs 1.00 baseline",
                        unit=_unit(summary, value_field),
                    )
                )

        if wants_pressure_chart and chart is not None:
            date_field = _datetime_field(chart) or _field(chart, ["date"])
            value_field = _field(chart, ["congestion_index"])
            if date_field and value_field:
                single_observation = chart.row_count == 1
                visuals.append(
                    CartesianVisualization(
                        id="pressure_time_series",
                        title="Port pressure snapshot" if single_observation else "Port pressure over time",
                        dataset_id=chart.id,
                        table_fallback_dataset_id=table.id if table else chart.id,
                        accessible_summary=(
                            "The dated pressure_v2 observation is shown against the documented 1.00 baseline context."
                            if single_observation
                            else "Validated pressure_v2 observations in chronological order relative to the 1.00 baseline."
                        ),
                        chart_type=(
                            "bar"
                            if plan.requested_visual == VisualizationIntent.BAR or single_observation
                            else "area"
                            if plan.requested_visual == VisualizationIntent.AREA
                            else "line"
                        ),
                        x_field=date_field,
                        y_fields=[value_field],
                        y_unit=_unit(chart, value_field),
                    )
                )
        if not visuals and summary is not None:
            value_field = _field(summary, ["congestion_index"])
            if value_field:
                visuals.append(
                    KPIVisualization(
                        id="pressure_kpi",
                        title="Port pressure",
                        dataset_id=summary.id,
                        table_fallback_dataset_id=table.id if table else summary.id,
                        accessible_summary="The validated pressure_v2 value; 1.00 is the matching historical baseline.",
                        value_field=value_field,
                        label="Pressure vs 1.00 baseline",
                        unit=_unit(summary, value_field),
                    )
                )

    elif operation == QueryOperation.VESSEL_TYPE_COMPOSITION and chart is not None:
        composition = (
            table
            if table is not None and "share_percent" in dataset_fields(table)
            else chart
        )
        category = _field(composition, ["vessel_type_norm", "vessel_type", "category"])
        value = _field(
            composition,
            ["arrival_count", "share_percent", "share_pct", "share", "arrivals_vessels", "value"],
        )
        scope = _field(composition, ["scope", "port", "port_key"])
        if category and value and scope:
            visuals.append(
                CartesianVisualization(
                    id="composition_chart",
                    title="Arrival composition by vessel type",
                    dataset_id=composition.id,
                    table_fallback_dataset_id=table.id if table else composition.id,
                    accessible_summary="A stacked breakdown of arrival-event counts by vessel type.",
                    chart_type="stacked_bar",
                    x_field=scope,
                    y_fields=[value],
                    series_field=category,
                    y_unit=next((c.unit for c in composition.columns if c.field == value), None),
                    stacked=True,
                )
            )

    elif operation == QueryOperation.ARRIVAL_PATTERN and chart is not None:
        x = _field(chart, ["hour"])
        y = _field(chart, ["day_of_week"])
        value = _field(chart, ["arrival_count", "arrivals_vessels", "value"])
        if x and y and value:
            visuals.append(
                HeatmapVisualization(
                    id="weekday_hour_heatmap",
                    title="Arrivals by weekday and hour",
                    dataset_id=chart.id,
                    table_fallback_dataset_id=table.id if table else chart.id,
                    accessible_summary="Calendar-ordered weekday rows and UTC hour columns show historical arrival intensity.",
                    x_field=x,
                    y_field=y,
                    value_field=value,
                    unit=next((c.unit for c in chart.columns if c.field == value), None),
                )
            )

    elif operation == QueryOperation.CORRELATION and chart is not None:
        x = _field(chart, ["arrival_count", "arrivals_vessels"])
        y = _field(chart, ["median_dwell_minutes"])
        if x and y:
            visuals.append(
                CartesianVisualization(
                    id="correlation_scatter",
                    title="Arrivals and dwell association",
                    dataset_id=chart.id,
                    table_fallback_dataset_id=table.id if table else chart.id,
                    accessible_summary="Each point is one paired historical day; proximity does not imply causation.",
                    chart_type="scatter",
                    x_field=x,
                    y_fields=[y],
                    x_unit=next((c.unit for c in chart.columns if c.field == x), None),
                    y_unit=next((c.unit for c in chart.columns if c.field == y), None),
                )
            )

    elif (
        operation == QueryOperation.DWELL_SUMMARY
        and plan.aggregation == "mean"
        and summary is not None
    ):
        value = _field(summary, ["mean_dwell_hours"])
        if value:
            visuals.append(
                KPIVisualization(
                    id="mean_dwell_kpi",
                    title="Mean completed dwell",
                    dataset_id=summary.id,
                    table_fallback_dataset_id=table.id if table else summary.id,
                    accessible_summary="Mean completed dwell for the bounded port-call scope.",
                    value_field=value,
                    label="Mean completed dwell",
                    unit=_unit(summary, value),
                )
            )

    elif operation in {QueryOperation.DWELL_SUMMARY, QueryOperation.DWELL_DISTRIBUTION, QueryOperation.ROUTE_TRAVEL_TIME} and chart is not None:
        distribution_source = (
            by_id.get("distribution_observations")
            if operation == QueryOperation.DWELL_SUMMARY
            else chart
        ) or chart
        value = _field(
            distribution_source,
            [
                "dwell_minutes",
                "bin_midpoint_minutes",
                "duration_h",
                "median_duration_h",
                "median_dwell_minutes",
            ],
        )
        category = _field(distribution_source, ["percentile", "vessel_type_norm", "vessel_type", "port_key"])
        if value:
            visuals.append(
                DistributionVisualization(
                    id="distribution_chart",
                    title="Distribution",
                    dataset_id=distribution_source.id,
                    table_fallback_dataset_id=table.id if table else distribution_source.id,
                    accessible_summary="Distribution of the validated duration values in the selected scope.",
                    chart_type=(
                        "histogram"
                        if operation in {QueryOperation.DWELL_SUMMARY, QueryOperation.DWELL_DISTRIBUTION}
                        else "percentile"
                    ),
                    value_field=value,
                    count_field=_field(distribution_source, ["calls"]) if operation == QueryOperation.DWELL_DISTRIBUTION else None,
                    category_field=category,
                    bins=20 if operation == QueryOperation.DWELL_DISTRIBUTION else None,
                    unit=next((c.unit for c in distribution_source.columns if c.field == value), None),
                )
            )

    elif operation == QueryOperation.MMSI_PORT_STAYS:
        if summary is not None:
            value_field = _field(summary, ["duration_h"])
            if value_field:
                visuals.append(
                    KPIVisualization(
                        id="port_stay_duration_kpi",
                        title="Time in port",
                        dataset_id=summary.id,
                        table_fallback_dataset_id=table.id if table else summary.id,
                        accessible_summary="Total validated time in port for the selected vessel and date scope.",
                        value_field=value_field,
                        label="Total dwell",
                        unit=_unit(summary, value_field),
                    )
                )

        if plan.requested_visual in {
            VisualizationIntent.AUTO,
            VisualizationIntent.LINE,
            VisualizationIntent.AREA,
            VisualizationIntent.BAR,
        } and chart is not None:
            time_field = _datetime_field(chart) or _field(chart, ["arrival_time"])
            value_field = _field(chart, ["dwell_minutes"])
            if time_field and value_field:
                visuals.append(
                    CartesianVisualization(
                        id="port_stay_duration_chart",
                        title="Port-stay duration",
                        dataset_id=chart.id,
                        table_fallback_dataset_id=table.id if table else chart.id,
                        accessible_summary="Port-stay durations ordered by recorded arrival time.",
                        chart_type=(
                            "bar"
                            if plan.requested_visual in {VisualizationIntent.AUTO, VisualizationIntent.BAR}
                            else "area"
                            if plan.requested_visual == VisualizationIntent.AREA
                            else "line"
                        ),
                        x_field=time_field,
                        y_fields=[value_field],
                        y_unit=_unit(chart, value_field),
                    )
                )

    elif operation in {
        QueryOperation.FORECAST_ARRIVALS,
        QueryOperation.FORECAST_CONGESTION,
        QueryOperation.FORECAST_COMPARISON,
    } and chart is not None:
        predicted = _field(chart, ["predicted", "value"])
        lower = _field(chart, ["lower"])
        upper = _field(chart, ["upper"])
        date_field: Optional[str] = None
        if operation == QueryOperation.FORECAST_COMPARISON:
            category = _field(chart, ["port", "day_of_week"])
            if category and predicted:
                calendar = category == "day_of_week"
                visuals.append(
                    CartesianVisualization(
                        id="forecast_comparison_chart",
                        title="Forecast comparison",
                        dataset_id=chart.id,
                        table_fallback_dataset_id=table.id if table else chart.id,
                        accessible_summary=(
                            "Point predictions are compared by category; lower and upper uncertainty intervals remain available in the data-table fallback."
                        ),
                        chart_type="bar",
                        x_field=category,
                        y_fields=[predicted],
                        orientation="vertical" if calendar else "horizontal",
                        sort="calendar" if calendar else "descending",
                        y_unit=_unit(chart, predicted) or "index",
                    )
                )
        else:
            date_field = _field(chart, ["date"])
        if operation != QueryOperation.FORECAST_COMPARISON and date_field and predicted:
            unit = _unit(chart, predicted) or (
                "vessels" if operation == QueryOperation.FORECAST_ARRIVALS else "index"
            )
            interval_usable = bool(
                lower
                and upper
                and _usable_interval(
                    chart,
                    point_field=predicted,
                    lower_field=lower,
                    upper_field=upper,
                )
            )
            if interval_usable:
                visuals.append(
                    ForecastVisualization(
                        id="forecast_chart",
                        title="Forecast and uncertainty interval",
                        dataset_id=chart.id,
                        table_fallback_dataset_id=table.id if table else chart.id,
                        accessible_summary="Predicted values with lower and upper uncertainty bounds.",
                        date_field=date_field,
                        predicted_field=predicted,
                        lower_field=str(lower),
                        upper_field=str(upper),
                        actual_field=_field(chart, ["actual"]),
                        unit=unit,
                    )
                )
            else:
                actual = _field(chart, ["actual"])
                y_fields = [field for field in (actual, predicted) if field]
                visuals.append(
                    CartesianVisualization(
                        id="forecast_prediction_chart",
                        title="Forecast",
                        dataset_id=chart.id,
                        table_fallback_dataset_id=table.id if table else chart.id,
                        accessible_summary=(
                            "Observed and predicted values are shown. Interval bounds were unavailable for display."
                        ),
                        chart_type="line",
                        x_field=date_field,
                        y_fields=y_fields,
                        y_unit=unit,
                    )
                )

    elif operation == QueryOperation.ARRIVAL_ANOMALY and chart is not None:
        x = _field(chart, ["date"])
        values = [field for field in ("arrival_count", "arrivals_vessels", "threshold") if field in dataset_fields(chart)]
        if x and values:
            visuals.append(
                CartesianVisualization(
                    id="anomaly_chart",
                    title="Observed arrivals and anomaly threshold",
                    dataset_id=chart.id,
                    table_fallback_dataset_id=table.id if table else chart.id,
                    accessible_summary="Observed arrivals are plotted with the detection threshold; crossings are potential anomalies.",
                    chart_type="line",
                    x_field=x,
                    y_fields=values,
                    y_unit="count",
                    highlight=CartesianHighlight(
                        condition_field="is_anomaly",
                        value_field="arrival_count" if "arrival_count" in values else "arrivals_vessels",
                        label="Detected arrival spike",
                    ),
                )
            )

    elif operation == QueryOperation.AIS_JUMP and table is not None:
        lat = _field(table, ["latitude"])
        lon = _field(table, ["longitude"])
        if lat and lon:
            visuals.append(
                MapVisualization(
                    id="anomaly_map",
                    title="Potential AIS coordinate jumps",
                    dataset_id=table.id,
                    table_fallback_dataset_id=table.id,
                    accessible_summary=(
                        "Map of coordinates associated with potential sudden AIS jumps; the validated anomaly rows remain available as the data-table fallback."
                    ),
                    latitude_field=lat,
                    longitude_field=lon,
                    label_field=_field(table, ["mmsi", "stable_id"]),
                    value_field=_field(table, ["distance_km"]),
                )
            )
        if chart is not None:
            time_field = _datetime_field(chart) or _field(chart, ["timestamp", "timestamp_full"])
            if time_field:
                for field, title in (
                    ("distance_km", "Jump distance over time"),
                    ("implied_speed_kn", "Implied speed over time"),
                    ("speed_kn", "Reported speed over time"),
                ):
                    if field not in dataset_fields(chart) or not _has_finite_values(chart, field):
                        continue
                    visuals.append(
                        CartesianVisualization(
                            id=f"ais_jump_{field}",
                            title=title,
                            dataset_id=chart.id,
                            table_fallback_dataset_id=table.id,
                            accessible_summary=(
                                "Chronological movement values support the mapped anomaly events; each unit family is plotted on its own axis."
                            ),
                            chart_type="line",
                            x_field=time_field,
                            y_fields=[field],
                            y_unit=_unit(chart, field),
                        )
                    )

    elif plan.eta_watch_intent is not None and table is not None:
        fields = dataset_fields(table)
        label_field = _field(table, ["vessel_label", "vessel_name", "mmsi"])
        eta_timeline = by_id.get("eta_timeline") or table
        eta_timeline_fields = dataset_fields(eta_timeline)
        eta_label_field = _field(
            eta_timeline,
            ["vessel_label", "vessel_name", "mmsi"],
        )
        eta_field = _field(
            eta_timeline,
            ["reported_eta_utc", "ais_eta_utc", "eta_utc"],
        )
        latitude = _field(table, ["latitude"])
        longitude = _field(table, ["longitude"])
        position_time = _field(
            table,
            ["position_time_utc", "ais_location_time_utc", "observation_time_utc"],
        )
        intent = plan.eta_watch_intent

        if intent == ETAWatchIntent.DESTINATION_LOAD:
            destination = _field(
                table,
                ["destination_name", "destination_locode", "port_locode"],
            )
            count_field = _field(table, ["inbound_vessels"])
            if destination and count_field and _has_finite_values(table, count_field):
                visuals.append(
                    CartesianVisualization(
                        id="eta_destination_load",
                        title="AIS-reported inbound destination load",
                        dataset_id=table.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary=(
                            "Verified destination groups ranked by the number of "
                            "matching vessel-reported AIS signals in the requested UTC window."
                        ),
                        chart_type="bar",
                        x_field=destination,
                        y_fields=[count_field],
                        orientation="horizontal",
                        sort="descending",
                        y_unit=_unit(table, count_field) or "vessels",
                    )
                )
        elif intent == ETAWatchIntent.ETA_REVISIONS:
            change_field = _field(table, ["eta_change_minutes"])
            if label_field and change_field and _has_finite_values(table, change_field):
                visuals.append(
                    CartesianVisualization(
                        id="eta_reported_revisions",
                        title="Vessel-reported ETA revisions",
                        dataset_id=table.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary=(
                            "Signed changes between vessel-reported AIS ETA observations; "
                            "positive values moved later and negative values moved earlier. "
                            "This is not a confirmed operational delay."
                        ),
                        chart_type="bar",
                        x_field=label_field,
                        y_fields=[change_field],
                        orientation="horizontal",
                        # The executor already orders revision rows by absolute
                        # magnitude. Preserve that priority instead of
                        # re-sorting signed values and pushing a large earlier
                        # revision below a smaller later revision.
                        sort="none",
                        y_unit=_unit(table, change_field) or "minutes",
                        reference_lines=[
                            ReferenceLineSpec(
                                id="eta_revision_zero",
                                label="No ETA change",
                                axis="y",
                                value=0.0,
                                unit=_unit(table, change_field) or "minutes",
                                line_style="solid",
                            )
                        ],
                    )
                )
        else:
            map_supported = bool(
                latitude
                and longitude
                and _has_finite_values(table, latitude)
                and _has_finite_values(table, longitude)
            )
            if map_supported and intent in {
                ETAWatchIntent.SHIFT_HANDOVER,
                ETAWatchIntent.INBOUND_WATCHLIST,
                ETAWatchIntent.LOW_SPEED_EXCEPTIONS,
                ETAWatchIntent.VESSEL_STATUS,
            } and (
                intent != ETAWatchIntent.INBOUND_WATCHLIST
                or "position" in plan.dimensions
            ):
                visuals.append(
                    MapVisualization(
                        id="eta_watch_positions",
                        title=(
                            "Low-speed vessel observations"
                            if intent == ETAWatchIntent.LOW_SPEED_EXCEPTIONS
                            else "Last validated AIS positions"
                        ),
                        dataset_id=table.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary=(
                            "Independent source-reported AIS position points. No route, "
                            "port call, arrival, or destination path is inferred."
                        ),
                        latitude_field=latitude,
                        longitude_field=longitude,
                        label_field=label_field,
                        value_field=_field(table, ["speed_kn", "sog_kn"]),
                        geometry_mode="points",
                        timestamp_field=position_time,
                    )
                )
            if intent == ETAWatchIntent.LOW_SPEED_EXCEPTIONS:
                speed_field = _field(table, ["speed_kn", "sog_kn"])
                if (
                    label_field
                    and speed_field
                    and _has_finite_values(table, speed_field)
                ):
                    threshold = float(plan.speed_threshold_kn or 2.0)
                    visuals.append(
                        CartesianVisualization(
                            id="eta_low_speed_ranking",
                            title="Low-speed vessels requiring attention",
                            dataset_id=table.id,
                            table_fallback_dataset_id=table.id,
                            accessible_summary=(
                                "Validated vessel-reported speeds ranked from slowest "
                                "to fastest, with the requested low-speed threshold "
                                "shown for operational triage."
                            ),
                            chart_type="bar",
                            x_field=label_field,
                            y_fields=[speed_field],
                            orientation="horizontal",
                            sort="ascending",
                            y_unit=_unit(table, speed_field) or "knots",
                            reference_lines=[
                                ReferenceLineSpec(
                                    id="eta_low_speed_threshold",
                                    label=f"Threshold {threshold:g} kn",
                                    axis="y",
                                    value=threshold,
                                    unit=_unit(table, speed_field) or "knots",
                                )
                            ],
                        )
                    )
            elif (
                eta_field
                and eta_label_field
                and _has_values(eta_timeline, eta_field)
            ):
                visuals.append(
                    TimelineVisualization(
                        id="eta_watch_timeline",
                        title=(
                            "Next vessel-reported ETA"
                            if eta_timeline.row_count == 1
                            else "Due-soon vessel-reported ETAs"
                            if intent == ETAWatchIntent.SHIFT_HANDOVER
                            else "Vessel-reported ETA schedule"
                        ),
                        dataset_id=eta_timeline.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary=(
                            "Only fresh, valid, future-facing vessel-reported AIS ETAs "
                            "are placed on separate vessel lanes in UTC. Rows without "
                            "a valid ETA remain in the verified table and are not plotted."
                        ),
                        time_field=eta_field,
                        label_field=eta_label_field,
                        lane_field=(
                            eta_label_field
                            if eta_timeline.row_count > 1
                            else None
                        ),
                        detail_fields=[
                            field
                            for field in (
                                "destination_name",
                                "destination_locode",
                                "destination_raw",
                                "speed_kn",
                                "position_time_utc",
                                "eta_change_minutes",
                                "is_low_speed",
                                "is_position_stale",
                                "is_missing_eta",
                            )
                            if field in eta_timeline_fields
                        ],
                    )
                )

        # Shift handovers and exception triage always retain a compact status
        # table alongside the analytical chart. Other intents also receive the
        # table when it is the only faithful visual.
        if intent in {
            ETAWatchIntent.SHIFT_HANDOVER,
            ETAWatchIntent.LOW_SPEED_EXCEPTIONS,
            ETAWatchIntent.SIGNAL_QUALITY,
        } or not visuals:
            visuals.append(
                TableVisualization(
                    id="eta_watch_status_table",
                    title=(
                        "Signals needing confirmation"
                        if intent == ETAWatchIntent.SIGNAL_QUALITY
                        else "ETA Watch operational rows"
                    ),
                    dataset_id=table.id,
                    table_fallback_dataset_id=table.id,
                    accessible_summary=(
                        "Validated vessel-broadcast rows with requested operational "
                        "fields and explicit missing or stale states."
                    ),
                    visible_fields=[
                        field
                        for field in (
                            "vessel_label",
                            "mmsi",
                            "destination_name",
                            "destination_locode",
                            "reported_eta_utc",
                            "latitude",
                            "longitude",
                            "speed_kn",
                            "position_time_utc",
                            "eta_change_minutes",
                            "is_low_speed",
                            "is_position_stale",
                            "is_missing_eta",
                        )
                        if field in fields
                    ],
                )
            )

    elif operation in {
        QueryOperation.FIRST_ARRIVAL,
        QueryOperation.LAST_ARRIVAL,
        QueryOperation.FIRST_DEPARTURE,
        QueryOperation.FIRST_ROUTE_VESSEL,
        QueryOperation.LIVE_PORT_ARRIVALS,
        QueryOperation.VESSEL_ETA,
        QueryOperation.VESSEL_DELAY,
        QueryOperation.ETA_COMPARISON,
    } and table is not None:
        live_eta = operation in {
            QueryOperation.LIVE_PORT_ARRIVALS,
            QueryOperation.VESSEL_ETA,
            QueryOperation.VESSEL_DELAY,
            QueryOperation.ETA_COMPARISON,
        }
        ais_destination_only = bool(
            live_eta
            and "ais_eta_utc" in dataset_fields(table)
            and _has_values(table, "ais_eta_utc")
            and (
                "official_eta_utc" not in dataset_fields(table)
                or not _has_values(table, "official_eta_utc")
            )
        )
        time_field = (
            "ais_eta_utc"
            if ais_destination_only
            else _field(
                table,
                ["official_eta_utc", "arrival_time", "departure_time", "arrival_date"],
            )
        )
        label_field = _field(
            table,
            ["vessel_name", "mmsi", "voyage_id", "port_key", "port_locode"],
        )
        if time_field and label_field:
            visuals.append(
                TimelineVisualization(
                    id="event_timeline",
                    title=(
                        "Vessel-reported AIS destination ETAs"
                        if ais_destination_only
                        else
                        "Official ETA schedule and AIS announcements"
                        if live_eta
                        else "Recorded events"
                    ),
                    dataset_id=table.id,
                    table_fallback_dataset_id=table.id,
                    accessible_summary=(
                        "Fresh Fintraffic-observed vessel broadcasts ordered by their self-reported "
                        "AIS ETA. These destination signals are neither an official port schedule nor "
                        "an arrival confirmation or prediction."
                        if ais_destination_only
                        else
                        "Portnet official scheduled arrivals ordered in UTC; validated AIS vessel-announced "
                        "ETAs and nullable announced variance remain available in the data table."
                        if live_eta
                        else "Events ordered by their recorded UTC timestamp."
                    ),
                    time_field=time_field,
                    label_field=label_field,
                    detail_fields=[
                        field
                        for field in (
                            "port_locode",
                            "ais_destination",
                            "ais_destination_match",
                            "ais_eta_utc",
                            "ais_metadata_time_utc",
                            "ais_location_time_utc",
                            "sog_kn",
                            "announced_delay_minutes",
                            "variance_status",
                            "port_key",
                            "duration_h",
                        )
                        if field in dataset_fields(table)
                    ],
                )
            )
            delay_field = _field(table, ["announced_delay_minutes"])
            if (
                operation
                in {QueryOperation.VESSEL_DELAY, QueryOperation.ETA_COMPARISON}
                and delay_field
                and _has_finite_values(table, delay_field)
            ):
                visuals.insert(
                    0,
                    CartesianVisualization(
                        id="announced_eta_variance",
                        title="Announced ETA variance",
                        dataset_id=table.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary=(
                            "Validated AIS vessel-reported ETA minus Portnet official scheduled ETA; "
                            "positive minutes mean later. Rows without a validated match remain null in the table."
                        ),
                        chart_type="bar",
                        x_field=label_field,
                        y_fields=[delay_field],
                        orientation="horizontal",
                        sort="descending",
                        y_unit=_unit(table, delay_field) or "minutes",
                    ),
                )
            latitude = _field(table, ["latitude"])
            longitude = _field(table, ["longitude"])
            if (
                operation == QueryOperation.VESSEL_ETA
                and "position" in plan.dimensions
                and latitude
                and longitude
                and _has_finite_values(table, latitude)
                and _has_finite_values(table, longitude)
            ):
                visuals.insert(
                    0,
                    MapVisualization(
                        id="current_vessel_position",
                        title=(
                            "Latest AIS-observed vessel position"
                            if ais_destination_only
                            else "Current reported vessel position"
                        ),
                        dataset_id=table.id,
                        table_fallback_dataset_id=table.id,
                        accessible_summary=(
                            "Fresh Fintraffic-observed AIS positions shown as independent points; "
                            "no route, arrival, or port-call status is inferred."
                            if ais_destination_only
                            else "Fresh source-reported AIS positions shown as independent points; "
                            "no route is inferred."
                        ),
                        latitude_field=latitude,
                        longitude_field=longitude,
                        label_field=label_field,
                        value_field=_field(table, ["sog_kn"]),
                        geometry_mode="points",
                        timestamp_field=_field(table, ["ais_location_time_utc"]),
                    ),
                )

    elif operation == QueryOperation.MIXED_PORT_ROUTE_COMPARISON:
        for dataset in datasets:
            if not dataset.id.startswith("metric_"):
                continue
            x = _field(dataset, ["scope_label", "port", "route"])
            y = _field(dataset, ["value"])
            if x and y:
                visuals.append(
                    CartesianVisualization(
                        id=f"visual_{dataset.id}",
                        title=dataset.id.removeprefix("metric_").replace("_", " ").title(),
                        dataset_id=dataset.id,
                        table_fallback_dataset_id=table.id if table else dataset.id,
                        accessible_summary="Values are isolated to one unit family to avoid a misleading mixed axis.",
                        chart_type="bar",
                        x_field=x,
                        y_fields=[y],
                        orientation="horizontal",
                        sort="descending",
                        y_unit=_unit(dataset, y),
                    )
                )

    if not visuals and table is not None:
        visuals.append(
            TableVisualization(
                id="result_table",
                title="Result data",
                dataset_id=table.id,
                table_fallback_dataset_id=table.id,
                accessible_summary="Validated result rows are available as a data table.",
                visible_fields=[column.field for column in table.columns],
            )
        )
    if not visuals:
        return [_omitted("not_meaningful", "A graph would not add reliable information for this answer.")]
    return [
        _validated(
            _with_v21_bindings(visualization, by_id, plan, caveats),
            by_id,
            plan,
        )
        for visualization in visuals
    ]
