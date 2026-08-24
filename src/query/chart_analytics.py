"""Deterministic analytics used only to enrich validated chart payloads.

The functions in this module never rewrite the canonical answer.  Every
reported number is computed from rows included in the response and is copied
into an immutable fact slot before it is exposed as a chart insight.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    CartesianVisualization,
    ChartInsight,
    ColumnSpec,
    DatasetSpec,
    FactSlot,
    ForecastVisualization,
    QueryOperation,
    QueryPlan,
    VisualizationSpec,
)


def finite_number(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dataset_fields(dataset: DatasetSpec) -> set[str]:
    return {column.field for column in dataset.columns}


def _field(dataset: DatasetSpec, candidates: Iterable[str]) -> Optional[str]:
    fields = _dataset_fields(dataset)
    return next((candidate for candidate in candidates if candidate in fields), None)


def _unit(dataset: DatasetSpec, field: str) -> Optional[str]:
    return next((column.unit for column in dataset.columns if column.field == field), None)


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
    fields = _dataset_fields(dataset)
    selected: List[str] = []
    for pollutant in requested:
        for field in pollutant_fields.get(str(pollutant).strip().upper(), []):
            if field in fields and field not in selected:
                selected.append(field)
    return selected


def _append_column(
    dataset: DatasetSpec,
    *,
    field: str,
    label: str,
    data_type: str,
    values: Sequence[Any],
    unit: Optional[str] = None,
) -> DatasetSpec:
    if len(values) != dataset.row_count:
        raise ValueError(f"{field} has {len(values)} values for {dataset.row_count} rows")
    if field in _dataset_fields(dataset):
        return dataset
    rows = [
        {**row, field: value}
        for row, value in zip(dataset.rows, values)
    ]
    columns = [
        *dataset.columns,
        ColumnSpec(
            field=field,
            label=label,
            data_type=data_type,  # type: ignore[arg-type]
            unit=unit,
        ),
    ]
    return dataset.model_copy(update={"columns": columns, "rows": rows})


def _stable_row_ids(dataset: DatasetSpec) -> DatasetSpec:
    if "row_id" in _dataset_fields(dataset):
        return dataset
    values: List[str] = []
    for index, row in enumerate(dataset.rows):
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
        digest = hashlib.sha256(f"{dataset.id}:{index}:{payload}".encode("utf-8")).hexdigest()[:18]
        values.append(f"{dataset.id}_{digest}")
    return _append_column(
        dataset,
        field="row_id",
        label="Row ID",
        data_type="string",
        values=values,
    )


def _rolling_median(values: Sequence[Optional[float]], window: int = 7) -> List[Optional[float]]:
    output: List[Optional[float]] = []
    for index in range(len(values)):
        start = index - window + 1
        frame = values[max(0, start) : index + 1]
        if start < 0 or any(value is None for value in frame):
            output.append(None)
            continue
        output.append(float(median(value for value in frame if value is not None)))
    return output


def ols_metadata(dataset: DatasetSpec, x_field: str, y_field: str) -> Optional[Dict[str, float]]:
    pairs = [
        (x, y)
        for row in dataset.rows
        if (x := finite_number(row.get(x_field))) is not None
        and (y := finite_number(row.get(y_field))) is not None
    ]
    if len(pairs) < 3:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        return None
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in pairs
    ) / denominator
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * value for value in x_values]
    total_sum_squares = sum((value - y_mean) ** 2 for value in y_values)
    residual_sum_squares = sum(
        (observed - predicted) ** 2
        for observed, predicted in zip(y_values, fitted)
    )
    r_squared = (
        max(0.0, min(1.0, 1.0 - residual_sum_squares / total_sum_squares))
        if total_sum_squares > 0
        else 0.0
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "count": float(len(pairs)),
    }


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    q1 = _quantile(ordered, 0.25)
    q3 = _quantile(ordered, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inliers = [value for value in ordered if lower_fence <= value <= upper_fence]
    return {
        "minimum": ordered[0],
        "q1": q1,
        "median": _quantile(ordered, 0.5),
        "q3": q3,
        "maximum": ordered[-1],
        "lower_whisker": inliers[0] if inliers else ordered[0],
        "upper_whisker": inliers[-1] if inliers else ordered[-1],
        "p90": _quantile(ordered, 0.9),
        "count": float(len(ordered)),
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
    }


def enrich_chart_datasets(plan: QueryPlan, datasets: List[DatasetSpec]) -> List[DatasetSpec]:
    """Add reproducible row identifiers and defensible derived chart series."""

    enriched: List[DatasetSpec] = []
    for dataset in datasets:
        current = dataset
        fields = _dataset_fields(current)
        if plan.operation == QueryOperation.CARBON and "date" in fields:
            selected_fields = _carbon_metric_fields(plan, current)
            if selected_fields:
                point_field = selected_fields[0]
                point_values = [
                    finite_number(row.get(point_field))
                    for row in current.rows
                ]
                finite_values = [value for value in point_values if value is not None]
                if finite_values:
                    peak = max(finite_values)
                    # Keep the existing row identifiers stable: the peak flag is
                    # a presentation-only enrichment, not part of row identity.
                    current = _stable_row_ids(current)
                    current = _append_column(
                        current,
                        field="is_peak",
                        label="Peak Selected Carbon Metric",
                        data_type="boolean",
                        values=[
                            value is not None
                            and math.isclose(value, peak, rel_tol=1e-12, abs_tol=1e-12)
                            for value in point_values
                        ],
                    )
                    fields = _dataset_fields(current)
        if (
            current.id == "chart"
            and plan.operation in {QueryOperation.ARRIVALS, QueryOperation.PEAK_ARRIVAL_DAY}
            and {"date", "arrival_count"}.issubset(fields)
            and current.row_count >= 7
        ):
            arrivals = [finite_number(row.get("arrival_count")) for row in current.rows]
            current = _append_column(
                current,
                field="rolling_median_7",
                label="7-day Rolling Median",
                data_type="number",
                values=_rolling_median(arrivals),
                unit=_unit(current, "arrival_count"),
            )
            fields = _dataset_fields(current)
        if (
            current.id == "chart"
            and plan.operation == QueryOperation.CORRELATION
            and {"arrival_count", "median_dwell_minutes"}.issubset(fields)
        ):
            fit = ols_metadata(current, "arrival_count", "median_dwell_minutes")
            if fit is not None:
                fitted = [
                    (
                        fit["intercept"] + fit["slope"] * value
                        if (value := finite_number(row.get("arrival_count"))) is not None
                        and finite_number(row.get("median_dwell_minutes")) is not None
                        else None
                    )
                    for row in current.rows
                ]
                current = _append_column(
                    current,
                    field="ols_fitted_median_dwell_minutes",
                    label="OLS Fitted Median Dwell",
                    data_type="number",
                    values=fitted,
                    unit=_unit(current, "median_dwell_minutes"),
                )
        enriched.append(current)

    observations = next(
        (dataset for dataset in enriched if dataset.id == "distribution_observations"),
        None,
    )
    if observations is not None:
        value_field = _field(observations, ["dwell_minutes", "duration_h"])
        if value_field:
            values = [
                value
                for row in observations.rows
                if (value := finite_number(row.get(value_field))) is not None
                and (
                    value > 0
                    if value_field == "dwell_minutes"
                    else value >= 0
                )
            ]
            if values:
                summary = _distribution_summary(values)
                flagged = [
                    (
                        value < summary["lower_fence"] or value > summary["upper_fence"]
                        if (value := finite_number(row.get(value_field))) is not None
                        else False
                    )
                    for row in observations.rows
                ]
                observations = _append_column(
                    observations,
                    field="is_outlier",
                    label="Tukey Outlier",
                    data_type="boolean",
                    values=flagged,
                )
                enriched = [
                    observations if dataset.id == observations.id else dataset
                    for dataset in enriched
                ]
                summary_row = {
                    key: int(value) if key == "count" else value
                    for key, value in summary.items()
                    if key not in {"lower_fence", "upper_fence"}
                }
                summary_columns = [
                    ColumnSpec(
                        field=key,
                        label=key.replace("_", " ").title(),
                        data_type="integer" if key == "count" else "number",
                        unit=None if key == "count" else _unit(observations, value_field),
                    )
                    for key in summary_row
                ]
                enriched.append(
                    DatasetSpec(
                        id="distribution_summary",
                        columns=summary_columns,
                        rows=[summary_row],
                        row_count=1,
                    )
                )
                outlier_rows = [
                    dict(row)
                    for row in observations.rows
                    if row.get("is_outlier") is True
                ]
                if outlier_rows:
                    enriched.append(
                        DatasetSpec(
                            id="distribution_outliers",
                            columns=list(observations.columns),
                            rows=outlier_rows,
                            row_count=len(outlier_rows),
                        )
                    )

    return [_stable_row_ids(dataset) for dataset in enriched]


_FORECAST_QUALITY_PATTERN = re.compile(
    r"MASE=(?P<mase>\d+(?:\.\d+)?),\s*80%\s+interval\s+coverage=(?P<coverage>\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def forecast_quality(caveats: Iterable[str]) -> Optional[Dict[str, float]]:
    for caveat in caveats:
        match = _FORECAST_QUALITY_PATTERN.search(str(caveat))
        if not match:
            continue
        mase = float(match.group("mase"))
        coverage = float(match.group("coverage"))
        if mase < 1.0 and 0.70 <= coverage <= 0.90:
            return {
                "mase": mase,
                "interval_coverage": coverage,
                "interval_level": 0.8,
            }
    return None


def _format_number(value: float, *, decimals: int = 2) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def _metric_label(field: str) -> str:
    return {
        "ttw_co2e_t": "TTW CO2e",
        "wtw_co2e_t": "WTW CO2e",
        "co2_t": "CO2",
        "nox_kg": "NOx",
        "sox_kg": "SOx",
        "pm_kg": "PM",
    }.get(field, field.replace("_", " "))


def _fact(
    name: str,
    value: Any,
    *,
    unit: Optional[str],
    visualization_id: str,
) -> FactSlot:
    return FactSlot(
        name=name,
        value=value,
        unit=unit,
        entity=visualization_id,
        source="computed",
    )


def build_chart_insights(
    *,
    plan: QueryPlan,
    datasets: List[DatasetSpec],
    visualizations: List[VisualizationSpec],
    evidence_ids: Iterable[str] = (),
) -> Tuple[List[ChartInsight], List[FactSlot]]:
    """Return at most three deterministic observations plus their fact slots."""

    by_id = {dataset.id: dataset for dataset in datasets}
    evidence = list(dict.fromkeys(str(item) for item in evidence_ids if str(item)))[:5]
    insights: List[ChartInsight] = []
    facts: List[FactSlot] = []

    def add(
        *,
        visualization_id: str,
        insight_type: str,
        statement: str,
        values: Sequence[Tuple[str, Any, Optional[str]]],
    ) -> None:
        if len(insights) >= 3:
            return
        fact_names: List[str] = []
        for suffix, value, unit in values:
            name = f"chart.{visualization_id}.{suffix}"
            facts.append(_fact(name, value, unit=unit, visualization_id=visualization_id))
            fact_names.append(name)
        insights.append(
            ChartInsight(
                id=f"insight_{len(insights) + 1}_{visualization_id}",
                visualization_id=visualization_id,
                insight_type=insight_type,  # type: ignore[arg-type]
                statement=statement,
                fact_names=fact_names,
                evidence_ids=evidence,
            )
        )

    primary = next(
        (
            visual
            for visual in visualizations
            if visual.kind not in {"omitted", "table"} and visual.dataset_id in by_id
        ),
        None,
    )

    arrival_visual = next(
        (
            visual
            for visual in visualizations
            if isinstance(visual, CartesianVisualization)
            and visual.dataset_id in by_id
            and any(field in {"arrival_count", "arrivals_vessels", "value"} for field in visual.y_fields)
        ),
        None,
    )

    if (
        plan.operation in {QueryOperation.ARRIVALS, QueryOperation.PEAK_ARRIVAL_DAY}
        and arrival_visual is not None
    ):
        primary = arrival_visual
        dataset = by_id[primary.dataset_id or ""]
        value_field = _field(dataset, ["arrival_count", "arrivals_vessels", "value"])
        date_field = _field(dataset, ["date"])
        if value_field:
            observations = [
                (row.get(date_field) if date_field else None, value)
                for row in dataset.rows
                if (value := finite_number(row.get(value_field))) is not None
            ]
            if observations:
                peak_date, peak_value = max(observations, key=lambda item: item[1])
                period_median = float(median(value for _, value in observations))
                unit = _unit(dataset, value_field)
                display_unit = "arrival events" if unit == "count" else unit or "vessels"
                date_clause = f" on {str(peak_date)[:10]}" if peak_date else ""
                add(
                    visualization_id=primary.id,
                    insight_type="peak",
                    statement=(
                        f"Peak arrivals were {_format_number(peak_value)} {display_unit}"
                        f"{date_clause}; the period median was {_format_number(period_median)}."
                    ),
                    values=[
                        ("peak_value", peak_value, unit),
                        ("peak_date", peak_date, None),
                        ("period_median", period_median, unit),
                    ],
                )
                rolling = [
                    value
                    for row in dataset.rows
                    if (value := finite_number(row.get("rolling_median_7"))) is not None
                ]
                if len(rolling) >= 2:
                    change = rolling[-1] - rolling[0]
                    direction = "rose" if change > 0 else "fell" if change < 0 else "was unchanged"
                    add(
                        visualization_id=primary.id,
                        insight_type="trend",
                        statement=(
                            f"The validated 7-day rolling median {direction} by "
                            f"{_format_number(abs(change))} {display_unit} from its first complete window."
                        ),
                        values=[
                            ("rolling_first", rolling[0], unit),
                            ("rolling_last", rolling[-1], unit),
                            ("rolling_change", change, unit),
                        ],
                    )

    elif plan.operation in {
        QueryOperation.CONGESTION,
        QueryOperation.PRESSURE_BY_VESSEL_TYPE,
        QueryOperation.PEAK_CONGESTION_DAY,
    } and primary is not None:
        dataset = by_id[primary.dataset_id or ""]
        value_field = _field(dataset, ["congestion_index", "pressure_index"])
        if value_field:
            values = [
                value
                for row in dataset.rows
                if (value := finite_number(row.get(value_field))) is not None
            ]
            if values:
                peak = max(values)
                deviation = peak - 1.0
                status = "port pressure index"
                add(
                    visualization_id=primary.id,
                    insight_type="baseline_deviation",
                    statement=(
                        f"Peak {status} was {_format_number(peak)}—"
                        f"{_format_number(abs(deviation))} {'above' if deviation >= 0 else 'below'} "
                        "the 1.00 baseline."
                    ),
                    values=[
                        ("peak_pressure", peak, "index"),
                        ("baseline", 1.0, "index"),
                        ("peak_deviation", deviation, "index"),
                        ("pressure_status", status, None),
                    ],
                )

    elif plan.operation == QueryOperation.CARBON:
        carbon_visual = next(
            (
                visual
                for visual in visualizations
                if isinstance(visual, CartesianVisualization)
                and {"ttw_co2e_t", "wtw_co2e_t"}.issubset(set(visual.y_fields))
                and visual.dataset_id in by_id
            ),
            None,
        )
        if carbon_visual is not None:
            dataset = by_id[carbon_visual.dataset_id or ""]
            paired_values = [
                (ttw, wtw)
                for row in dataset.rows
                if (ttw := finite_number(row.get("ttw_co2e_t"))) is not None
                and (wtw := finite_number(row.get("wtw_co2e_t"))) is not None
            ]
            # The statement covers all returned rows, so do not silently treat
            # a missing boundary value as zero or compute over a partial subset.
            if paired_values and len(paired_values) == len(dataset.rows):
                ttw = sum(pair[0] for pair in paired_values)
                wtw = sum(pair[1] for pair in paired_values)
                delta = wtw - ttw
                add(
                    visualization_id=carbon_visual.id,
                    insight_type="boundary_delta",
                    statement=(
                        f"WTW CO2e was {_format_number(abs(delta))} tCO2e "
                        f"{'above' if delta >= 0 else 'below'} TTW across the returned rows."
                    ),
                    values=[
                        ("ttw_total", ttw, "tCO2e"),
                        ("wtw_total", wtw, "tCO2e"),
                        ("boundary_delta", delta, "tCO2e"),
                    ],
                )

        selected_visual = None
        for visual in visualizations:
            if not isinstance(visual, CartesianVisualization) or visual.dataset_id not in by_id:
                continue
            selected_fields = _carbon_metric_fields(
                plan,
                by_id[visual.dataset_id or ""],
            )
            if selected_fields and selected_fields[0] in visual.y_fields:
                selected_visual = visual
                break
        if selected_visual is not None:
            dataset = by_id[selected_visual.dataset_id or ""]
            point_field = _carbon_metric_fields(plan, dataset)[0]
            date_field = _field(dataset, ["date"])
            peak_rows = [
                row
                for row in dataset.rows
                if row.get("is_peak") is True
                and finite_number(row.get(point_field)) is not None
            ]
            if date_field and peak_rows:
                peak_row = peak_rows[0]
                peak_value = finite_number(peak_row.get(point_field))
                assert peak_value is not None
                peak_date = peak_row.get(date_field)
                unit = _unit(dataset, point_field)
                metric_label = _metric_label(point_field)
                add(
                    visualization_id=selected_visual.id,
                    insight_type="peak",
                    statement=(
                        f"Peak {metric_label} was {_format_number(peak_value)} "
                        f"{unit or 'units'} on {str(peak_date)[:10]} across the returned rows."
                    ),
                    values=[
                        ("peak_value", peak_value, unit),
                        ("peak_date", peak_date, None),
                    ],
                )

    elif plan.operation in {
        QueryOperation.DWELL_SUMMARY,
        QueryOperation.DWELL_DISTRIBUTION,
        QueryOperation.ROUTE_TRAVEL_TIME,
    }:
        summary = by_id.get("distribution_summary")
        visual = next(
            (item for item in visualizations if item.kind == "distribution"),
            None,
        )
        if summary and visual and summary.rows:
            row = summary.rows[0]
            q1 = finite_number(row.get("q1"))
            med = finite_number(row.get("median"))
            q3 = finite_number(row.get("q3"))
            p90 = finite_number(row.get("p90"))
            count = finite_number(row.get("count"))
            if None not in {q1, med, q3, p90, count}:
                assert q1 is not None and med is not None and q3 is not None
                assert p90 is not None and count is not None
                unit = next(
                    (column.unit for column in summary.columns if column.field == "median"),
                    None,
                )
                outlier_count = by_id.get("distribution_outliers").row_count if by_id.get("distribution_outliers") else 0
                add(
                    visualization_id=visual.id,
                    insight_type="distribution_summary",
                    statement=(
                        f"Median {_format_number(med)} {unit or ''}, p90 {_format_number(p90)}, "
                        f"and IQR {_format_number(q3 - q1)} across {int(count):,} observations; "
                        f"Tukey's rule flags {outlier_count:,} outlier(s)."
                    ).replace("  ", " "),
                    values=[
                        ("median", med, unit),
                        ("p90", p90, unit),
                        ("iqr", q3 - q1, unit),
                        ("observation_count", int(count), "observations"),
                        ("outlier_count", outlier_count, "observations"),
                    ],
                )

    elif plan.operation == QueryOperation.CORRELATION:
        visual = next(
            (
                item
                for item in visualizations
                if isinstance(item, CartesianVisualization) and item.fitted_series
            ),
            None,
        )
        if visual:
            fit = visual.fitted_series[0]
            dataset = by_id.get(visual.dataset_id or "")
            count = (
                sum(
                    1
                    for row in dataset.rows
                    if finite_number(row.get(fit.x_field)) is not None
                    and finite_number(row.get(visual.y_fields[0])) is not None
                )
                if dataset
                else 0
            )
            if fit.slope is not None and fit.r_squared is not None:
                add(
                    visualization_id=visual.id,
                    insight_type="association",
                    statement=(
                        f"The backend OLS fit has slope {_format_number(fit.slope, decimals=3)} "
                        f"and R² {_format_number(fit.r_squared, decimals=3)} across {count:,} paired rows; "
                        "this describes association, not causation."
                    ),
                    values=[
                        ("ols_slope", fit.slope, None),
                        ("ols_intercept", fit.intercept, None),
                        ("ols_r_squared", fit.r_squared, None),
                        ("paired_rows", count, "rows"),
                    ],
                )

    elif plan.operation in {
        QueryOperation.FORECAST_ARRIVALS,
        QueryOperation.FORECAST_CONGESTION,
    }:
        visual = next(
            (
                item
                for item in visualizations
                if isinstance(item, ForecastVisualization) and item.quality_metrics is not None
            ),
            None,
        )
        if visual and visual.quality_metrics:
            quality = visual.quality_metrics
            add(
                visualization_id=visual.id,
                insight_type="forecast_quality",
                statement=(
                    f"The displayed forecast passed validation with MASE "
                    f"{_format_number(quality.mase, decimals=3)} and "
                    f"{quality.interval_level:.0%} interval coverage "
                    f"{quality.interval_coverage:.1%}."
                ),
                values=[
                    ("mase", quality.mase, None),
                    ("interval_level", quality.interval_level, "fraction"),
                    ("interval_coverage", quality.interval_coverage, "fraction"),
                    ("forecast_boundary", visual.forecast_boundary, None),
                ],
            )

    elif plan.operation == QueryOperation.ARRIVAL_ANOMALY and primary is not None:
        dataset = by_id[primary.dataset_id or ""]
        flagged: List[Tuple[Any, float, float, float]] = []
        for row in dataset.rows:
            observed = finite_number(row.get("arrival_count", row.get("arrivals_vessels")))
            threshold = finite_number(row.get("threshold"))
            if row.get("is_anomaly") is True and observed is not None and threshold is not None:
                flagged.append((row.get("date"), observed, threshold, observed - threshold))
        if flagged:
            top = max(flagged, key=lambda item: item[3])
            add(
                visualization_id=primary.id,
                insight_type="threshold_exceedance",
                statement=(
                    f"{len(flagged):,} flagged event(s) exceeded the threshold; the largest exceedance was "
                    f"{_format_number(top[3])} arrival events on {str(top[0])[:10]}."
                ),
                values=[
                    ("flagged_count", len(flagged), "events"),
                    ("largest_exceedance", top[3], "count"),
                    ("largest_exceedance_date", top[0], None),
                    ("largest_observed", top[1], "count"),
                    ("largest_threshold", top[2], "count"),
                ],
            )

    elif plan.operation == QueryOperation.AIS_JUMP and primary is not None:
        dataset = by_id[primary.dataset_id or ""]
        distances = [
            (row, value)
            for row in dataset.rows
            if (value := finite_number(row.get("distance_km"))) is not None
        ]
        if distances:
            row, distance = max(distances, key=lambda item: item[1])
            add(
                visualization_id=primary.id,
                insight_type="movement_anomaly",
                statement=(
                    f"The largest provenance-backed flagged segment spans "
                    f"{_format_number(distance)} km at {str(row.get('timestamp_full') or '')[:19]}."
                ),
                values=[
                    ("largest_distance", distance, "km"),
                    ("largest_distance_timestamp", row.get("timestamp_full"), None),
                ],
            )

    elif plan.operation == QueryOperation.VESSEL_TYPE_COMPOSITION and primary is not None:
        dataset = by_id[primary.dataset_id or ""]
        value_field = _field(
            dataset,
            ["share_percent", "share_pct", "share", "arrival_count", "arrivals_vessels", "value"],
        )
        category_field = _field(dataset, ["vessel_type_norm", "vessel_type", "category"])
        if value_field and category_field:
            ranked = sorted(
                (
                    (str(row.get(category_field)), value)
                    for row in dataset.rows
                    if (value := finite_number(row.get(value_field))) is not None
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if ranked:
                label, value = ranked[0]
                unit = _unit(dataset, value_field)
                add(
                    visualization_id=primary.id,
                    insight_type="dominant_share",
                    statement=(
                        f"{label} is the dominant returned category at "
                        f"{_format_number(value)} {unit or ''}."
                    ).strip(),
                    values=[
                        ("dominant_category", label, None),
                        ("dominant_value", value, unit),
                    ],
                )

    elif plan.operation in {
        QueryOperation.TOP_PORTS,
        QueryOperation.ARRIVALS_MULTI,
        QueryOperation.PORT_COMPARISON,
        QueryOperation.BUSIEST_WEEKDAY,
        QueryOperation.BUSIEST_HOUR,
        QueryOperation.WEEKDAY_COMPARISON,
        QueryOperation.CONGESTION_WEEKDAY_COMPARISON,
    } and primary is not None:
        dataset = by_id[primary.dataset_id or ""]
        value_field = primary.y_fields[0] if isinstance(primary, CartesianVisualization) else None
        category_field = primary.x_field if isinstance(primary, CartesianVisualization) else None
        if value_field and category_field:
            ranked = sorted(
                (
                    (str(row.get(category_field)), value)
                    for row in dataset.rows
                    if (value := finite_number(row.get(value_field))) is not None
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if len(ranked) >= 2:
                winner, winner_value = ranked[0]
                runner_up, runner_value = ranked[1]
                margin = winner_value - runner_value
                unit = _unit(dataset, value_field)
                add(
                    visualization_id=primary.id,
                    insight_type="ranking_margin",
                    statement=(
                        f"{winner} leads {runner_up} by {_format_number(margin)} {unit or 'units'} "
                        "across the returned rows."
                    ),
                    values=[
                        ("winner", winner, None),
                        ("winner_value", winner_value, unit),
                        ("runner_up", runner_up, None),
                        ("runner_up_value", runner_value, unit),
                        ("winner_margin", margin, unit),
                    ],
                )

    return insights[:3], facts
