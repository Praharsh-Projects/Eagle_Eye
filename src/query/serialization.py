"""Finite, deterministic serialization helpers for query envelopes."""

from __future__ import annotations

import math
import re
from statistics import mean, median
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .models import ColumnSpec, DatasetSpec, FactSlot


_UNIT_BY_FIELD = {
    "arrival_count": "count",
    "daily_distinct_vessels": "vessels",
    "arrivals_vessels": "vessels",
    "arrivals_events": "events",
    "dwell_minutes": "minutes",
    "median_dwell_minutes": "minutes",
    "mean_dwell_minutes": "minutes",
    "mean_dwell_hours": "hours",
    "median_dwell_hours": "hours",
    "complete_dwell_count": "count",
    "duration_h": "hours",
    "median_duration_h": "hours",
    "congestion_index": "index",
    "pressure_index": "index",
    "occupancy_vessels": "vessels",
    "distance_km": "km",
    "speed_kn": "knots",
    "implied_speed_kn": "knots",
    "sog_kn": "knots",
    "announced_delay_minutes": "minutes",
    "latitude": "degrees",
    "longitude": "degrees",
    "predicted": "index",
    "actual": "index",
    "lower": "index",
    "upper": "index",
    "share": "fraction",
    "share_pct": "percent",
    "co2_t": "tCO2",
    "ttw_co2e_t": "tCO2e",
    "wtt_co2e_t": "tCO2e",
    "wtw_co2e_t": "tCO2e",
    "nox_kg": "kg",
    "sox_kg": "kg",
    "pm_kg": "kg",
}


def unit_for_field(field: str) -> Optional[str]:
    """Return the canonical unit for a semantic field, including CI bounds."""

    if field in _UNIT_BY_FIELD:
        return _UNIT_BY_FIELD[field]
    base = re.sub(r"_(?:lower|upper)$", "", field)
    return _UNIT_BY_FIELD.get(base)


def finite_json_value(value: Any) -> Any:
    if value is None:
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, int)):
        return value
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): finite_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [finite_json_value(item) for item in value]
    return str(value)


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    non_null = series.dropna()
    if not non_null.empty and all(isinstance(value, (pd.Timestamp, datetime, date)) for value in non_null.head(20)):
        return "datetime"
    return "string"


def _humanize(field: str) -> str:
    return re.sub(r"\s+", " ", field.replace("_", " ")).strip().title()


def dataframe_to_dataset(
    frame: Optional[pd.DataFrame],
    *,
    dataset_id: str,
    max_rows: int = 5000,
    unit_overrides: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[DatasetSpec]:
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    if not isinstance(work.index, pd.RangeIndex):
        index_name = str(work.index.name or "index")
        if index_name in work.columns:
            work = work.reset_index(drop=True)
        else:
            work = work.reset_index()
    work.columns = [str(column) for column in work.columns]
    work = work.head(max_rows)
    overrides = unit_overrides or {}
    columns = [
        ColumnSpec(
            field=column,
            label=_humanize(column),
            data_type=_infer_type(work[column]),
            unit=overrides[column] if column in overrides else unit_for_field(column),
        )
        for column in work.columns
    ]
    rows: List[Dict[str, Any]] = []
    for raw in work.to_dict(orient="records"):
        rows.append({str(key): finite_json_value(value) for key, value in raw.items()})
    return DatasetSpec(id=dataset_id, columns=columns, rows=rows, row_count=len(rows))


def dataset_fields(dataset: DatasetSpec) -> set[str]:
    return {column.field for column in dataset.columns}


def _answer_unit(answer: str, end: int) -> Optional[str]:
    suffix = (answer[end:] or "").lstrip().lower()
    unit_patterns = (
        (r"^%", "percent"),
        (r"^(?:h|hr|hrs|hour|hours)\b", "hours"),
        (r"^(?:min|mins|minute|minutes)\b", "minutes"),
        (r"^(?:km|kilometre|kilometres|kilometer|kilometers)\b", "km"),
        (r"^(?:kn|knot|knots)\b", "knots"),
        (r"^(?:vessel|vessels)\b", "vessels"),
        (r"^(?:arrival|arrivals)\b", "arrivals"),
        (r"^(?:event|events)\b", "events"),
        (r"^(?:day|days|day bucket|day buckets)\b", "days"),
        (r"^(?:voyage|voyages)\b", "voyages"),
        (r"^x\b", "ratio"),
    )
    for pattern, unit in unit_patterns:
        if re.search(pattern, suffix):
            return unit
    # Comparisons often render as ``Monday=55 vs Friday=57 arrivals``.  Carry
    # the shared unit backward within the same short clause so both immutable
    # numeric slots remain typed.
    clause = re.split(r"[.;\n]", suffix, maxsplit=1)[0][:80]
    for pattern, unit in unit_patterns:
        shared_pattern = rf"(?<![A-Za-z_]){pattern.removeprefix('^')}"
        if re.search(shared_pattern, clause):
            return unit
    return None


def extract_answer_facts(
    answer: str,
    *,
    source: str = "computed",
    state: Optional[str] = None,
    operation: Optional[str] = None,
    metric: Optional[str] = None,
    entities: Optional[Dict[str, Any]] = None,
    include_answer_numbers: bool = True,
) -> List[FactSlot]:
    """Build immutable semantic slots before exposing rendered answer numbers.

    The answer remains presentation text.  These slots make its result state,
    source, entities, scope and comparison polarity auditable so later wording
    cannot silently reverse or broaden a validated result.
    """
    facts: List[FactSlot] = []
    semantic_values: List[tuple[str, Any, Optional[str]]] = [
        ("source_type", source, operation),
    ]
    if state:
        semantic_values.append(("result_state", state, operation))
    if operation:
        semantic_values.append(("operation", operation, operation))
    if metric:
        semantic_values.append(("metric", metric, operation))
    for name, value in (entities or {}).items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (list, tuple)):
            semantic_values.extend((f"{name}_{index}", item, str(item)) for index, item in enumerate(value, 1))
        else:
            semantic_values.append((name, value, str(value)))
    for name, value, entity in semantic_values:
        facts.append(
            FactSlot(
                name=name,
                value=finite_json_value(value),
                entity=entity,
                source=source,  # type: ignore[arg-type]
            )
        )

    winner_patterns = (
        r"\b([A-Z]{2}[A-Z0-9]{3})\s+was\s+higher\b",
        r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+is\s+busier\b",
        r"\b([A-Za-z][A-Za-z -]{1,40}?)\s+is\s+the\s+largest\b",
    )
    for pattern in winner_patterns:
        winner = re.search(pattern, answer or "", flags=re.IGNORECASE)
        if winner:
            value = winner.group(1).strip()
            facts.extend(
                [
                    FactSlot(name="comparison_winner", value=value, entity=operation, source=source),  # type: ignore[arg-type]
                    FactSlot(name="comparison_polarity", value="higher", entity=value, source=source),  # type: ignore[arg-type]
                ]
            )
            break
    if state in {"NO_DATA", "NO_CURRENT_DATA", "UNSUPPORTED", "CLARIFICATION_REQUIRED", "ERROR"}:
        facts.append(FactSlot(name="result_polarity", value="unavailable", entity=operation, source=source))  # type: ignore[arg-type]

    if not include_answer_numbers:
        return facts

    for index, match in enumerate(re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", answer or ""), start=1):
        token = match.group(0)
        normalized = token.replace(",", "").lstrip("+")
        try:
            value: Any = float(normalized)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = token
        facts.append(
            FactSlot(
                name=f"answer_number_{index}",
                value=value,
                unit=_answer_unit(answer, match.end()),
                entity=operation,
                source=source,  # type: ignore[arg-type]
            )
        )
    return facts


def extract_dataset_facts(
    datasets: Iterable[DatasetSpec],
    *,
    operation: str,
    source: str = "computed",
) -> List[FactSlot]:
    """Derive named immutable facts from validated rows, never answer prose.

    The legacy ``answer_number_*`` facts remain available for compatibility.
    These additional facts implement measurement_contract.v2 and give the
    evaluator stable semantic names for nested ranking and time-series values.
    """

    by_id = {dataset.id: dataset for dataset in datasets}
    table = by_id.get("table")
    chart = by_id.get("chart")
    observations = by_id.get("distribution_observations")
    facts: List[FactSlot] = []

    def add(name: str, value: Any, unit: Optional[str] = None, entity: Optional[str] = None) -> None:
        safe = finite_json_value(value)
        if safe is None or safe == [] or safe == {}:
            return
        facts.append(
            FactSlot(
                name=name,
                value=safe,
                unit=unit,
                entity=entity or operation,
                source=source,  # type: ignore[arg-type]
            )
        )

    def number(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def date_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text) else text

    if operation in {"arrivals", "peak_arrival_day"} and table is not None:
        daily: List[Dict[str, Any]] = []
        for row in table.rows:
            value = number(row.get("arrival_count"))
            date_value = date_token(row.get("date"))
            if value is None:
                continue
            if date_value:
                daily.append(
                    {
                        "date_utc": date_value,
                        "arrival_count": int(value) if value.is_integer() else value,
                    }
                )
        if daily:
            total = sum(float(row["arrival_count"]) for row in daily)
            add("arrival_count", int(total) if total.is_integer() else total, "count")
            add("daily_arrivals", daily, "count")
            peak = max(daily, key=lambda row: float(row["arrival_count"]))
            add("peak_arrival_count", peak["arrival_count"], "count")
            peak_dates = [
                row["date_utc"]
                for row in daily
                if float(row["arrival_count"]) == float(peak["arrival_count"])
            ]
            add("peak_dates_utc", peak_dates)
            # Retain the original singular fact for compatibility with clients
            # that presented the first peak date before measurement_contract.v2.
            add("peak_arrival_date", peak["date_utc"])

    if operation in {"arrivals_multi", "port_comparison"} and table is not None:
        rows: List[Dict[str, Any]] = []
        for row in table.rows:
            metric = str(row.get("metric") or "arrival_count")
            value = number(row.get("arrival_count" if "arrival_count" in row else "value"))
            port = str(row.get("port") or row.get("port_key") or "").strip()
            if metric != "arrival_count" or value is None or not port:
                continue
            rows.append(
                {
                    "port_locode": port,
                    "arrival_count": int(value) if value.is_integer() else value,
                }
            )
        rows.sort(key=lambda row: (-float(row["arrival_count"]), str(row["port_locode"])))
        if rows:
            add("port_arrival_counts", rows, "count")
            winner_value = float(rows[0]["arrival_count"])
            winners = [row["port_locode"] for row in rows if float(row["arrival_count"]) == winner_value]
            add("winner_port_locodes", winners)
            if len(rows) >= 2:
                margin = winner_value - float(rows[1]["arrival_count"])
                add("absolute_margin", int(margin) if margin.is_integer() else margin, "count")

    if operation == "top_ports" and table is not None:
        ranking: List[Dict[str, Any]] = []
        for row in table.rows:
            value = number(row.get("arrival_count"))
            locode = str(row.get("port_key") or row.get("locode_norm") or "").strip()
            label = str(row.get("port_label") or "").strip()
            name = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip() or label
            if value is None or not locode:
                continue
            ranking.append(
                {
                    "port_locode": locode,
                    "port_name": name,
                    "arrival_count": int(value) if value.is_integer() else value,
                }
            )
        ranking.sort(key=lambda row: (-float(row["arrival_count"]), str(row["port_locode"])))
        for rank, row in enumerate(ranking, 1):
            row["rank"] = rank
        add("port_ranking", ranking, "count")

    if operation == "vessel_type_composition" and table is not None:
        counts: Dict[str, Any] = {}
        for row in table.rows:
            value = number(row.get("arrival_count"))
            category = str(row.get("vessel_type_norm") or row.get("category") or "").strip().lower()
            if value is None or not category:
                continue
            normalized = "cargo" if "cargo" in category else "tanker" if "tanker" in category else category
            counts[normalized] = int(value) if value.is_integer() else value
        add("vessel_type_arrival_counts", counts, "count")

    if operation in {"dwell_summary", "dwell_distribution"} and observations is not None:
        values = [
            value
            for row in observations.rows
            if (value := number(row.get("dwell_minutes"))) is not None and 0 < value <= 45 * 24 * 60
        ]
        if values:
            add("complete_dwell_count", len(values), "count")
            add("mean_dwell_hours", mean(values) / 60.0, "hours")
            add("median_dwell_hours", median(values) / 60.0, "hours")

    if operation in {
        "busiest_weekday",
        "busiest_hour",
        "weekday_comparison",
        "arrival_pattern",
        "arrival_anomaly",
    } and (chart or table) is not None:
        dataset = chart or table
        assert dataset is not None
        named_rows = [
            finite_json_value(row)
            for row in dataset.rows
            if number(row.get("arrival_count")) is not None
        ]
        add("arrival_series", named_rows, "count")

    return facts


def dedupe_strings(items: Iterable[str], limit: int = 20) -> List[str]:
    output: List[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in output:
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output
