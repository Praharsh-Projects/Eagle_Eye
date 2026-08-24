"""Restored Eagle Eye Streamlit product interface.

The FastAPI canonical service and React workspace remain available as a
rollback path; the familiar Streamlit analyst interface is the default again.
"""

from __future__ import annotations

import time
import os
import re
import json
import uuid
import unicodedata
from html import escape
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pandas 3 infers Arrow-backed strings by default. PyArrow 25 can segfault when
# those arrays are operated on from Streamlit's script thread on macOS, so the
# UI runtime keeps strings as ordinary Python objects after Parquet reads.
pd.options.future.infer_string = False

# Streamlit converts Altair and dataframe payloads through PyArrow. The current
# macOS runtime can segfault in that native conversion even after the underlying
# analytics has completed, so UI charts use Matplotlib images instead.
alt = None


def _style_matplotlib_axis(ax: Any) -> None:
    ax.set_facecolor("#071727")
    ax.figure.patch.set_facecolor("#071727")
    ax.grid(True, color="#183147", alpha=0.7, linewidth=0.8)
    ax.tick_params(colors="#dcecff", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color("#8fe8ff")
    ax.yaxis.label.set_color("#8fe8ff")
    ax.title.set_color("#eef7ff")


def _safe_line_chart(data: Any, *, width: str = "stretch") -> None:
    del width
    frame = data.to_frame() if isinstance(data, pd.Series) else pd.DataFrame(data).copy()
    if frame.empty:
        st.info("No chartable series for this response.")
        return
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="all")
    if numeric.empty:
        st.info("No chartable numeric series for this response.")
        return
    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    _style_matplotlib_axis(ax)
    for column in numeric.columns:
        ax.plot(numeric.index, numeric[column], marker="o", markersize=3.5, linewidth=2.2, label=str(column))
    if len(numeric.columns) > 1:
        legend = ax.legend(frameon=False, loc="upper left")
        for label in legend.get_texts():
            label.set_color("#dcecff")
    fig.autofmt_xdate(rotation=0)
    st.pyplot(fig, width="stretch", clear_figure=True)
    plt.close(fig)


def _safe_bar_chart(data: Any, *, width: str = "stretch") -> None:
    del width
    frame = data.to_frame() if isinstance(data, pd.Series) else pd.DataFrame(data).copy()
    if frame.empty:
        st.info("No chartable series for this response.")
        return
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.empty:
        st.info("No chartable numeric series for this response.")
        return
    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    _style_matplotlib_axis(ax)
    numeric.plot(kind="bar", ax=ax, color=["#5aa9ff", "#8fe8ff", "#35d39a", "#ffb55e"])
    ax.tick_params(axis="x", rotation=0)
    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.set_frame_on(False)
        for label in legend.get_texts():
            label.set_color("#dcecff")
    st.pyplot(fig, width="stretch", clear_figure=True)
    plt.close(fig)


def _safe_dataframe(
    data: Any,
    *,
    width: str = "stretch",
    hide_index: bool = True,
    **_: Any,
) -> None:
    del width
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        st.info("No tabular rows for this response.")
        return
    max_rows = 500
    visible = frame.head(max_rows)
    table_html = visible.to_html(index=not hide_index, border=0, escape=True, classes="eagle-safe-table")
    st.markdown(
        """
        <style>
        .eagle-safe-table {width:100%; border-collapse:collapse; font-size:0.85rem; color:#dcecff;}
        .eagle-safe-table th {text-align:left; color:#8fe8ff; border-bottom:1px solid #29465f; padding:0.55rem;}
        .eagle-safe-table td {border-bottom:1px solid #183147; padding:0.5rem; vertical-align:top;}
        .eagle-safe-table tr:hover {background:#0d2236;}
        </style>
        """ + table_html,
        unsafe_allow_html=True,
    )
    if len(frame) > max_rows:
        st.caption(f"Showing the first {max_rows:,} of {len(frame):,} rows.")

from src.api.server import _build_state
from src.app.streamlit_query_bridge import (
    canonical_presentation,
    dataset_frame,
    distinct_visual_summary,
    run_canonical_query,
)
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.carbon.presentation import (
    build_comparison_bar_table,
    build_emissions_findings,
    build_reduction_suggestions,
    classify_level,
    compute_emissions_metrics,
    derive_threshold_bands,
    emissions_measurement_note,
    extract_chart_findings,
    format_kgco2e,
    format_percent,
    format_tco2e,
    safe_percent_delta,
    sanitize_threshold_percentiles,
    scale_tco2e,
    to_emissions_display_table,
)
from src.carbon.query import (
    CARBON_STATE_COMPUTED,
    CARBON_STATE_COMPUTED_ZERO,
    CARBON_STATE_FORECAST_ONLY,
    CARBON_STATE_NOT_COMPUTABLE,
    CARBON_STATE_RETRIEVAL_ONLY,
    CARBON_STATE_UNSUPPORTED,
    CarbonQueryEngine,
    CarbonResult,
)
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.qa.intent import IntentResult, classify_question
from src.query.models import AnswerEnvelope, AnswerState, ExportRequest, FeedbackRequest
from src.query.service import QueryService
from src.rag.retriever import QueryFilters, RAGRetriever
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.cloud_bootstrap import ensure_bundle, ensure_file_manifest
from src.utils.confidence import extract_confidence_label
from src.utils.config import load_config
from src.utils.runtime import chroma_remote_settings, force_local_vector_env
from src.utils.serialization import compact_traffic_evidence


SAMPLE_QUERIES_BY_CATEGORY: Dict[str, List[str]] = {
    "Traffic Monitoring": [
        "How many vessel arrivals were recorded at SEGOT in March 2022?",
        "How many vessel arrivals were recorded at Karlshamn and Karlskrona in March 2022?",
        "What is the first arrival seen at Karlshamn on 2022-03-22?",
        "What is the last arrival seen at Karlshamn on 2022-03-22?",
        "Show first departure from Karlshamn in March 2022.",
        "What is median and p90 route travel time from PLSZZ to PLSWI in 2021-02?",
        "Compare arrivals at PLSZZ and PLSWI and route durations from PLSZZ to PLSWI and PLSWI to SEMMA in 2021-02.",
        "Which weekday is usually busiest at LVVNT?",
        "Compare Friday and Monday arrivals at GDANSK in March 2022.",
        "Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28.",
        "How many tanker arrivals were recorded at LVVNT between 2022-03-01 and 2022-03-10?",
        "What was the peak arrival day at SEGOT in March 2022?",
        "Which port had more arrivals in March 2022: LVVNT or SEGOT?",
        "Show cargo-ship arrivals at GDANSK during 2022-03.",
    ],
    "Vessel Investigation": [
        "For MMSI 245286000, how long was the vessel in port on 2021-01-01?",
        "Show suspicious AIS jumps for MMSI 246521000 on 2022-03-10.",
        "For MMSI 212575000, summarize suspicious AIS jumps on 2021-01-01.",
        "List any AIS jump anomalies for MMSI 266232000 between 2021-01-01 and 2021-01-03.",
        "Show movement anomalies for MMSI 246650000 in March 2022.",
        "How many anomaly events were detected for MMSI 255806245 in 2022-03?",
        "For MMSI 304833000, show port-stay duration evidence during 2022-03.",
        "Investigate unusual AIS jumps in March 2022.",
    ],
    "ETA & Delay": [
        "Prepare a Sweden-bound shift handover for the next 12 hours: due soon, low-speed exceptions, ETA changes, and stale signals.",
        "Which AIS-visible vessels reporting Swedish destinations are due in the next 6 hours? Show vessel, destination, reported ETA, last position, speed, and observation time.",
        "Show the next AIS-visible vessel reporting an ETA for Stockholm, Gothenburg, Nynäshamn, Malmö, or Trelleborg.",
        "Which Sweden-bound vessels due in the next 6 hours are moving below 2 knots?",
        "Which Swedish destinations have the most AIS-reported inbound vessels in the next 24 hours?",
        "Which Sweden-bound vessels changed their reported ETA by more than 30 minutes in the last hour?",
        "Which Sweden-bound vessels have a stale position report or no valid reported ETA?",
        "Where is the next AIS-visible vessel reporting a Swedish destination, and what ETA is it transmitting?",
        "Build a 12-hour Baltic inbound watchlist for Tallinn, Riga, Klaipėda, Gdańsk, Helsinki, and Turku.",
        "Which Baltic-bound vessels are due in the next 6 hours, and where were they last observed?",
    ],
    "Port Pressure": [
        "What is the port pressure index at SEGOT in March 2022?",
        "Show port pressure trend at LVVNT between 2022-02-01 and 2022-02-28.",
        "Compare port pressure between SEGOT and GDANSK in March 2022.",
        "Which days had the highest pressure at LVVNT in 2022-03?",
        "Is pressure at SEGVX above baseline on 2022-03-10?",
        "Show pressure by vessel type at GDANSK for 2022-03.",
        "Compare Friday versus Monday port pressure at SEGOT.",
        "List top 5 high-pressure days at LVVNT in 2022-03.",
    ],
    "Carbon Emissions": [
        "What are TTW emissions at SEGOT in March 2022 for CO2e, NOx, SOx, and PM?",
        "Show WTW CO2e emissions at LVVNT between 2022-02-01 and 2022-02-28.",
        "Carbon emissions for SEGOT by month in 2022.",
        "Report TTW CO2e and NOx at LVVNT for 2022-03 grouped by day.",
        "Show WTW CO2e at SEGVX between 2022-03-01 and 2022-03-31.",
        "What are call-level emissions for MMSI 209468000 and call_id 209468000_2021-01-06T10-17-56_SETRG?",
        "Estimate carbon emissions for a tanker in manoeuvring mode for 2 hours at 6 knots.",
        "Compare TTW versus WTW CO2e totals at SETRG for March 2022.",
        "Show monthly WTW CO2e trend for SETRG in 2022.",
        "Give carbon evidence IDs used for LVVNT emissions in 2022-02.",
    ],
    "Unsupported Scope": [
        "What is crane utilization at berth 3 in SEGOT today?",
        "What is gate queue length at Port of Gdansk right now?",
        "How many TEU were handled per hour at berth 5 yesterday?",
        "What is yard occupancy percentage at terminal block C right now?",
        "Show quay crane productivity at LVVNT in March 2022.",
        "What is truck turn-time at the gate for SEGOT today?",
        "Give exact berth-level queue length for vessel arrivals at GDANSK.",
    ],
}

QUERY_CATEGORY_PAGE_ORDER: List[str] = [
    "Traffic Monitoring",
    "Vessel Investigation",
    "ETA & Delay",
    "Port Pressure",
    "Carbon Emissions",
]

QUERY_CATEGORY_HELP: Dict[str, Dict[str, List[str] | str]] = {
    "Traffic Monitoring": {
        "overview": "Use deterministic KPI aggregations for arrivals, busiest periods, and port comparisons.",
        "what_to_enter": [
            "A traffic question with port + date range (example: `How many arrivals at SEGOT in March 2022?`).",
            "Optional filters: port, date range, vessel type, anomaly flag.",
        ],
        "expected_output": [
            "Deterministic arrivals/count metrics (`Computed`).",
            "Evidence rows and chart for the selected scope.",
            "Source provenance and retrieval details (if vector retrieval is active).",
        ],
        "test_steps": [
            "Sample input: `How many vessel arrivals were recorded at SEGOT in March 2022?`",
            "Optional filters: port=`SEGOT`, date range=`2022-03-01` to `2022-03-31`.",
            "Expected: deterministic arrivals count, evidence rows, and a time-series chart.",
        ],
        "calculation": [
            "Data source: `arrivals_daily.parquet` and `arrivals_hourly.parquet`.",
            "Method: pandas groupby/filter over deterministic KPI tables.",
            "Evidence: retrieved vector rows are explanatory; numeric truth is from KPI aggregation.",
        ],
    },
    "Vessel Investigation": {
        "overview": "Inspect a single vessel (MMSI-based) for dwell behavior and AIS jump anomalies.",
        "what_to_enter": [
            "A vessel-focused question with MMSI and date/window.",
            "Optional filters: date range and anomaly flag.",
        ],
        "expected_output": [
            "Detected anomalies/dwell evidence for matching vessel rows.",
            "Event lines with timestamps and movement context.",
            "Clear no-data response when MMSI/date scope has no rows.",
        ],
        "test_steps": [
            "Sample input: `Show suspicious AIS jumps for MMSI 246521000 on 2022-03-10.`",
            "Optional filters: date range around the target day.",
            "Expected: anomaly count + event table with displacement/speed evidence when available.",
        ],
        "calculation": [
            "Data source: `events.parquet` (preferred) or indexed metadata.",
            "Method: deterministic screening rules over chronological AIS rows.",
            "Threshold logic: large displacement in short time window / implied speed rule.",
        ],
    },
    "ETA & Delay": {
        "overview": "Monitor fresh vessel-reported destinations, ETAs, positions, speeds, ETA revisions, and signal quality through a Sweden-first Baltic operational watch.",
        "what_to_enter": [
            "Ask for a shift handover, inbound watchlist, destination load, exception check, or a specific vessel by name, MMSI, or IMO.",
            "Use a current UTC window of 6, 12, 24, or at most 48 hours. Sweden is the default scope; named Baltic ports can be combined.",
        ],
        "expected_output": [
            "A decision brief followed by the first actionable vessels, exact observation times, matching maps or ETA rails, and the complete source table.",
            "ETA revisions and low-speed cases are monitoring signals, not confirmed delay or an official port schedule.",
            "A specific warming, stale, unavailable, or unsupported reason when the requested broadcast fields do not pass freshness checks.",
        ],
        "test_steps": [
            "Sample input: `Prepare a Sweden-bound shift handover for the next 12 hours: due soon, low-speed exceptions, ETA changes, and stale signals.`",
            "Optional filters: select a supported port or vessel identifier to narrow the live watch.",
            "Expected: current UTC observations, matched and displayed totals, source health, actionable rows, matching charts, and no historical substitution.",
        ],
        "calculation": [
            "Numeric authority: normalized AISStream position and ship-static broadcasts joined by MMSI.",
            "Freshness boundary: requested position, destination, and ETA fields must independently pass the current-source checks.",
            "ETA change compares the same vessel's validated reported ETA snapshots; it is not a prediction or confirmed delay.",
            "AIS coverage is non-exhaustive. This page never substitutes historical congestion calculations for live evidence.",
        ],
    },
    "Port Pressure": {
        "overview": "Assess operational pressure in port time windows using deterministic index calculations and arrivals signals.",
        "what_to_enter": [
            "A pressure query with port and date range (example: `pressure index at SEGOT in March 2022`).",
            "Optional: vessel type filter for lane-specific pressure view.",
        ],
        "expected_output": [
            "Pressure/congestion index trend (`Computed`).",
            "Evidence rows backing the pressure calculation.",
            "Operational recommendation bullets tied to computed drivers.",
        ],
        "test_steps": [
            "Sample input: `What is the port pressure index at SEGOT in March 2022?`",
            "Optional filters: add vessel type and date range to narrow pressure scope.",
            "Expected: pressure index trend, evidence rows, and pressure-focused recommendations.",
        ],
        "calculation": [
            "Data source: `congestion_daily.parquet`, `arrivals_daily.parquet`, `arrivals_hourly.parquet`.",
            "Method: deterministic pressure-index calculation over filtered rows.",
            "Output unit: pressure is an index (dimensionless).",
        ],
    },
    "Carbon Emissions": {
        "overview": "Compute deterministic carbon outputs when carbon inventory rows match the query scope.",
        "what_to_enter": [
            "A carbon query with boundary keyword (`TTW`/`WTW`), pollutants, and scope (port/date or mmsi+call_id).",
            "For call-level output, include both MMSI and call_id.",
        ],
        "expected_output": [
            "Carbon metrics, tables, and charts when structured rows match the scope.",
            "A direct no-data reason when no matching finite rows exist.",
            "Structured carbon evidence and any retrieved supporting traffic evidence.",
        ],
        "test_steps": [
            "Sample input: `Report TTW CO2e and NOx at LVVNT for 2022-03 grouped by day.`",
            "Try boundary keywords (`TTW`/`WTW`) and pollutants (`CO2`, `CH4`, `N2O`, `CO2e`, `NOx`, `SOx`, `PM`).",
            "Expected: carbon metrics and charts, or a direct reason when no matching rows exist.",
        ],
        "calculation": [
            "Data source: `carbon_segments.parquet`, `carbon_emissions_segment.parquet`, `carbon_emissions_daily_port.parquet`, `carbon_emissions_call.parquet`.",
            "Method: deterministic sums over structurally valid matched segment/call rows.",
            "Finite zero values remain distinct from a scope with no matching data.",
        ],
    },
    "Unsupported Scope": {
        "overview": "Validate refusal behavior for requests outside dataset capability (berth cranes, gate queues, TEU throughput).",
        "what_to_enter": [
            "An intentionally unsupported operations query (crane utilization, gate queue, TEU throughput).",
        ],
        "expected_output": [
            "Explicit unsupported refusal (`UNSUPPORTED`).",
            "No fabricated numeric output.",
        ],
        "test_steps": [
            "Sample input: `What is crane utilization at berth 3 in SEGOT today?`",
            "Expected: explicit unsupported response (no fabricated numeric output).",
        ],
        "calculation": [
            "Rule: intent classifier maps unsupported operational asks to refusal path.",
            "Data limitations are shown directly to avoid unsupported claims.",
        ],
    },
}

QUERY_CATEGORY_REQUIRED_DATA: Dict[str, List[str]] = {
    "Traffic Monitoring": [
        "AIS or port-call arrivals feeds",
        "Canonical port reference (UN/LOCODE)",
    ],
    "Vessel Investigation": [
        "AIS position time-series (MMSI keyed)",
        "Port-call windows for dwell cross-check",
    ],
    "ETA & Delay": [
        "AISStream position and ship-static broadcasts",
        "Curated Baltic destination aliases and UN/LOCODE references",
        "Bounded recent AIS snapshot history for ETA revisions",
    ],
    "Port Pressure": [
        "Arrivals daily/hourly tables",
        "Port-call expected-arrivals feeds for live mode",
    ],
    "Carbon Emissions": [
        "Carbon segment/call deterministic artifacts",
        "Fuel/emission factor registry with versioning",
        "Vessel engine/fuel enrichment for more complete calculations",
    ],
    "Unsupported Scope": [
        "Not supported by current dataset (berth crane/gate queue/TEU telemetry missing).",
    ],
}

EXTERNAL_DATA_REQUIREMENTS: List[Dict[str, str]] = [
    {
        "Dataset": "AIS positions / track history",
        "Where to get": "MarineTraffic API or VesselFinder API",
        "Place in project": "data/raw/ais/",
    },
    {
        "Dataset": "Port calls / berth calls",
        "Where to get": "MarineTraffic Port/Berth Calls or VesselFinder PortCalls",
        "Place in project": "data/raw/port_calls/ and data/raw/berth_calls/",
    },
    {
        "Dataset": "Vessel master + engine/fuel enrichment",
        "Where to get": "VesselFinder MasterData / MarineTraffic + manual enrichment",
        "Place in project": "data/raw/vessel_master/ then curated to vessel profiles",
    },
    {
        "Dataset": "UN/LOCODE reference",
        "Where to get": "UNECE UN/LOCODE download (`release.zip` + updates)",
        "Place in project": "data/raw/reference/unlocode/",
    },
    {
        "Dataset": "Weather / sea-state",
        "Where to get": "Open-Meteo Marine API (primary), ERA5 (backfill)",
        "Place in project": "data/raw/weather/ then data/curated/weather_hourly/",
    },
    {
        "Dataset": "Regulatory polygons (ECA)",
        "Where to get": "Marine Regions / IMO-referenced layers",
        "Place in project": "data/raw/regulatory/",
    },
    {
        "Dataset": "MRV calibration",
        "Where to get": "EMSA THETIS-MRV public data",
        "Place in project": "data/raw/mrv/",
    },
]

PORT_ALIAS_TO_CODE: Dict[str, str] = {
    "gothenburg": "SEGOT",
    "goteborg": "SEGOT",
    "goteborgs": "SEGOT",
    "port of gothenburg": "SEGOT",
    "karlshamn": "SEKAN",
    "port of karlshamn": "SEKAN",
    "karlskrona": "SEKAA",
    "port of karlskrona": "SEKAA",
    "gdansk": "PLGDN",
    "gdynia": "PLGDY",
    "klaipeda": "LTKLJ",
    "riga": "LVRIX",
    "kotka": "FIKTK",
    "swinoujscie": "PLSWI",
    "szczecin": "PLSZZ",
    "sodertalje": "SESOE",
}

PORT_PARSE_STOPWORDS = {
    "DAILY",
    "TREND",
    "INDEX",
    "LEVEL",
    "ISTHE",
    "MONTHLY",
    "WEEKLY",
    "CARBON",
    "EMISSIONS",
    "EMISSION",
    "CONGESTION",
    "FORECAST",
    "PREDICT",
    "EXPECTED",
    "SHOW",
    "WHAT",
    "WHICH",
}
BALTIC_LOCODE_PREFIXES = {"SE", "FI", "LV", "LT", "PL", "EE", "DK", "DE", "NO", "RU"}


@dataclass
class EvidenceBundle:
    lines: List[str]
    rows: List[Dict[str, Any]]
    trace: Dict[str, Any]


def _ordered_percentile_frame(
    frame: pd.DataFrame,
    *,
    value_field: str,
    category_field: Optional[str],
) -> pd.DataFrame:
    """Return percentile values in semantic percentile order for plotting."""

    if value_field not in frame.columns:
        return pd.DataFrame(columns=["category", "value"])
    values = pd.to_numeric(frame[value_field], errors="coerce")
    if category_field and category_field in frame.columns:
        categories = frame[category_field].astype(str)
    else:
        categories = pd.Series(
            [f"Value {index + 1}" for index in range(len(frame))],
            index=frame.index,
            dtype="object",
        )
    ordered = pd.DataFrame(
        {
            "category": categories,
            "value": values,
            "_position": range(len(frame)),
        },
        index=frame.index,
    ).dropna(subset=["value"])
    if ordered.empty:
        return ordered[["category", "value"]]

    def _percentile_number(label: Any) -> float:
        match = re.search(r"(?:^|\D)(\d+(?:\.\d+)?)\s*%?", str(label))
        return float(match.group(1)) if match else float("inf")

    ordered["_percentile"] = ordered["category"].map(_percentile_number)
    ordered = ordered.sort_values(
        ["_percentile", "_position"],
        kind="stable",
    )
    return ordered[["category", "value"]].reset_index(drop=True)


def _canonical_source_metadata(
    envelope: AnswerEnvelope,
    *,
    source_label: str,
    source_detail: str,
) -> tuple[str, str]:
    """Describe attached RAG evidence without weakening numeric authority."""

    if envelope.state not in {AnswerState.COMPUTED, AnswerState.PARTIAL} or not envelope.evidence:
        return source_label, source_detail
    label = "Structured data + supporting evidence"
    detail = (
        f"{source_detail} Retrieved AIS, document, or web rows provide supporting context only; "
        "validated structured datasets remain the numeric authority."
    )
    return label, detail


def _evidence_rows_for_display(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a readable AIS/document evidence table for the public Evidence tab."""

    frame = pd.DataFrame(rows).dropna(axis=1, how="all")
    if frame.empty:
        return frame
    field_labels = {
        "evidence_id": "Evidence ID",
        "source_type": "Source type",
        "title": "Source / event",
        "excerpt": "Evidence excerpt",
        "url": "Source URL",
        "event_kind": "Event kind",
        "timestamp_full": "Timestamp",
        "timestamp": "Timestamp",
        "date": "Date",
        "locode": "Port",
        "locode_norm": "Port",
        "port_name": "Port name",
        "name": "Vessel",
        "mmsi": "MMSI",
        "imo": "IMO",
        "vessel_type": "Vessel type",
        "destination_norm": "Destination",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "speed": "Speed (kn)",
        "page": "Page",
        "source": "Document source",
    }
    priority = list(field_labels)
    visible: List[str] = []
    visible_labels: set[str] = set()
    for field in priority:
        label = field_labels[field]
        if field in frame.columns and label not in visible_labels:
            visible.append(field)
            visible_labels.add(label)
    if not visible:
        visible = list(frame.columns[:10])
    display = frame[visible].copy()
    if "excerpt" in display.columns:
        display["excerpt"] = display["excerpt"].map(
            lambda value: str(value)[:500] if pd.notna(value) else value
        )
    return display.rename(columns=field_labels)


@st.cache_resource(show_spinner="Loading the validated Eagle Eye runtime…")
def _init_query_service() -> QueryService:
    """Share the same canonical runtime factory used by FastAPI and QA."""

    return _build_state()["query_service"]


def _render_canonical_visualizations(
    envelope: AnswerEnvelope,
    *,
    compact: bool = False,
    show_title: bool = True,
) -> None:
    """Render validated visualization specs inside the established report layout."""

    if show_title:
        st.markdown("<div class='ee-section-title'>Visual</div>", unsafe_allow_html=True)

    def _fallback(dataset_id: Optional[str], message: str) -> None:
        st.error(message)
        if dataset_id:
            fallback = dataset_frame(envelope, dataset_id)
            if not fallback.empty:
                _safe_dataframe(fallback, width="stretch", hide_index=True)

    def _axes(height: float = 3.4) -> tuple[Any, Any]:
        fig, ax = plt.subplots(
            figsize=(8.8 if compact else 10.5, min(height, 2.8) if compact else height),
            constrained_layout=True,
        )
        _style_matplotlib_axis(ax)
        return fig, ax

    if not envelope.visualizations:
        st.info("No validated visualization was returned for this response.")
        return

    for visual in envelope.visualizations:
        kind = visual.kind
        summary = visual.accessible_summary
        fallback_id = visual.table_fallback_dataset_id or visual.dataset_id
        if kind == "omitted":
            reason = str(getattr(visual, "reason", summary)).strip()
            st.info(reason)
            distinct_summary = distinct_visual_summary(reason, summary)
            if distinct_summary:
                st.caption(distinct_summary)
            continue

        frame = dataset_frame(envelope, str(visual.dataset_id or ""))
        if frame.empty:
            _fallback(fallback_id, "The validated chart dataset is unavailable. Showing its data fallback when possible.")
            st.caption(summary)
            continue

        try:
            if kind == "kpi":
                value = frame.iloc[0][visual.value_field]
                rendered = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)
                if visual.unit:
                    rendered = f"{rendered} {visual.unit}"
                st.metric(visual.label, rendered)

            elif kind == "cartesian":
                fig, ax = _axes()
                x_field = visual.x_field
                y_fields = list(visual.y_fields)
                work = frame.dropna(subset=[x_field]).copy()
                if (
                    visual.series_field
                    and visual.series_field in work.columns
                    and visual.chart_type in {"bar", "grouped_bar", "stacked_bar"}
                    and len(y_fields) == 1
                ):
                    pivoted = work.pivot_table(
                        index=x_field,
                        columns=visual.series_field,
                        values=y_fields[0],
                        aggfunc="sum",
                        fill_value=0,
                    )
                    y_fields = [str(column) for column in pivoted.columns]
                    pivoted.columns = y_fields
                    work = pivoted.reset_index()
                if visual.chart_type in {"line", "area"}:
                    x_values = pd.to_datetime(work[x_field], errors="coerce")
                    if x_values.notna().sum() != len(work):
                        x_values = work[x_field].astype(str)
                    for index, field in enumerate(y_fields):
                        values = pd.to_numeric(work[field], errors="coerce")
                        ax.plot(x_values, values, marker="o", markersize=3, linewidth=2.2, label=field)
                        if visual.chart_type == "area" and index == 0:
                            ax.fill_between(x_values, values, alpha=0.16, color="#5aa9ff")
                    ax.set_xlabel(x_field.replace("_", " ").title())
                    ax.set_ylabel((visual.y_unit or y_fields[0]).replace("_", " ").title())
                    fig.autofmt_xdate(rotation=0)
                elif visual.chart_type == "scatter":
                    y_field = y_fields[0]
                    ax.scatter(
                        pd.to_numeric(work[x_field], errors="coerce"),
                        pd.to_numeric(work[y_field], errors="coerce"),
                        color="#74d7ff",
                        edgecolors="#e7fff8",
                        alpha=0.8,
                    )
                    ax.set_xlabel((visual.x_unit or x_field).replace("_", " ").title())
                    ax.set_ylabel((visual.y_unit or y_field).replace("_", " ").title())
                else:
                    labels = work[x_field].astype(str).tolist()
                    positions = list(range(len(labels)))
                    horizontal = visual.orientation == "horizontal"
                    stacked = visual.stacked or visual.chart_type == "stacked_bar"
                    running = [0.0] * len(labels)
                    width = 0.76 / max(1, len(y_fields))
                    for index, field in enumerate(y_fields):
                        values = pd.to_numeric(work[field], errors="coerce").fillna(0.0).tolist()
                        color = ["#5aa9ff", "#8fe8ff", "#35d39a", "#ffb55e"][index % 4]
                        offset = [position + (index - (len(y_fields) - 1) / 2) * width for position in positions]
                        if horizontal:
                            ax.barh(
                                positions if stacked else offset,
                                values,
                                height=0.72 if stacked else width,
                                left=running if stacked else None,
                                label=field,
                                color=color,
                            )
                        else:
                            ax.bar(
                                positions if stacked else offset,
                                values,
                                width=0.72 if stacked else width,
                                bottom=running if stacked else None,
                                label=field,
                                color=color,
                            )
                        if stacked:
                            running = [left + value for left, value in zip(running, values)]
                    if horizontal:
                        ax.set_yticks(positions, labels)
                    else:
                        ax.set_xticks(positions, labels, rotation=0)
                if len(y_fields) > 1:
                    legend = ax.legend(frameon=False, loc="best")
                    for label in legend.get_texts():
                        label.set_color("#dcecff")
                ax.set_title(visual.title)
                st.pyplot(fig, width="stretch", clear_figure=True)
                plt.close(fig)

            elif kind == "forecast":
                fig, ax = _axes()
                work = frame.copy()
                dates = pd.to_datetime(work[visual.date_field], errors="coerce")
                predicted = pd.to_numeric(work[visual.predicted_field], errors="coerce")
                lower = pd.to_numeric(work[visual.lower_field], errors="coerce")
                upper = pd.to_numeric(work[visual.upper_field], errors="coerce")
                if visual.actual_field and visual.actual_field in work.columns:
                    ax.plot(dates, pd.to_numeric(work[visual.actual_field], errors="coerce"), color="#74d7ff", label="Actual")
                ax.plot(dates, predicted, color="#ffb55e", marker="o", label="Predicted")
                ax.fill_between(dates, lower, upper, color="#ffb55e", alpha=0.16, label="80% interval")
                legend = ax.legend(frameon=False)
                for label in legend.get_texts():
                    label.set_color("#dcecff")
                ax.set_title(visual.title)
                fig.autofmt_xdate(rotation=0)
                st.pyplot(fig, width="stretch", clear_figure=True)
                plt.close(fig)

            elif kind == "distribution":
                fig, ax = _axes()
                values = pd.to_numeric(frame[visual.value_field], errors="coerce").dropna()
                if visual.chart_type == "percentile":
                    percentiles = _ordered_percentile_frame(
                        frame,
                        value_field=visual.value_field,
                        category_field=visual.category_field,
                    )
                    if percentiles.empty:
                        raise ValueError("No finite percentile values are available")
                    positions = list(range(len(percentiles)))
                    ax.barh(
                        positions,
                        percentiles["value"].tolist(),
                        color="#5aa9ff",
                        edgecolor="#8fe8ff",
                    )
                    ax.scatter(
                        percentiles["value"].tolist(),
                        positions,
                        color="#e7fff8",
                        edgecolors="#071727",
                        zorder=3,
                    )
                    ax.set_yticks(positions, percentiles["category"].astype(str).tolist())
                    ax.invert_yaxis()
                    ax.set_ylabel(
                        (visual.category_field or "Percentile").replace("_", " ").title()
                    )
                elif visual.count_field:
                    counts = pd.to_numeric(frame.loc[values.index, visual.count_field], errors="coerce").fillna(0)
                    ax.bar(values.astype(str), counts, color="#5aa9ff")
                    ax.set_ylabel("Count")
                elif visual.chart_type == "boxplot":
                    ax.boxplot(values, vert=False, patch_artist=True, boxprops={"facecolor": "#5aa9ff"})
                else:
                    ax.hist(values, bins=visual.bins or 12, color="#5aa9ff", edgecolor="#8fe8ff")
                    ax.set_ylabel("Count")
                ax.set_xlabel((visual.unit or visual.value_field).replace("_", " ").title())
                ax.set_title(visual.title)
                st.pyplot(fig, width="stretch", clear_figure=True)
                plt.close(fig)

            elif kind == "heatmap":
                fig, ax = _axes(4.6)
                heat = frame.pivot_table(
                    index=visual.y_field,
                    columns=visual.x_field,
                    values=visual.value_field,
                    aggfunc="mean",
                )
                image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="Blues")
                ax.set_xticks(range(len(heat.columns)), [str(value) for value in heat.columns])
                ax.set_yticks(range(len(heat.index)), [str(value) for value in heat.index])
                ax.set_xlabel(visual.x_field.replace("_", " ").title())
                ax.set_ylabel(visual.y_field.replace("_", " ").title())
                ax.set_title(visual.title)
                colorbar = fig.colorbar(image, ax=ax)
                colorbar.ax.tick_params(colors="#dcecff")
                st.pyplot(fig, width="stretch", clear_figure=True)
                plt.close(fig)

            elif kind == "map":
                fig, ax = _axes(4.4)
                ax.scatter(
                    pd.to_numeric(frame[visual.longitude_field], errors="coerce"),
                    pd.to_numeric(frame[visual.latitude_field], errors="coerce"),
                    color="#35d39a",
                    edgecolors="#e7fff8",
                )
                if visual.label_field and visual.label_field in frame.columns:
                    for _, row in frame.iterrows():
                        ax.annotate(
                            str(row[visual.label_field]),
                            (row[visual.longitude_field], row[visual.latitude_field]),
                            color="#dcecff",
                            xytext=(4, 4),
                            textcoords="offset points",
                        )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(visual.title)
                st.pyplot(fig, width="stretch", clear_figure=True)
                plt.close(fig)

            elif kind == "timeline":
                fig, ax = _axes()
                work = frame.copy()
                times = pd.to_datetime(work[visual.time_field], errors="coerce")
                ax.scatter(times, [1] * len(work), color="#74d7ff")
                for index, (_, row) in enumerate(work.iterrows()):
                    ax.annotate(
                        str(row[visual.label_field]),
                        (times.iloc[index], 1),
                        color="#dcecff",
                        rotation=25,
                        xytext=(4, 8),
                        textcoords="offset points",
                    )
                ax.set_yticks([])
                ax.set_title(visual.title)
                fig.autofmt_xdate(rotation=0)
                st.pyplot(fig, width="stretch", clear_figure=True)
                plt.close(fig)

            elif kind == "table":
                visible = [field for field in visual.visible_fields if field in frame.columns]
                _safe_dataframe(frame[visible] if visible else frame, width="stretch", hide_index=True)

            else:
                _fallback(fallback_id, f"Unsupported validated visualization kind: {kind}.")
        except Exception as exc:
            _fallback(
                fallback_id,
                f"The chart could not be rendered ({type(exc).__name__}). The validated data is shown instead.",
            )
        st.caption(summary)


_GRAPH_VISUALIZATION_KINDS = frozenset(
    {"cartesian", "forecast", "distribution", "heatmap", "map", "timeline"}
)


def _primary_graph_visualization(envelope: AnswerEnvelope) -> Optional[Any]:
    """Return the first validated graph, excluding scalar and fallback-only specs."""

    return next(
        (
            visualization
            for visualization in envelope.visualizations
            if visualization.kind in _GRAPH_VISUALIZATION_KINDS
        ),
        None,
    )


def _render_primary_canonical_visualization(
    envelope: AnswerEnvelope,
    *,
    compact: bool = False,
    show_title: bool = True,
) -> bool:
    """Render only the primary graph when the canonical response genuinely has one."""

    primary = _primary_graph_visualization(envelope)
    if primary is None:
        return False
    primary_envelope = envelope.model_copy(update={"visualizations": [primary]})
    _render_canonical_visualizations(
        primary_envelope,
        compact=compact,
        show_title=show_title,
    )
    return True


@st.cache_resource
def _init_kpi_engine(processed_dir: str, cache_key: str = "") -> KPIQueryEngine:
    _ = cache_key
    return KPIQueryEngine(processed_dir=processed_dir).preload()


@st.cache_resource
def _init_forecast_engine(processed_dir: str, cache_key: str = "") -> ForecastEngine:
    _ = cache_key
    engine = ForecastEngine(processed_dir=processed_dir)
    engine.kpi.preload()
    return engine


@st.cache_resource
def _init_carbon_engine(
    processed_dir: str,
    factor_registry_path: str,
    monte_carlo_draws: int,
    cache_key: str = "",
) -> CarbonQueryEngine:
    _ = cache_key
    return CarbonQueryEngine(
        processed_dir=processed_dir,
        factor_registry_path=factor_registry_path,
        monte_carlo_draws=monte_carlo_draws,
        auto_build=True,
    ).preload()


def _validate_sample_queries_runtime(carbon_engine: Optional[CarbonQueryEngine]) -> None:
    if carbon_engine is None:
        return
    if carbon_engine.calls is None or carbon_engine.calls.empty:
        print("[sample-validation] carbon_emissions_call.parquet is empty; call-level sample queries may no_data.")
        return

    # Keep startup validation out of pandas' Arrow comparison path. Some macOS
    # pyarrow builds can crash natively when an Arrow-backed string Series is
    # compared from Streamlit's script thread.
    mmsi_values = carbon_engine.calls["mmsi"].fillna("").tolist()
    call_id_values = carbon_engine.calls["call_id"].fillna("").tolist()
    available_calls = {
        (str(mmsi).strip(), str(call_id).strip())
        for mmsi, call_id in zip(mmsi_values, call_id_values)
    }

    for category, samples in SAMPLE_QUERIES_BY_CATEGORY.items():
        for sample in samples:
            if "call-level emissions" not in sample.lower():
                continue
            mmsi_hit = re.search(r"\bmmsi\s+(\d{6,9})\b", sample, flags=re.IGNORECASE)
            call_hit = re.search(r"\bcall[_\-\s]?id[\s:=_\-]*([A-Za-z0-9_\-:.]+)\b", sample, flags=re.IGNORECASE)
            if not mmsi_hit or not call_hit:
                print(f"[sample-validation] {category}: call-level sample could not be parsed -> {sample}")
                continue
            mmsi = mmsi_hit.group(1).strip()
            call_id = re.sub(r"^[\s:_\-]+", "", call_hit.group(1).strip())
            if (mmsi, call_id) not in available_calls:
                print(
                    "[sample-validation] "
                    f"{category}: missing call-level sample data for mmsi={mmsi}, call_id={call_id}."
                )


@st.cache_resource
def _init_retriever(
    persist_dir: str,
    config_path: str,
    force_local_vector: bool = False,
) -> RAGRetriever:
    if force_local_vector:
        with force_local_vector_env():
            return RAGRetriever(persist_dir=persist_dir, config_path=config_path)
    return RAGRetriever(persist_dir=persist_dir, config_path=config_path)


def _resolve_processed_dir(preferred_dir: Path) -> tuple[Path, bool]:
    required = preferred_dir / "arrivals_daily.parquet"
    if required.exists():
        return preferred_dir, False
    fallback = Path("demo_data/processed")
    if (fallback / "arrivals_daily.parquet").exists():
        return fallback, True
    return preferred_dir, False


def _resolve_persist_dir(preferred_dir: Path) -> tuple[Path, bool]:
    required = preferred_dir / "chroma.sqlite3"
    if required.exists():
        return preferred_dir, False
    fallback = Path("demo_data/chroma")
    if (fallback / "chroma.sqlite3").exists():
        return fallback, True
    return preferred_dir, False


def _remote_vector_enabled(config: Dict[str, Any]) -> bool:
    try:
        return chroma_remote_settings(config=config) is not None
    except Exception:
        return False


def _parse_anomaly_filter(value: str) -> Optional[bool]:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _artifact_signature(paths: List[Path]) -> str:
    parts: List[str] = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}")
        except FileNotFoundError:
            parts.append(f"{path}:missing")
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _load_openai_api_key_from_runtime() -> tuple[Optional[str], str]:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key, "env"

    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""

    if secret_key:
        os.environ["OPENAI_API_KEY"] = secret_key
        return secret_key, "streamlit_secrets"
    return None, "missing"


def _load_runtime_setting(name: str) -> tuple[str, str]:
    value = str(os.getenv(name, "")).strip()
    if value:
        return value, "env"
    try:
        secret_value = str(st.secrets.get(name, "")).strip()
    except Exception:
        secret_value = ""
    if secret_value:
        os.environ[name] = secret_value
        return secret_value, "streamlit_secrets"
    return "", "missing"


def _maybe_bootstrap_processed_bundle(preferred_dir: Path) -> tuple[bool, str]:
    required_files = [
        "arrivals_daily.parquet",
        "arrivals_hourly.parquet",
        "congestion_daily.parquet",
        "dwell_time.parquet",
        "occupancy_hourly.parquet",
        "port_catalog.parquet",
        "kpi_capabilities.json",
    ]
    if all((preferred_dir / name).exists() for name in required_files):
        return False, f"Processed runtime assets already exist in {preferred_dir}."

    bundle_url, source = _load_runtime_setting("APP_PROCESSED_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_PROCESSED_BUNDLE_URL configured."

    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=preferred_dir,
        required_files=required_files,
    )
    if source != "missing":
        message = f"{message} Source: {source}."
    return changed, message


def _maybe_bootstrap_events_bundle(preferred_dir: Path) -> tuple[bool, str]:
    required_files = ["events.parquet"]
    if all((preferred_dir / name).exists() for name in required_files):
        return False, f"Events runtime asset already exists in {preferred_dir}."

    bundle_url, source = _load_runtime_setting("APP_EVENTS_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_EVENTS_BUNDLE_URL configured."

    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=preferred_dir,
        required_files=required_files,
    )
    if source != "missing":
        message = f"{message} Source: {source}."
    return changed, message


def _maybe_bootstrap_chroma_bundle(preferred_dir: Path) -> tuple[bool, str]:
    required_files = ["chroma.sqlite3", "traffic_metadata_index.csv"]
    if all((preferred_dir / name).exists() for name in required_files):
        return False, f"Chroma runtime assets already exist in {preferred_dir}."

    manifest_url, manifest_source = _load_runtime_setting("APP_CHROMA_MANIFEST_URL")
    if manifest_url:
        changed, message = ensure_file_manifest(
            url=manifest_url,
            target_dir=preferred_dir,
            required_files=required_files,
            timeout_seconds=3600,
        )
        if manifest_source != "missing":
            message = f"{message} Source: {manifest_source}."
        return changed, message

    bundle_url, source = _load_runtime_setting("APP_CHROMA_BUNDLE_URL")
    if not bundle_url:
        return False, "No APP_CHROMA_MANIFEST_URL or APP_CHROMA_BUNDLE_URL configured."

    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=preferred_dir,
        required_files=required_files,
        timeout_seconds=1800,
    )
    if source != "missing":
        message = f"{message} Source: {source}."
    return changed, message


def _pick_filter(override: Optional[str], extracted: Optional[str]) -> Optional[str]:
    if override is not None and override.strip():
        return override.strip()
    if extracted is not None and str(extracted).strip():
        return str(extracted).strip()
    return None


def _normalize_text_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^port of\s+", "", text)
    return text


def _resolve_port_token_with_mode(port_token: Optional[str], kpi: KPIQueryEngine) -> tuple[Optional[str], str]:
    token = (port_token or "").strip()
    if not token:
        return None, "missing"
    if re.fullmatch(r"[A-Za-z]{2}\s?[A-Za-z]{3}", token):
        return token.upper().replace(" ", ""), "exact_code"
    norm = _normalize_text_token(token)
    alias_code = PORT_ALIAS_TO_CODE.get(norm)
    if alias_code:
        return alias_code, "alias"

    catalog = kpi.port_catalog
    if catalog.empty:
        return token, "unresolved"

    code = token.upper().replace(" ", "")

    work = catalog.copy()
    for col in ("port_key", "locode_norm", "port_label", "port_name_norm"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    if "arrivals_total" not in work.columns:
        work["arrivals_total"] = 0
    work["arrivals_total"] = pd.to_numeric(work["arrivals_total"], errors="coerce").fillna(0)
    work["source_kind"] = work.get("source_kind", "").fillna("").astype(str).str.lower()
    work["locode_norm"] = work.get("locode_norm", "").fillna("").astype(str).str.upper()
    work["is_structured_port"] = (
        (work["source_kind"] == "port_call")
        & work["locode_norm"].str.fullmatch(r"[A-Z]{5}")
    )

    exact_code = work[
        (work["port_key"].str.upper() == code) | (work["locode_norm"].str.upper() == code)
    ]
    if not exact_code.empty:
        row = exact_code.sort_values("arrivals_total", ascending=False).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip(), "exact_code"

    work["port_label_norm"] = work["port_label"].map(_normalize_text_token)
    work["port_name_norm_clean"] = work["port_name_norm"].map(_normalize_text_token)
    exact_label = work[
        (work["port_label_norm"] == norm) | (work["port_name_norm_clean"] == norm)
    ]
    if not exact_label.empty:
        if exact_label["is_structured_port"].any():
            exact_label = exact_label[exact_label["is_structured_port"]]
        row = exact_label.sort_values("arrivals_total", ascending=False).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip(), "exact_label"

    contains = work[
        work["port_label_norm"].str.contains(norm, regex=False)
        | work["port_name_norm_clean"].str.contains(norm, regex=False)
    ]
    if not contains.empty:
        if contains["is_structured_port"].any():
            contains = contains[contains["is_structured_port"]]
        row = contains.sort_values("arrivals_total", ascending=False).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip(), "contains"

    def _best_similarity(row: pd.Series) -> float:
        cand_a = str(row.get("port_label_norm", ""))
        cand_b = str(row.get("port_name_norm_clean", ""))
        return max(
            SequenceMatcher(None, norm, cand_a).ratio() if cand_a else 0.0,
            SequenceMatcher(None, norm, cand_b).ratio() if cand_b else 0.0,
        )

    work["similarity"] = work.apply(_best_similarity, axis=1)
    fuzzy = work[work["similarity"] >= 0.80]
    if not fuzzy.empty:
        if fuzzy["is_structured_port"].any():
            fuzzy = fuzzy[fuzzy["is_structured_port"]]
        row = fuzzy.sort_values(["similarity", "arrivals_total"], ascending=[False, False]).iloc[0]
        return str(row.get("port_key") or row.get("locode_norm") or token).strip(), "fuzzy"

    return token, "unresolved"


def _resolve_port_token(port_token: Optional[str], kpi: KPIQueryEngine) -> Optional[str]:
    resolved, _ = _resolve_port_token_with_mode(port_token, kpi)
    return resolved


def _resolve_ports(port_tokens: List[str], kpi: KPIQueryEngine) -> List[str]:
    resolved: List[str] = []
    for token in port_tokens:
        mapped = _resolve_port_token(token, kpi) or token
        if mapped not in resolved:
            resolved.append(mapped)
    return resolved


def _is_known_port_token(port_token: Optional[str], kpi: KPIQueryEngine) -> bool:
    token = (port_token or "").strip()
    if not token:
        return False
    code = token.upper().replace(" ", "")
    catalog = kpi.port_catalog
    if catalog.empty:
        return bool(re.fullmatch(r"[A-Z]{5}", code))

    work = catalog.copy()
    for col in ("port_key", "locode_norm"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

    mask = (work["port_key"] == code) | (work["locode_norm"] == code)
    return bool(mask.any())


def _extract_port_tokens_from_question(question: str) -> List[str]:
    candidates: List[str] = []
    for m in re.finditer(r"\b([A-Za-z]{2})\s*([A-Za-z]{3})\b", question):
        token = f"{m.group(1)}{m.group(2)}".upper()
        if token[:2] in BALTIC_LOCODE_PREFIXES and token not in PORT_PARSE_STOPWORDS and token not in candidates:
            candidates.append(token)

    for alias in PORT_ALIAS_TO_CODE.keys():
        if alias in _normalize_text_token(question):
            if alias not in candidates:
                candidates.append(alias)

    return candidates[:8]


def _resolve_scope_with_aggressive_port_fallback(
    question: str,
    entities: Dict[str, Any],
    user_filters: Dict[str, Any],
    kpi: KPIQueryEngine,
) -> Dict[str, Any]:
    raw_port = _pick_filter(user_filters.get("port"), entities.get("port"))
    start = _pick_filter(user_filters.get("date_from"), entities.get("date_from"))
    end = _pick_filter(user_filters.get("date_to"), entities.get("date_to"))

    ranked_inputs: List[Tuple[str, str]] = []
    raw_user_port = str(user_filters.get("port") or "").strip()
    raw_entity_port = str(entities.get("port") or "").strip()
    for source, token in (
        [("user_filter", user_filters.get("port")), ("entity_primary", entities.get("port"))]
        + [("entity_port", item) for item in list(entities.get("ports") or [])]
        + [("question_scan", item) for item in _extract_port_tokens_from_question(question)]
    ):
        t = str(token or "").strip()
        if t and all(existing[1] != t for existing in ranked_inputs):
            ranked_inputs.append((source, t))

    resolved_candidates: List[Dict[str, Any]] = []
    for source, token in ranked_inputs:
        mapped, mode = _resolve_port_token_with_mode(token, kpi)
        valid = _is_known_port_token(mapped, kpi)
        resolved_candidates.append(
            {
                "source": source,
                "token": token,
                "resolved": mapped,
                "mode": mode,
                "valid": valid,
            }
        )

    resolved_port = None
    resolved_mode = "missing"
    selected_candidate: Optional[Dict[str, Any]] = None
    for item in resolved_candidates:
        if item["valid"]:
            resolved_port = str(item["resolved"]).strip()
            resolved_mode = str(item.get("mode") or "missing")
            selected_candidate = item
            break
    if resolved_port is None:
        if raw_user_port:
            resolved_port, resolved_mode = _resolve_port_token_with_mode(raw_user_port, kpi)
            resolved_port = resolved_port or raw_user_port
        else:
            resolved_port = None

    normalized_raw = str(raw_port or "").upper().replace(" ", "")
    normalized_resolved = str(resolved_port or "").upper().replace(" ", "")
    selected_token = str((selected_candidate or {}).get("token") or "")
    selected_source = str((selected_candidate or {}).get("source") or "")
    correction_applied = bool(
        resolved_port
        and (
            normalized_raw != normalized_resolved
            or selected_token != str(raw_port or "")
            or selected_source not in {"", "user_filter", "entity_primary"}
            or resolved_mode in {"alias", "exact_label", "contains", "fuzzy"}
        )
    )
    correction_note = None
    if correction_applied:
        resolution_labels = {
            "alias": "canonical alias mapping",
            "exact_label": "catalog label mapping",
            "contains": "catalog substring fallback",
            "fuzzy": "fuzzy similarity fallback",
            "exact_code": "validated catalog code",
        }
        fallback_label = resolution_labels.get(resolved_mode, "validated fallback candidate")
        source_token = selected_token or str(raw_port or "")
        correction_note = (
            f"Resolved scope correction: port token `{source_token}` was mapped to `{resolved_port}` "
            f"using {fallback_label}."
        )
    elif raw_entity_port and not raw_user_port and resolved_port is None:
        correction_note = (
            f"Resolved scope correction: ignored ambiguous parsed port token `{raw_entity_port}` because no valid catalog match was found."
        )

    return {
        "raw_port": raw_port,
        "port": resolved_port,
        "date_from": start,
        "date_to": end,
        "resolved_candidates": resolved_candidates,
        "resolved_mode": resolved_mode,
        "correction_applied": correction_applied,
        "correction_note": correction_note,
    }


def _derive_answer_source(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence: EvidenceBundle,
) -> tuple[str, str]:
    if isinstance(result, CarbonResult):
        if result.result_state in {CARBON_STATE_NOT_COMPUTABLE, CARBON_STATE_UNSUPPORTED}:
            return (
                "Not computable from available carbon data",
                "No deterministic carbon inventory matched the requested scope.",
            )
        if result.result_state == CARBON_STATE_RETRIEVAL_ONLY:
            return (
                "Retrieved supporting traffic evidence only",
                "Traffic retrieval found relevant context, but numeric carbon emissions could not be computed reliably.",
            )
        if result.result_state == CARBON_STATE_FORECAST_ONLY:
            return (
                "Forecast request not computable",
                "Carbon forecast was requested but no deterministic carbon forecast model is configured in this runtime.",
            )
        label = "Structured carbon data"
        if evidence.rows:
            label = "Structured carbon data + supporting evidence"
        detail = (
            "Carbon metrics are deterministic inventory outputs from AIS + port-call segmentation. "
            "Vector retrieval is optional supporting evidence."
        )
        return label, detail

    if isinstance(result, ForecastResult):
        return (
            "Historical forecast data",
            "Forecast values use the filtered historical time series for the applied scope.",
        )

    label = "Structured data" + (" + supporting evidence" if evidence.rows else "")
    detail = "Numeric output comes from the filtered structured rows for the applied scope."
    return label, detail


def _make_rag_filters(
    entities: Dict[str, Any],
    overrides: Dict[str, Any],
    include_dates: bool = True,
) -> QueryFilters:
    port_token = _pick_filter(overrides.get("port"), entities.get("port"))

    locode = None
    destination = None
    port_name = None
    if port_token:
        if re.fullmatch(r"[A-Za-z]{2}\s?[A-Za-z]{3}", port_token):
            locode = port_token
        else:
            destination = port_token
            port_name = port_token

    date_from = _pick_filter(overrides.get("date_from"), entities.get("date_from")) if include_dates else None
    date_to = _pick_filter(overrides.get("date_to"), entities.get("date_to")) if include_dates else None

    return QueryFilters(
        mmsi=entities.get("mmsi"),
        imo=entities.get("imo"),
        locode=locode,
        port_name=port_name,
        destination=destination,
        vessel_type=_pick_filter(overrides.get("vessel_type"), entities.get("vessel_type")),
        date_from=date_from,
        date_to=date_to,
    )


def _retrieve_evidence(
    retriever: Optional[RAGRetriever],
    question: str,
    entities: Dict[str, Any],
    overrides: Dict[str, Any],
    top_k: int,
    include_dates: bool,
) -> EvidenceBundle:
    if retriever is None:
        return EvidenceBundle(
            lines=[],
            rows=[],
            trace={
                "retrieval_status": "disabled",
                "reason": "Retriever disabled for this deterministic request.",
            },
        )

    try:
        filters = _make_rag_filters(entities=entities, overrides=overrides, include_dates=include_dates)
        started = time.perf_counter()
        result = retriever.query_traffic(question=question, filters=filters, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        return EvidenceBundle(
            lines=[],
            rows=[],
            trace={
                "retrieval_status": "error",
                "reason": f"Vector retrieval failed: {exc}",
            },
        )

    lines: List[str] = []
    rows: List[Dict[str, Any]] = []
    for item in result.evidence[:top_k]:
        chunk_id = str(
            item.metadata.get("chunk_id")
            or item.metadata.get("stable_id")
            or item.metadata.get("id")
            or ""
        )
        dist_txt = f"{float(item.distance):.4f}" if item.distance is not None else "n/a"
        lines.append(
            f"`vector_id={item.id}` | `chunk_id={chunk_id or 'n/a'}` | `dist={dist_txt}` | "
            f"{compact_traffic_evidence(item.metadata, item.text)}"
        )
        rows.append(
            {
                "vector_id": item.id,
                "chunk_id": chunk_id or None,
                "distance": item.distance,
                "timestamp": item.metadata.get("timestamp_full") or item.metadata.get("date"),
                "port": item.metadata.get("locode_norm")
                or item.metadata.get("locode")
                or item.metadata.get("port_name")
                or item.metadata.get("destination_norm"),
                "vessel_type": item.metadata.get("vessel_type_norm")
                or item.metadata.get("vessel_type"),
                "mmsi": item.metadata.get("mmsi"),
            }
        )

    active_filters = {
        key: value
        for key, value in {
            "mmsi": filters.mmsi,
            "imo": filters.imo,
            "locode": filters.locode,
            "port_name": filters.port_name,
            "destination": filters.destination,
            "vessel_type": filters.vessel_type,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "lat_min": filters.lat_min,
            "lat_max": filters.lat_max,
            "lon_min": filters.lon_min,
            "lon_max": filters.lon_max,
        }.items()
        if value not in (None, "", [])
    }

    trace = {
        "retrieval_status": "ok" if rows else "no_hits",
        "reason": "Vector rows retrieved successfully." if rows else "No vector rows matched the query and filters.",
        "collection": retriever.config["index"]["traffic_collection"],
        "mode": result.mode,
        "vector_backend": getattr(retriever, "vector_backend", "unknown"),
        "query_latency_ms": round(latency_ms, 2),
        "returned_items": len(result.evidence),
        "top_k_requested": top_k,
        "where_filter": result.where_filter,
        "active_filters": active_filters,
    }
    return EvidenceBundle(lines=lines, rows=rows, trace=trace)


def _extract_prediction_triplet(
    result: ForecastResult,
    target_date: Optional[str],
    target_dow: Optional[str],
) -> Optional[Tuple[float, float, float]]:
    if result.forecast is None or result.forecast.empty:
        return None

    df = result.forecast.copy()
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
    rows = df.dropna(subset=["date"])

    if target_date:
        target_ts = pd.to_datetime(target_date, errors="coerce", utc=True)
        if pd.notna(target_ts):
            target_ts = pd.Timestamp(target_ts).floor("D")
            picked = rows[rows["date"] == target_ts]
            if picked.empty:
                picked = rows.tail(1)
            return (
                float(picked["predicted"].mean()),
                float(picked["lower"].mean()),
                float(picked["upper"].mean()),
            )

    if target_dow:
        dow_rows = rows[rows["date"].dt.day_name() == target_dow.title()]
        if dow_rows.empty:
            dow_rows = rows.tail(1)
        return (
            float(dow_rows["predicted"].mean()),
            float(dow_rows["lower"].mean()),
            float(dow_rows["upper"].mean()),
        )

    tail = rows.tail(1)
    return (
        float(tail["predicted"].mean()),
        float(tail["lower"].mean()),
        float(tail["upper"].mean()),
    )


def _compare_forecast_ports(
    forecaster: ForecastEngine,
    ports: List[str],
    target_date: Optional[str],
    target_dow: Optional[str],
    horizon_weeks: int,
) -> AnalyticsResult:
    return forecaster.compare_congestion_ports(
        ports=ports,
        target_date=target_date,
        target_dow=target_dow,
        horizon_weeks=horizon_weeks,
    )


def _handle_ask_question(
    question: str,
    intent_result: IntentResult,
    kpi: KPIQueryEngine,
    forecaster: ForecastEngine,
    carbon: CarbonQueryEngine,
    retriever: Optional[RAGRetriever],
    top_k_evidence: int,
    user_filters: Dict[str, Any],
    events_path: Optional[Path],
) -> tuple[Union[AnalyticsResult, ForecastResult, CarbonResult], EvidenceBundle]:
    """Retired pre-v2 dispatcher retained only for legacy offline contract imports.

    The interactive Streamlit page must never call this function. Both submit
    paths route through ``run_canonical_query`` and ``QueryService.query``.
    """

    entities = dict(intent_result.entities or {})
    q_lower = question.lower()

    scope = _resolve_scope_with_aggressive_port_fallback(
        question=question,
        entities=entities,
        user_filters=user_filters,
        kpi=kpi,
    )
    raw_port = scope.get("raw_port")
    port = scope.get("port")
    start = scope.get("date_from")
    end = scope.get("date_to")
    vessel_type = _pick_filter(user_filters.get("vessel_type"), entities.get("vessel_type"))
    dow = entities.get("dow")
    target_date = entities.get("target_date")
    window = entities.get("window")
    metric = entities.get("metric", "arrivals_vessels")
    aggregation = entities.get("aggregation")
    result_limit = int(entities.get("limit") or 1)
    mmsi = entities.get("mmsi")
    horizon_weeks = int(entities.get("horizon_weeks") or 4)
    source_scope = entities.get("source_scope")

    ports: List[str] = [str(p).strip() for p in entities.get("ports") or [] if str(p).strip()]
    if port and port not in ports:
        ports.insert(0, port)
    has_valid_explicit_port = any(
        _is_known_port_token(_resolve_port_token(token, kpi), kpi)
        for token in ports
    )
    for cand in scope.get("resolved_candidates", []):
        if not cand.get("valid"):
            continue
        if has_valid_explicit_port and cand.get("source") == "question_scan":
            continue
        resolved = str(cand.get("resolved") or "").strip()
        if resolved and resolved not in ports:
            ports.append(resolved)
    ports = _resolve_ports(ports, kpi)
    route_pairs_raw = list(entities.get("route_pairs") or [])
    route_pairs_resolved: List[Dict[str, str]] = []
    for pair in route_pairs_raw:
        origin_raw = str(pair.get("origin") or "").strip()
        destination_raw = str(pair.get("destination") or "").strip()
        if not origin_raw or not destination_raw:
            continue
        origin_resolved = _resolve_ports([origin_raw], kpi)[0]
        destination_resolved = _resolve_ports([destination_raw], kpi)[0]
        if not origin_resolved or not destination_resolved:
            continue
        if origin_resolved == destination_resolved:
            continue
        candidate = {"origin": origin_resolved, "destination": destination_resolved}
        if candidate not in route_pairs_resolved:
            route_pairs_resolved.append(candidate)

    origin_port_raw = _pick_filter(user_filters.get("origin_port"), entities.get("origin_port"))
    destination_port_raw = _pick_filter(user_filters.get("destination_port"), entities.get("destination_port"))
    origin_port = _resolve_ports([origin_port_raw], kpi)[0] if origin_port_raw else (ports[0] if len(ports) >= 1 else None)
    destination_port = _resolve_ports([destination_port_raw], kpi)[0] if destination_port_raw else (ports[1] if len(ports) >= 2 else None)
    entities["ports"] = ports
    entities["port"] = port
    entities["origin_port"] = origin_port
    entities["destination_port"] = destination_port
    entities["route_pairs"] = route_pairs_resolved
    entities["date_from"] = start
    entities["date_to"] = end
    extraction_diag = dict(entities.get("extraction_diagnostics") or {})
    extraction_diag["resolved_scope"] = {
        "raw_port": raw_port,
        "resolved_port": port,
        "date_from": start,
        "date_to": end,
        "origin_port": origin_port,
        "destination_port": destination_port,
        "route_pairs": route_pairs_resolved,
        "correction_applied": bool(scope.get("correction_applied")),
        "candidates": scope.get("resolved_candidates", []),
    }
    entities["extraction_diagnostics"] = extraction_diag
    intent_result.entities = entities

    if start and end:
        start_ts = pd.to_datetime(start, errors="coerce", utc=True)
        end_ts = pd.to_datetime(end, errors="coerce", utc=True)
        if pd.notna(start_ts) and pd.notna(end_ts) and start_ts > end_ts:
            return (
                KPIQueryEngine.no_data("Invalid date range: `From date` is after `To date`."),
                EvidenceBundle(lines=[], rows=[], trace={}),
            )

    evidence_overrides = dict(user_filters)
    if port:
        evidence_overrides["port"] = port
    if start:
        evidence_overrides["date_from"] = start
    if end:
        evidence_overrides["date_to"] = end

    if intent_result.intent == "G":
        return (
            KPIQueryEngine.unsupported(intent_result.reason),
            EvidenceBundle(lines=[], rows=[], trace={}),
        )

    if raw_port and not port:
        return (
            KPIQueryEngine.no_data(
                f"Requested port `{raw_port}` is not present in the canonical port catalog; no broader scope was substituted."
            ),
            EvidenceBundle(lines=[], rows=[], trace={}),
        )

    if intent_result.intent == "H":
        carbon_filters = dict(user_filters)
        if port:
            carbon_filters["port"] = port
        if start:
            carbon_filters["date_from"] = start
        if end:
            carbon_filters["date_to"] = end
        try:
            result = carbon.from_question_entities(
                question=question,
                entities=entities,
                user_filters=carbon_filters,
                resolved_scope={"port": port, "date_from": start, "date_to": end},
            )
        except TypeError as exc:
            # Backward-compatible fallback if an older runtime/module copy
            # does not yet accept `resolved_scope`.
            if "resolved_scope" not in str(exc):
                raise
            result = carbon.from_question_entities(
                question=question,
                entities=entities,
                user_filters=carbon_filters,
            )
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if isinstance(result, CarbonResult):
            if result.result_state == CARBON_STATE_NOT_COMPUTABLE and evidence.rows:
                result.result_state = CARBON_STATE_RETRIEVAL_ONLY
                result.source_label = "Supporting traffic evidence"
                result.confidence_label = "not_applicable"
                result.confidence_reason = "No structured carbon rows are available for this scope."
                result.coverage_notes.append(
                    "Traffic evidence was retrieved, but numeric carbon emissions could not be computed reliably."
                )
                result.diagnostics = dict(result.diagnostics or {})
                result.diagnostics["result_state"] = CARBON_STATE_RETRIEVAL_ONLY
                result.diagnostics["sanity_status"] = result.diagnostics.get("sanity_status", "warning")
            elif result.status == "ok" and evidence.rows and result.result_state in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                result.source_label = "Hybrid (computed + retrieved supporting evidence)"
            if scope.get("correction_note"):
                result.coverage_notes.append(str(scope["correction_note"]))
                result.diagnostics = dict(result.diagnostics or {})
                result.diagnostics["scope_correction_note"] = str(scope["correction_note"])
        return result, evidence

    if intent_result.intent == "A":
        if aggregation == "peak_day" and metric == "congestion_index":
            result = kpi.get_peak_congestion_days(
                port=port,
                start=start,
                end=end,
                dow=dow,
                window=window,
                limit=result_limit,
            )
        elif "pressure" in q_lower and "vessel type" in q_lower:
            result = kpi.get_pressure_by_vessel_type(port=port, start=start, end=end)
        elif aggregation == "first_route_vessel":
            result = kpi.get_first_route_vessel(
                origin_port=origin_port,
                destination_port=destination_port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif aggregation == "first_arrival":
            result = kpi.get_first_arrival(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif aggregation == "last_arrival":
            result = kpi.get_last_arrival(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif aggregation == "first_departure":
            result = kpi.get_first_departure(
                port=port or origin_port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif aggregation == "route_travel_time_summary":
            result = kpi.get_route_travel_time_summary(
                origin_port=origin_port,
                destination_port=destination_port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
            )
        elif len(ports) >= 2 and not any(token in q_lower for token in ("compare", "vs", "versus", "more than", "less than")):
            result = kpi.get_arrivals_multi(
                ports=ports,
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
                window=window,
                source_scope=source_scope,
            )
        elif aggregation == "peak_day":
            result = kpi.get_peak_arrival_day(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
                source_scope=source_scope,
            )
        elif mmsi and any(token in q_lower for token in ("how long", "dwell", "in port", "port stay", "port-stay", "stayed")):
            result = kpi.get_mmsi_port_stays(mmsi=str(mmsi), start=start, end=end, port=port)
        elif "top" in q_lower and "port" in q_lower:
            result = kpi.top_ports_by_arrivals(
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
                source_scope=source_scope,
            )
        elif "dwell" in q_lower:
            result = kpi.get_avg_dwell_time(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow)
        elif "congestion" in q_lower or "pressure" in q_lower:
            result = kpi.get_congestion(port=port, start=start, end=end, dow=dow, window=window)
        else:
            result = kpi.get_arrivals(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
                window=window,
                source_scope=source_scope,
            )
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if scope.get("correction_note"):
            result.coverage_notes.append(str(scope["correction_note"]))
        return result, evidence

    if intent_result.intent == "B":
        if aggregation == "peak_day" and metric == "congestion_index":
            result = kpi.get_peak_congestion_days(
                port=port,
                start=start,
                end=end,
                dow=dow,
                window=window,
                limit=result_limit,
            )
        elif metric == "congestion_index" and entities.get("dow") and entities.get("dow_compare"):
            result = kpi.compare_congestion_weekdays(
                port=port,
                start=start,
                end=end,
                day_a=entities["dow"],
                day_b=entities["dow_compare"],
            )
        elif aggregation == "peak_day":
            result = kpi.get_peak_arrival_day(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                window=window,
                source_scope=source_scope,
            )
        elif entities.get("dow") and entities.get("dow_compare"):
            result = kpi.compare_weekdays(
                port=port,
                start=start,
                end=end,
                day_a=entities["dow"],
                day_b=entities["dow_compare"],
                vessel_type=vessel_type,
                source_scope=source_scope,
            )
        elif "hour" in q_lower:
            result = kpi.get_busiest_hour(port=port, start=start, end=end, vessel_type=vessel_type)
        else:
            result = kpi.get_busiest_dow(
                port=port,
                start=start,
                end=end,
                vessel_type=vessel_type,
                source_scope=source_scope,
            )

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if scope.get("correction_note"):
            result.coverage_notes.append(str(scope["correction_note"]))
        return result, evidence

    if intent_result.intent == "C":
        if entities.get("dow") and entities.get("dow_compare") and len(ports) <= 1:
            result = forecaster.compare_congestion_weekdays(
                port=port or "",
                day_a=str(entities["dow"]),
                day_b=str(entities["dow_compare"]),
                horizon_weeks=horizon_weeks,
            )
            evidence = _retrieve_evidence(
                retriever=retriever,
                question=question,
                entities=entities,
                overrides=evidence_overrides,
                top_k=top_k_evidence,
                include_dates=False,
            )
            if scope.get("correction_note"):
                result.coverage_notes.append(str(scope["correction_note"]))
            return result, evidence

        if len(ports) >= 2 and any(token in q_lower for token in ("compare", "vs", "versus", "more than", "less than")):
            result = _compare_forecast_ports(
                forecaster=forecaster,
                ports=ports,
                target_date=target_date,
                target_dow=dow,
                horizon_weeks=horizon_weeks,
            )
            evidence = _retrieve_evidence(
                retriever=retriever,
                question=question,
                entities=entities,
                overrides=evidence_overrides,
                top_k=top_k_evidence,
                include_dates=False,
            )
            if scope.get("correction_note"):
                result.coverage_notes.append(str(scope["correction_note"]))
            return result, evidence

        if target_date:
            result = forecaster.forecast_congestion_for_date(
                port=port or "",
                target_date=target_date,
                horizon_weeks=horizon_weeks,
            )
        else:
            result = forecaster.forecast_congestion(
                port=port or "",
                target_dow=dow or "Friday",
                horizon_weeks=horizon_weeks,
            )

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=False,
        )
        if scope.get("correction_note"):
            result.coverage_notes.append(str(scope["correction_note"]))
        return result, evidence

    if intent_result.intent == "D":
        if route_pairs_resolved and ports:
            result = kpi.compare_ports_and_routes(
                ports=ports,
                route_pairs=route_pairs_resolved,
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
                window=window,
            )
        elif route_pairs_resolved and not ports:
            route_rows = []
            for pair in route_pairs_resolved:
                route_result = kpi.get_route_travel_time_summary(
                    origin_port=str(pair.get("origin") or ""),
                    destination_port=str(pair.get("destination") or ""),
                    start=start,
                    end=end,
                    vessel_type=vessel_type,
                    window=window,
                )
                if route_result.status != "ok" or route_result.table is None:
                    continue
                summary = route_result.answer
                route_rows.append({"route": f"{pair.get('origin')}->{pair.get('destination')}", "summary": summary})
            if route_rows:
                route_df = pd.DataFrame(route_rows)
                result = AnalyticsResult(
                    status="ok",
                    answer=f"Computed route comparison across {len(route_df):,} route(s).",
                    table=route_df,
                    chart=None,
                    coverage_notes=["Route comparison uses median and p90 duration summaries from matched route-event records."],
                    caveats=[],
                )
            else:
                result = KPIQueryEngine.no_data("No route-duration rows were available for the requested route comparison.")
        else:
            result = kpi.compare_ports(
                ports=ports,
                metric=metric,
                start=start,
                end=end,
                vessel_type=vessel_type,
                dow=dow,
                source_scope=source_scope,
            )
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if scope.get("correction_note"):
            result.coverage_notes.append(str(scope["correction_note"]))
        return result, evidence

    if intent_result.intent == "E":
        if not start and not end and not kpi.congestion.empty:
            latest = pd.to_datetime(kpi.congestion["date"], errors="coerce", utc=True).max()
            if pd.notna(latest):
                start = end = latest.strftime("%Y-%m-%d")
        target = start or end
        result = kpi.diagnose_congestion(port=port, target_date=target)
        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if scope.get("correction_note"):
            result.coverage_notes.append(str(scope["correction_note"]))
        return result, evidence

    if intent_result.intent == "F":
        if metric == "ais_jump" or any(
            token in q_lower
            for token in ("jump", "spoof", "teleport", "impossible", "movement anomal", "position anomal")
        ):
            filters = _make_rag_filters(entities=entities, overrides=evidence_overrides, include_dates=True)
            jump_result: Dict[str, Any]
            jump_source = ""
            if events_path and events_path.exists():
                jump_result = detect_sudden_jump_events_from_parquet(
                    events_path=events_path,
                    mmsi=filters.mmsi,
                    date_from=filters.date_from,
                    date_to=filters.date_to,
                )
                jump_source = "row-level AIS events parquet"
            elif retriever is not None:
                jump_result = retriever.detect_sudden_jumps(filters=filters)
                jump_source = "AIS metadata index"
            else:
                return (
                    AnalyticsResult(
                        status="no_data",
                        answer="I don't have row-level AIS evidence in the current runtime to verify jump anomalies.",
                        table=None,
                        chart=None,
                        coverage_notes=[],
                        caveats=[
                            "This query needs either a working vector retriever or events.parquet in the cloud runtime.",
                            "Configure remote Chroma or set APP_EVENTS_BUNDLE_URL for event-level anomaly detection.",
                        ],
                    ),
                    EvidenceBundle(lines=[], rows=[], trace={}),
                )

            count = int(jump_result.get("count", 0))
            events = pd.DataFrame(jump_result.get("events") or [])
            chart = None
            table = None
            if not events.empty:
                table_cols = [
                    c
                    for c in [
                        "mmsi",
                        "timestamp_full",
                        "distance_km",
                        "implied_speed_kn",
                        "dt_minutes",
                        "trigger_rule",
                        "latitude",
                        "longitude",
                        "prev_latitude",
                        "prev_longitude",
                        "port",
                        "stable_id",
                    ]
                    if c in events.columns
                ]
                table = events[table_cols].copy()
                if {"timestamp_full", "distance_km"}.issubset(events.columns):
                    chart = (
                        events.assign(
                            timestamp_dt=pd.to_datetime(events["timestamp_full"], errors="coerce", utc=True)
                        )
                        .dropna(subset=["timestamp_dt"])
                        .sort_values("timestamp_dt")
                        .set_index("timestamp_dt")[["distance_km"]]
                    )

            if count > 0:
                answer = f"Detected {count} potential sudden AIS coordinate jumps in the filtered range."
            else:
                answer = "No sudden AIS coordinate jumps were detected in the filtered range."

            result = AnalyticsResult(
                status="ok",
                answer=answer,
                table=table,
                chart=chart,
                coverage_notes=[
                    f"Rows used: {count}",
                    f"Data sources used: {jump_source}",
                ],
                caveats=[
                    "Jump rule: distance >= 20 km within 30 minutes, or implied speed >= 40 kn with >= 5 km displacement.",
                    "This screening rule identifies unusual movement; it does not establish spoofing.",
                ],
            )
        else:
            result = kpi.detect_arrival_spikes(port=port, start=start, end=end)

        evidence = _retrieve_evidence(
            retriever=retriever,
            question=question,
            entities=entities,
            overrides=evidence_overrides,
            top_k=top_k_evidence,
            include_dates=True,
        )
        if (
            metric == "ais_jump"
            and result.table is not None
            and not result.table.empty
            and not evidence.rows
        ):
            trace = dict(evidence.trace or {})
            trace["retrieval_status"] = "computed_only"
            trace["reason"] = (
                "Vector retrieval returned no hits; evidence is from row-level AIS jump computation."
            )
            evidence = EvidenceBundle(lines=evidence.lines, rows=evidence.rows, trace=trace)
        return result, evidence

    result = kpi.get_arrivals(port=port, start=start, end=end, vessel_type=vessel_type, dow=dow, window=window)
    evidence = _retrieve_evidence(
        retriever=retriever,
        question=question,
        entities=entities,
        overrides=evidence_overrides,
        top_k=top_k_evidence,
        include_dates=True,
    )
    if scope.get("correction_note"):
        result.coverage_notes.append(str(scope["correction_note"]))
    return result, evidence


def _render_compact_result(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence: EvidenceBundle,
    show_technical: bool,
    canonical_envelope: Optional[AnswerEnvelope] = None,
    intent_result: Optional[IntentResult] = None,
    carbon_engine: Optional[CarbonQueryEngine] = None,
    threshold_percentiles: Tuple[float, float, float] = (0.25, 0.50, 0.75),
    question: Optional[str] = None,
    screenshot_mode: bool = False,
    surface_all_canonical_visuals: bool = False,
    query_service: Optional[QueryService] = None,
) -> None:
    def _apply_result_ui_polish_styles() -> None:
        st.markdown(
            """
<style>
.ee-result-title {
    margin: 0.15rem 0 0.65rem 0;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--ee-text);
}
.ee-capture-note {
    margin: 0 0 0.55rem 0;
    font-size: 0.92rem;
    color: var(--ee-muted);
}
.ee-section-title {
    margin: 0.0rem 0 0.4rem 0;
    font-size: 1.02rem;
    font-weight: 650;
    color: var(--ee-text);
}
.ee-subsection-title {
    margin: 0.5rem 0 0.25rem 0;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--ee-muted);
}
div[data-testid="stMetric"] {
    background: var(--ee-surface);
    border: 1px solid var(--ee-border);
    border-radius: 6px;
    padding: 0.45rem 0.6rem;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 0.82rem;
    color: var(--ee-muted);
}
div[data-testid="stMetricValue"] > div {
    font-size: 1.05rem;
}
</style>
            """,
            unsafe_allow_html=True,
        )

    def _section_title(label: str) -> None:
        st.markdown(f"<div class='ee-section-title'>{label}</div>", unsafe_allow_html=True)

    def _subsection_title(label: str) -> None:
        st.markdown(f"<div class='ee-subsection-title'>{label}</div>", unsafe_allow_html=True)

    def _meta_card(label: str, value: str, hint: Optional[str] = None) -> None:
        hint_html = f"<div class='ee-meta-hint'>{escape(hint)}</div>" if hint else ""
        st.markdown(
            f"""
            <div class="ee-meta-card">
                <div class="ee-meta-label">{escape(label)}</div>
                <div class="ee-meta-value">{escape(value)}</div>
                {hint_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    _apply_result_ui_polish_styles()
    st.markdown("<div class='ee-result-title'>Analysis result</div>", unsafe_allow_html=True)
    canonical_view = canonical_presentation(canonical_envelope) if canonical_envelope is not None else None

    def _fallback_evidence_from_result(
        value: Union[AnalyticsResult, ForecastResult, CarbonResult],
        max_items: int = 5,
    ) -> List[str]:
        lines: List[str] = []

        if isinstance(value, CarbonResult):
            for eid in (value.evidence_ids or [])[:max_items]:
                lines.append(f"carbon_evidence_id={eid}")
            if value.table is not None and not value.table.empty:
                head = value.table.head(min(3, max_items)).copy()
                metric_col = "wtw_co2e_t" if value.boundary in {"WTW", "TTW_WTW"} else "ttw_co2e_t"
                if metric_col not in head.columns:
                    metric_col = "co2_t" if "co2_t" in head.columns else metric_col
                for _, row in head.iterrows():
                    tokens = []
                    if "date" in head.columns and pd.notna(row.get("date")):
                        tokens.append(f"date_utc={pd.to_datetime(row.get('date'), errors='coerce', utc=True).strftime('%Y-%m-%d')}")
                    if metric_col in head.columns and pd.notna(row.get(metric_col)):
                        tokens.append(f"{metric_col}={format_tco2e(float(row.get(metric_col)))}")
                    if "port_key" in head.columns and pd.notna(row.get("port_key")):
                        tokens.append(f"port={row.get('port_key')}")
                    if tokens:
                        lines.append(" | ".join(tokens))
            return lines[:max_items]

        if isinstance(value, ForecastResult):
            anchor_values_note = next(
                (n for n in value.coverage_notes if n.startswith("Analog values used:")),
                None,
            )
            anchor_dates_note = next(
                (n for n in value.coverage_notes if n.startswith("Analog dates used:")),
                None,
            )
            if anchor_values_note:
                lines.append(anchor_values_note)
            elif anchor_dates_note:
                lines.append(anchor_dates_note)

            if value.history is not None and not value.history.empty:
                hist = value.history.copy()
                if "date" in hist.columns and "actual" in hist.columns:
                    hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.floor("D")
                    hist = hist.dropna(subset=["date", "actual"]).sort_values("date")
                    for _, row in hist.tail(max_items).iterrows():
                        lines.append(
                            f"Historical point | {row['date'].strftime('%Y-%m-%d')} | value={float(row['actual']):.2f}"
                        )

            if value.forecast is not None and not value.forecast.empty:
                fdf = value.forecast.copy()
                if "date" in fdf.columns and "predicted" in fdf.columns:
                    fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce", utc=True).dt.floor("D")
                    fdf = fdf.dropna(subset=["date", "predicted"]).sort_values("date")
                    for _, row in fdf.tail(min(2, max_items)).iterrows():
                        lower = float(row["lower"]) if "lower" in row and pd.notna(row["lower"]) else float("nan")
                        upper = float(row["upper"]) if "upper" in row and pd.notna(row["upper"]) else float("nan")
                        lines.append(
                            f"Forecast target | {row['date'].strftime('%Y-%m-%d')} | "
                            f"pred={float(row['predicted']):.2f}, range={lower:.2f}-{upper:.2f}"
                        )
            return lines[:max_items]

        if value.table is not None and not value.table.empty:
            tdf = value.table.head(max_items).copy()
            for _, row in tdf.iterrows():
                fragments: List[str] = []
                for col in tdf.columns[:4]:
                    cell = row[col]
                    if pd.isna(cell):
                        continue
                    if isinstance(cell, pd.Timestamp):
                        rendered = cell.strftime("%Y-%m-%d")
                    else:
                        rendered = str(cell)
                    fragments.append(f"{col}={rendered}")
                if fragments:
                    lines.append(" | ".join(fragments))
        return lines[:max_items]

    def _to_analyst_evidence_line(line: str) -> str:
        if "|" not in line:
            return line
        return line.split("|", maxsplit=3)[-1].strip()

    carbon_metrics: Dict[str, Optional[float]] = {}
    carbon_level_label = "n/a"
    carbon_change_vs_median_pct: Optional[float] = None
    carbon_change_vs_baseline_pct: Optional[float] = None
    carbon_ci_width_rel: Optional[float] = None
    carbon_bands = derive_threshold_bands([])
    carbon_findings: List[Dict[str, str]] = []
    carbon_suggestions: List[str] = []
    carbon_chart_findings: List[Any] = []
    carbon_note_unit = "tCO2e"
    carbon_hist_series: pd.Series = pd.Series(dtype=float)
    min_baseline_denominator = 1.0
    if carbon_engine is not None:
        try:
            min_baseline_denominator = float(carbon_engine.sanity_config.get("min_baseline_denominator_tco2e", 1.0))
        except Exception:
            min_baseline_denominator = 1.0

    carbon_result_state = result.result_state if isinstance(result, CarbonResult) else ""
    carbon_is_computed = isinstance(result, CarbonResult) and carbon_result_state in {
        CARBON_STATE_COMPUTED,
        CARBON_STATE_COMPUTED_ZERO,
    }
    carbon_is_unavailable = isinstance(result, CarbonResult) and not carbon_is_computed
    carbon_state_message = ""

    if isinstance(result, CarbonResult):
        if carbon_is_computed:
            carbon_metrics = compute_emissions_metrics(result.table, result.boundary)
            current_total = float(carbon_metrics.get("total_tco2e") or 0.0)
            scaled_current = scale_tco2e(current_total)
            carbon_note_unit = scaled_current.unit
            metric_col = "wtw_co2e_t" if result.boundary in {"WTW", "TTW_WTW"} else "ttw_co2e_t"
            if metric_col not in (result.table.columns if result.table is not None else []):
                metric_col = "co2_t"
            if carbon_engine is not None and not carbon_engine.daily_port.empty and metric_col in carbon_engine.daily_port.columns:
                hist = carbon_engine.daily_port.copy()
                if result.table is not None and "port_key" in result.table.columns and result.table["port_key"].notna().any():
                    ports = sorted(set(result.table["port_key"].dropna().astype(str)))
                    hist = hist[hist["port_key"].astype(str).isin(ports)]
                carbon_hist_series = pd.to_numeric(hist[metric_col], errors="coerce").dropna()
            elif result.table is not None and metric_col in result.table.columns:
                carbon_hist_series = pd.to_numeric(result.table[metric_col], errors="coerce").dropna()

            carbon_bands = derive_threshold_bands(
                values=carbon_hist_series.tolist(),
                percentiles=threshold_percentiles,
            )
            carbon_level_label = classify_level(current_total, carbon_bands)

            if len(carbon_hist_series) > 0:
                hist_median = float(carbon_hist_series.median())
                hist_mean = float(carbon_hist_series.mean())
                carbon_change_vs_median_pct = safe_percent_delta(
                    current_value=current_total,
                    baseline_value=hist_median,
                    min_denominator=min_baseline_denominator,
                )
                carbon_change_vs_baseline_pct = safe_percent_delta(
                    current_value=current_total,
                    baseline_value=hist_mean,
                    min_denominator=min_baseline_denominator,
                )

            first_metric = result.uncertainty_interval.get("CO2e") or result.uncertainty_interval.get("CO2")
            if first_metric:
                point = float(first_metric.get("point", 0.0))
                lower = float(first_metric.get("lower", 0.0))
                upper = float(first_metric.get("upper", 0.0))
                if point > 0:
                    carbon_ci_width_rel = max(0.0, (upper - lower) / point)

            target_note = next((n for n in result.coverage_notes if n.startswith("Coverage window:")), None)
            target_ts: Optional[pd.Timestamp] = None
            if target_note and " to " in target_note:
                try:
                    target_ts = pd.to_datetime(target_note.split(" to ")[-1], errors="coerce", utc=True)
                except Exception:
                    target_ts = None

            carbon_chart_findings = extract_chart_findings(
                chart_df=result.chart if result.chart is not None else pd.DataFrame(),
                target_ts=target_ts,
                max_findings=5,
            )
            carbon_findings = build_emissions_findings(
                current_tco2e=current_total,
                level=carbon_level_label,
                change_vs_median_pct=carbon_change_vs_median_pct,
                source_label=result.source_label,
                ci_width_rel=carbon_ci_width_rel,
                chart_findings=carbon_chart_findings,
            )
            carbon_suggestions = build_reduction_suggestions(
                level=carbon_level_label,
                change_vs_median_pct=carbon_change_vs_median_pct,
                ci_width_rel=carbon_ci_width_rel,
                source_label=result.source_label,
            )
            if carbon_change_vs_baseline_pct is None or carbon_change_vs_median_pct is None:
                carbon_findings.append(
                    {
                        "type": "inferred",
                        "text": "Baseline denominator is too small for meaningful percentage comparison in this scope.",
                    }
                )
        else:
            state_reason_map = {
                CARBON_STATE_NOT_COMPUTABLE: "No deterministic carbon inventory matched the requested scope.",
                CARBON_STATE_RETRIEVAL_ONLY: "Traffic evidence was retrieved, but numeric carbon emissions could not be computed reliably.",
                CARBON_STATE_FORECAST_ONLY: "Forecast was requested, but no deterministic carbon forecast model is available for this runtime.",
                CARBON_STATE_UNSUPPORTED: "This carbon request is outside the supported deterministic scope.",
            }
            carbon_state_message = state_reason_map.get(
                carbon_result_state,
                "No deterministic carbon output is available for this response.",
            )
            carbon_findings = [
                {"type": "status", "text": carbon_state_message},
            ]
            if evidence.rows:
                carbon_findings.append(
                    {
                        "type": "status",
                        "text": "Retrieved evidence is traffic-related and not sufficient for numeric carbon accounting.",
                    }
                )
            carbon_suggestions = [
                "Improve carbon data coverage for this scope before using emissions totals operationally.",
                "Add validated fuel/engine/activity factors and call-linked rows for the selected period.",
                "Use retrieved traffic evidence as context only, not as numeric carbon truth.",
            ]

    def _build_recommendation_triggers(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
        triggers: List[str] = []
        if canonical_envelope is not None and canonical_envelope.state not in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
        }:
            return [
                f"Trigger: canonical result state is {canonical_envelope.state.value}; no computed operational conclusion is available."
            ]
        if isinstance(value, CarbonResult):
            if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                triggers.append("Trigger: deterministic carbon inventory is unavailable for this scope.")
                if value.result_state == CARBON_STATE_RETRIEVAL_ONLY:
                    triggers.append("Trigger: only retrieval-based supporting traffic evidence is available.")
                return triggers
            if carbon_metrics.get("total_tco2e") is not None:
                triggers.append(f"Trigger: total emissions={format_tco2e(float(carbon_metrics['total_tco2e']))}.")
            triggers.append(f"Trigger: relative level={carbon_level_label} ({carbon_bands.source_label}).")
            if carbon_change_vs_median_pct is not None:
                triggers.append(f"Trigger: change vs historical median={format_percent(carbon_change_vs_median_pct)}.")
            if carbon_ci_width_rel is not None:
                triggers.append(f"Trigger: uncertainty CI width={format_percent(carbon_ci_width_rel * 100.0)}.")
            triggers.append(f"Trigger: source label={value.source_label}.")
            return triggers

        if isinstance(value, ForecastResult) and value.forecast is not None and not value.forecast.empty:
            pred = float(value.forecast["predicted"].mean())
            upper = float(value.forecast["upper"].mean()) if "upper" in value.forecast.columns else pred
            lower = float(value.forecast["lower"].mean()) if "lower" in value.forecast.columns else pred
            if pred >= 1.8:
                triggers.append("Trigger: forecast congestion index >= 1.80 (high-pressure band).")
            elif pred >= 1.3:
                triggers.append("Trigger: forecast congestion index in 1.30-1.79 (elevated band).")
            else:
                triggers.append("Trigger: forecast congestion index < 1.30 (normal-to-low band).")
            triggers.append(f"Trigger: uncertainty interval {lower:.2f}-{upper:.2f} used to size staffing buffer.")
            return triggers

        answer_text = value.answer.lower()
        if "jump" in answer_text or "anomaly" in answer_text:
            if value.table is not None and not value.table.empty:
                max_dist = (
                    float(pd.to_numeric(value.table.get("distance_km"), errors="coerce").max())
                    if "distance_km" in value.table.columns
                    else None
                )
                triggers.append("Trigger: the AIS jump screening rule matched at least one event.")
                if max_dist is not None and not pd.isna(max_dist):
                    triggers.append(f"Trigger: max detected displacement {max_dist:.2f} km in a short interval.")
            else:
                triggers.append("Trigger: the AIS jump screening rule found no qualifying events.")
            return triggers

        if value.chart is not None and not value.chart.empty:
            first_col = [c for c in value.chart.columns if c != "date"]
            if first_col:
                metric_col = first_col[0]
                metric_values = pd.to_numeric(value.chart[metric_col], errors="coerce").dropna()
                if not metric_values.empty:
                    triggers.append(
                        f"Trigger: planning recommendation based on observed {metric_col} range "
                        f"{metric_values.min():.2f}-{metric_values.max():.2f}."
                    )
        if not triggers:
            triggers.append("Trigger: recommendation generated from filtered KPI summary for the selected window.")
        return triggers

    def _build_method_steps(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
        if canonical_view is not None:
            return list(canonical_view.method_steps)
        steps: List[str] = []
        if isinstance(value, CarbonResult):
            steps.append(f"Result state: {value.result_state}.")
            if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                steps.append("Deterministic carbon computation: unavailable for this scope.")
                if value.result_state == CARBON_STATE_RETRIEVAL_ONLY:
                    steps.append("Retrieved evidence is traffic-only context and not numeric carbon source-of-truth.")
                elif value.result_state == CARBON_STATE_FORECAST_ONLY:
                    steps.append("Forecast-only carbon query detected; no deterministic carbon forecast model configured.")
                for note in value.coverage_notes[:4]:
                    steps.append(note)
                return steps
            steps.append("Applied deterministic AIS + port-call mode segmentation (transit/manoeuvring/berth/anchorage).")
            steps.append(f"Boundary: {value.boundary}; Pollutants: {', '.join(value.pollutants)}.")
            steps.append(f"Computed values: {value.source_label}.")
            steps.append("Forecast values: not used unless an explicit forecast request is made.")
            steps.append("Narrative findings are kept separate from the numeric result.")
            steps.append("Retrieved evidence: optional supporting rows only, not numeric source-of-truth.")
            steps.append(f"Factor params version: {value.params_version}.")
            for note in value.coverage_notes[:6]:
                steps.append(note)
            return steps

        if isinstance(value, ForecastResult):
            steps.append("Applied active filters (port/date/vessel-type) to the congestion history for this query.")
            for note in value.coverage_notes:
                if (
                    note.startswith("Coverage window:")
                    or note.startswith("Rows used:")
                    or note.startswith("Target date:")
                    or note.startswith("Forecast target weekday:")
                    or note.startswith("Analog dates used:")
                    or note.startswith("Analog values used:")
                    or note.startswith("Resolved scope correction:")
                    or note.startswith("Meaning:")
                ):
                    steps.append(note)
            method = next((n for n in value.coverage_notes if n.startswith("Method:")), None)
            if method:
                steps.append(method)
            steps.append("Computed the requested value and available lower-upper interval.")
        else:
            steps.append("Applied active filters (port/date/vessel-type/anomaly) to KPI tables.")
            for note in value.coverage_notes:
                if (
                    note.startswith("Coverage window:")
                    or note.startswith("Rows used:")
                    or note.startswith("Data sources used:")
                    or note.startswith("Resolved scope correction:")
                ):
                    steps.append(note)
            if value.table is not None and not value.table.empty:
                steps.append(f"Aggregated filtered rows into {len(value.table):,} output row(s) using deterministic pandas operations.")
            else:
                steps.append("Computed deterministic metric directly from the filtered subset.")

        assumptions = [c for c in value.caveats if not c.lower().startswith("confidence:")]
        for assumption in assumptions[:2]:
            steps.append(f"Assumption: {assumption}")

        deduped: List[str] = []
        for step in steps:
            if step and step not in deduped:
                deduped.append(step)
        return deduped

    def _build_port_actions(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
        actions: List[str] = []
        if canonical_envelope is not None and canonical_envelope.state not in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
        }:
            if canonical_envelope.state == AnswerState.CLARIFICATION_REQUIRED:
                return ["Provide the missing scope requested in the answer, then run the analysis again."]
            if canonical_envelope.state == AnswerState.NO_CURRENT_DATA:
                return ["Choose a historical date range within the validated dataset coverage."]
            if canonical_envelope.state == AnswerState.UNSUPPORTED:
                return ["Reformulate the request using an operation listed in Eagle Eye's supported capabilities."]
            return ["Adjust the query scope or use a supported operation before taking an operational action."]
        if isinstance(value, CarbonResult):
            if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                return [
                    "Improve carbon data coverage for the selected scope before interpreting emissions numerically.",
                    "Add validated vessel fuel/engine/activity factors for periods with missing deterministic carbon rows.",
                    "Use retrieved traffic evidence as context only until deterministic carbon inventory is available.",
                ]
            if carbon_suggestions:
                return carbon_suggestions
            return [
                "Use staggered arrival windows to reduce peak waiting and anchorage emissions.",
                "Use shore-power and idle-engine reduction where berth dwell is long.",
                "Re-check uncertainty drivers before committing to high-impact interventions.",
            ]

        if isinstance(value, ForecastResult) and value.forecast is not None and not value.forecast.empty:
            pred = float(value.forecast["predicted"].mean())
            upper = float(value.forecast["upper"].mean()) if "upper" in value.forecast.columns else pred
            lower = float(value.forecast["lower"].mean()) if "lower" in value.forecast.columns else pred
            spread = max(0.0, upper - pred)

            if pred >= 1.8:
                actions.append("Activate high-traffic playbook: reserve extra berth windows and pre-book pilot/tug shifts.")
                actions.append("Advance-notify terminal and gate teams to smooth truck and yard peaks.")
            elif pred >= 1.3:
                actions.append("Pre-allocate buffer berth slots and increase watchstanding in VTS for the target window.")
                actions.append("Coordinate with agents to stagger ETAs for vessels with flexible arrival windows.")
            else:
                actions.append("Run the normal berth plan while keeping one reserve slot for late-arrival clustering.")

            actions.append(
                f"Use predicted range {lower:.2f}-{upper:.2f} to set staffing floors/ceilings instead of a single-point plan."
            )
            if spread >= 0.6:
                actions.append("Maintain operational contingency: uncertainty is wide, so add tug/pilot standby margin.")
            actions.append("Use retrieved analog evidence rows to brief operations with concrete historical precedents.")
            return actions

        answer_text = value.answer.lower()
        if "jump" in answer_text or "anomaly" in answer_text:
            actions.append("Open AIS integrity checks for listed MMSI and validate with external tracking feeds.")
            actions.append("Flag suspicious tracks for VTS review before acting on route deviations.")
            actions.append("Prioritize vessels with repeated jump flags for manual watchlist monitoring.")
        else:
            actions.append("Use the daily/weekly pattern in the chart to plan shift staffing and pilot windows.")
            actions.append("Re-run this query with tighter vessel-type filters for targeted operational planning.")
            actions.append("Apply port-vs-port comparisons to rebalance pilot/tug resources across nearby ports.")
        return actions

    def _build_evidence_backed_answer(
        value: Union[AnalyticsResult, ForecastResult, CarbonResult],
        bundle: EvidenceBundle,
    ) -> Optional[str]:
        if canonical_envelope is not None:
            return None
        if value.status == "ok" or not bundle.rows:
            return None
        df = pd.DataFrame(bundle.rows)
        if df.empty:
            return None
        port_text = "unknown port"
        if "port" in df.columns and df["port"].notna().any():
            top_port = (
                df["port"].dropna().astype(str).value_counts().index.tolist()[:1]
            )
            if top_port:
                port_text = top_port[0]
        date_span = ""
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            if not ts.empty:
                date_span = f" between {ts.min().strftime('%Y-%m-%d')} and {ts.max().strftime('%Y-%m-%d')}"
        return (
            f"Direct KPI aggregation returned no exact match, but vector retrieval found {len(df)} relevant records "
            f"for {port_text}{date_span}. This is evidence-backed retrieval output, not a deterministic aggregate."
        )

    def _to_naive_datetime(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert(None)

    def _metric_label(column_name: str) -> str:
        return str(column_name).replace("_", " ").strip().title()

    def _value_format(series: pd.Series) -> str:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return ".2f"
        rounded = numeric.round()
        if (numeric - rounded).abs().max() < 1e-9:
            return ".0f"
        if numeric.abs().max() >= 100:
            return ".1f"
        return ".2f"

    def _apply_chart_theme(chart: Any, height: int) -> Any:
        if alt is None:
            return chart
        return (
            chart.properties(height=height)
            .configure(background="transparent")
            .configure_view(strokeOpacity=0)
            .configure_axis(
                domain=False,
                tickColor="#1d344b",
                tickSize=3,
                grid=True,
                gridColor="#183147",
                gridOpacity=0.75,
                labelColor="#dcecff",
                labelFontSize=11,
                labelFont="Avenir Next, Helvetica Neue, sans-serif",
                titleColor="#8fe8ff",
                titleFontSize=12,
                titleFont="Avenir Next, Helvetica Neue, sans-serif",
                titleFontWeight="bold",
            )
            .configure_axisX(labelPadding=10)
            .configure_axisY(labelPadding=8)
            .configure_legend(
                orient="top",
                direction="horizontal",
                labelColor="#dcecff",
                titleColor="#8fe8ff",
                symbolType="stroke",
            )
        )

    def _render_time_series_chart(
        plot_df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        y_title: str,
        height: int,
        compact: bool,
        color: str = "#74d7ff",
        fill: str = "#2b8cff",
        tooltip_title: Optional[str] = None,
        annotate_peak: bool = True,
    ) -> None:
        if alt is None:
            fallback = plot_df[[x_col, y_col]].copy().set_index(x_col)
            _safe_line_chart(fallback, width="stretch")
            return

        safe_df = plot_df.dropna(subset=[x_col, y_col]).copy()
        if safe_df.empty:
            st.info("No chartable series for this response.")
            return

        value_fmt = _value_format(safe_df[y_col])
        tick_count = 5 if compact else 7
        base = alt.Chart(safe_df)
        area = base.mark_area(
            color=fill,
            opacity=0.14,
            interpolate="monotone",
            line=False,
        ).encode(
            x=alt.X(
                f"{x_col}:T",
                title="Date",
                axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=tick_count, labelFlush=False),
            ),
            y=alt.Y(f"{y_col}:Q", title=y_title),
        )
        line = base.mark_line(
            color=color,
            strokeWidth=3,
            interpolate="monotone",
            strokeCap="round",
            strokeJoin="round",
        ).encode(
            x=alt.X(
                f"{x_col}:T",
                title="Date",
                axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=tick_count, labelFlush=False),
            ),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title="Date"),
                alt.Tooltip(f"{y_col}:Q", title=tooltip_title or y_title, format=value_fmt),
            ],
        )
        points = base.mark_point(
            filled=True,
            fill=color,
            color="#e7faff",
            strokeWidth=1.5,
            size=58 if compact else 76,
        ).encode(
            x=f"{x_col}:T",
            y=f"{y_col}:Q",
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title="Date"),
                alt.Tooltip(f"{y_col}:Q", title=tooltip_title or y_title, format=value_fmt),
            ],
        )
        chart = area + line + points

        if annotate_peak and not compact and len(safe_df) > 2:
            peak_idx = pd.to_numeric(safe_df[y_col], errors="coerce").idxmax()
            peak_df = safe_df.loc[[peak_idx], [x_col, y_col]].copy()
            peak_df["label"] = "Peak"
            peak_point = alt.Chart(peak_df).mark_point(
                filled=True,
                fill="#8fe8ff",
                color="#ffffff",
                strokeWidth=2,
                size=150,
            ).encode(
                x=f"{x_col}:T",
                y=f"{y_col}:Q",
                tooltip=[
                    alt.Tooltip(f"{x_col}:T", title="Peak date"),
                    alt.Tooltip(f"{y_col}:Q", title=tooltip_title or y_title, format=value_fmt),
                ],
            )
            peak_label = alt.Chart(peak_df).mark_text(
                align="left",
                dx=10,
                dy=-10,
                color="#8fe8ff",
                fontSize=11,
                fontWeight="bold",
            ).encode(
                x=f"{x_col}:T",
                y=f"{y_col}:Q",
                text="label:N",
            )
            chart = chart + peak_point + peak_label

        st.altair_chart(_apply_chart_theme(chart, height), width="stretch")

    def _render_horizontal_bar_chart(
        plot_df: pd.DataFrame,
        *,
        category_col: str,
        value_col: str,
        x_title: str,
        height: int,
        compact: bool,
        base_color: str = "#6fcfff",
        highlight_color: str = "#8fe8ff",
        tooltip_title: Optional[str] = None,
    ) -> None:
        numeric = pd.to_numeric(plot_df[value_col], errors="coerce")
        safe_df = plot_df.copy()
        safe_df[value_col] = numeric
        safe_df = safe_df.dropna(subset=[value_col]).copy()
        if safe_df.empty:
            st.info("No chartable series for this response.")
            return

        safe_df[category_col] = safe_df[category_col].astype(str)
        safe_df = safe_df.sort_values(value_col, ascending=False).reset_index(drop=True)
        safe_df["accent"] = safe_df[value_col] == safe_df[value_col].max()
        label_order = safe_df[category_col].tolist()
        value_fmt = _value_format(safe_df[value_col])

        if alt is None:
            fallback = safe_df.set_index(category_col)[[value_col]]
            _safe_bar_chart(fallback, width="stretch")
            return

        axis_limit = 110 if compact else 180
        base = alt.Chart(safe_df)
        bars = base.mark_bar(cornerRadius=8, size=28 if compact else 24).encode(
            y=alt.Y(
                f"{category_col}:N",
                title=None,
                sort=label_order,
                axis=alt.Axis(labelLimit=axis_limit),
            ),
            x=alt.X(
                f"{value_col}:Q",
                title=x_title,
                axis=alt.Axis(tickCount=5, format=value_fmt),
            ),
            color=alt.condition(
                "datum.accent",
                alt.value(highlight_color),
                alt.value(base_color),
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title="Category"),
                alt.Tooltip(f"{value_col}:Q", title=tooltip_title or x_title, format=value_fmt),
            ],
        )
        labels = base.mark_text(
            align="left",
            baseline="middle",
            dx=8,
            color="#eef7ff",
            fontSize=11,
            fontWeight="bold",
        ).encode(
            y=alt.Y(f"{category_col}:N", sort=label_order),
            x=alt.X(f"{value_col}:Q"),
            text=alt.Text(f"{value_col}:Q", format=value_fmt),
        )
        st.altair_chart(_apply_chart_theme(bars + labels, height), width="stretch")

    def _render_chart(
        value: Union[AnalyticsResult, ForecastResult, CarbonResult],
        *,
        compact: bool = False,
        show_title: bool = True,
    ) -> None:
        if canonical_envelope is not None:
            _render_canonical_visualizations(
                canonical_envelope,
                compact=compact,
                show_title=show_title,
            )
            return
        if show_title:
            _section_title("Visual")

        series_height = 210 if compact else 340
        history_height = 190 if compact else 320
        band_height = 180 if compact else 260

        if isinstance(value, CarbonResult):
            if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                st.info("Relative emissions chart is unavailable because this response has no deterministic carbon computation.")
                return
            if value.chart is None or value.chart.empty:
                st.info("No chartable carbon series for this response.")
                return
            chart_df = value.chart.copy()
            st.caption(emissions_measurement_note("tCO2e"))
            if isinstance(chart_df.index, pd.DatetimeIndex):
                plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "x"})
                plot_df["x"] = _to_naive_datetime(plot_df["x"]).dt.floor("D")
                value_col = [c for c in plot_df.columns if c != "x"][0]
                display_col = value_col.replace("_", " ").upper()
                if alt is not None:
                    safe_df = plot_df.dropna(subset=["x", value_col]).copy()
                    value_fmt = _value_format(safe_df[value_col])
                    tick_count = 5 if compact else 7
                    base = alt.Chart(safe_df)
                    area = base.mark_area(
                        color="#1aa7a3",
                        opacity=0.14,
                        interpolate="monotone",
                        line=False,
                    ).encode(
                        x=alt.X(
                            "x:T",
                            title="Date",
                            axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=tick_count, labelFlush=False),
                        ),
                        y=alt.Y(f"{value_col}:Q", title=f"{display_col} (tCO2e)"),
                    )
                    line = base.mark_line(
                        color="#35d39a",
                        strokeWidth=3,
                        interpolate="monotone",
                        strokeCap="round",
                        strokeJoin="round",
                    ).encode(
                        x=alt.X(
                            "x:T",
                            title="Date",
                            axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=tick_count, labelFlush=False),
                        ),
                        y=alt.Y(f"{value_col}:Q", title=f"{display_col} (tCO2e)"),
                        tooltip=[
                            alt.Tooltip("x:T", title="Date"),
                            alt.Tooltip(f"{value_col}:Q", title=f"{display_col} (tCO2e)", format=value_fmt),
                        ],
                    )
                    points = base.mark_point(
                        filled=True,
                        fill="#35d39a",
                        color="#e7fff8",
                        strokeWidth=1.5,
                        size=58 if compact else 76,
                    ).encode(
                        x="x:T",
                        y=f"{value_col}:Q",
                        tooltip=[
                            alt.Tooltip("x:T", title="Date"),
                            alt.Tooltip(f"{value_col}:Q", title=f"{display_col} (tCO2e)", format=value_fmt),
                        ],
                    )
                    chart = area + line + points

                    if not compact and len(safe_df) > 2:
                        peak_idx = pd.to_numeric(safe_df[value_col], errors="coerce").idxmax()
                        peak_df = safe_df.loc[[peak_idx], ["x", value_col]].copy()
                        peak_df["label"] = "Peak"
                        peak_point = alt.Chart(peak_df).mark_point(
                            filled=True,
                            fill="#8ff7d0",
                            color="#ffffff",
                            strokeWidth=2,
                            size=150,
                        ).encode(
                            x="x:T",
                            y=f"{value_col}:Q",
                            tooltip=[
                                alt.Tooltip("x:T", title="Peak date"),
                                alt.Tooltip(f"{value_col}:Q", title=f"{display_col} (tCO2e)", format=value_fmt),
                            ],
                        )
                        peak_label = alt.Chart(peak_df).mark_text(
                            align="left",
                            dx=10,
                            dy=-10,
                            color="#8ff7d0",
                            fontSize=11,
                            fontWeight="bold",
                        ).encode(x="x:T", y=f"{value_col}:Q", text="label:N")
                        chart = chart + peak_point + peak_label

                    if carbon_chart_findings:
                        ann = pd.DataFrame(
                            [
                                {
                                    "x": pd.Timestamp(item.timestamp).tz_convert(None)
                                    if pd.Timestamp(item.timestamp).tzinfo
                                    else pd.Timestamp(item.timestamp),
                                    "y": float(item.value),
                                    "finding": item.finding,
                                }
                                for item in carbon_chart_findings[:5]
                            ]
                        )
                        finding_points = alt.Chart(ann).mark_point(
                            color="#ff8f70",
                            size=120,
                            filled=True,
                        ).encode(
                            x="x:T",
                            y="y:Q",
                            tooltip=["finding:N", alt.Tooltip("y:Q", title=f"{display_col} (tCO2e)", format=".3f")],
                        )
                        finding_labels = (
                            alt.Chart(ann.head(3))
                            .mark_text(align="left", dx=10, dy=-10, color="#ffd1c3", fontSize=11)
                            .encode(x="x:T", y="y:Q", text="finding:N")
                        )
                        chart = chart + finding_points + finding_labels
                    st.altair_chart(_apply_chart_theme(chart, series_height), width="stretch")
                else:
                    _safe_line_chart(chart_df, width="stretch")
            else:
                plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "bucket"})
                value_col = [c for c in plot_df.columns if c != "bucket"][0]
                display_col = value_col.replace("_", " ").upper()
                if alt is not None:
                    chart = (
                        alt.Chart(
                            plot_df.assign(
                                highlight=lambda d: pd.to_numeric(d[value_col], errors="coerce")
                                == pd.to_numeric(d[value_col], errors="coerce").max()
                            )
                        )
                        .mark_bar(cornerRadius=8, size=28 if compact else 24)
                        .encode(
                            y=alt.Y("bucket:N", title=None, sort="-x"),
                            x=alt.X(f"{value_col}:Q", title=f"{display_col} (tCO2e)"),
                            color=alt.condition("datum.highlight", alt.value("#4de0a7"), alt.value("#2cc18d")),
                            tooltip=[
                                alt.Tooltip("bucket:N", title="Bucket"),
                                alt.Tooltip(f"{value_col}:Q", title=f"{display_col} (tCO2e)", format=".3f"),
                            ],
                        )
                    )
                    label_layer = (
                        alt.Chart(plot_df)
                        .mark_text(align="left", baseline="middle", dx=8, color="#eef7ff", fontWeight="bold")
                        .encode(
                            y=alt.Y("bucket:N", sort="-x"),
                            x=alt.X(f"{value_col}:Q"),
                            text=alt.Text(f"{value_col}:Q", format=_value_format(plot_df[value_col])),
                        )
                    )
                    numeric_vals = pd.to_numeric(plot_df[value_col], errors="coerce")
                    ann_rows: List[Dict[str, Any]] = []
                    if numeric_vals.notna().any():
                        max_idx = int(numeric_vals.idxmax())
                        min_idx = int(numeric_vals.idxmin())
                        ann_rows.append(
                            {
                                "bucket": str(plot_df.loc[max_idx, "bucket"]),
                                "value": float(numeric_vals.loc[max_idx]),
                                "finding": "Finding: Highest emissions in this window.",
                            }
                        )
                        if min_idx != max_idx:
                            ann_rows.append(
                                {
                                    "bucket": str(plot_df.loc[min_idx, "bucket"]),
                                    "value": float(numeric_vals.loc[min_idx]),
                                    "finding": "Finding: Lowest emissions in this window.",
                                }
                            )
                    if ann_rows:
                        ann_df = pd.DataFrame(ann_rows).head(3)
                        points = alt.Chart(ann_df).mark_point(color="#ff8f70", size=120, filled=True).encode(
                            y=alt.Y("bucket:N", title=None, sort="-x"),
                            x="value:Q",
                            tooltip=["finding:N", alt.Tooltip("value:Q", title=f"{display_col} (tCO2e)", format=".3f")],
                        )
                        labels = (
                            alt.Chart(ann_df)
                            .mark_text(align="left", dx=10, dy=-10, color="#ffd1c3")
                            .encode(y=alt.Y("bucket:N", sort="-x"), x="value:Q", text="finding:N")
                        )
                        st.altair_chart(_apply_chart_theme(chart + label_layer + points + labels, series_height), width="stretch")
                    else:
                        st.altair_chart(_apply_chart_theme(chart + label_layer, series_height), width="stretch")
                else:
                    _safe_dataframe(chart_df, width="stretch", hide_index=True)
            st.caption("Tooltip values and axis are unit-labelled in tCO2e for auditability.")
            return

        if isinstance(value, ForecastResult):
            hist = pd.DataFrame()
            if value.history is not None and not value.history.empty and {"date", "actual"}.issubset(value.history.columns):
                hist = value.history[["date", "actual"]].copy()
                hist["date"] = _to_naive_datetime(hist["date"]).dt.floor("D")
                hist = hist.dropna(subset=["date"]).sort_values("date")

            fc = pd.DataFrame()
            if value.forecast is not None and not value.forecast.empty and {"date", "predicted"}.issubset(value.forecast.columns):
                cols = [c for c in ("date", "predicted", "lower", "upper") if c in value.forecast.columns]
                fc = value.forecast[cols].copy()
                fc["date"] = _to_naive_datetime(fc["date"]).dt.floor("D")
                fc = fc.dropna(subset=["date"]).sort_values("date")

            if hist.empty and fc.empty:
                st.info("No chartable series for this response.")
                return

            gap_days = 0
            if not hist.empty and not fc.empty:
                gap_days = int((fc["date"].min() - hist["date"].max()).days)

            if gap_days > 60:
                st.caption("Recent historical series used for baseline.")
                hist_tail = hist.tail(90).copy()
                if alt is not None and not hist_tail.empty:
                    _render_time_series_chart(
                        hist_tail,
                        x_col="date",
                        y_col="actual",
                        y_title="Observed value",
                        height=history_height,
                        compact=compact,
                        color="#74d7ff",
                        fill="#2f77c7",
                        tooltip_title="Observed value",
                        annotate_peak=not compact,
                    )
                elif not hist_tail.empty:
                    _safe_line_chart(hist_tail.set_index("date")[["actual"]], width="stretch")

                if not fc.empty:
                    st.caption("Prediction interval for requested target date.")
                    fc_tail = fc.tail(1).copy()
                    if alt is not None:
                        band = (
                            alt.Chart(fc_tail)
                            .mark_rule(color="#ffb55e", strokeWidth=5)
                            .encode(
                                x=alt.X("date:T", title="Target date", axis=alt.Axis(format="%b %d", labelAngle=0)),
                                y=alt.Y("lower:Q", title="Forecast"),
                                y2="upper:Q",
                                tooltip=[
                                    "date:T",
                                    alt.Tooltip("predicted:Q", format=".2f"),
                                    alt.Tooltip("lower:Q", format=".2f"),
                                    alt.Tooltip("upper:Q", format=".2f"),
                                ],
                            )
                        )
                        point = alt.Chart(fc_tail).mark_point(color="#74d7ff", size=150, filled=True).encode(
                            x="date:T",
                            y="predicted:Q",
                        )
                        label = alt.Chart(fc_tail).mark_text(
                            align="left",
                            dx=10,
                            dy=-10,
                            color="#ffcf90",
                            fontWeight="bold",
                        ).encode(x="date:T", y="predicted:Q", text=alt.Text("predicted:Q", format=".2f"))
                        st.altair_chart(_apply_chart_theme(band + point + label, band_height), width="stretch")
                    else:
                        _safe_dataframe(fc_tail, width="stretch", hide_index=True)
                return

            hist_tail = hist.tail(120).copy()
            frames: List[pd.DataFrame] = []
            if not hist_tail.empty:
                frames.append(hist_tail.assign(series="actual", value=hist_tail["actual"])[["date", "series", "value"]])
            if not fc.empty:
                frames.append(fc.assign(series="predicted", value=fc["predicted"])[["date", "series", "value"]])
            long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            if alt is not None and not long_df.empty:
                line = (
                    alt.Chart(long_df)
                    .mark_line(point=True, interpolate="monotone", strokeWidth=3)
                    .encode(
                        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=6)),
                        y=alt.Y("value:Q", title="Value"),
                        color=alt.Color(
                            "series:N",
                            scale=alt.Scale(domain=["actual", "predicted"], range=["#74d7ff", "#ffb55e"]),
                            legend=alt.Legend(title=None),
                        ),
                        tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".2f")],
                    )
                )
                chart = line
                if not fc.empty and {"lower", "upper"}.issubset(fc.columns):
                    interval = (
                        alt.Chart(fc)
                        .mark_area(color="#ffb55e", opacity=0.12)
                        .encode(x="date:T", y="lower:Q", y2="upper:Q")
                    )
                    chart = interval + chart
                st.altair_chart(_apply_chart_theme(chart, series_height), width="stretch")
            elif not long_df.empty:
                wide = long_df.pivot_table(index="date", columns="series", values="value", aggfunc="mean").sort_index()
                _safe_line_chart(wide, width="stretch")
            return

        if value.chart is None or value.chart.empty:
            st.info("No chartable series for this response.")
            return

        chart_df = value.chart.copy()
        if isinstance(chart_df.index, pd.DatetimeIndex):
            plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "x"})
            plot_df["x"] = _to_naive_datetime(plot_df["x"]).dt.floor("D")
            value_col = [c for c in plot_df.columns if c != "x"][0]
            if alt is not None:
                _render_time_series_chart(
                    plot_df,
                    x_col="x",
                    y_col=value_col,
                    y_title=_metric_label(value_col),
                    height=series_height,
                    compact=compact,
                    color="#74d7ff",
                    fill="#2f77c7",
                    tooltip_title=_metric_label(value_col),
                    annotate_peak=True,
                )
            else:
                _safe_line_chart(chart_df, width="stretch")
        else:
            plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "x"})
            value_col = [c for c in plot_df.columns if c != "x"][0]
            if alt is not None:
                _render_horizontal_bar_chart(
                    plot_df,
                    category_col="x",
                    value_col=value_col,
                    x_title=_metric_label(value_col),
                    height=series_height,
                    compact=compact,
                    base_color="#5aa9ff",
                    highlight_color="#8fe8ff",
                    tooltip_title=_metric_label(value_col),
                )
            else:
                _safe_bar_chart(chart_df, width="stretch")

    # Every production Streamlit submission reaches this canonical branch.  It
    # intentionally renders the same public result hierarchy as the React
    # workspace and leaves method classifications in the envelope trace.
    if canonical_envelope is not None:
        view = canonical_presentation(canonical_envelope)
        source_label, source_detail = _canonical_source_metadata(
            canonical_envelope,
            source_label=view.source_label,
            source_detail=view.source_detail,
        )
        successful = canonical_envelope.state in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
            AnswerState.RETRIEVED,
            AnswerState.GENERAL,
        }

        if question:
            _meta_card("Question", question, "Submitted analysis request.")
        st.markdown(
            f"""
            <div class="ee-answer-card">
                <div class="ee-answer-label">Answer</div>
                <div class="ee-answer-copy">{escape(view.answer)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if canonical_envelope.visualizations:
            with st.container(border=True):
                _section_title("Charts")
                _render_chart(result, compact=screenshot_mode, show_title=False)
        else:
            st.info("No chart is available for this result.")

        if canonical_envelope.chart_insights:
            with st.expander("Observed in the chart", expanded=True):
                for insight in canonical_envelope.chart_insights:
                    st.markdown(f"- {insight.statement}")

        # Keep answer and charts ahead of metadata for the documented mobile
        # reading order. Successful outputs do not display a result-state badge.
        if successful:
            source_col, freshness_col = st.columns(2, gap="small")
            with source_col:
                _meta_card("Source", source_label, source_detail)
            with freshness_col:
                _meta_card("Freshness", canonical_envelope.freshness.message)
        else:
            outcome_col, source_col, freshness_col = st.columns(3, gap="small")
            with outcome_col:
                _meta_card("Outcome", view.state_label)
            with source_col:
                _meta_card("Source", source_label, source_detail)
            with freshness_col:
                _meta_card("Freshness", canonical_envelope.freshness.message)

        evidence_tab, data_scope_tab = st.tabs(["Evidence", "Data & scope"])
        with evidence_tab:
            if evidence.lines:
                for line in [_to_analyst_evidence_line(item) for item in evidence.lines]:
                    st.markdown(f"- {line}")
            if evidence.rows:
                _safe_dataframe(
                    _evidence_rows_for_display(evidence.rows),
                    width="stretch",
                    hide_index=True,
                )
            if not evidence.lines and not evidence.rows:
                st.info("No supporting evidence rows were attached to this response.")

        with data_scope_tab:
            with st.expander("Applied query scope", expanded=False):
                for line in view.applied_scope:
                    st.markdown(f"- {line}")
            with st.expander("Dataset freshness", expanded=False):
                for line in view.freshness:
                    st.markdown(f"- {line}")
            if canonical_envelope.immutable_facts:
                fact_rows = [fact.model_dump(mode="json") for fact in canonical_envelope.immutable_facts]
                with st.expander("Facts", expanded=False):
                    _safe_dataframe(pd.DataFrame(fact_rows), width="stretch", hide_index=True)
            for dataset in canonical_envelope.datasets:
                frame = dataset_frame(canonical_envelope, dataset.id)
                if frame.empty:
                    continue
                with st.expander(
                    f"Data table · {dataset.id} · {dataset.row_count:,} row(s)",
                    expanded=False,
                ):
                    _safe_dataframe(frame.head(200), width="stretch", hide_index=True)
                    if len(frame) > 200:
                        st.caption(f"Showing 200 of {len(frame):,} rows.")

        if query_service is not None:
            action_col, issue_col = st.columns(2, gap="small")
            with action_col:
                if canonical_envelope.datasets:
                    dataset_ids = [dataset.id for dataset in canonical_envelope.datasets]
                    selected_dataset = st.selectbox(
                        "Export data table",
                        options=dataset_ids,
                        key=f"export_dataset_{canonical_envelope.turn_id}",
                    )
                    export_format = st.selectbox(
                        "Export format",
                        options=["csv", "json"],
                        key=f"export_format_{canonical_envelope.turn_id}",
                    )
                    if st.button(
                        "Create export",
                        key=f"create_export_{canonical_envelope.turn_id}",
                    ):
                        try:
                            exported = query_service.export(
                                ExportRequest(
                                    conversation_id=canonical_envelope.conversation_id,
                                    turn_id=canonical_envelope.turn_id,
                                    dataset_id=selected_dataset,
                                    format=export_format,
                                )
                            )
                            st.success(
                                f"Created {exported.format.upper()} export with "
                                f"{exported.row_count:,} row(s): {exported.path}"
                            )
                        except Exception:
                            st.error("The export could not be created for this result.")
            with issue_col:
                issue_note = st.text_input(
                    "Report issue (optional note)",
                    key=f"issue_note_{canonical_envelope.turn_id}",
                )
                if st.button(
                    "Report issue",
                    key=f"report_issue_{canonical_envelope.turn_id}",
                ):
                    try:
                        accepted = query_service.submit_feedback(
                            FeedbackRequest(
                                prompt=canonical_envelope.question,
                                trace_id=canonical_envelope.trace.trace_id,
                                note=issue_note or None,
                            )
                        )
                        st.success(f"Issue report accepted: {accepted.feedback_id}")
                    except Exception:
                        st.error("The issue report could not be recorded.")
        return

    retrieved_lines = evidence.lines
    computed_lines = _fallback_evidence_from_result(result)
    display_lines = retrieved_lines if show_technical else [_to_analyst_evidence_line(line) for line in retrieved_lines]
    trace = evidence.trace or {}
    if canonical_view is not None:
        source_label, source_detail = _canonical_source_metadata(
            canonical_envelope,
            source_label=canonical_view.source_label,
            source_detail=canonical_view.source_detail,
        )
        confidence_label = canonical_view.confidence_label
        status_label = canonical_view.state_label
        answer_text = canonical_view.answer
    else:
        source_label, source_detail = _derive_answer_source(result, evidence)
        confidence_label = extract_confidence_label(result)
        status_label = (
            result.result_state.replace("_", " ").title()
            if isinstance(result, CarbonResult)
            else str(result.status).replace("_", " ").title()
        )
        answer_text = str(result.answer)
    evidence_backed = _build_evidence_backed_answer(result, evidence)
    forecast_meaning = (
        next((n for n in result.coverage_notes if n.startswith("Meaning:")), None)
        if isinstance(result, ForecastResult)
        else None
    )

    def _render_response_contract_section() -> None:
        st.markdown("1. **Retrieved facts**")
        if display_lines:
            for line in display_lines[:3]:
                st.markdown(f"- {line}")
        else:
            st.markdown("- No retrieved fact rows for this scope.")
        st.markdown("2. **Observed history**")
        if computed_lines:
            for line in computed_lines[:3]:
                st.markdown(f"- {line}")
        else:
            st.markdown("- No deterministic observed-history rows for this scope.")
        st.markdown("3. **Forecast inputs**")
        if isinstance(result, ForecastResult):
            forecast_notes = [
                n
                for n in (result.coverage_notes or [])
                if "forecast" in n.lower() or "analog" in n.lower() or "target date" in n.lower()
            ]
            if forecast_notes:
                for line in forecast_notes[:3]:
                    st.markdown(f"- {line}")
            else:
                st.markdown("- Forecast model inputs were used (historical seasonal analogs).")
        else:
            st.markdown("- Forecast inputs not required for this response path.")
        st.markdown("4. **Analysis result**")
        st.markdown(f"- {answer_text}")
        st.markdown("5. **Scope notes**")
        if result.caveats:
            for line in result.caveats[:3]:
                st.markdown(f"- {line}")
        else:
            st.markdown("- No additional scope notes were surfaced for this response.")
        st.markdown("6. **Available uncertainty interval**")
        if isinstance(result, CarbonResult) and result.uncertainty_interval:
            for key, payload in list(result.uncertainty_interval.items())[:2]:
                st.markdown(
                    f"- {key}: {float(payload.get('lower', 0.0)):.3f} to {float(payload.get('upper', 0.0)):.3f}"
                )
        st.markdown("7. **Evidence / provenance IDs**")
        if isinstance(result, CarbonResult):
            ids = list(result.evidence_ids or [])[:5]
            if ids:
                st.markdown(f"- carbon_evidence_ids: {', '.join(ids)}")
            elif evidence.rows:
                st.markdown(f"- retrieved_vector_rows: {len(evidence.rows)}")
            else:
                st.markdown("- No evidence IDs or retrieved provenance rows for this scope.")
        elif evidence.rows:
            st.markdown(f"- retrieved_vector_rows: {len(evidence.rows)}")
        else:
            st.markdown("- No provenance rows available for this response.")

    def _render_snapshot_panel() -> None:
        def _render_answer() -> None:
            if question:
                _meta_card("Question", question, "Submitted analysis request.")
            st.markdown(
                f"""
                <div class="ee-answer-card">
                    <div class="ee-answer-label">Answer</div>
                    <div class="ee-answer-copy">{escape(answer_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _render_answer()
        if isinstance(result, CarbonResult):
            if result.result_state in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                st.caption(emissions_measurement_note("tCO2e"))
            else:
                st.caption(
                    "Unit for valid carbon outputs: tCO2e. "
                    "No valid deterministic carbon output is available for this scope."
                )
        if evidence_backed:
            st.info(evidence_backed)
        if forecast_meaning:
            st.info(forecast_meaning.split(":", 1)[1].strip())

        chart_host = st.container(border=True)
        with chart_host:
            if surface_all_canonical_visuals and canonical_envelope is not None:
                _section_title("Charts")
            _render_chart(result, compact=screenshot_mode, show_title=False)

        # The answer and every validated chart intentionally precede metadata so
        # narrow layouts never make a valid visualization appear to be missing.
        meta_col1, meta_col2 = st.columns(2, gap="small")
        with meta_col1:
            _meta_card("Source", source_label, source_detail)
        with meta_col2:
            _meta_card("Evidence", f"{len(display_lines)} supporting row(s)")

    _render_snapshot_panel()
    if screenshot_mode:
        detail_host = st.expander("Detailed record", expanded=False)
    else:
        detail_host = st.container()
    with detail_host:
        evidence_tab, data_scope_tab, recommendations_tab = st.tabs(
            ["Evidence", "Data & scope", "Recommendations"]
        )
    summary_tab = data_scope_tab
    visuals_tab = data_scope_tab
    actions_tab = recommendations_tab
    audit_tab = recommendations_tab

    with summary_tab:
        if canonical_view is not None:
            with st.expander("Applied query scope", expanded=False):
                for line in canonical_view.applied_scope:
                    st.markdown(f"- {line}")
            with st.expander("Dataset freshness", expanded=False):
                for line in canonical_view.freshness:
                    st.markdown(f"- {line}")
        elif result.coverage_notes:
            with st.expander("Coverage notes", expanded=False):
                for line in result.coverage_notes:
                    st.markdown(f"- {line}")
        if show_technical:
            st.caption(
                f"Response path: {status_label} | Computed evidence: {len(computed_lines)} | "
                f"Retrieved evidence: {len(display_lines)}"
            )

    with visuals_tab:
        if result.table is not None and not result.table.empty and not isinstance(result, CarbonResult):
            with st.container(border=True):
                _subsection_title("Data Preview")
                _safe_dataframe(result.table.head(40), width="stretch", hide_index=True)
                st.caption("Preview of the rows behind this response path.")

        if isinstance(result, CarbonResult):
            with st.container(border=True):
                _section_title("Carbon Contract")
                st.write(
                    f"Boundary: `{result.boundary}` | Pollutants: `{', '.join(result.pollutants)}` | "
                    f"Params version: `{result.params_version}` | Result state: `{result.result_state}`"
                )
                st.caption(
                    "Computed values are deterministic inventory outputs; forecast and narrative insights are shown separately."
                )
                if carbon_is_unavailable:
                    st.warning(carbon_state_message or "No deterministic carbon inventory matched the requested scope.")

                c1, c2, c3 = st.columns(3)
                c4, c5, c6 = st.columns(3)
                if carbon_is_computed and carbon_metrics:
                    total_val = float(carbon_metrics.get("total_tco2e") or 0.0)
                    intensity_val = carbon_metrics.get("intensity_kg_per_call")

                    c1.metric("Total emissions", format_tco2e(total_val))
                    c1.caption(emissions_measurement_note(carbon_note_unit))

                    c2.metric(
                        "Emissions intensity",
                        f"{format_kgco2e(float(intensity_val))}/vessel-call" if intensity_val is not None else "n/a",
                    )
                    c2.caption(emissions_measurement_note("kgCO2e/vessel-call"))

                    c3.metric("Forecast emissions", "n/a (not requested)")
                    c3.caption(emissions_measurement_note("tCO2e/forecast-window"))

                    c4.metric("Relative level", carbon_level_label)
                    c4.caption(f"Thresholds are {carbon_bands.source_label}.")

                    c5.metric(
                        "Change vs baseline",
                        format_percent(carbon_change_vs_baseline_pct) if carbon_change_vs_baseline_pct is not None else "n/a",
                        delta=format_percent(carbon_change_vs_baseline_pct) if carbon_change_vs_baseline_pct is not None else None,
                    )
                    c5.caption(
                        "Baseline = historical mean for selected scope."
                        if carbon_change_vs_baseline_pct is not None
                        else "Baseline too small for meaningful percentage comparison."
                    )

                    c6.metric(
                        "Change vs historical median",
                        format_percent(carbon_change_vs_median_pct) if carbon_change_vs_median_pct is not None else "n/a",
                        delta=format_percent(carbon_change_vs_median_pct) if carbon_change_vs_median_pct is not None else None,
                    )
                    c6.caption(
                        "Median is computed from historical dataset values."
                        if carbon_change_vs_median_pct is not None
                        else "Not enough carbon data for a stable median comparison."
                    )

                    _subsection_title("Emissions Level (Relative Scale)")
                    st.caption(
                        "Low/Moderate/High/Very High classification relative to this dataset percentiles (P25/P50/P75)."
                    )
                    if alt is not None:
                        bar_df = build_comparison_bar_table(current_value=total_val, bands=carbon_bands)
                        marker_df = pd.DataFrame([{"x": total_val, "label": f"Current: {format_tco2e(total_val)}"}])

                        bars = (
                            alt.Chart(bar_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("start:Q", title="Emissions (tCO2e)"),
                                x2="end:Q",
                                y=alt.Y("level:N", sort=["Very High", "High", "Moderate", "Low"], title=None),
                                color=alt.Color(
                                    "level:N",
                                    scale=alt.Scale(
                                        domain=["Low", "Moderate", "High", "Very High"],
                                        range=["#22c55e", "#84cc16", "#f59e0b", "#ef4444"],
                                    ),
                                    legend=None,
                                ),
                                tooltip=[
                                    alt.Tooltip("level:N", title="Level"),
                                    alt.Tooltip("start:Q", title="Start (tCO2e)", format=".2f"),
                                    alt.Tooltip("end:Q", title="End (tCO2e)", format=".2f"),
                                ],
                            )
                            .properties(height=220)
                        )
                        marker = alt.Chart(marker_df).mark_rule(color="#f8fafc", strokeWidth=3).encode(x="x:Q")
                        label = (
                            alt.Chart(marker_df)
                            .mark_text(align="left", dy=-8, dx=6, color="#f8fafc")
                            .encode(x="x:Q", y=alt.value(0), text="label:N")
                        )
                        st.altair_chart(bars + marker + label, width="stretch")
                        st.caption(
                            f"Interpretation: current emissions are `{carbon_level_label}` relative to this dataset "
                            f"(P25={carbon_bands.p25:.2f}, P50={carbon_bands.p50:.2f}, P75={carbon_bands.p75:.2f} tCO2e)."
                        )
                    else:
                        st.info(
                            f"Relative level: {carbon_level_label} | P25={carbon_bands.p25:.2f}, "
                            f"P50={carbon_bands.p50:.2f}, P75={carbon_bands.p75:.2f}, current={total_val:.2f} tCO2e"
                        )
                    st.caption("Threshold basis: relative to this dataset (not an external regulatory limit).")
                else:
                    c1.metric("Total emissions", "N/A")
                    c1.caption("Unit for valid carbon outputs: tCO2e.")
                    c2.metric("Emissions intensity", "N/A")
                    c2.caption("Unit for valid carbon outputs: kgCO2e/vessel-call.")
                    c3.metric("Forecast emissions", "N/A")
                    c3.caption("Unit for valid forecast outputs: tCO2e/forecast-window.")
                    c4.metric("Relative level", "Unavailable")
                    c4.caption("Relative emissions level unavailable for this scope.")
                    c5.metric("Change vs baseline", "N/A")
                    c5.caption("Not enough carbon data for comparison.")
                    c6.metric("Change vs historical median", "N/A")
                    c6.caption("Not enough carbon data for comparison.")

                if carbon_is_computed and result.table is not None and not result.table.empty:
                    _subsection_title("Emissions Table")
                    display_table = to_emissions_display_table(result.table)
                    _safe_dataframe(display_table, width="stretch", hide_index=True)
                    st.caption("All emissions columns are standardized and explicitly unit-labelled.")

                if carbon_is_computed and result.uncertainty_interval:
                    rows = []
                    for key, payload in result.uncertainty_interval.items():
                        rows.append(
                            {
                                "metric": key,
                                "point": format_tco2e(float(payload.get("point", 0.0)))
                                if key.upper().startswith("CO2")
                                else f"{float(payload.get('point', 0.0)):.2f} kg",
                                "lower": format_tco2e(float(payload.get("lower", 0.0)))
                                if key.upper().startswith("CO2")
                                else f"{float(payload.get('lower', 0.0)):.2f} kg",
                                "upper": format_tco2e(float(payload.get("upper", 0.0)))
                                if key.upper().startswith("CO2")
                                else f"{float(payload.get('upper', 0.0)):.2f} kg",
                            }
                        )
                    _safe_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    st.caption(emissions_measurement_note("tCO2e"))

                _subsection_title("Findings")
                if carbon_findings:
                    for item in carbon_findings:
                        st.markdown(f"- `{item.get('type', 'deterministic')}` {item.get('text', '')}")
                elif carbon_is_computed:
                    st.info("No strong deterministic findings were available for this carbon scope.")
                else:
                    st.info("No deterministic carbon findings are available for this scope.")

    with evidence_tab:
        with st.container(border=True):
            _section_title("Evidence")
            if isinstance(result, CarbonResult):
                _subsection_title("Deterministic carbon evidence")
                if carbon_is_computed and computed_lines:
                    for line in computed_lines:
                        st.markdown(f"- {line}")
                    st.caption(emissions_measurement_note("tCO2e"))
                else:
                    st.info("No deterministic carbon evidence for this scope.")

                if display_lines:
                    _subsection_title("Retrieved supporting traffic evidence")
                    for line in display_lines:
                        st.markdown(f"- {line}")
                    if carbon_is_unavailable:
                        st.caption(
                            "Supporting traffic evidence is contextual only and not a numeric carbon source-of-truth."
                        )
            else:
                if computed_lines:
                    _subsection_title("Computed evidence used for this answer")
                    for line in computed_lines:
                        st.markdown(f"- {line}")
                if display_lines:
                    _subsection_title("Retrieved supporting evidence")
                    for line in display_lines:
                        st.markdown(f"- {line}")
                if not display_lines and not computed_lines:
                    st.info("No evidence rows were available for this response.")

            if evidence.rows:
                _subsection_title("Retrieved evidence rows (AIS / document / web)")
                _safe_dataframe(
                    _evidence_rows_for_display(evidence.rows),
                    width="stretch",
                    hide_index=True,
                )
                if canonical_envelope is not None and canonical_envelope.state in {
                    AnswerState.COMPUTED,
                    AnswerState.PARTIAL,
                }:
                    st.caption(
                        "These retrieved rows support interpretation and provenance. "
                        "Validated structured datasets remain the numeric authority for the answer."
                    )

            if evidence.rows and not show_technical:
                st.caption(
                    "Detailed vector identifiers and distances are retained in automated audit artifacts."
                )

    method_steps = _build_method_steps(result)
    with actions_tab:
        if method_steps:
            with st.container(border=True):
                _section_title("Recommendation basis")
                for idx, step in enumerate(method_steps, start=1):
                    st.markdown(f"{idx}. {step}")

        with st.container(border=True):
            if isinstance(result, CarbonResult):
                _section_title("How To Reduce Emissions")
                if carbon_suggestions:
                    for action in carbon_suggestions[:5]:
                        st.markdown(f"- {action}")
                else:
                    st.markdown(
                        "- Insufficient strong evidence for targeted actions; maintain baseline operations and monitor."
                    )
                if result.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                    st.caption(
                        "These are data-quality recommendations because deterministic carbon computation is unavailable."
                    )
            else:
                _section_title("Port Operations Recommendations")
                for action in _build_port_actions(result):
                    st.markdown(f"- {action}")

            _subsection_title("Recommendation Triggers")
            for trigger in _build_recommendation_triggers(result):
                st.markdown(f"- {trigger}")

        with st.container(border=True):
            with st.expander("Result details", expanded=False):
                _render_response_contract_section()

    with audit_tab:
        with st.container(border=True):
            _section_title("Retrieval Provenance")
            status = str(trace.get("retrieval_status", "unknown")).upper()
            reason = str(trace.get("reason", "No retrieval status available."))
            st.write(f"Status: `{status}`")
            st.write(reason)
            st.caption(
                f"Response path: {status_label} | Computed evidence rows: {len(computed_lines)} | Retrieved evidence rows: {len(display_lines)}"
            )
            if isinstance(result, CarbonResult):
                sanity = str((result.diagnostics or {}).get("sanity_status", "n/a"))
                st.write(f"Carbon sanity status: `{sanity}`")
                warning_items = list((result.diagnostics or {}).get("warnings") or [])
                if warning_items:
                    for item in warning_items[:5]:
                        st.markdown(f"- {item}")
            if trace:
                st.write(
                    f"Collection: `{trace.get('collection', 'n/a')}` | "
                    f"Backend: `{trace.get('vector_backend', 'n/a')}` | "
                    f"Mode: `{trace.get('mode', 'n/a')}` | "
                    f"Latency: `{trace.get('query_latency_ms', 'n/a')} ms` | "
                    f"Returned: `{trace.get('returned_items', 0)}`"
                )
                where_used = trace.get("where_filter")
                if where_used not in (None, "", {}):
                    st.write(f"Where filter: `{where_used}`")

        if show_technical:
            if intent_result is not None:
                st.markdown("**Intent extraction diagnostics**")
                st.json(
                    {
                        "intent": intent_result.intent,
                        "reason": intent_result.reason,
                        "entities": intent_result.entities,
                        "extraction_diagnostics": (intent_result.entities or {}).get("extraction_diagnostics", {}),
                    }
                )
            if isinstance(result, CarbonResult):
                st.markdown("**Carbon technical audit**")
                st.write(
                    f"params_version=`{result.params_version}` | "
                    f"result_state=`{result.result_state}` | "
                    f"confidence=`{result.confidence_label}` | "
                    f"reason=`{result.confidence_reason}`"
                )
                if result.evidence_ids:
                    st.write("Evidence IDs:", ", ".join(result.evidence_ids[:20]))
                if result.segment_ids:
                    st.write("Segment IDs:", ", ".join(result.segment_ids[:20]))
                if result.export_csv_path or result.export_json_path:
                    st.write(
                        f"Exports: csv=`{result.export_csv_path or 'n/a'}`, "
                        f"json=`{result.export_json_path or 'n/a'}`"
                    )
                diag = dict(result.diagnostics or {})
                if diag:
                    st.markdown("**Carbon sanity diagnostics**")
                    summary_keys = [
                        "raw_rows_before_dedup",
                        "rows_after_dedup",
                        "duplicates_removed_rows",
                        "unique_vessel_calls",
                        "total_duration_hours",
                        "median_duration_hours",
                        "total_tco2e",
                        "mean_tco2e_per_call",
                        "median_tco2e_per_call",
                        "duplicated_call_ids_detected",
                        "sanity_status",
                    ]
                    diag_summary = {k: diag.get(k) for k in summary_keys if k in diag}
                    if diag_summary:
                        st.json(diag_summary)
                    call_trace = diag.get("trace_single_call")
                    if call_trace:
                        st.markdown("**Single-call trace**")
                        st.json(call_trace)
            if evidence.rows:
                trace_df = pd.DataFrame(evidence.rows)
                cols = [
                    c
                    for c in ["vector_id", "chunk_id", "distance", "timestamp", "port", "vessel_type", "mmsi"]
                    if c in trace_df.columns
                ]
                _safe_dataframe(trace_df[cols], width="stretch", hide_index=True)
            else:
                st.info("No vector rows available for this query. Evidence above may come from deterministic KPI computation.")
            with st.expander("Raw retrieval trace JSON", expanded=False):
                st.json(trace)
        else:
            st.caption(
                "Detailed vector and retrieval traces are retained in automated audit artifacts."
            )





def _render_page_overview(
    *,
    kpi: KPIQueryEngine,
    carbon: CarbonQueryEngine,
    retriever_reason: str,
) -> None:
    st.markdown("<div class='ee-workspace-heading'>Coverage summary</div>", unsafe_allow_html=True)
    st.caption("Historical datasets and analytical surfaces currently available to this workspace.")
    with st.expander("Workspace directory", expanded=False):
        st.markdown("- `Analysis Desk`: cross-domain, evidence-grounded maritime analysis.")
        st.markdown("- `Traffic Monitoring`, `Vessel Investigation`, `ETA & Delay`, `Port Pressure`, `Carbon Emissions`: query-category pages.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ports", int(kpi.port_catalog["port_key"].nunique()) if not kpi.port_catalog.empty else 0)
    c2.metric("Arrivals rows", int(len(kpi.arrivals_daily)))
    c3.metric("Carbon daily rows", int(len(carbon.daily_port)) if carbon.daily_port is not None else 0)
    st.caption("Structured datasets provide numeric authority. Retrieved material provides explanatory context.")
    if retriever_reason:
        with st.expander("Evidence service", expanded=False):
            st.caption(retriever_reason)








def _render_page_benchmarking() -> None:
    st.subheader("Benchmarking")
    st.caption("Industry-competitiveness checks: retrieval quality, numeric integrity, state gating, latency, and refusal correctness.")
    with st.expander("How to test benchmarking", expanded=False):
        st.markdown("- Run benchmark suite to regenerate `evaluation/latest/summary.json` and `per_query_results.csv`.")
        st.markdown("- Validate categories: traffic, vessel, ETA/delay, port pressure, carbon deterministic, no-data, unsupported.")
        st.markdown("- Confirm p50/p95 latency and result-state correctness.")
    summary_path = Path("evaluation/latest/summary.json")
    per_query_path = Path("evaluation/latest/per_query_results.csv")
    query_catalog_path = Path("evaluation/latest/query_catalog_v2.json")
    if query_catalog_path.exists():
        st.markdown("**Query catalog (category coverage)**")
        try:
            st.json(json.loads(query_catalog_path.read_text(encoding="utf-8")))
        except Exception as exc:
            st.warning(f"Could not parse query catalog JSON: {exc}")
    if summary_path.exists():
        st.markdown("**Summary**")
        st.json(pd.read_json(summary_path, typ="series").to_dict())
    else:
        st.info("No `evaluation/latest/summary.json` found.")
    if per_query_path.exists():
        st.markdown("**Per-query results**")
        _safe_dataframe(pd.read_csv(per_query_path).head(500), width="stretch", hide_index=True)
    else:
        st.info("No `evaluation/latest/per_query_results.csv` found.")




def _render_page_chat(
    *,
    query_service: QueryService,
    carbon_engine: CarbonQueryEngine,
    top_k_evidence: int,
    show_technical: bool,
    threshold_percentiles: Tuple[float, float, float],
) -> None:
    st.markdown("<div class='ee-workspace-heading'>New analysis</div>", unsafe_allow_html=True)
    st.caption(
        "Submit a maritime question to the canonical analysis service. Structured datasets remain "
        "the numeric authority; matching local evidence is attached for context and provenance."
    )
    with st.expander("Analysis guidance", expanded=False):
        st.markdown("- Enter a natural-language maritime question.")
        st.markdown("- Optionally keep query scope tight with port/date filters in query pages for better precision.")
        st.markdown("- Expected output: direct answer, charts, source, freshness, evidence, and data scope.")
        st.markdown("- Numeric results are produced only by validated analytical paths.")
    chat_history: List[Dict[str, Any]] = list(st.session_state.get("chat_history", []))
    if chat_history:
        with st.expander("Analysis log", expanded=True):
            for turn in chat_history[-8:]:
                st.markdown(f"**You:** {str(turn.get('question', '')).strip()}")
                st.markdown(f"**Eagle Eye:** {str(turn.get('answer', '')).strip()}")

    st.text_input(
        "Analysis request",
        key="chat_message_page",
        placeholder="Describe the maritime question, location, and time period",
    )
    col_send, col_new = st.columns([1, 1])
    chat_send = col_send.button("Analyze", type="primary", key="chat_send_page")
    chat_new = col_new.button("New analysis", key="chat_new_page")
    stored_bundle = st.session_state.get("chat_latest_result_bundle")

    def _render_chat_result(bundle: Dict[str, Any]) -> None:
        envelope = bundle["canonical_envelope"]
        _render_compact_result(
            result=bundle["result"],
            evidence=bundle["evidence"],
            show_technical=show_technical,
            canonical_envelope=envelope,
            intent_result=None,
            carbon_engine=carbon_engine,
            threshold_percentiles=threshold_percentiles,
            question=envelope.question,
            surface_all_canonical_visuals=True,
            query_service=query_service,
        )

    if chat_new:
        st.session_state["chat_history"] = []
        st.session_state["chat_conversation_id"] = ""
        st.session_state.pop("chat_latest_result_bundle", None)
        st.success("Started a new analysis.")
        return
    if not chat_send:
        if stored_bundle:
            _render_chat_result(stored_bundle)
        return

    message = str(st.session_state.get("chat_message_page", "")).strip()
    if not message:
        st.warning("Enter an analysis request first.")
        return
    conversation_id = str(st.session_state.get("chat_conversation_id", "")).strip()
    if not conversation_id:
        conversation_id = f"streamlit_chat_{uuid.uuid4().hex[:16]}"
        st.session_state["chat_conversation_id"] = conversation_id
    canonical = run_canonical_query(
        query_service,
        question=message,
        conversation_id=conversation_id,
        top_k_evidence=top_k_evidence,
        user_filters={},
    )
    envelope = canonical.envelope
    result = canonical.result
    evidence = EvidenceBundle(
        lines=canonical.evidence.lines,
        rows=canonical.evidence.rows,
        trace=canonical.evidence.trace,
    )
    history = list(st.session_state.get("chat_history", []))
    history.append(
        {
            "turn_id": envelope.turn_id,
            "question": envelope.question,
            "answer": envelope.answer,
            "result_state": envelope.state.value,
            "source_type": envelope.mode.value,
            "confidence": envelope.confidence,
            "evidence_lines": canonical.evidence.lines,
        }
    )
    st.session_state["chat_history"] = history[-8:]
    latest_bundle = {
        "canonical_envelope": envelope,
        "result": result,
        "evidence": evidence,
    }
    st.session_state["chat_latest_result_bundle"] = latest_bundle
    _render_chat_result(latest_bundle)


def _query_category_key(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _display_page_label(page: str) -> str:
    """Keep stable internal page values while using analyst-facing labels."""

    return "Analysis Desk" if page == "Chat Assistant" else page


def _render_query_category_intro(category: str) -> None:
    guide = QUERY_CATEGORY_HELP.get(category, {})
    overview = str(guide.get("overview", "")).strip()
    overview_text = (
        overview if overview else "Deterministic maritime analytics for the selected query category."
    )
    st.markdown(
        f"""
        <div class="ee-query-context">
            <div class="ee-query-context-label">Analytical scope</div>
            <div class="ee-query-context-copy">{escape(overview_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_query_category_guide(category: str) -> None:
    """Keep the complete page guide available without blocking the query workflow."""

    guide = QUERY_CATEGORY_HELP.get(category, {})
    overview = str(guide.get("overview", "")).strip()
    overview_text = (
        overview if overview else "Deterministic maritime analytics for the selected query category."
    )
    with st.expander("About this analysis", expanded=False):
        purpose_col, output_col, method_col = st.columns(3, gap="large")
        with purpose_col:
            st.markdown("**Purpose and input**")
            st.markdown(f"- {overview_text}")
            for line in guide.get("test_steps", [])[:2]:
                st.markdown(f"- {line}")
            for line in guide.get("what_to_enter", []):
                st.markdown(f"- {line}")
        with output_col:
            st.markdown("**Expected output**")
            for line in guide.get("expected_output", []):
                st.markdown(f"- {line}")
        with method_col:
            st.markdown("**Method and data**")
            for line in guide.get("calculation", []):
                st.markdown(f"- {line}")
            for line in QUERY_CATEGORY_REQUIRED_DATA.get(category, []):
                st.markdown(f"- {line}")

def _apply_global_app_styles() -> None:
    st.markdown(
        """
<style>
:root {
    --ee-bg-1: #07111d;
    --ee-bg-2: #0b1b2e;
    --ee-bg-3: #10253a;
    --ee-panel: rgba(8, 20, 33, 0.92);
    --ee-panel-soft: rgba(12, 28, 46, 0.92);
    --ee-border: rgba(124, 193, 224, 0.18);
    --ee-border-strong: rgba(135, 231, 255, 0.34);
    --ee-text: #eef7ff;
    --ee-muted: #b7cadf;
    --ee-accent: #89e8ff;
    --ee-accent-2: #4dc7ff;
    --ee-accent-3: #16b9a3;
}

@keyframes eeFadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes eeChipFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-2px); }
}

@keyframes eeSweep {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.stApp {
    color: var(--ee-text);
    background:
        radial-gradient(circle at 14% 12%, rgba(22, 185, 163, 0.09), transparent 20%),
        radial-gradient(circle at 82% 18%, rgba(77, 199, 255, 0.08), transparent 22%),
        linear-gradient(180deg, var(--ee-bg-1) 0%, var(--ee-bg-2) 58%, var(--ee-bg-3) 100%);
}

.block-container {
    max-width: 1520px;
    padding-top: 1.05rem;
    padding-bottom: 2.75rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

h1, h2, h3 {
    letter-spacing: -0.025em;
}

p, li, label, [data-testid="stMarkdownContainer"] {
    line-height: 1.58;
}

div[data-testid="stSidebar"] {
    border-right: 1px solid var(--ee-border);
    background: linear-gradient(180deg, #111722 0%, #191d29 100%);
}

div[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--ee-border);
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(9, 21, 36, 0.96) 0%, rgba(10, 26, 43, 0.96) 100%);
    box-shadow: 0 18px 42px rgba(2, 8, 23, 0.20);
    transition: transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease;
    animation: eeFadeUp 0.35s ease both;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    border-color: var(--ee-border-strong);
    box-shadow: 0 22px 54px rgba(2, 8, 23, 0.24);
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(124, 193, 224, 0.16);
    border-radius: 18px;
    background: rgba(8, 18, 32, 0.64);
}

div[data-testid="stExpander"] details summary p {
    font-weight: 600;
}

div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 999px;
    border: 1px solid rgba(135, 231, 255, 0.14);
    background: rgba(10, 23, 40, 0.58);
    color: var(--ee-muted);
    padding: 0.5rem 1rem;
    height: 2.6rem;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, rgba(22, 185, 163, 0.18), rgba(77, 199, 255, 0.22));
    color: var(--ee-text);
    border-color: rgba(135, 231, 255, 0.34);
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid rgba(135, 231, 255, 0.24);
    font-weight: 680;
    padding: 0.58rem 1rem;
    background: rgba(8, 18, 32, 0.78);
    color: var(--ee-text);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(92deg, #0ea5e9 0%, #2563eb 42%, #14b8a6 100%);
    color: white;
    border-color: rgba(103, 232, 249, 0.42);
}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-testid="stDateInputField"] input {
    border-radius: 15px !important;
    font-size: 0.99rem !important;
    background: rgba(5, 15, 28, 0.76) !important;
    border-color: rgba(135, 231, 255, 0.18) !important;
    color: var(--ee-text) !important;
}

.stTextArea textarea {
    line-height: 1.65 !important;
}

div[data-testid="stMetric"] {
    background: rgba(148, 204, 225, 0.06);
    border: 1px solid rgba(148, 204, 225, 0.20);
    border-radius: 16px;
    padding: 0.55rem 0.7rem;
}

div[data-testid="stMetricLabel"] > div {
    font-size: 0.82rem;
    color: var(--ee-muted);
}

div[data-testid="stMetricValue"] > div {
    font-size: 1.08rem;
}

div[data-testid="stVegaLiteChart"],
div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 14px 34px rgba(2, 8, 23, 0.16);
}

.ee-app-hero,
.ee-page-hero,
.ee-answer-card,
.ee-meta-card {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--ee-border);
    border-radius: 24px;
}

.ee-app-hero {
    margin: 0.15rem 0 1rem 0;
    padding: 1.2rem 1.25rem 1.05rem 1.25rem;
    background: linear-gradient(135deg, rgba(7, 18, 31, 0.96) 0%, rgba(9, 27, 46, 0.94) 56%, rgba(12, 35, 56, 0.94) 100%);
    box-shadow: 0 24px 52px rgba(2, 8, 23, 0.24);
}

.ee-app-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 85% 18%, rgba(77, 199, 255, 0.12), transparent 16%),
        linear-gradient(180deg, transparent 72%, rgba(124, 193, 224, 0.08) 72%, rgba(124, 193, 224, 0.08) 73%, transparent 73%),
        repeating-linear-gradient(90deg, transparent 0 11%, rgba(124, 193, 224, 0.045) 11% 11.3%, transparent 11.3% 22%);
    opacity: 0.85;
}

.ee-app-hero > * {
    position: relative;
    z-index: 1;
}

.ee-app-grid,
.ee-page-hero-grid {
    display: grid;
    gap: 1rem;
    align-items: stretch;
}

.ee-app-grid {
    grid-template-columns: minmax(0, 1.6fr) minmax(290px, 0.92fr);
}

.ee-page-hero-grid {
    grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.65fr);
}

.ee-app-kicker,
.ee-page-kicker,
.ee-page-status-label,
.ee-sidecar-kicker {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ee-accent);
    margin-bottom: 0.35rem;
}

.ee-app-title {
    font-size: clamp(2rem, 1.3rem + 1.8vw, 3.1rem);
    font-weight: 820;
    line-height: 1.02;
    color: var(--ee-text);
    margin-bottom: 0.45rem;
}

.ee-app-copy,
.ee-page-copy {
    max-width: 72ch;
    color: var(--ee-muted);
    font-size: 1rem;
    line-height: 1.62;
}

.ee-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.95rem;
}

.ee-page-chip-row {
    margin-top: 0.8rem;
}

.ee-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.42rem 0.76rem;
    border-radius: 999px;
    background: rgba(135, 231, 255, 0.08);
    border: 1px solid rgba(135, 231, 255, 0.18);
    color: #d7f7ff;
    font-size: 0.84rem;
    font-weight: 620;
    animation: eeChipFloat 6s ease-in-out infinite;
}

.ee-chip:nth-child(2) {
    animation-delay: 1.2s;
}

.ee-chip:nth-child(3) {
    animation-delay: 2.4s;
}

.ee-sidecar {
    border: 1px solid rgba(135, 231, 255, 0.18);
    border-radius: 22px;
    padding: 0.95rem 1rem;
    background: linear-gradient(180deg, rgba(8, 18, 32, 0.86) 0%, rgba(11, 25, 41, 0.86) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.ee-sidecar-title,
.ee-page-status-value {
    font-size: 1.1rem;
    font-weight: 740;
    color: var(--ee-text);
    margin-bottom: 0.3rem;
}

.ee-sidecar-copy,
.ee-page-status-copy {
    color: var(--ee-muted);
    font-size: 0.92rem;
    line-height: 1.55;
}

.ee-sidecar-top {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: flex-start;
}

.ee-radar {
    position: relative;
    width: 124px;
    aspect-ratio: 1;
    border-radius: 50%;
    background: radial-gradient(circle at center, rgba(135, 231, 255, 0.12) 0%, rgba(135, 231, 255, 0.02) 42%, rgba(135, 231, 255, 0.0) 62%), rgba(8, 18, 32, 0.66);
    border: 1px solid rgba(135, 231, 255, 0.20);
    overflow: hidden;
    flex-shrink: 0;
}

.ee-radar::before,
.ee-radar::after {
    content: "";
    position: absolute;
    inset: 14px;
    border-radius: 50%;
    border: 1px solid rgba(135, 231, 255, 0.18);
}

.ee-radar::after {
    inset: 32px;
}

.ee-radar-sweep {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 58%;
    height: 3px;
    transform-origin: left center;
    background: linear-gradient(90deg, rgba(135, 231, 255, 0.0) 0%, rgba(135, 231, 255, 0.5) 100%);
    animation: eeSweep 7s linear infinite;
}

.ee-radar-line {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, transparent calc(50% - 0.5px), rgba(135, 231, 255, 0.14) calc(50% - 0.5px), rgba(135, 231, 255, 0.14) calc(50% + 0.5px), transparent calc(50% + 0.5px)),
        linear-gradient(transparent calc(50% - 0.5px), rgba(135, 231, 255, 0.12) calc(50% - 0.5px), rgba(135, 231, 255, 0.12) calc(50% + 0.5px), transparent calc(50% + 0.5px));
}

.ee-sidecar-stats {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.65rem;
    margin-top: 0.9rem;
}

.ee-sidecar-stat {
    padding: 0.72rem 0.78rem;
    border-radius: 16px;
    border: 1px solid rgba(135, 231, 255, 0.12);
    background: rgba(5, 15, 28, 0.48);
}

.ee-sidecar-stat-value {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ee-text);
    margin-bottom: 0.16rem;
}

.ee-sidecar-stat-label {
    font-size: 0.77rem;
    color: var(--ee-muted);
    line-height: 1.45;
}

.ee-page-hero {
    margin: 0.15rem 0 0.85rem 0;
    padding: 1rem 1rem 0.92rem 1rem;
    background: linear-gradient(180deg, rgba(8, 18, 32, 0.92) 0%, rgba(10, 25, 43, 0.92) 100%);
}

.ee-page-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 88% 50%, rgba(77, 199, 255, 0.10), transparent 16%),
        linear-gradient(180deg, transparent 74%, rgba(124, 193, 224, 0.06) 74%, rgba(124, 193, 224, 0.06) 75%, transparent 75%);
    opacity: 0.9;
}

.ee-page-hero > * {
    position: relative;
    z-index: 1;
}

.ee-page-title {
    font-size: 1.4rem;
    font-weight: 760;
    color: var(--ee-text);
    margin-bottom: 0.3rem;
}

.ee-page-status-card {
    border: 1px solid rgba(135, 231, 255, 0.16);
    border-radius: 18px;
    padding: 0.85rem 0.95rem;
    background: rgba(5, 15, 28, 0.44);
    align-self: center;
}

.ee-answer-card {
    padding: 1rem 1.05rem;
    background: linear-gradient(135deg, rgba(9, 20, 35, 0.98) 0%, rgba(12, 32, 52, 0.96) 100%);
}

.ee-answer-label,
.ee-meta-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ee-accent);
    margin-bottom: 0.45rem;
}

.ee-answer-copy {
    color: var(--ee-text);
    font-size: 1.1rem;
    line-height: 1.58;
    white-space: pre-wrap;
}

.ee-meta-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
}

.ee-meta-card {
    padding: 0.85rem 0.95rem;
    background: rgba(7, 17, 31, 0.76);
}

.ee-meta-value {
    font-size: 1.04rem;
    font-weight: 650;
    color: var(--ee-text);
    line-height: 1.45;
}

.ee-meta-hint {
    margin-top: 0.3rem;
    font-size: 0.84rem;
    color: var(--ee-muted);
    line-height: 1.5;
}

.ee-result-title {
    margin: 0.15rem 0 0.65rem 0;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--ee-text);
}

.ee-section-title {
    margin: 0 0 0.4rem 0;
    font-size: 1.02rem;
    font-weight: 680;
    color: var(--ee-text);
}

.ee-subsection-title {
    margin: 0.5rem 0 0.25rem 0;
    font-size: 0.92rem;
    font-weight: 600;
    color: #d4deee;
}

@media (max-width: 1080px) {
    .ee-app-grid,
    .ee-page-hero-grid {
        grid-template-columns: 1fr;
    }

    .ee-sidecar-top {
        flex-direction: column;
    }

    .ee-radar {
        margin-left: 0;
    }
}

/* Eagle Eye maritime operations system.
   These final rules intentionally replace the former promotional/glass treatment. */
:root {
    --ee-canvas: #071014;
    --ee-surface: #0D171D;
    --ee-surface-raised: #111E25;
    --ee-border: #27363E;
    --ee-border-strong: #3B505A;
    --ee-text: #E8EFF1;
    --ee-muted: #9FB0B8;
    --ee-accent: #28B7A5;
    --ee-accent-strong: #54CDBE;
    --ee-success: #61B98B;
    --ee-warning: #D9AA57;
    --ee-danger: #DE7373;
    color-scheme: dark;
}

html,
body,
[class*="css"] {
    font-family: "Avenir Next", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
}

code,
pre,
kbd,
.ee-status-key,
.ee-product-line,
.ee-sidebar-wordmark {
    font-family: "SFMono-Regular", "SF Mono", Consolas, monospace;
}

.stApp {
    color: var(--ee-text) !important;
    background: var(--ee-canvas) !important;
}

.block-container {
    max-width: 1440px;
    padding: 1rem 1.4rem 3rem !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

h1,
h2,
h3,
p,
li,
label,
[data-testid="stMarkdownContainer"] {
    letter-spacing: normal;
}

div[data-testid="stSidebar"] {
    background: #091217 !important;
    border-right: 1px solid var(--ee-border) !important;
}

div[data-testid="stSidebar"] .block-container {
    padding: 1rem 0.9rem 2rem !important;
}

.ee-sidebar-brand {
    padding: 0.25rem 0.25rem 1rem;
    margin-bottom: 0.35rem;
    border-bottom: 1px solid var(--ee-border);
}

.ee-sidebar-wordmark {
    color: var(--ee-text);
    font-size: 0.9rem;
    font-weight: 760;
    letter-spacing: 0.15em;
}

.ee-sidebar-caption {
    margin-top: 0.3rem;
    color: var(--ee-muted);
    font-size: 0.77rem;
}

div[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: var(--ee-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
    min-height: 2.35rem;
    margin: 0.1rem 0;
    padding: 0.42rem 0.55rem;
    border-left: 2px solid transparent;
    border-radius: 0 5px 5px 0;
    color: var(--ee-muted);
}

div[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    color: var(--ee-text);
    border-left-color: var(--ee-accent);
    background: #102129;
}

.ee-masthead {
    display: grid;
    grid-template-columns: minmax(170px, 0.48fr) minmax(360px, 1.45fr) minmax(260px, 0.82fr);
    gap: 1.25rem;
    align-items: center;
    margin: 0 0 1.15rem;
    padding: 0.9rem 0 1rem;
    border-top: 1px solid var(--ee-border);
    border-bottom: 1px solid var(--ee-border);
}

.ee-masthead-brand {
    padding-right: 1rem;
    border-right: 1px solid var(--ee-border);
}

.ee-wordmark {
    color: var(--ee-text);
    font-size: 1rem;
    font-weight: 780;
    letter-spacing: 0.11em;
}

.ee-product-line {
    margin-top: 0.25rem;
    color: var(--ee-accent-strong);
    font-size: 0.67rem;
    letter-spacing: 0.12em;
}

.ee-masthead-title {
    color: var(--ee-text);
    font-size: clamp(1.45rem, 1.05rem + 1vw, 2rem);
    font-weight: 680;
    line-height: 1.15;
}

.ee-masthead-copy {
    max-width: 68ch;
    margin-top: 0.32rem;
    color: var(--ee-muted);
    font-size: 0.9rem;
    line-height: 1.5;
}

.ee-masthead-status {
    display: grid;
    gap: 0.48rem;
    padding-left: 1rem;
    border-left: 1px solid var(--ee-border);
}

.ee-status-item {
    display: grid;
    grid-template-columns: 5.3rem 1fr;
    gap: 0.55rem;
    align-items: baseline;
}

.ee-status-key {
    color: var(--ee-muted);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.ee-status-value {
    color: var(--ee-text);
    font-size: 0.78rem;
    line-height: 1.35;
}

.ee-status-ready .ee-status-value::before,
.ee-status-attention .ee-status-value::before {
    display: inline-block;
    width: 0.45rem;
    height: 0.45rem;
    margin-right: 0.45rem;
    border-radius: 50%;
    content: "";
}

.ee-status-ready .ee-status-value::before {
    background: var(--ee-success);
}

.ee-status-attention .ee-status-value::before {
    background: var(--ee-warning);
}

.ee-query-context {
    margin: 0 0 0.8rem;
    padding: 0.68rem 0.82rem;
    border-left: 3px solid var(--ee-accent);
    background: #0A151A;
}

.ee-query-context-label,
.ee-workbench-label,
.ee-answer-label,
.ee-meta-label {
    margin-bottom: 0.25rem;
    color: var(--ee-accent-strong) !important;
    font-size: 0.7rem !important;
    font-weight: 720 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase;
}

.ee-query-context-copy {
    color: var(--ee-muted);
    font-size: 0.9rem;
    line-height: 1.5;
}

.ee-workspace-heading {
    margin: 0.1rem 0 0.2rem;
    color: var(--ee-text);
    font-size: 1.18rem;
    font-weight: 680;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--ee-border) !important;
    border-radius: 7px !important;
    background: var(--ee-surface) !important;
    box-shadow: none !important;
    transform: none !important;
    animation: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--ee-border-strong) !important;
    box-shadow: none !important;
    transform: none !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--ee-border) !important;
    border-radius: 6px !important;
    background: transparent !important;
}

div[data-testid="stTabs"] [role="tablist"] {
    gap: 0;
    border-bottom: 1px solid var(--ee-border);
}

div[data-testid="stTabs"] button[role="tab"] {
    height: 2.65rem;
    padding: 0.5rem 0.85rem;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--ee-muted) !important;
    font-weight: 620;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-bottom-color: var(--ee-accent) !important;
    color: var(--ee-text) !important;
    background: transparent !important;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.45rem;
    border: 1px solid var(--ee-border-strong) !important;
    border-radius: 6px !important;
    background: var(--ee-surface-raised) !important;
    color: var(--ee-text) !important;
    box-shadow: none !important;
    font-weight: 650;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--ee-accent) !important;
    color: var(--ee-text) !important;
}

.stButton > button[kind="primary"] {
    border-color: var(--ee-accent) !important;
    background: var(--ee-accent) !important;
    color: #071014 !important;
}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-testid="stDateInputField"] input {
    border: 1px solid var(--ee-border-strong) !important;
    border-radius: 6px !important;
    background: #081318 !important;
    color: var(--ee-text) !important;
    box-shadow: none !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #81949D !important;
    opacity: 1;
}

*:focus-visible {
    outline: 2px solid var(--ee-accent-strong) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

div[data-testid="stMetric"] {
    min-height: 5.2rem;
    padding: 0.7rem 0.8rem !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 6px !important;
    background: var(--ee-surface) !important;
}

div[data-testid="stMetricLabel"] > div {
    color: var(--ee-muted) !important;
    font-size: 0.76rem !important;
}

div[data-testid="stMetricValue"] > div {
    color: var(--ee-text) !important;
    font-size: 1.08rem !important;
}

div[data-testid="stVegaLiteChart"],
div[data-testid="stDataFrame"],
div[data-testid="stImage"] {
    overflow: hidden;
    border: 1px solid var(--ee-border);
    border-radius: 7px !important;
    box-shadow: none !important;
}

.ee-answer-card,
.ee-meta-card {
    border: 1px solid var(--ee-border) !important;
    border-radius: 7px !important;
    background: var(--ee-surface) !important;
    box-shadow: none !important;
}

.ee-answer-card {
    padding: 0.95rem 1rem !important;
    border-left: 3px solid var(--ee-accent) !important;
}

.ee-answer-copy,
.ee-meta-value,
.ee-result-title,
.ee-section-title {
    color: var(--ee-text) !important;
}

.ee-answer-copy {
    font-size: 1.02rem !important;
    line-height: 1.58 !important;
}

.ee-meta-grid {
    gap: 0.55rem !important;
}

.ee-meta-card {
    padding: 0.62rem 0.72rem !important;
}

.ee-meta-hint,
.ee-capture-note,
.ee-subsection-title {
    color: var(--ee-muted) !important;
}

.ee-app-hero,
.ee-page-hero,
.ee-sidecar,
.ee-radar,
.ee-chip-row {
    display: none !important;
}

@media (max-width: 900px) {
    .block-container {
        padding: 0.8rem 1rem 2.25rem !important;
    }

    .ee-masthead {
        grid-template-columns: minmax(150px, 0.5fr) minmax(0, 1.5fr);
    }

    .ee-masthead-status {
        grid-column: 1 / -1;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        padding: 0.7rem 0 0;
        border-top: 1px solid var(--ee-border);
        border-left: 0;
    }
}

@media (max-width: 560px) {
    .block-container {
        padding: 0.65rem 0.72rem 2rem !important;
    }

    .ee-masthead {
        grid-template-columns: 1fr;
        gap: 0.75rem;
        padding-top: 0.7rem;
    }

    .ee-masthead-brand {
        padding: 0 0 0.65rem;
        border-right: 0;
        border-bottom: 1px solid var(--ee-border);
    }

    .ee-masthead-status {
        grid-template-columns: 1fr;
    }

    .ee-status-item {
        grid-template-columns: 4.5rem 1fr;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        padding-inline: 0.55rem;
        font-size: 0.78rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header(
    selected_page: str,
    *,
    freshness_label: str,
    evidence_active: bool,
) -> None:
    page_scopes = {
        "Overview": "Coverage, runtime readiness, and the available analytical workspaces.",
        "Chat Assistant": "Cross-domain maritime analysis with deterministic facts and grounded evidence.",
        "Traffic Monitoring": "Port calls, arrival patterns, route history, and vessel traffic comparisons.",
        "Vessel Investigation": "Vessel stays, movement histories, and observed AIS anomalies.",
        "ETA & Delay": "Sweden-first Baltic inbound watchlists, reported ETAs, vessel positions, revisions, and signal-quality exceptions.",
        "Port Pressure": "Observed arrival pressure, dwell signals, and baseline comparisons.",
        "Carbon Emissions": "Deterministic TTW and WTW inventory analysis with explicit boundaries.",
    }
    visible_page = _display_page_label(selected_page)
    page_scope = page_scopes.get(selected_page, "Historical maritime analysis and evidence.")
    evidence_label = "Available" if evidence_active else "Unavailable"
    evidence_state = "ready" if evidence_active else "attention"
    st.markdown(
        f"""
        <div class="ee-masthead">
            <div class="ee-masthead-brand">
                <div class="ee-wordmark">EAGLE EYE</div>
                <div class="ee-product-line">MARITIME OPERATIONS</div>
            </div>
            <div class="ee-masthead-page">
                <div class="ee-masthead-title">{escape(visible_page)}</div>
                <div class="ee-masthead-copy">{escape(page_scope)}</div>
            </div>
            <div class="ee-masthead-status" aria-label="Workspace status">
                <div class="ee-status-item">
                    <span class="ee-status-key">DATA</span>
                    <span class="ee-status-value">{escape(freshness_label)}</span>
                </div>
                <div class="ee-status-item ee-status-{evidence_state}">
                    <span class="ee-status-key">EVIDENCE</span>
                    <span class="ee-status-value">{escape(evidence_label)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_query_category_page(
    *,
    category: str,
    query_service: QueryService,
    kpi_engine: KPIQueryEngine,
    forecast_engine: ForecastEngine,
    carbon_engine: CarbonQueryEngine,
    events_path: Optional[Path],
    top_k_evidence: int,
    show_technical: bool,
    threshold_percentiles: Tuple[float, float, float],
    ensure_retriever: Callable[[], tuple[Optional[RAGRetriever], str]],
    default_from_date: pd.Timestamp,
    default_to_date: pd.Timestamp,
    screenshot_mode: bool = False,
) -> None:
    key = _query_category_key(category)
    state_question_key = f"ask_question_{key}"
    state_sample_key = f"sample_select_{key}"
    state_result_bundle_key = f"ask_result_bundle_{key}"
    state_use_range_key = f"use_date_range_{key}"
    state_port_key = f"ui_port_{key}"
    state_vessel_type_key = f"ui_vessel_type_{key}"
    state_anomaly_key = f"ui_anomaly_{key}"
    state_from_key = f"ui_from_{key}"
    state_to_key = f"ui_to_{key}"

    if screenshot_mode:
        st.markdown(f"**{category}**")
        st.caption("Capture mode is enabled. Use the compact snapshot result to fit answer, chart, and evidence in one frame.")
    else:
        _render_query_category_intro(category)
    samples = SAMPLE_QUERIES_BY_CATEGORY.get(category, [])
    if not samples:
        st.warning(f"No sample queries configured for category `{category}`.")
        return
    if state_question_key not in st.session_state:
        st.session_state[state_question_key] = samples[0]
    if state_sample_key not in st.session_state:
        st.session_state[state_sample_key] = samples[0]
    if state_anomaly_key not in st.session_state:
        st.session_state[state_anomaly_key] = "any"
    if state_use_range_key not in st.session_state:
        st.session_state[state_use_range_key] = False
    if state_from_key not in st.session_state:
        st.session_state[state_from_key] = default_from_date.date()
    if state_to_key not in st.session_state:
        st.session_state[state_to_key] = default_to_date.date()
    stored_bundle = st.session_state.get(state_result_bundle_key)

    def _render_stored_result(bundle: Dict[str, Any]) -> None:
        _render_compact_result(
            result=bundle["result"],
            evidence=bundle["evidence"],
            show_technical=show_technical,
            canonical_envelope=bundle.get("canonical_envelope"),
            intent_result=bundle.get("intent_result"),
            carbon_engine=carbon_engine,
            threshold_percentiles=threshold_percentiles,
            question=bundle.get("question"),
            screenshot_mode=screenshot_mode,
            surface_all_canonical_visuals=True,
            query_service=query_service,
        )

    def _render_query_controls() -> bool:
        with st.container(border=True):
            st.markdown("<div class='ee-workbench-label'>Analysis request</div>", unsafe_allow_html=True)
            selected = st.selectbox(
                "Sample analysis",
                options=samples,
                index=samples.index(st.session_state[state_sample_key]) if st.session_state[state_sample_key] in samples else 0,
                key=state_sample_key,
            )
            if st.button("Load sample query", key=f"load_sample_{key}"):
                st.session_state[state_question_key] = selected

            st.text_area("Question", key=state_question_key, height=80 if screenshot_mode else 112)
            st.caption(
                "Port pressure is an index calculated from the available historical AIS and port-call data."
            )

            with st.expander("Optional filters", expanded=False):
                st.text_input(
                    "Port / LOCODE / name",
                    key=state_port_key,
                    help="Examples: SEGOT, Gothenburg, Port of Gothenburg",
                )
                st.checkbox(
                    "Apply date range filter",
                    key=state_use_range_key,
                    help="Use calendar inputs to avoid date formatting errors.",
                )
                if st.session_state[state_use_range_key]:
                    col_from, col_to = st.columns(2)
                    col_from.date_input("From date", key=state_from_key, format="YYYY-MM-DD")
                    col_to.date_input("To date", key=state_to_key, format="YYYY-MM-DD")
                st.text_input("Vessel type", key=state_vessel_type_key)
                st.selectbox("Anomaly flag", options=["any", "true", "false"], key=state_anomaly_key)

            return bool(st.button("Analyze", type="primary", key=f"ask_btn_{key}"))

    if screenshot_mode and stored_bundle:
        _render_stored_result(stored_bundle)
        with st.expander("Query controls", expanded=False):
            ask_clicked = _render_query_controls()
    else:
        ask_clicked = _render_query_controls()

    if not ask_clicked:
        if stored_bundle and not screenshot_mode:
            _render_stored_result(stored_bundle)
        _render_query_category_guide(category)
        return

    question = str(st.session_state.get(state_question_key, "")).strip()
    if not question:
        st.warning("Enter a question first.")
        return

    ui_date_from = ""
    ui_date_to = ""
    if bool(st.session_state.get(state_use_range_key)):
        from_obj = st.session_state.get(state_from_key)
        to_obj = st.session_state.get(state_to_key)
        if from_obj:
            ui_date_from = pd.Timestamp(from_obj).strftime("%Y-%m-%d")
        if to_obj:
            ui_date_to = pd.Timestamp(to_obj).strftime("%Y-%m-%d")

    user_filters: Dict[str, Any] = {
        "port": str(st.session_state.get(state_port_key, "") or "").strip() or None,
        "date_from": ui_date_from or None,
        "date_to": ui_date_to or None,
        "vessel_type": str(st.session_state.get(state_vessel_type_key, "") or "").strip() or None,
        "anomaly": _parse_anomaly_filter(str(st.session_state.get(state_anomaly_key, "any"))),
    }
    conversation_key = f"canonical_conversation_{key}"
    conversation_id = str(st.session_state.get(conversation_key, "")).strip()
    if not conversation_id:
        conversation_id = f"streamlit_{key}_{uuid.uuid4().hex[:16]}"
        st.session_state[conversation_key] = conversation_id
    canonical = run_canonical_query(
        query_service,
        question=question,
        conversation_id=conversation_id,
        top_k_evidence=top_k_evidence,
        user_filters=user_filters,
    )
    result = canonical.result
    evidence = EvidenceBundle(
        lines=canonical.evidence.lines,
        rows=canonical.evidence.rows,
        trace=canonical.evidence.trace,
    )
    st.session_state[state_result_bundle_key] = {
        "question": question,
        "intent_result": None,
        "canonical_envelope": canonical.envelope,
        "result": result,
        "evidence": evidence,
    }
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Eagle Eye", layout="wide")
    _apply_global_app_styles()

    config_path = "config/config.yaml"
    config = load_config(config_path)
    configured_processed_dir = Path(config.get("predict", {}).get("processed_dir", "data/processed"))
    processed_bootstrap_changed, processed_bootstrap_message = _maybe_bootstrap_processed_bundle(
        configured_processed_dir
    )
    events_bootstrap_changed, events_bootstrap_message = _maybe_bootstrap_events_bundle(
        configured_processed_dir
    )
    default_processed_dir, using_demo_processed = _resolve_processed_dir(configured_processed_dir)
    configured_persist_dir = Path(config["paths"].get("persist_dir", "data/chroma"))
    chroma_bootstrap_changed = False
    chroma_bootstrap_message = ""
    requested_vector_mode = str(
        os.getenv("VECTOR_DB_MODE", config.get("vector_db", {}).get("mode", "local"))
    ).strip().lower()
    using_remote_vector = _remote_vector_enabled(config)
    if using_remote_vector:
        persist_dir = configured_persist_dir
        using_demo_chroma = False
    else:
        chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_bundle(
            configured_persist_dir
        )
        persist_dir, using_demo_chroma = _resolve_persist_dir(configured_persist_dir)

    selected_page = "Traffic Monitoring"
    with st.sidebar:
        st.markdown(
            """
            <div class="ee-sidebar-brand">
                <div class="ee-sidebar-wordmark">EAGLE EYE</div>
                <div class="ee-sidebar-caption">Maritime operations workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        analyst_pages = [
            "Overview",
            "Chat Assistant",
            "Traffic Monitoring",
            "Vessel Investigation",
            "ETA & Delay",
            "Port Pressure",
            "Carbon Emissions",
        ]
        page_options = analyst_pages
        selected_page = st.radio(
            "Workspace",
            options=page_options,
            index=page_options.index("Chat Assistant"),
            format_func=_display_page_label,
        )
        with st.expander("Workspace settings", expanded=False):
            top_k_evidence = st.slider(
                "Evidence top K",
                min_value=0,
                max_value=8,
                value=5,
                help="Retrieve five supporting AIS/document rows by default. Set 0 only when you explicitly want evidence retrieval disabled.",
            )
            screenshot_mode = st.toggle(
                "Screenshot mode",
                value=False,
                help="Condense answer, chart, and evidence into a single capture-friendly layout.",
            )
            st.caption("Specific ports and date ranges improve analytical precision.")
        show_technical = False
        with st.expander("Data loading status", expanded=False):
            if using_demo_processed:
                st.info("Running with bundled demo processed data (`demo_data/processed`).")
                if "APP_PROCESSED_BUNDLE_URL" in processed_bootstrap_message or "Downloaded" in processed_bootstrap_message:
                    st.warning(processed_bootstrap_message)
            elif processed_bootstrap_changed:
                st.info(processed_bootstrap_message)
            elif "No APP_PROCESSED_BUNDLE_URL configured." not in processed_bootstrap_message:
                st.warning(processed_bootstrap_message)
            if events_bootstrap_changed:
                st.info(events_bootstrap_message)
            if using_remote_vector:
                st.info("Using remote Chroma service (configured via CHROMA_* / VECTOR_DB_MODE).")
            elif chroma_bootstrap_changed:
                st.info(chroma_bootstrap_message)
            if using_demo_chroma:
                st.info("Running with bundled demo vector index (`demo_data/chroma`).")
                st.caption("Full retrieval parity with local requires a remote Chroma service because the full local vector store is too large for cloud packaging.")
            elif chroma_bootstrap_message and "No APP_CHROMA_BUNDLE_URL configured." not in chroma_bootstrap_message:
                st.caption(chroma_bootstrap_message)
            if requested_vector_mode in {"remote", "http"} and not using_remote_vector:
                st.warning("VECTOR_DB_MODE is remote but CHROMA_HOST is missing/invalid; using local/demo index.")
            if not using_demo_processed and not processed_bootstrap_changed:
                st.caption(f"Processed runtime path: {default_processed_dir}")

    try:
        # Every page reuses the canonical service's preloaded engines. Keeping
        # a second set of dataframes doubled resident memory and left a lazy
        # PyArrow read on the category-query rerun path.
        query_service = _init_query_service()
        kpi_engine = query_service.kpi
        forecast_engine = query_service.forecaster
        carbon_engine = query_service.carbon
        carbon_cfg = config.get("carbon", {})
        threshold_percentiles = sanitize_threshold_percentiles(
            carbon_cfg.get("relative_level_percentiles", [0.25, 0.50, 0.75])
        )
        _validate_sample_queries_runtime(carbon_engine)
    except Exception as exc:
        st.error(f"Could not initialize data engines: {exc}")
        st.info("Run `./run_demo_pipeline.sh` first.")
        st.stop()

    retriever: Optional[RAGRetriever] = getattr(query_service, "retriever", None)
    retriever_reason = str(getattr(query_service, "retriever_reason", "") or "").strip()
    api_key, key_source = _load_openai_api_key_from_runtime()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_conversation_id" not in st.session_state:
        st.session_state["chat_conversation_id"] = ""
    if "chat_message" not in st.session_state:
        st.session_state["chat_message"] = ""
    def _ensure_retriever() -> tuple[Optional[RAGRetriever], str]:
        nonlocal persist_dir, using_demo_chroma, chroma_bootstrap_changed, chroma_bootstrap_message
        try:
            active = _init_retriever(persist_dir=str(persist_dir), config_path=config_path)
            if api_key:
                reason = f"RAG evidence active (API key source: {key_source}, backend: {active.vector_backend})."
            else:
                reason = (
                    f"Local RAG evidence active (backend: {active.vector_backend}; no OpenAI key required). "
                    "Model-backed web synthesis remains off."
                )
            return active, reason
        except Exception as exc:
            reason = f"Retriever init failed: {exc}"
            if using_remote_vector:
                chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_bundle(
                    configured_persist_dir
                )
                fallback_persist_dir, fallback_using_demo_chroma = _resolve_persist_dir(configured_persist_dir)
                if (fallback_persist_dir / "chroma.sqlite3").exists():
                    try:
                        active = _init_retriever(
                            persist_dir=str(fallback_persist_dir),
                            config_path=config_path,
                            force_local_vector=True,
                        )
                        persist_dir = fallback_persist_dir
                        using_demo_chroma = fallback_using_demo_chroma
                        return active, (
                            f"Remote retriever failed ({exc}). "
                            f"Fell back to local vector store at {fallback_persist_dir} "
                            f"(backend: {active.vector_backend})."
                        )
                    except Exception as local_exc:
                        reason = f"Remote retriever failed ({exc}); local fallback failed ({local_exc})."
            return None, reason

    if retriever is None:
        retriever, retriever_reason = _ensure_retriever()
    elif not retriever_reason:
        backend = getattr(retriever, "vector_backend", "unknown")
        retriever_reason = (
            f"RAG evidence active (API key source: {key_source}, backend: {backend})."
            if api_key
            else (
                f"Local RAG evidence active (backend: {backend}; no OpenAI key required). "
                "Model-backed web synthesis remains off."
            )
        )
    if retriever is not None:
        query_service.retriever = retriever
    query_service.retriever_reason = retriever_reason

    with st.sidebar:
        with st.expander("Runtime status", expanded=False):
            if retriever is not None:
                st.caption(retriever_reason)
            elif retriever_reason:
                st.warning(retriever_reason)
            if getattr(carbon_engine, "available", False):
                st.caption(f"Carbon layer active (params: {carbon_engine.params_version.get('version', 'unknown')}).")
                if carbon_engine.daily_port is not None and not carbon_engine.daily_port.empty and "date" in carbon_engine.daily_port.columns:
                    carbon_dates = pd.to_datetime(carbon_engine.daily_port["date"], errors="coerce", utc=True).dropna()
                    if not carbon_dates.empty:
                        st.caption(
                            f"Carbon date coverage: {carbon_dates.min().date()} to {carbon_dates.max().date()}."
                        )
                if carbon_engine.daily_port is not None and not carbon_engine.daily_port.empty and "port_key" in carbon_engine.daily_port.columns:
                    st.caption(f"Carbon ports in scope: {int(carbon_engine.daily_port['port_key'].nunique())}.")
            else:
                st.warning("Carbon layer artifacts not found. Build with `python -m src.carbon.build --processed_dir data/processed`.")

    st.session_state["retriever_reason"] = retriever_reason
    events_path = configured_processed_dir / "events.parquet"
    if not events_path.exists():
        events_path = default_processed_dir / "events.parquet"

    default_from_date = pd.Timestamp.now().floor("D") - pd.Timedelta(days=30)
    default_to_date = pd.Timestamp.now().floor("D")
    if not kpi_engine.arrivals_daily.empty and "date" in kpi_engine.arrivals_daily.columns:
        date_series = pd.to_datetime(kpi_engine.arrivals_daily["date"], errors="coerce", utc=True).dropna()
        if not date_series.empty:
            default_from_date = date_series.min().floor("D")
            default_to_date = date_series.max().floor("D")

    if not (screenshot_mode and selected_page in QUERY_CATEGORY_PAGE_ORDER):
        freshness_label = (
            f"{default_from_date.strftime('%Y-%m-%d')} — {default_to_date.strftime('%Y-%m-%d')}"
        )
        _render_app_header(
            selected_page,
            freshness_label=freshness_label,
            evidence_active=retriever is not None,
        )

    if selected_page == "Overview":
        _render_page_overview(
            kpi=kpi_engine,
            carbon=carbon_engine,
            retriever_reason=retriever_reason,
        )
        return

    if selected_page == "Chat Assistant":
        _render_page_chat(
            query_service=query_service,
            carbon_engine=carbon_engine,
            top_k_evidence=top_k_evidence,
            show_technical=show_technical,
            threshold_percentiles=threshold_percentiles,
        )
        return

    if selected_page in QUERY_CATEGORY_PAGE_ORDER:
        _render_query_category_page(
            category=selected_page,
            query_service=query_service,
            kpi_engine=kpi_engine,
            forecast_engine=forecast_engine,
            carbon_engine=carbon_engine,
            events_path=events_path if events_path.exists() else None,
            top_k_evidence=top_k_evidence,
            show_technical=show_technical,
            threshold_percentiles=threshold_percentiles,
            ensure_retriever=_ensure_retriever,
            default_from_date=default_from_date,
            default_to_date=default_to_date,
            screenshot_mode=screenshot_mode,
        )
        return

    st.warning("Unknown page selection. Falling back to Traffic Monitoring.")
    _render_query_category_page(
        category="Traffic Monitoring",
        query_service=query_service,
        kpi_engine=kpi_engine,
        forecast_engine=forecast_engine,
        carbon_engine=carbon_engine,
        events_path=events_path if events_path.exists() else None,
        top_k_evidence=top_k_evidence,
        show_technical=show_technical,
        threshold_percentiles=threshold_percentiles,
        ensure_retriever=_ensure_retriever,
        default_from_date=default_from_date,
        default_to_date=default_to_date,
        screenshot_mode=screenshot_mode,
    )
    return


if __name__ == "__main__":
    main()
