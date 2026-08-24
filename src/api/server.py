"""FastAPI deployment path for Eagle Eye."""

from __future__ import annotations

import asyncio
import os
import json
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

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
from src.carbon.presentation import (
    build_emissions_findings,
    build_reduction_suggestions,
    classify_level,
    compute_emissions_metrics,
    derive_threshold_bands,
    extract_chart_findings,
    format_percent,
    format_tco2e,
    safe_percent_delta,
    sanitize_threshold_percentiles,
)
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.live_eta.aisstream import AISStreamCollector
from src.query.context import ConversationStore
from src.query.models import (
    AnswerEnvelope,
    AnswerState,
    ExportRequest,
    ExportResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryFiltersPayload,
    QueryRequest,
)
from src.query.planner import QueryPlanner
from src.query.service import QueryService
from src.rag.retriever import RAGRetriever
from src.rag.synthesis import build_local_synthesizer
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.cloud_bootstrap import ensure_bundle, ensure_file_manifest
from src.utils.confidence import extract_confidence_label
from src.utils.config import load_config
from src.utils.runtime import chroma_remote_settings, force_local_vector_env
from src.utils.redaction import redact_sensitive_text


class AskFiltersPayload(BaseModel):
    port: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    vessel_type: Optional[str] = None
    vessel_name: Optional[str] = None
    mmsi: Optional[str] = None
    imo: Optional[str] = None
    anomaly: Optional[bool] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k_evidence: int = Field(default=5, ge=0, le=10)
    filters: AskFiltersPayload = Field(default_factory=AskFiltersPayload)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    top_k_evidence: int = Field(default=5, ge=0, le=10)
    filters: AskFiltersPayload = Field(default_factory=AskFiltersPayload)


class CarbonEstimateRequest(BaseModel):
    vessel_type: Optional[str] = None
    mode: str = "transit"
    duration_h: float = Field(default=1.0, gt=0.0, le=240.0)
    speed_kn: float = Field(default=10.0, ge=0.0, le=60.0)
    mcr_kw: Optional[float] = Field(default=None, gt=0.0)
    ref_speed_kn: Optional[float] = Field(default=None, gt=0.0)
    aux_power_kw: Optional[float] = Field(default=None, ge=0.0)
    fuel_type: Optional[str] = None
    engine_family: Optional[str] = None
    boundary: str = "TTW"
    pollutants: Optional[List[str]] = None




@dataclass
class EvidenceBundle:
    lines: List[str]
    rows: List[Dict[str, Any]]
    trace: Dict[str, Any]


def _resolve_processed_dir(preferred_dir: Path) -> tuple[Path, bool]:
    if (preferred_dir / "arrivals_daily.parquet").exists():
        return preferred_dir, False
    fallback = Path("demo_data/processed")
    if (fallback / "arrivals_daily.parquet").exists():
        return fallback, True
    return preferred_dir, False


def _resolve_persist_dir(preferred_dir: Path) -> tuple[Path, bool]:
    if (preferred_dir / "chroma.sqlite3").exists():
        return preferred_dir, False
    fallback = Path("demo_data/chroma")
    if (fallback / "chroma.sqlite3").exists():
        return fallback, True
    return preferred_dir, False


def _load_runtime_setting(name: str) -> str:
    return str(os.getenv(name, "")).strip()


def _runtime_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}




def _maybe_bootstrap_bundle(
    env_name: str,
    target_dir: Path,
    required_files: List[str],
) -> tuple[bool, str]:
    if all((target_dir / rel_path).exists() for rel_path in required_files):
        return False, f"{env_name} assets already exist in {target_dir}."
    bundle_url = _load_runtime_setting(env_name)
    if not bundle_url:
        return False, f"No {env_name} configured."
    changed, message = ensure_bundle(
        url=bundle_url,
        target_dir=target_dir,
        required_files=required_files,
    )
    return changed, message


def _maybe_bootstrap_chroma_runtime(target_dir: Path) -> tuple[bool, str]:
    required_files = ["chroma.sqlite3", "traffic_metadata_index.csv"]
    if all((target_dir / name).exists() for name in required_files):
        return False, f"Chroma runtime assets already exist in {target_dir}."

    manifest_url = _load_runtime_setting("APP_CHROMA_MANIFEST_URL")
    if manifest_url:
        return ensure_file_manifest(
            url=manifest_url,
            target_dir=target_dir,
            required_files=required_files,
            timeout_seconds=3600,
        )

    bundle_url = _load_runtime_setting("APP_CHROMA_BUNDLE_URL")
    if bundle_url:
        return ensure_bundle(
            url=bundle_url,
            target_dir=target_dir,
            required_files=required_files,
            timeout_seconds=3600,
        )

    return False, "No APP_CHROMA_MANIFEST_URL or APP_CHROMA_BUNDLE_URL configured."


def _init_retriever(
    persist_dir: Path,
    config_path: str,
    force_local_vector: bool = False,
) -> RAGRetriever:
    if force_local_vector:
        with force_local_vector_env():
            return RAGRetriever(persist_dir=persist_dir, config_path=config_path)
    return RAGRetriever(persist_dir=persist_dir, config_path=config_path)


def _fallback_evidence_from_result(
    value: Union[AnalyticsResult, ForecastResult, CarbonResult],
    max_items: int = 5,
) -> List[str]:
    lines: List[str] = []
    if isinstance(value, CarbonResult):
        for eid in (value.evidence_ids or [])[:max_items]:
            lines.append(f"carbon_evidence_id={eid}")
        if value.table is not None and not value.table.empty:
            head = value.table.head(min(3, max_items))
            for _, row in head.iterrows():
                tokens = []
                for col in head.columns[:4]:
                    cell = row[col]
                    if pd.isna(cell):
                        continue
                    tokens.append(f"{col}={cell}")
                if tokens:
                    lines.append(" | ".join(tokens))
        return lines[:max_items]

    if isinstance(value, ForecastResult):
        anchor_values_note = next((n for n in value.coverage_notes if n.startswith("Analog values used:")), None)
        anchor_dates_note = next((n for n in value.coverage_notes if n.startswith("Analog dates used:")), None)
        if anchor_values_note:
            lines.append(anchor_values_note)
        elif anchor_dates_note:
            lines.append(anchor_dates_note)
        if value.history is not None and not value.history.empty and {"date", "actual"}.issubset(value.history.columns):
            hist = value.history.copy()
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.floor("D")
            hist = hist.dropna(subset=["date", "actual"]).sort_values("date")
            for _, row in hist.tail(max_items).iterrows():
                lines.append(f"Historical point | {row['date'].strftime('%Y-%m-%d')} | value={float(row['actual']):.2f}")
        return lines[:max_items]

    if value.table is not None and not value.table.empty:
        table = value.table.head(max_items).copy()
        for _, row in table.iterrows():
            parts: List[str] = []
            for col in table.columns[:4]:
                cell = row[col]
                if pd.isna(cell):
                    continue
                if isinstance(cell, pd.Timestamp):
                    rendered = cell.strftime("%Y-%m-%d")
                else:
                    rendered = str(cell)
                parts.append(f"{col}={rendered}")
            if parts:
                lines.append(" | ".join(parts))
    return lines[:max_items]


def _build_method_steps(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
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
        steps.append(f"Source label: {value.source_label}.")
        steps.append(f"Confidence: {value.confidence_label} ({value.confidence_reason})")
        if value.params_version:
            steps.append(f"Factor registry version: {value.params_version}.")
        for note in value.coverage_notes[:6]:
            steps.append(note)
        return steps

    if isinstance(value, ForecastResult):
        steps.append("Applied active filters (port/date/vessel-type) to congestion history.")
        for note in value.coverage_notes:
            if (
                note.startswith("Coverage window:")
                or note.startswith("Rows used:")
                or note.startswith("Target date:")
                or note.startswith("Forecast target weekday:")
                or note.startswith("Analog dates used:")
                or note.startswith("Analog values used:")
                or note.startswith("Meaning:")
                or note.startswith("Method:")
            ):
                steps.append(note)
        steps.append("Computed point estimate and uncertainty interval for the requested target.")
    else:
        steps.append("Applied active filters to KPI tables.")
        for note in value.coverage_notes:
            if note.startswith("Coverage window:") or note.startswith("Rows used:") or note.startswith("Data sources used:"):
                steps.append(note)
        if value.table is not None and not value.table.empty:
            steps.append(f"Aggregated filtered rows into {len(value.table):,} output row(s).")
        else:
            steps.append("Computed deterministic metric directly from filtered subset.")

    for assumption in [c for c in value.caveats if not c.lower().startswith("confidence:")][:2]:
        steps.append(f"Assumption: {assumption}")

    out: List[str] = []
    for step in steps:
        if step and step not in out:
            out.append(step)
    return out


def _build_port_actions(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> List[str]:
    actions: List[str] = []
    if isinstance(value, CarbonResult):
        if value.result_state not in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
            return [
                "Improve carbon data coverage for the selected scope before interpreting emissions numerically.",
                "Add validated vessel fuel/engine/activity factors for periods with missing deterministic carbon rows.",
                "Use retrieved traffic evidence as context only until deterministic carbon inventory is available.",
            ]
        metric = value.uncertainty_interval.get("CO2e") or value.uncertainty_interval.get("CO2") or {}
        point = float(metric.get("point", 0.0))
        upper = float(metric.get("upper", point))
        if point >= 50:
            actions.append("Prioritize shore-power and berth energy optimization on the highest-emitting call windows.")
            actions.append("Coordinate speed and arrival windows with pilots to reduce manoeuvring fuel burn.")
        elif point >= 15:
            actions.append("Apply targeted slow-steaming and auxiliary-load management for vessels in this corridor.")
            actions.append("Flag high-intensity days for emissions-aware berth allocation.")
        else:
            actions.append("Keep baseline operating plan; monitor for drift against this emissions baseline.")
        if upper > point * 1.4:
            actions.append("Uncertainty is wide; refresh with updated AIS coverage before operational decisions.")
        actions.append("Use evidence IDs in technical audit mode to validate factor/fallback assumptions.")
        return actions

    if isinstance(value, ForecastResult) and value.forecast is not None and not value.forecast.empty:
        pred = float(value.forecast["predicted"].mean())
        upper = float(value.forecast["upper"].mean()) if "upper" in value.forecast.columns else pred
        lower = float(value.forecast["lower"].mean()) if "lower" in value.forecast.columns else pred
        spread = max(0.0, upper - pred)
        confidence = extract_confidence_label(value).lower()
        if pred >= 1.8:
            actions.append("Activate high-traffic playbook: reserve extra berth windows and pre-book pilot/tug shifts.")
            actions.append("Advance-notify terminal and gate teams to smooth truck and yard peaks.")
        elif pred >= 1.3:
            actions.append("Pre-allocate buffer berth slots and increase watchstanding in VTS for the target window.")
            actions.append("Coordinate with agents to stagger ETAs for vessels with flexible arrival windows.")
        else:
            actions.append("Run normal berth plan but keep one fallback slot for late-arrival clustering.")
        actions.append(f"Use predicted range {lower:.2f}-{upper:.2f} to set staffing floors/ceilings instead of a single-point plan.")
        if spread >= 0.6:
            actions.append("Maintain operational contingency: uncertainty is wide, so add tug/pilot standby margin.")
        if "low" in confidence:
            actions.append("Refresh this forecast 24-48 hours before execution because confidence is currently low.")
        return actions

    answer_text = value.answer.lower()
    if "jump" in answer_text or "anomaly" in answer_text:
        actions.append("Open AIS integrity checks for listed MMSI and validate with external tracking feeds.")
        actions.append("Flag suspicious tracks for VTS review before acting on route deviations.")
    else:
        actions.append("Use the daily/weekly pattern in the chart to plan shift staffing and pilot windows.")
        actions.append("Re-run this query with tighter vessel-type filters for targeted operational planning.")
    return actions


def _serialize_chart(chart: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
    if chart is None or chart.empty:
        return None
    df = chart.reset_index()
    records = []
    for row in df.to_dict(orient="records"):
        item: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                item[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                item[key] = value
        records.append(item)
    return records


def _pick_chart(value: Union[AnalyticsResult, ForecastResult, CarbonResult]) -> Optional[pd.DataFrame]:
    if isinstance(value, AnalyticsResult):
        return value.chart
    if isinstance(value, CarbonResult):
        return value.chart
    if value.forecast is not None and not value.forecast.empty:
        return value.forecast
    if value.history is not None and not value.history.empty:
        return value.history
    return None


def _serialize_result(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence: EvidenceBundle,
    threshold_percentiles: Tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": result.status,
        "answer": result.answer,
        "confidence": extract_confidence_label(result),
        "coverage_notes": result.coverage_notes,
        "caveats": result.caveats,
        "method_steps": _build_method_steps(result),
        "recommendations": _build_port_actions(result),
        "evidence": {
            "computed": _fallback_evidence_from_result(result),
            "retrieved_lines": evidence.lines,
            "retrieved_rows": evidence.rows,
        },
        "chart": _serialize_chart(_pick_chart(result)),
        "retrieval_provenance": evidence.trace,
    }
    if isinstance(result, CarbonResult):
        computed_states = {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}
        is_computed = result.result_state in computed_states
        diagnostics = dict(result.diagnostics or {})
        min_baseline_denominator = 1.0
        try:
            min_baseline_denominator = float(diagnostics.get("min_baseline_denominator_tco2e", 1.0))
        except Exception:
            min_baseline_denominator = 1.0
        payload["carbon"] = {
            "result_state": result.result_state,
            "boundary": result.boundary,
            "pollutants": result.pollutants,
            "source_label": result.source_label,
            "confidence_label": result.confidence_label,
            "confidence_reason": result.confidence_reason,
            "uncertainty_interval": result.uncertainty_interval,
            "params_version": result.params_version,
            "evidence_ids": result.evidence_ids,
            "segment_ids": result.segment_ids,
            "diagnostics": diagnostics,
            "export_csv_path": result.export_csv_path,
            "export_json_path": result.export_json_path,
            "units": {
                "absolute_emissions": "tCO2e (auto-scales to ktCO2e/MtCO2e in UI)",
                "intensity_per_call": "kgCO2e/vessel-call",
                "per_day": "tCO2e/day",
                "per_hour": "kgCO2e/hour",
                "threshold_basis": "relative to this dataset",
            },
            "deterministic_carbon_evidence": _fallback_evidence_from_result(result),
            "retrieved_supporting_traffic_evidence": evidence.lines,
        }
        if not is_computed:
            payload["chart"] = None
            state_reason = {
                CARBON_STATE_NOT_COMPUTABLE: "No deterministic carbon inventory matched the requested scope.",
                CARBON_STATE_RETRIEVAL_ONLY: "Traffic evidence was retrieved, but numeric carbon emissions could not be computed reliably.",
                CARBON_STATE_FORECAST_ONLY: "Carbon forecast was requested, but deterministic carbon forecast outputs are unavailable.",
                CARBON_STATE_UNSUPPORTED: "Carbon query is outside the supported deterministic scope.",
            }.get(result.result_state, "No deterministic carbon output is available for this scope.")
            suggestions = [
                "Improve carbon data coverage for this scope before interpreting emissions numerically.",
                "Add validated fuel/engine/activity factors before interpreting carbon totals.",
                "Use retrieved traffic evidence as context only, not as numeric carbon truth.",
            ]
            payload["carbon"]["availability"] = {
                "computable": False,
                "message": state_reason,
            }
            payload["carbon"]["relative_scale"] = None
            payload["carbon"]["metrics"] = {
                "total_emissions": None,
                "intensity_kgco2e_per_vessel_call": None,
                "tco2e_per_day": None,
                "kgco2e_per_hour": None,
                "relative_level": None,
                "change_vs_baseline": None,
                "change_vs_historical_median": None,
            }
            payload["carbon"]["findings"] = [
                {"type": "status", "text": state_reason},
                {
                    "type": "status",
                    "text": "Retrieved traffic evidence is contextual and does not provide deterministic numeric carbon accounting.",
                }
                if result.result_state == CARBON_STATE_RETRIEVAL_ONLY
                else {"type": "status", "text": "Interpret this response as unavailable rather than low-emission."},
            ]
            payload["carbon"]["emissions_reduction_suggestions"] = suggestions
            payload["recommendations"] = suggestions
            return payload

        metrics = compute_emissions_metrics(result.table, result.boundary)
        total = float(metrics.get("total_tco2e") or 0.0)
        table_metric_col = "wtw_co2e_t" if result.boundary in {"WTW", "TTW_WTW"} else "ttw_co2e_t"
        if result.table is not None and table_metric_col not in result.table.columns:
            table_metric_col = "co2_t" if "co2_t" in result.table.columns else table_metric_col
        hist_values = (
            pd.to_numeric(result.table.get(table_metric_col), errors="coerce").dropna().tolist()
            if result.table is not None and table_metric_col in result.table.columns
            else []
        )
        bands = derive_threshold_bands(hist_values, percentiles=threshold_percentiles)
        level = classify_level(total, bands)
        hist_median = float(np.median(hist_values)) if hist_values else None
        hist_mean = float(np.mean(hist_values)) if hist_values else None
        change_vs_median_pct = safe_percent_delta(
            current_value=total,
            baseline_value=hist_median,
            min_denominator=min_baseline_denominator,
        )
        change_vs_baseline_pct = safe_percent_delta(
            current_value=total,
            baseline_value=hist_mean,
            min_denominator=min_baseline_denominator,
        )
        ci_item = result.uncertainty_interval.get("CO2e") or result.uncertainty_interval.get("CO2") or {}
        point = float(ci_item.get("point", 0.0))
        lower = float(ci_item.get("lower", 0.0))
        upper = float(ci_item.get("upper", 0.0))
        ci_width_rel = ((upper - lower) / point) if point > 0 else None
        chart_df = _pick_chart(result)
        chart_findings = extract_chart_findings(chart_df if chart_df is not None else pd.DataFrame(), target_ts=None, max_findings=5)
        findings = build_emissions_findings(
            current_tco2e=total,
            level=level,
            change_vs_median_pct=change_vs_median_pct,
            source_label=result.source_label,
            ci_width_rel=ci_width_rel,
            chart_findings=chart_findings,
        )
        if change_vs_baseline_pct is None or change_vs_median_pct is None:
            findings.append(
                {
                    "type": "inferred",
                    "text": "Baseline denominator is too small for meaningful percentage comparison in this scope.",
                }
            )
        suggestions = build_reduction_suggestions(
            level=level,
            change_vs_median_pct=change_vs_median_pct,
            ci_width_rel=ci_width_rel,
            source_label=result.source_label,
        )
        payload["carbon"]["availability"] = {"computable": True, "message": "Deterministic carbon inventory computed."}
        payload["carbon"]["relative_scale"] = {
            "classification": level,
            "basis": bands.source_label,
            "thresholds_tco2e": {"p25": bands.p25, "p50": bands.p50, "p75": bands.p75},
            "current_tco2e": total,
            "current_display": format_tco2e(total),
        }
        payload["carbon"]["metrics"] = {
            "total_emissions": format_tco2e(total),
            "intensity_kgco2e_per_vessel_call": (
                f"{float(metrics['intensity_kg_per_call']):.1f} kgCO2e/vessel-call"
                if metrics.get("intensity_kg_per_call") is not None
                else None
            ),
            "tco2e_per_day": (
                f"{float(metrics['tco2e_per_day']):.2f} tCO2e/day"
                if metrics.get("tco2e_per_day") is not None
                else None
            ),
            "kgco2e_per_hour": (
                f"{float(metrics['kgco2e_per_hour']):.2f} kgCO2e/hour"
                if metrics.get("kgco2e_per_hour") is not None
                else None
            ),
            "relative_level": level,
            "change_vs_baseline": format_percent(change_vs_baseline_pct) if change_vs_baseline_pct is not None else None,
            "change_vs_historical_median": (
                format_percent(change_vs_median_pct) if change_vs_median_pct is not None else None
            ),
        }
        payload["carbon"]["findings"] = findings
        payload["carbon"]["emissions_reduction_suggestions"] = suggestions
        payload["recommendations"] = suggestions
    return payload


def _build_state() -> Dict[str, Any]:
    config_path = "config/config.yaml"
    config = load_config(config_path)
    configured_processed_dir = Path(config.get("predict", {}).get("processed_dir", "data/processed"))
    carbon_cfg = config.get("carbon", {})
    threshold_percentiles = sanitize_threshold_percentiles(
        carbon_cfg.get("relative_level_percentiles", [0.25, 0.50, 0.75])
    )
    _maybe_bootstrap_bundle(
        "APP_PROCESSED_BUNDLE_URL",
        configured_processed_dir,
        [
            "arrivals_daily.parquet",
            "arrivals_hourly.parquet",
            "congestion_daily.parquet",
            "dwell_time.parquet",
            "occupancy_hourly.parquet",
            "port_catalog.parquet",
            "kpi_capabilities.json",
        ],
    )
    _maybe_bootstrap_bundle(
        "APP_EVENTS_BUNDLE_URL",
        configured_processed_dir,
        ["events.parquet"],
    )

    requested_vector_mode = str(os.getenv("VECTOR_DB_MODE", config.get("vector_db", {}).get("mode", "local"))).strip().lower()
    try:
        using_remote_vector = chroma_remote_settings(config=config) is not None
    except Exception:
        using_remote_vector = False
    configured_persist_dir = Path(config["paths"].get("persist_dir", "data/chroma"))
    chroma_bootstrap_changed = False
    chroma_bootstrap_message = ""
    if not using_remote_vector:
        chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_runtime(
            configured_persist_dir
        )

    processed_dir, using_demo_processed = _resolve_processed_dir(configured_processed_dir)
    if using_remote_vector:
        persist_dir = configured_persist_dir
        using_demo_chroma = False
    else:
        persist_dir, using_demo_chroma = _resolve_persist_dir(configured_persist_dir)

    # Materialize deterministic tables before the service is shared across
    # worker threads. This bounds first-query latency and prevents lazy native
    # PyArrow reads during Streamlit reruns.
    kpi_engine = KPIQueryEngine(processed_dir=processed_dir).preload()
    forecast_engine = ForecastEngine(processed_dir=processed_dir)
    forecast_engine.kpi = kpi_engine
    carbon_engine = CarbonQueryEngine(
        processed_dir=processed_dir,
        factor_registry_path=carbon_cfg.get("factor_registry_path", "config/carbon_factors.v1.json"),
        monte_carlo_draws=int(carbon_cfg.get("monte_carlo_draws", 500)),
        auto_build=True,
    ).preload()

    retriever = None
    retriever_reason = ""
    enable_model_responses = _runtime_flag("EAGLE_EYE_ENABLE_MODEL_RESPONSES", default=False)
    api_key = _load_runtime_setting("OPENAI_API_KEY") if enable_model_responses else ""
    chat_openai: Optional[OpenAI] = None
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            chat_openai = OpenAI(api_key=api_key)
        except Exception:
            chat_openai = None
    try:
        retriever = _init_retriever(persist_dir=persist_dir, config_path=config_path)
        retriever_reason = (
            f"Retriever active (vector backend: {retriever.vector_backend}; "
            f"query backend: {retriever.query_backend})."
        )
    except Exception as exc:
        retriever_reason = redact_sensitive_text(f"Retriever init failed: {exc}")
        if using_remote_vector:
            chroma_bootstrap_changed, chroma_bootstrap_message = _maybe_bootstrap_chroma_runtime(
                configured_persist_dir
            )
            fallback_persist_dir, fallback_using_demo_chroma = _resolve_persist_dir(
                configured_persist_dir
            )
            if (fallback_persist_dir / "chroma.sqlite3").exists():
                try:
                    retriever = _init_retriever(
                        persist_dir=fallback_persist_dir,
                        config_path=config_path,
                        force_local_vector=True,
                    )
                    persist_dir = fallback_persist_dir
                    using_demo_chroma = fallback_using_demo_chroma
                    retriever_reason = (
                        f"Remote retriever failed ({redact_sensitive_text(exc)}). "
                        f"Fell back to local vector store at {fallback_persist_dir} "
                        f"(vector backend: {retriever.vector_backend}; "
                        f"query backend: {retriever.query_backend})."
                    )
                except Exception as local_exc:
                    retriever_reason = redact_sensitive_text(
                        f"Remote retriever failed ({exc}); local fallback failed ({local_exc})."
                    )

    try:
        local_synthesizer = build_local_synthesizer(config)
        local_synthesizer_reason = (
            f"Configured {local_synthesizer.provider}/{local_synthesizer.model}."
            if local_synthesizer is not None
            else "Local synthesis disabled."
        )
    except Exception as exc:
        local_synthesizer = None
        local_synthesizer_reason = redact_sensitive_text(
            f"Local synthesis configuration failed: {exc}"
        )

    events_path = configured_processed_dir / "events.parquet"
    if not events_path.exists():
        events_path = processed_dir / "events.parquet"
    chat_max_turns = int(config.get("chat", {}).get("max_history_turns", 8))
    model_cfg = config.get("models", {})
    planner_model = _load_runtime_setting("EAGLE_EYE_PLANNER_MODEL") or str(
        model_cfg.get("planner_model", "gpt-5.6-terra")
    )
    general_model = _load_runtime_setting("EAGLE_EYE_GENERAL_MODEL") or str(
        model_cfg.get("general_model", planner_model)
    )
    research_model = _load_runtime_setting("EAGLE_EYE_RESEARCH_MODEL") or str(
        model_cfg.get("research_model", "gpt-5.6-sol")
    )
    reasoning_effort = _load_runtime_setting("EAGLE_EYE_REASONING_EFFORT") or "medium"
    sqlite_path = Path(
        _load_runtime_setting("EAGLE_EYE_SQLITE_PATH") or "data/runtime/eagle_eye.sqlite3"
    )
    conversation_store = ConversationStore(sqlite_path, max_history_turns=chat_max_turns)
    # ETA Watch uses one backend-only AISStream collector. The legacy
    # Fintraffic adapter remains importable for stored-result compatibility
    # and its focused tests, but it is not a public runtime source.
    aisstream_enabled = _runtime_flag("EAGLE_EYE_AISSTREAM_ENABLED", default=True)
    aisstream_collector: Optional[AISStreamCollector] = None
    if aisstream_enabled:
        try:
            aisstream_collector = AISStreamCollector(
                sqlite_path=Path(
                    _load_runtime_setting("EAGLE_EYE_AISSTREAM_SQLITE_PATH")
                    or "data/runtime/aisstream.sqlite3"
                ),
                api_key=_load_runtime_setting("AISSTREAM_API_KEY"),
                history_hours=int(
                    _load_runtime_setting("EAGLE_EYE_AISSTREAM_HISTORY_HOURS")
                    or "24"
                ),
                stale_after_minutes=int(
                    _load_runtime_setting("EAGLE_EYE_AISSTREAM_MAX_AGE_MINUTES")
                    or "10"
                ),
            )
        except (TypeError, ValueError):
            aisstream_collector = None
    planner = QueryPlanner(
        openai_client=chat_openai,
        model=planner_model,
        reasoning_effort=reasoning_effort,
        enable_openai=enable_model_responses,
    )
    query_service = QueryService(
        kpi=kpi_engine,
        forecaster=forecast_engine,
        carbon=carbon_engine,
        conversation_store=conversation_store,
        retriever=retriever,
        retriever_reason=retriever_reason,
        events_path=events_path,
        planner=planner,
        openai_client=chat_openai,
        general_model=general_model,
        research_model=research_model,
        reasoning_effort=reasoning_effort,
        enable_model_responses=enable_model_responses,
        local_synthesizer=local_synthesizer,
        live_eta=aisstream_collector,
        export_dir=_load_runtime_setting("EAGLE_EYE_EXPORT_DIR") or "data/exports",
        processed_dir=processed_dir,
    )
    return {
        "config_path": config_path,
        "threshold_percentiles": threshold_percentiles,
        "processed_dir": str(processed_dir),
        "persist_dir": str(persist_dir),
        "using_demo_processed": using_demo_processed,
        "using_demo_chroma": using_demo_chroma,
        "using_remote_vector": using_remote_vector,
        "requested_vector_mode": requested_vector_mode,
        "chroma_bootstrap_changed": chroma_bootstrap_changed,
        "chroma_bootstrap_message": chroma_bootstrap_message,
        "kpi": kpi_engine,
        "forecast": forecast_engine,
        "carbon": carbon_engine,
        "retriever": retriever,
        "retriever_reason": retriever_reason,
        "retrieval_backend": retriever.retrieval_backend if retriever else None,
        "local_synthesizer": local_synthesizer,
        "local_synthesizer_reason": local_synthesizer_reason,
        "events_path": str(events_path),
        "query_service": query_service,
        "conversation_store": conversation_store,
        "aisstream_collector": aisstream_collector,
        "aisstream_enabled": aisstream_enabled,
        "fintraffic_adapter": None,
        "fintraffic_enabled": False,
        "planner_model": planner_model,
        "general_model": general_model,
        "research_model": research_model,
        "reasoning_effort": reasoning_effort,
        "model_responses_enabled": enable_model_responses,
        "chat_max_turns": chat_max_turns,
    }


@asynccontextmanager
async def _lifespan(application: FastAPI):
    runtime = _build_state()
    application.state.runtime = runtime
    collector = runtime.get("aisstream_collector")
    collector_task: Optional[asyncio.Task[Any]] = None
    if isinstance(collector, AISStreamCollector):
        capabilities = collector.capabilities()
        if capabilities.get("api_key_configured"):
            collector_task = asyncio.create_task(
                collector.run(),
                name="eagle-eye-aisstream-collector",
            )
            runtime["aisstream_task"] = collector_task
    try:
        yield
    finally:
        if isinstance(collector, AISStreamCollector):
            await collector.stop()
        if collector_task is not None and not collector_task.done():
            collector_task.cancel()
        if collector_task is not None:
            with suppress(asyncio.CancelledError):
                await collector_task


app = FastAPI(title="Eagle Eye API", version="2.0.0", lifespan=_lifespan)


def _runtime_state() -> Dict[str, Any]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime = _build_state()
        app.state.runtime = runtime
    return runtime


def _web_dist_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "web" / "dist"


def _compat_result(envelope: AnswerEnvelope) -> Dict[str, Any]:
    status = (
        "ok"
        if envelope.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}
        else "unsupported"
        if envelope.state == AnswerState.UNSUPPORTED
        else "no_data"
    )
    chart_dataset = next((item for item in envelope.datasets if item.id == "chart"), None)
    caveats = list(envelope.caveats)
    if status != "ok" and envelope.answer not in caveats:
        caveats.append(envelope.answer)
    return {
        "status": status,
        "answer": envelope.answer,
        "confidence": envelope.confidence,
        "coverage_notes": [envelope.freshness.message],
        "caveats": caveats,
        "method_steps": [
            f"Canonical route: {envelope.mode.value}.",
            f"Canonical operation: {envelope.plan.operation.value}.",
        ],
        "recommendations": [],
        "evidence": {
            "computed": [fact.model_dump(mode="json") for fact in envelope.facts],
            "retrieved_lines": [item.excerpt for item in envelope.evidence if item.excerpt],
            "retrieved_rows": [item.model_dump(mode="json") for item in envelope.evidence],
        },
        "chart": chart_dataset.rows if chart_dataset else None,
        "retrieval_provenance": {
            "trace_id": envelope.trace.trace_id,
            "planner_source": envelope.trace.planner_source,
        },
        "canonical": envelope.model_dump(mode="json"),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    state = _runtime_state()
    return {
        "status": "ok",
        "processed_dir": state["processed_dir"],
        "persist_dir": state["persist_dir"],
        "using_demo_processed": state["using_demo_processed"],
        "using_demo_chroma": state["using_demo_chroma"],
        "using_remote_vector": state["using_remote_vector"],
        "requested_vector_mode": state["requested_vector_mode"],
        "chroma_bootstrap_changed": state["chroma_bootstrap_changed"],
        "chroma_bootstrap_message": state["chroma_bootstrap_message"],
        "retriever_reason": state["retriever_reason"],
        "retrieval_backend": state["retrieval_backend"],
        "local_synthesizer_reason": state["local_synthesizer_reason"],
        "events_available": bool(Path(state["events_path"]).exists()),
        "carbon_available": bool(state["carbon"].available),
        "carbon_params_version": state["carbon"].params_version.get("version"),
        "planner_model": state["planner_model"],
        "general_model": state["general_model"],
        "research_model": state["research_model"],
        "reasoning_effort": state["reasoning_effort"],
        "model_responses_enabled": state["model_responses_enabled"],
        "conversation_store": "sqlite",
        "conversation_count": state["conversation_store"].count_conversations(),
        "fintraffic_enabled": state.get("fintraffic_enabled", False),
        "fintraffic_available": state.get("fintraffic_adapter") is not None,
    }


@app.get("/")
def root() -> Any:
    index = _web_dist_dir() / "index.html"
    if index.exists():
        return FileResponse(index)
    state = _runtime_state()
    return {
        "name": "Eagle Eye API",
        "docs": "/docs",
        "health": "/health",
        "query_v2": "/api/v2/query",
        "query_stream_v2": "/api/v2/query/stream",
        "capabilities_v2": "/api/v2/capabilities",
        "exports_v2": "/api/v2/exports",
        "feedback_v2": "/api/v2/feedback",
        "ask": "/ask",
        "chat": "/api/v1/chat",
        "carbon_ports": "/api/v1/carbon/ports/{port_id}/emissions",
        "carbon_call": "/api/v1/carbon/vessels/{mmsi}/calls/{call_id}",
        "carbon_estimate": "/api/v1/carbon/estimate",
        "carbon_evidence": "/api/v1/carbon/evidence/{evidence_id}",
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    state = _runtime_state()
    envelope = state["query_service"].query(
        QueryRequest(
            question=req.question,
            top_k_evidence=req.top_k_evidence,
            filters=QueryFiltersPayload.model_validate(req.filters.model_dump()),
        )
    )
    return {
        "question": envelope.question,
        "intent": envelope.plan.model_dump(mode="json"),
        "result": _compat_result(envelope),
    }


@app.post("/api/v1/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    state = _runtime_state()
    envelope = state["query_service"].query(
        QueryRequest(
            question=req.message,
            conversation_id=req.conversation_id,
            top_k_evidence=req.top_k_evidence,
            filters=QueryFiltersPayload.model_validate(req.filters.model_dump()),
        )
    )
    return {
        "conversation_id": envelope.conversation_id,
        "turn_id": envelope.turn_id,
        "message": envelope.question,
        "answer": envelope.answer,
        "result_state": envelope.state.value,
        "source_type": "Computed" if envelope.state in {AnswerState.COMPUTED, AnswerState.PARTIAL} else "Retrieved" if envelope.state == AnswerState.RETRIEVED else "System",
        "confidence": envelope.confidence,
        "assumptions_used": envelope.caveats,
        "evidence": {
            "lines": [item.excerpt for item in envelope.evidence if item.excerpt],
            "rows": [item.model_dump(mode="json") for item in envelope.evidence],
        },
        "tool_trace": envelope.trace.model_dump(mode="json"),
        "intent": envelope.plan.model_dump(mode="json"),
        "deterministic_result": _compat_result(envelope),
    }


@app.post("/api/v2/query", response_model=AnswerEnvelope)
def query_v2(req: QueryRequest) -> AnswerEnvelope:
    return _runtime_state()["query_service"].query(req)


@app.post("/api/v2/query/stream")
def query_stream_v2(req: QueryRequest) -> StreamingResponse:
    def events():
        yield "event: progress\ndata: " + json.dumps({"stage": "accepted"}) + "\n\n"
        yield "event: progress\ndata: " + json.dumps({"stage": "executing"}) + "\n\n"
        envelope = _runtime_state()["query_service"].query(req)
        yield "event: text\ndata: " + json.dumps({"delta": envelope.answer}) + "\n\n"
        yield "event: final\ndata: " + envelope.model_dump_json() + "\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v2/capabilities")
def capabilities_v2() -> Dict[str, Any]:
    return _runtime_state()["query_service"].capabilities()


@app.post("/api/v2/exports", response_model=ExportResponse)
def exports_v2(req: ExportRequest) -> ExportResponse:
    try:
        return _runtime_state()["query_service"].export(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=redact_sensitive_text(exc)) from exc


@app.post("/api/v2/feedback", response_model=FeedbackResponse, status_code=202)
def feedback_v2(req: FeedbackRequest) -> FeedbackResponse:
    return _runtime_state()["query_service"].submit_feedback(req)
















def _parse_pollutants_query(value: Optional[str]) -> List[str]:
    if not value:
        return ["CO2e", "NOx", "SOx", "PM"]
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    return items or ["CO2e", "NOx", "SOx", "PM"]


@app.get("/api/v1/carbon/ports/{port_id}/emissions")
def carbon_port_emissions(
    port_id: str,
    from_date: Optional[str] = Query(default=None, alias="from"),
    to_date: Optional[str] = Query(default=None, alias="to"),
    group_by: str = Query(default="day"),
    boundary: str = Query(default="TTW"),
    pollutants: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    result = engine.query_port_emissions(
        port_id=port_id,
        date_from=from_date,
        date_to=to_date,
        group_by=group_by,
        boundary=boundary,
        pollutants=_parse_pollutants_query(pollutants),
        include_uncertainty=True,
        include_evidence=True,
    )
    return {
        "port_id": port_id,
        "result": _serialize_result(
            result,
            EvidenceBundle(lines=[], rows=[], trace={}),
            threshold_percentiles=state["threshold_percentiles"],
        ),
    }


@app.get("/api/v1/carbon/vessels/{mmsi}/calls/{call_id}")
def carbon_vessel_call(
    mmsi: str,
    call_id: str,
    boundary: str = Query(default="TTW"),
    pollutants: Optional[str] = Query(default=None),
    include_uncertainty: bool = Query(default=True),
    include_evidence: bool = Query(default=True),
) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    result = engine.query_vessel_call(
        mmsi=mmsi,
        call_id=call_id,
        boundary=boundary,
        pollutants=_parse_pollutants_query(pollutants),
        include_uncertainty=include_uncertainty,
        include_evidence=include_evidence,
    )
    return {
        "mmsi": mmsi,
        "call_id": call_id,
        "result": _serialize_result(
            result,
            EvidenceBundle(lines=[], rows=[], trace={}),
            threshold_percentiles=state["threshold_percentiles"],
        ),
    }


@app.post("/api/v1/carbon/estimate")
def carbon_estimate(req: CarbonEstimateRequest) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    result = engine.estimate_with_assumptions(req.model_dump())
    return {
        "result": _serialize_result(
            result,
            EvidenceBundle(lines=[], rows=[], trace={}),
            threshold_percentiles=state["threshold_percentiles"],
        )
    }


@app.get("/api/v1/carbon/evidence/{evidence_id}")
def carbon_evidence(evidence_id: str) -> Dict[str, Any]:
    state = _runtime_state()
    engine: CarbonQueryEngine = state["carbon"]
    payload = engine.get_evidence(evidence_id)
    if payload.get("status") != "ok":
        raise HTTPException(status_code=404, detail=payload.get("reason", "Evidence not found"))
    return payload


# Register the commercial build only after every API/documentation route so an
# SPA fallback can never mask an API 404 or intercept OpenAPI.
_DIST_DIR = _web_dist_dir()
if (_DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST_DIR / "assets"), name="web-assets")


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    include_in_schema=False,
)
def web_spa_fallback(full_path: str) -> Any:
    first_segment = full_path.split("/", 1)[0].lower()
    if first_segment in {"api", "docs", "redoc", "health", "openapi.json"}:
        raise HTTPException(status_code=404, detail="Not found")
    dist = _web_dist_dir().resolve()
    if not dist.is_dir():
        raise HTTPException(status_code=404, detail="Commercial web build is not installed")
    candidate = (dist / full_path).resolve()
    try:
        candidate.relative_to(dist)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    if candidate.is_file():
        return FileResponse(candidate)
    index = dist / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Commercial web build is incomplete")
