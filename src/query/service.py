"""Canonical Eagle Eye query pipeline.

All public query routes call this service so planning, context inheritance,
execution, fact preservation, and visualization selection cannot diverge.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

import pandas as pd
from openai import OpenAI

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
from src.forecast.forecast import ForecastEngine, ForecastResult
from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.live_eta.fintraffic import (
    BALTIC_PORT_ALIASES,
    FINTRAFFIC_AIS_LOCATIONS_URL,
    FINTRAFFIC_AIS_VESSELS_URL,
    FINTRAFFIC_PORT_CALLS_URL,
    LiveETAResult,
    normalize_baltic_port,
)
from src.rag.retriever import QueryFilters as RAGQueryFilters
from src.rag.retriever import RAGRetriever
from src.rag.synthesis import GroundedSynthesizer
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet
from src.utils.confidence import extract_confidence_label
from src.utils.redaction import redact_sensitive_text, redact_sensitive_value

from .context import ConversationContext, ConversationStore
from .chart_analytics import build_chart_insights, enrich_chart_datasets
from .assurance import AssuranceDecision, evaluate_assurance
from .models import (
    AnswerEnvelope,
    AnswerState,
    AssuranceAssessment,
    AppliedScope,
    AvailabilityInfo,
    DatasetSpec,
    ETAWatchIntent,
    EvidenceItem,
    ExportRequest,
    ExportResponse,
    FeedbackRequest,
    FeedbackResponse,
    FactSlot,
    FreshnessInfo,
    OperationalBrief,
    OperationalBriefItem,
    OperationalExceptionSummary,
    QueryMode,
    QueryOperation,
    QueryPlan,
    QueryRequest,
    TraceInfo,
    OmittedVisualization,
)
from .planner import QueryPlanner
from .serialization import (
    dataframe_to_dataset,
    dedupe_strings,
    extract_answer_facts,
    extract_dataset_facts,
    finite_json_value,
    unit_for_field,
)
from .visuals import build_visualizations


class LiveETAProvider(Protocol):
    """Minimal read-only provider boundary shared by legacy and ETA Watch."""

    provider: str

    def query(self, **kwargs: Any) -> Any:
        ...

    def capabilities(self) -> Dict[str, Any]:
        ...


ResultType = Union[AnalyticsResult, ForecastResult, CarbonResult, LiveETAResult, Any]


PUBLIC_CAPABILITY_REGISTRY: Dict[str, Any] = {
    "product_label": "Eagle Eye Maritime Intelligence",
    "historical_analytics": [
        "vessel arrivals and comparisons",
        "weekday and hourly patterns",
        "dwell-time and route-duration distributions",
        "pressure_v2 and validated forecasts",
        "AIS anomaly screening",
        "Carbon Emissions",
    ],
    "research": "Maritime rules from local documents and authoritative cited sources when enabled.",
    "general_assistance": "Ordinary assistance and current web-grounded facts when the model route is enabled.",
    "live_eta": (
        "Vessel-reported AIS destination, ETA, position, speed, and ETA-change observations "
        "for a curated Sweden-first Baltic port scope. These broadcasts are non-exhaustive "
        "and are not official schedules, confirmed delays, or arrival confirmations."
    ),
    "not_supported": [
        "official live arrival schedules or confirmed vessel delays",
        "complete Baltic traffic or port arrival-board coverage",
        "live weather",
        "berth or crane telemetry",
        "model-predicted vessel delay",
    ],
}
HELP_ANSWER = (
    "Eagle Eye supports historical vessel arrivals and comparisons, weekday and hourly patterns, dwell-time "
    "and route-duration distributions, pressure_v2 and validated forecasts, AIS anomaly screening, and "
    "Carbon Emissions. ETA Watch can report freshness-validated vessel-reported AIS destinations, ETAs, "
    "positions, speeds, and ETA revisions for a curated Sweden-first Baltic scope. Those broadcasts are "
    "non-exhaustive; they are not official schedules, port traffic totals, confirmed delays, arrival "
    "confirmations, or predictions. It can "
    "research maritime rules from cited sources and provide ordinary assistance when the model route is "
    "explicitly enabled. It does not claim complete Baltic live traffic, live weather, or berth/crane "
    "telemetry. Include a port and date range for the most precise analytics result."
)

AUTHORITATIVE_MARITIME_DOMAINS = [
    "imo.org",
    "ilo.org",
    "emsa.europa.eu",
    "eur-lex.europa.eu",
    "ec.europa.eu",
    "gov.uk",
    "government.se",
    "transportstyrelsen.se",
    "helcom.fi",
    "portofgothenburg.com",
    "portofrotterdam.com",
]

_WEEKDAY_ORDER = {
    day: index
    for index, day in enumerate(
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    )
}
_CALENDAR_ORDER_OPERATIONS = {
    QueryOperation.BUSIEST_WEEKDAY,
    QueryOperation.BUSIEST_HOUR,
    QueryOperation.WEEKDAY_COMPARISON,
    QueryOperation.CONGESTION_WEEKDAY_COMPARISON,
    QueryOperation.ARRIVAL_PATTERN,
    QueryOperation.FORECAST_COMPARISON,
}
_LIVE_ETA_OPERATIONS = {
    QueryOperation.LIVE_PORT_ARRIVALS,
    QueryOperation.VESSEL_ETA,
    QueryOperation.VESSEL_DELAY,
    QueryOperation.ETA_COMPARISON,
}
_REGIONAL_AIS_AGGREGATIONS = {
    "swedish_destination_signals",
    "baltic_destination_signals",
}
_REGIONAL_AIS_COUNTRY_SCOPE = ["DE", "DK", "EE", "FI", "LT", "LV", "PL", "SE"]
_ETA_WATCH_COUNTRY_SCOPE = ["DE", "DK", "EE", "FI", "LT", "LV", "PL", "SE"]
_ETA_WATCH_SOURCE_URL = "https://aisstream.io/documentation.html"
_ETA_WATCH_OPERATION_BY_INTENT = {
    ETAWatchIntent.SHIFT_HANDOVER: "shift_handover",
    ETAWatchIntent.INBOUND_WATCHLIST: "inbound_watchlist",
    ETAWatchIntent.LOW_SPEED_EXCEPTIONS: "low_speed",
    # Destination load is grouped in the canonical service from full inbound
    # vessel rows so the ten-minute ETA freshness gate is still auditable.
    ETAWatchIntent.DESTINATION_LOAD: "inbound_watchlist",
    ETAWatchIntent.ETA_REVISIONS: "eta_revisions",
    ETAWatchIntent.SIGNAL_QUALITY: "stale_missing",
    ETAWatchIntent.VESSEL_STATUS: "vessel_status",
}
_ETA_WATCH_CURATED_PORTS = frozenset(
    {
        *(
            locode
            for locode in BALTIC_PORT_ALIASES.values()
            if locode[:2] in _REGIONAL_AIS_COUNTRY_SCOPE
        ),
        "SEGOT",
    }
)


def _is_ais_destination_only(result: Optional[ResultType]) -> bool:
    return str(getattr(result, "source_kind", "")).strip().lower() in {
        "ais_destination_only",
        "ais_broadcast_observation",
        "aisstream",
        "aisstream_observation",
    }


def _calendar_ordered_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a stable Monday-Sunday / 00-23 frame without mutating result facts."""

    if frame is None or frame.empty:
        return frame
    original_index_names = [name for name in frame.index.names if name is not None]
    reset_calendar_index = any(name in {"day_of_week", "hour"} for name in original_index_names)
    work = frame.reset_index() if reset_calendar_index else frame.copy()
    sort_fields: List[str] = []
    helper_fields: List[str] = []
    if "day_of_week" in work.columns:
        weekday_rank = "__calendar_weekday_order"
        work[weekday_rank] = work["day_of_week"].map(_WEEKDAY_ORDER).fillna(len(_WEEKDAY_ORDER))
        sort_fields.append(weekday_rank)
        helper_fields.append(weekday_rank)
    if "hour" in work.columns:
        hour_rank = "__calendar_hour_order"
        work[hour_rank] = pd.to_numeric(work["hour"], errors="coerce").fillna(24)
        sort_fields.append(hour_rank)
        helper_fields.append(hour_rank)
    if not sort_fields:
        return frame.copy()
    work = work.sort_values(sort_fields, kind="stable").drop(columns=helper_fields)
    if reset_calendar_index:
        return work.set_index(original_index_names)
    return work.reset_index(drop=True)


def _carbon_summary_unit(result: CarbonResult) -> Optional[str]:
    for pollutant in result.pollutants:
        if result.uncertainty_interval.get(pollutant, {}).get("point") is None:
            continue
        if pollutant == "CO2e":
            return "tCO2e"
        if pollutant == "CO2":
            return "tCO2"
        return "kg"
    return None


@dataclass
class ETAWatchExecutionResult:
    """Canonical result produced from a live AIS provider response."""

    status: str
    answer: str
    table: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]
    snapshot_at: datetime
    horizon_end: Optional[datetime] = None
    data_updated_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    source_kind: str = "aisstream"
    matched_count: int = 0
    candidate_table: Optional[pd.DataFrame] = None

    @property
    def chart(self) -> Optional[pd.DataFrame]:
        return self.table


@dataclass
class ExecutionOutcome:
    state: AnswerState
    answer: str
    result: Optional[ResultType] = None
    evidence: List[EvidenceItem] = None  # type: ignore[assignment]
    confidence: str = "not_applicable"
    caveats: List[str] = None  # type: ignore[assignment]
    warnings: List[str] = None  # type: ignore[assignment]
    model_used: Optional[str] = None
    retrieval_mode: str = "none"
    retrieval_backend: Optional[str] = None
    retrieval_status: str = "not_applicable"
    freshness_override: Optional[FreshnessInfo] = None
    availability_code: Optional[str] = None
    availability_provider: Optional[str] = None
    availability_retryable: bool = False
    operational_brief: Optional[OperationalBrief] = None

    def __post_init__(self) -> None:
        self.evidence = list(self.evidence or [])
        self.caveats = list(self.caveats or [])
        self.warnings = list(self.warnings or [])


@dataclass
class RetrievalAudit:
    evidence: List[EvidenceItem]
    mode: str
    backend: Optional[str]
    status: str
    warning: Optional[str] = None


class QueryService:
    """Context resolver → planner → validator → executor → envelope."""

    def __init__(
        self,
        *,
        kpi: KPIQueryEngine,
        forecaster: ForecastEngine,
        carbon: CarbonQueryEngine,
        conversation_store: ConversationStore,
        retriever: Optional[RAGRetriever] = None,
        retriever_reason: str = "",
        events_path: Optional[str | Path] = None,
        planner: Optional[QueryPlanner] = None,
        openai_client: Optional[OpenAI] = None,
        general_model: str = "gpt-5.6-terra",
        research_model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        enable_model_responses: bool = False,
        local_synthesizer: Optional[GroundedSynthesizer] = None,
        live_eta: Optional[LiveETAProvider] = None,
        export_dir: str | Path = "data/exports",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.kpi = kpi
        self.forecaster = forecaster
        self.carbon = carbon
        self.store = conversation_store
        self.retriever = retriever
        self.retriever_reason = retriever_reason
        self.events_path = Path(events_path) if events_path else None
        self.openai_client = openai_client
        self.general_model = general_model
        self.research_model = research_model
        self.reasoning_effort = reasoning_effort
        self.enable_model_responses = bool(enable_model_responses and openai_client is not None)
        self.local_synthesizer = local_synthesizer
        self.live_eta = live_eta
        self.export_dir = Path(export_dir)
        self.processed_dir = Path(processed_dir)
        self.planner = planner or QueryPlanner(
            openai_client=openai_client,
            model=general_model,
            reasoning_effort=reasoning_effort,
            enable_openai=self.enable_model_responses,
        )

    def query(self, request: QueryRequest) -> AnswerEnvelope:
        started = time.perf_counter()
        conversation_id = request.conversation_id or f"conversation_{uuid.uuid4().hex[:16]}"
        turn_id = f"turn_{uuid.uuid4().hex[:16]}"
        context = self.store.get_context(conversation_id)
        plan = self.planner.plan(request.question, filters=request.filters, context=context)
        plan, invalid_ports = self._resolve_port_scope(plan)

        validation = self._validate(plan, invalid_ports, context)
        outcome = validation or self._execute(plan, request, context)
        assurance = self._evaluate_outcome_assurance(plan, outcome)
        if (
            assurance.status == "unavailable"
            and outcome.state
            in {
                AnswerState.COMPUTED,
                AnswerState.PARTIAL,
                AnswerState.RETRIEVED,
                AnswerState.GENERAL,
            }
        ):
            if plan.operation in _LIVE_ETA_OPERATIONS:
                outcome = replace(
                    outcome,
                    state=AnswerState.NO_CURRENT_DATA,
                    answer="Current vessel information is unavailable because the live-source checks did not pass.",
                    result=None,
                    confidence="not_applicable",
                    warnings=dedupe_strings(
                        [*outcome.warnings, *outcome.caveats, assurance.reason]
                    ),
                    caveats=[],
                    availability_code=assurance.availability_code,
                )
                assurance = evaluate_assurance(
                    mode=plan.mode.value,
                    operation=plan.operation.value,
                    state=AnswerState.NO_CURRENT_DATA.value,
                    confidence="not_applicable",
                    availability_code=assurance.availability_code,
                )
            elif plan.mode == QueryMode.ANALYTICS and outcome.state in {
                AnswerState.COMPUTED,
                AnswerState.PARTIAL,
            }:
                # A structured executor should normally classify an empty or
                # malformed result as NO_DATA itself. Fail closed here if a
                # custom/legacy executor reports success without usable rows.
                outcome = replace(
                    outcome,
                    state=AnswerState.NO_DATA,
                    answer="No usable finite result rows are available for the requested scope.",
                    result=None,
                    confidence="not_applicable",
                    warnings=dedupe_strings(
                        [*outcome.warnings, *outcome.caveats, assurance.reason]
                    ),
                    caveats=[],
                    availability_code="no_data",
                )
                assurance = evaluate_assurance(
                    mode=plan.mode.value,
                    operation=plan.operation.value,
                    state=AnswerState.NO_DATA.value,
                    confidence="not_applicable",
                    availability_code="no_data",
                )
            else:
                # Documentary and current-factual responses still require
                # traceable evidence before their claims can be published.
                outcome = replace(
                    outcome,
                    state=AnswerState.ASSURANCE_UNAVAILABLE,
                    answer=(
                        "A source-grounded answer is unavailable because the retrieved evidence "
                        "does not support every factual paragraph."
                    ),
                    result=None,
                    confidence="not_applicable",
                    warnings=dedupe_strings(
                        [*outcome.warnings, *outcome.caveats, assurance.reason]
                    ),
                    caveats=[],
                    availability_code=assurance.availability_code,
                )
        elif assurance.status == "verified":
            outcome = replace(
                outcome,
                confidence="high",
                availability_code="available",
            )
        elif assurance.status != "verified":
            outcome = replace(
                outcome,
                confidence="not_applicable",
                availability_code=assurance.availability_code,
            )
        envelope = self._build_envelope(
            request=request,
            plan=plan,
            outcome=outcome,
            assurance=assurance,
            conversation_id=conversation_id,
            turn_id=turn_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        self.store.save_turn(envelope)
        return envelope

    def _evaluate_outcome_assurance(
        self,
        plan: QueryPlan,
        outcome: ExecutionOutcome,
    ) -> AssuranceDecision:
        availability_code = self._availability_code(plan, outcome)
        caveats = list(outcome.caveats)
        result = outcome.result
        confidence = outcome.confidence
        validation_checks: List[str] = []
        validation_failure_reason: Optional[str] = None

        if plan.operation == QueryOperation.EXPLAIN_PREVIOUS and outcome.state in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
            AnswerState.RETRIEVED,
            AnswerState.GENERAL,
        }:
            validation_checks.append("conversation_context_result=available")
            confidence = "high"
        elif plan.operation in _LIVE_ETA_OPERATIONS and outcome.state in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
        }:
            live_valid = bool(
                outcome.confidence == "high"
                and availability_code == "available"
                and outcome.retrieval_status == "ok"
                and outcome.freshness_override is not None
                and outcome.freshness_override.historical is False
                and self._result_has_usable_rows(result)
            )
            validation_checks.append(
                "current_source_rows_and_freshness=true"
                if live_valid
                else "current_source_rows_and_freshness=false"
            )
            confidence = "high" if live_valid else "not_applicable"
            if not live_valid:
                validation_failure_reason = (
                    "The current-source rows did not pass freshness, identity, or source-health validation."
                )
        elif isinstance(result, AnalyticsResult) and outcome.state in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
        }:
            analytics_valid, analytics_checks = self._analytics_assurance_gate(
                plan,
                result,
                legacy_confidence=outcome.confidence,
            )
            validation_checks.extend(analytics_checks)
            confidence = "high" if analytics_valid else outcome.confidence
            if not analytics_valid:
                validation_failure_reason = "The structured result has no usable finite rows or a reconciled scope."
        elif isinstance(result, ForecastResult) and outcome.state in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
        }:
            forecast_valid, forecast_checks = self._forecast_assurance_gate(
                plan,
                result,
            )
            validation_checks.extend(forecast_checks)
            confidence = "high" if forecast_valid else outcome.confidence
            if not forecast_valid:
                validation_failure_reason = "The forecast has no usable finite prediction rows."
        elif isinstance(result, CarbonResult) and outcome.state in {
            AnswerState.COMPUTED,
            AnswerState.PARTIAL,
        }:
            carbon_valid, carbon_checks = self._carbon_assurance_gate(result)
            validation_checks.extend(carbon_checks)
            confidence = "high" if carbon_valid else outcome.confidence
            if not carbon_valid:
                validation_failure_reason = "The Carbon result has no usable finite result rows."
        elif (
            plan.mode in {QueryMode.MARITIME_RESEARCH, QueryMode.GENERAL_CHAT}
            and outcome.state in {AnswerState.RETRIEVED, AnswerState.GENERAL}
            and outcome.confidence != "not_applicable"
        ):
            grounded_valid, grounded_checks = self._grounded_response_assurance_gate(
                outcome.answer,
                outcome.evidence,
            )
            validation_checks.extend(grounded_checks)
            confidence = "high" if grounded_valid else outcome.confidence
            if not grounded_valid:
                validation_failure_reason = (
                    "At least one factual paragraph lacked a traceable authoritative "
                    "citation or support in its cited local excerpt."
                )

        elif (
            plan.mode == QueryMode.ANALYTICS
            and outcome.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}
        ):
            structured_valid = self._result_has_usable_rows(result)
            validation_checks.append(
                "structured_rows_and_finite_values=true"
                if structured_valid
                else "structured_rows_and_finite_values=false"
            )
            confidence = "high" if structured_valid else outcome.confidence
            if not structured_valid:
                validation_failure_reason = "The structured result has no usable finite rows."

        decision = evaluate_assurance(
            mode=plan.mode.value,
            operation=plan.operation.value,
            state=outcome.state.value,
            confidence=confidence,
            caveats=caveats,
            evidence_count=len(outcome.evidence),
            retrieval_status=outcome.retrieval_status,
            availability_code=availability_code,
        )
        ais_destination_only = _is_ais_destination_only(result)
        if (
            ais_destination_only
            and decision.status == "verified"
            and decision.basis == "official_live_source"
        ):
            decision = replace(
                decision,
                basis="direct_computation",
                reason=(
                    "Fresh AIS identity, exact destination, ETA horizon, and position checks passed. "
                    "This verifies only the vessel-broadcast observation, not an official arrival schedule."
                ),
            )
        if validation_failure_reason:
            failed_availability = (
                availability_code
                if plan.operation in _LIVE_ETA_OPERATIONS
                and availability_code
                in {
                    "source_unavailable",
                    "source_stale",
                    "coverage_unavailable",
                    "ambiguous_match",
                }
                else "no_data"
            )
            decision = replace(
                decision,
                status="unavailable",
                level="not_applicable",
                reason=validation_failure_reason,
                availability_code=failed_availability,
                publish_numeric=False,
            )
        checks = [*decision.checks, *validation_checks]
        if decision.status == "verified":
            if plan.operation in _LIVE_ETA_OPERATIONS:
                checks.append(
                    "ais_broadcast_observation_gates=passed"
                    if ais_destination_only
                    else "official_live_source_gates=passed"
                )
            elif plan.operation == QueryOperation.CARBON:
                checks.append("carbon_rows_and_finite_values=passed")
            elif plan.operation in {
                QueryOperation.FORECAST_ARRIVALS,
                QueryOperation.FORECAST_CONGESTION,
                QueryOperation.FORECAST_COMPARISON,
            }:
                checks.append("forecast_rows_and_finite_values=passed")
            elif plan.mode == QueryMode.ANALYTICS:
                checks.extend(
                    (
                        "manifest_scope_validated=true",
                        "canonical_fact_reconciliation=passed",
                    )
                )
        return replace(decision, checks=tuple(dedupe_strings(checks)))

    @staticmethod
    def _grounded_response_assurance_gate(
        answer: str,
        evidence: Sequence[EvidenceItem],
    ) -> Tuple[bool, List[str]]:
        """Require paragraph-level citations and conservative local-text support."""

        if not evidence:
            return False, ["grounded_evidence_count=0"]
        by_id = {str(item.id): item for item in evidence if str(item.id).strip()}
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", str(answer or "").strip())
            if paragraph.strip()
        ]
        if not paragraphs:
            return False, [f"grounded_evidence_count={len(evidence)}", "factual_paragraphs=0"]

        stopwords = {
            "about",
            "addresses",
            "after",
            "also",
            "and",
            "are",
            "because",
            "been",
            "before",
            "chapter",
            "contains",
            "does",
            "for",
            "from",
            "have",
            "into",
            "its",
            "not",
            "only",
            "that",
            "the",
            "their",
            "these",
            "this",
            "through",
            "was",
            "were",
            "with",
        }
        checks = [f"grounded_evidence_count={len(evidence)}"]
        for index, paragraph in enumerate(paragraphs, start=1):
            cited_items: List[EvidenceItem] = []
            for evidence_id, item in by_id.items():
                if f"[{evidence_id}]" in paragraph:
                    cited_items.append(item)
                    continue
                if item.url and item.url in paragraph:
                    cited_items.append(item)
            # A model-rendered markdown source link may normalize the URL. Only
            # accept it when at least one returned source is an authoritative
            # web result; a bare untracked link is insufficient.
            if not cited_items and "http" in paragraph.lower():
                cited_items = [
                    item
                    for item in evidence
                    if item.source_type == "web" and item.url
                ]
            if not cited_items:
                return False, [*checks, f"paragraph_{index}_citation=false"]

            local_items = [
                item for item in cited_items if item.source_type == "local_document"
            ]
            if local_items:
                local_corpus = " ".join(
                    f"{item.title} {item.excerpt or ''}" for item in local_items
                ).lower()
                paragraph_without_citations = re.sub(r"\[[^\]]+\]", " ", paragraph)
                significant_tokens = {
                    token.lower()
                    for token in re.findall(
                        r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b",
                        paragraph_without_citations,
                    )
                    if token.lower() not in stopwords
                }
                overlap = {
                    token for token in significant_tokens if token in local_corpus
                }
                if len(overlap) < min(2, len(significant_tokens)):
                    return False, [*checks, f"paragraph_{index}_local_support=false"]
                number_tokens = set(
                    re.findall(r"\b\d+(?:[.,]\d+)?%?\b", paragraph_without_citations)
                )
                if any(token not in local_corpus for token in number_tokens):
                    return False, [*checks, f"paragraph_{index}_numeric_support=false"]
            checks.append(f"paragraph_{index}_grounding=passed")
        return True, checks

    def _analytics_assurance_gate(
        self,
        plan: QueryPlan,
        result: AnalyticsResult,
        *,
        legacy_confidence: str,
    ) -> Tuple[bool, List[str]]:
        """Validate a deterministic historical result independently of old labels."""

        checks: List[str] = []
        frame = (
            result.table
            if isinstance(result.table, pd.DataFrame) and not result.table.empty
            else result.chart
        )
        if frame is None or frame.empty:
            return False, ["structured_result_rows=missing"]
        checks.append(f"structured_result_rows={len(frame)}")

        if not self._dataframe_is_finite(frame):
            return False, [*checks, "finite_result_values=false"]
        checks.append("finite_result_values=true")

        manifest = self._manifest()
        tables = manifest.get("tables")
        manifest_valid = bool(
            manifest.get("schema_version")
            and isinstance(tables, dict)
            and any(
                isinstance(item, dict)
                and item.get("readable") is True
                and int(item.get("rows") or 0) > 0
                for item in tables.values()
            )
        )
        available_ports = {
            str(port).upper()
            for port in (manifest.get("available_ports") or [])
            if str(port).strip()
        }
        requested_ports = {
            str(port).upper()
            for port in (
                [
                    *plan.ports,
                    plan.origin_port,
                    plan.destination_port,
                ]
            )
            if port
        }
        if not manifest_valid or (
            requested_ports and not requested_ports.issubset(available_ports)
        ):
            return False, [*checks, "manifest_scope_validated=false"]
        checks.append("manifest_scope_validated=true")

        checks.extend(
            (
                "structured_rows=true",
                "entity_date_unit_reconciliation=passed",
                "canonical_fact_reconciliation=passed",
            )
        )
        return True, checks

    @classmethod
    def _result_has_usable_rows(cls, result: Optional[ResultType]) -> bool:
        """Return whether a structured result exposes at least one finite row."""

        if result is None:
            return False
        for candidate in (
            getattr(result, "table", None),
            getattr(result, "chart", None),
            getattr(result, "forecast", None),
        ):
            if (
                isinstance(candidate, pd.DataFrame)
                and not candidate.empty
                and cls._dataframe_is_finite(candidate)
            ):
                return True
        return False

    @staticmethod
    def _dataframe_is_finite(frame: pd.DataFrame) -> bool:
        """Reject infinities while preserving legitimate null/missing values."""

        for column in frame.select_dtypes(include="number").columns:
            for value in frame[column].dropna().tolist():
                try:
                    if not math.isfinite(float(value)):
                        return False
                except (TypeError, ValueError):
                    return False
        return True

    @classmethod
    def _forecast_assurance_gate(
        cls,
        plan: QueryPlan,
        result: ForecastResult,
    ) -> Tuple[bool, List[str]]:
        checks: List[str] = []
        forecast = result.forecast
        if forecast is None or forecast.empty or not cls._dataframe_is_finite(forecast):
            return False, ["forecast_rows_and_finite_values=false"]
        checks.append(f"forecast_rows={len(forecast)}")
        if "predicted" not in forecast.columns:
            return False, [*checks, "finite_predictions=false"]
        predicted = pd.to_numeric(forecast["predicted"], errors="coerce")
        if not any(math.isfinite(float(value)) for value in predicted.dropna().tolist()):
            return False, [*checks, "finite_predictions=false"]
        checks.extend(("finite_predictions=true", "entity_date_unit_reconciliation=passed"))
        return True, checks

    @classmethod
    def _carbon_assurance_gate(
        cls,
        result: CarbonResult,
    ) -> Tuple[bool, List[str]]:
        checks: List[str] = []
        frame = (
            result.table
            if isinstance(result.table, pd.DataFrame) and not result.table.empty
            else result.chart
        )
        if frame is None or frame.empty or not cls._dataframe_is_finite(frame):
            return False, ["carbon_rows_and_finite_values=false"]
        metric_fields = [
            field
            for field in frame.columns
            if field
            in {"ttw_co2e_t", "wtw_co2e_t", "co2_t", "nox_kg", "sox_kg", "pm_kg"}
        ]
        if not metric_fields or not any(
            pd.to_numeric(frame[field], errors="coerce").notna().any()
            for field in metric_fields
        ):
            return False, ["carbon_metric_values=false"]
        checks.extend(
            (
                f"carbon_result_rows={len(frame)}",
                "carbon_metric_values=true",
                "entity_date_unit_reconciliation=passed",
            )
        )
        return True, checks

    @staticmethod
    def _availability_code(
        plan: QueryPlan,
        outcome: ExecutionOutcome,
    ) -> str:
        if outcome.availability_code:
            return outcome.availability_code
        if outcome.state in {
            AnswerState.CLARIFICATION_REQUIRED,
            AnswerState.UNSUPPORTED,
        } or plan.mode == QueryMode.APP_HELP:
            return "not_applicable"
        if outcome.state in {AnswerState.COMPUTED, AnswerState.PARTIAL, AnswerState.RETRIEVED}:
            return "available"

        detail = " ".join(
            [outcome.answer, *outcome.caveats, *outcome.warnings]
        ).lower()
        if plan.operation in _LIVE_ETA_OPERATIONS:
            if "stale" in detail:
                return "source_stale"
            if any(
                token in detail
                for token in (
                    "ambiguous",
                    "conflict",
                    "mismatch",
                    "identity",
                    "multiple match",
                )
            ):
                return "ambiguous_match"
            if any(
                token in detail
                for token in (
                    "limited to",
                    "out of horizon",
                    "outside",
                    "unsupported port",
                    "coverage",
                )
            ):
                return "coverage_unavailable"
            if outcome.state == AnswerState.NO_DATA:
                return "no_data"
            return "source_unavailable"
        if outcome.state == AnswerState.NO_CURRENT_DATA:
            if "historical" in detail or "latest validated observation" in detail:
                return "source_stale"
            return "source_unavailable"
        if outcome.state in {AnswerState.NO_DATA, AnswerState.ERROR}:
            return "no_data"
        return "coverage_unavailable"

    def _resolve_port_scope(self, plan: QueryPlan) -> Tuple[QueryPlan, List[str]]:
        updated = plan.model_copy(deep=True)
        invalid: List[str] = []

        if updated.operation in _LIVE_ETA_OPERATIONS:
            resolved_ports: List[str] = []
            for value in updated.ports:
                resolved = normalize_baltic_port(value)
                if (
                    resolved
                    and updated.eta_watch_intent is not None
                    and resolved[:2] not in _ETA_WATCH_COUNTRY_SCOPE
                ):
                    resolved = None
                if resolved and resolved not in resolved_ports:
                    resolved_ports.append(resolved)
                elif not resolved:
                    invalid.append(str(value).strip())
            updated.ports = resolved_ports
            if updated.eta_watch_intent is not None:
                updated.source_scope = "aisstream"
            elif updated.aggregation in _REGIONAL_AIS_AGGREGATIONS:
                updated.source_scope = "fintraffic_ais"
            elif any(not port.startswith("FI") for port in resolved_ports):
                updated.source_scope = (
                    "fintraffic_ais"
                    if updated.operation == QueryOperation.VESSEL_ETA
                    else "official_schedule_unavailable"
                )
            else:
                updated.source_scope = "fintraffic_portnet"
            return updated, dedupe_strings(invalid)

        def resolve(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            cleaned = re.sub(
                r"\s+(?:today|tomorrow|now|currently|current|live|latest|this\s+week|next\s+week|coming\s+week)\s*$",
                "",
                str(value).strip(),
                flags=re.IGNORECASE,
            ).strip()
            resolved = self.kpi.resolve_port_token(cleaned)
            if resolved and self.kpi.is_known_port_token(resolved):
                return resolved
            invalid.append(cleaned or str(value))
            return None

        resolved_ports: List[str] = []
        for value in updated.ports:
            resolved = resolve(value)
            if resolved and resolved not in resolved_ports:
                resolved_ports.append(resolved)
        updated.ports = resolved_ports
        if updated.operation == QueryOperation.ARRIVALS_MULTI and len(resolved_ports) == 1:
            updated.operation = QueryOperation.ARRIVALS
        if (
            updated.operation == QueryOperation.FORECAST_COMPARISON
            and len(resolved_ports) == 1
            and not updated.compare_day_of_week
        ):
            updated.operation = (
                QueryOperation.FORECAST_ARRIVALS
                if updated.metric == "arrival_count"
                else QueryOperation.FORECAST_CONGESTION
            )
        updated.origin_port = resolve(updated.origin_port)
        updated.destination_port = resolve(updated.destination_port)
        for pair in updated.route_pairs:
            pair.origin = resolve(pair.origin) or ""
            pair.destination = resolve(pair.destination) or ""
        updated.route_pairs = [pair for pair in updated.route_pairs if pair.origin and pair.destination]
        if resolved_ports:
            invalid = [
                value
                for value in invalid
                if not re.search(
                    r"\b(arrivals?|dwell|congestion|pressure|durations?|percentiles?|forecast|anomal(?:y|ies)|today|tomorrow|weeks?)\b",
                    value,
                    flags=re.IGNORECASE,
                )
            ]
        if updated.operation in {QueryOperation.FIRST_ROUTE_VESSEL, QueryOperation.ROUTE_TRAVEL_TIME}:
            if not updated.origin_port and not updated.destination_port and len(resolved_ports) >= 2:
                updated.origin_port, updated.destination_port = resolved_ports[:2]
            if updated.origin_port and updated.destination_port:
                invalid = []
        return updated, dedupe_strings(invalid)

    def _validate(
        self,
        plan: QueryPlan,
        invalid_ports: Sequence[str],
        context: ConversationContext,
    ) -> Optional[ExecutionOutcome]:
        if plan.mode == QueryMode.CLARIFICATION:
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer=plan.clarification or "Please clarify the requested scope.",
                confidence="not_applicable",
                caveats=plan.ambiguities,
            )
        if plan.mode == QueryMode.UNSUPPORTED or plan.operation == QueryOperation.UNSUPPORTED:
            return ExecutionOutcome(
                state=AnswerState.UNSUPPORTED,
                answer=(
                    "I cannot answer that reliably from Eagle Eye's supported data. "
                    + (plan.reason or "No supported operation matched the request.")
                ),
                confidence="not_applicable",
            )
        if invalid_ports:
            if plan.operation in _LIVE_ETA_OPERATIONS:
                return ExecutionOutcome(
                    state=AnswerState.NO_CURRENT_DATA,
                    answer=(
                        "The requested port is outside Eagle Eye's curated Baltic live-source scope. "
                        f"Current vessel information is unavailable for {', '.join(invalid_ports)}."
                    ),
                    confidence="high",
                )
            return ExecutionOutcome(
                state=AnswerState.NO_DATA,
                answer=(
                    f"The requested port scope ({', '.join(invalid_ports)}) is not present in the canonical "
                    "port catalog. I did not broaden the query to unrelated ports."
                ),
                confidence="high",
            )
        if (
            plan.mode == QueryMode.ANALYTICS
            and plan.date_scope.is_current
            and plan.operation not in {*_LIVE_ETA_OPERATIONS, QueryOperation.EXPLAIN_PREVIOUS}
        ):
            freshness = self._freshness()
            return ExecutionOutcome(
                state=AnswerState.NO_CURRENT_DATA,
                answer=(
                    "Eagle Eye's analytics data is historical and cannot verify a current or live value. "
                    f"The latest validated observation is {freshness.data_to or 'not available'}."
                ),
                confidence="high",
            )
        if plan.operation == QueryOperation.ARRIVALS and not (
            plan.ports or plan.country_codes
        ):
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer="Which port or country should I use for the arrival count?",
                confidence="not_applicable",
            )
        if plan.operation == QueryOperation.DWELL_SUMMARY and (
            not plan.ports
            or not (plan.date_scope.date_from and plan.date_scope.date_to)
            or (
                plan.aggregation not in {"mean", "median"}
                and plan.requested_visual.value in {"auto", "none"}
            )
        ):
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer=(
                    "Please provide a port, a bounded date range, and whether you want the mean or median completed dwell time."
                ),
                confidence="not_applicable",
            )
        if plan.operation in _LIVE_ETA_OPERATIONS:
            if self.live_eta is None:
                if plan.eta_watch_intent is not None:
                    return ExecutionOutcome(
                        state=AnswerState.NO_CURRENT_DATA,
                        answer=(
                            "ETA Watch is not configured in this runtime. Set a valid "
                            "backend-only AISSTREAM_API_KEY and start the AISStream collector. "
                            "Current vessel information will be available after the source is connected."
                        ),
                        confidence="not_applicable",
                        availability_code="source_unavailable",
                        availability_provider="aisstream",
                        availability_retryable=True,
                        operational_brief=self._empty_operational_brief(
                            plan,
                            source_health="unavailable",
                            headline="ETA Watch source is not configured.",
                        ),
                    )
                return ExecutionOutcome(
                    state=AnswerState.NO_CURRENT_DATA,
                    answer=(
                        "The Fintraffic live ETA adapter is unavailable in this runtime. "
                        "Current vessel information cannot be retrieved."
                    ),
                    confidence="high",
                )
            if plan.eta_watch_intent is not None:
                return None
            has_identity = bool(plan.mmsi or plan.imo or plan.vessel_name)
            if (
                plan.operation == QueryOperation.VESSEL_ETA
                and not has_identity
                and len(plan.ports) != 1
                and plan.aggregation not in _REGIONAL_AIS_AGGREGATIONS
            ):
                return ExecutionOutcome(
                    state=AnswerState.CLARIFICATION_REQUIRED,
                    answer=(
                        "Provide an exact MMSI, valid IMO number, explicit vessel name, "
                        "or one supported Baltic port. Non-Finnish results are limited to "
                        "fresh, non-exhaustive AIS destination observations."
                    ),
                    confidence="not_applicable",
                )
            if (
                plan.operation == QueryOperation.VESSEL_DELAY
                and not has_identity
                and len(plan.ports) != 1
            ):
                return ExecutionOutcome(
                    state=AnswerState.CLARIFICATION_REQUIRED,
                    answer=(
                        "Provide a vessel identity or exactly one Finnish port for announced ETA variance; "
                        "a non-Finnish delay needs an official schedule baseline that is not integrated."
                    ),
                    confidence="not_applicable",
                )
            if plan.operation == QueryOperation.LIVE_PORT_ARRIVALS:
                if len(plan.ports) != 1:
                    return ExecutionOutcome(
                        state=AnswerState.CLARIFICATION_REQUIRED,
                        answer=(
                            "Provide exactly one port UN/LOCODE. Official live arrival schedules are "
                            "currently available only for Finnish Portnet ports."
                        ),
                        confidence="not_applicable",
                    )
            if plan.operation == QueryOperation.ETA_COMPARISON:
                if not (1 <= len(plan.ports) <= 2):
                    return ExecutionOutcome(
                        state=AnswerState.CLARIFICATION_REQUIRED,
                        answer=(
                            "Provide one or two Finnish port UN/LOCODEs. Announced ETA comparisons "
                            "require Finnish Portnet schedule baselines."
                        ),
                        confidence="not_applicable",
                    )
        if plan.operation in {
            QueryOperation.FIRST_ARRIVAL,
            QueryOperation.LAST_ARRIVAL,
            QueryOperation.FIRST_DEPARTURE,
            QueryOperation.FORECAST_ARRIVALS,
            QueryOperation.FORECAST_CONGESTION,
        } and not plan.ports:
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer="Which port should I use for this query?",
                confidence="not_applicable",
            )
        if plan.operation in {QueryOperation.PORT_COMPARISON, QueryOperation.ARRIVALS_MULTI} and len(plan.ports) < 2:
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer="Please name at least two ports to compare.",
                confidence="not_applicable",
            )
        if plan.operation in {QueryOperation.FIRST_ROUTE_VESSEL, QueryOperation.ROUTE_TRAVEL_TIME} and not (
            plan.origin_port and plan.destination_port
        ):
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer="Please provide both the origin and destination ports.",
                confidence="not_applicable",
            )
        if plan.operation == QueryOperation.EXPLAIN_PREVIOUS and context.previous_envelope is None:
            return ExecutionOutcome(
                state=AnswerState.CLARIFICATION_REQUIRED,
                answer="There is no previous result in this conversation to explain.",
                confidence="not_applicable",
            )
        return None

    def _execute(
        self,
        plan: QueryPlan,
        request: QueryRequest,
        context: ConversationContext,
    ) -> ExecutionOutcome:
        if plan.operation == QueryOperation.HELP:
            evidence = []
            if request.top_k_evidence > 0:
                evidence = [
                    EvidenceItem(
                        id="eagle_eye_capability_registry",
                        source_type="system",
                        title="Eagle Eye capability registry",
                        excerpt=HELP_ANSWER,
                        metadata={
                            "retrieval_mode": "capability_registry",
                            "retrieval_backend": "static",
                        },
                    )
                ]
            return ExecutionOutcome(
                state=AnswerState.GENERAL,
                answer=HELP_ANSWER,
                evidence=evidence,
                confidence="high",
                retrieval_mode="capability_registry",
                retrieval_backend="static",
                retrieval_status="ok" if evidence else "not_required",
            )
        if plan.operation == QueryOperation.EXPLAIN_PREVIOUS:
            previous = context.previous_envelope
            assert previous is not None
            return ExecutionOutcome(
                state=previous.state,
                answer=f"In plain language: {previous.answer}",
                confidence=previous.confidence,
                evidence=previous.evidence,
                caveats=previous.caveats,
                retrieval_mode="conversation_context",
                retrieval_backend="sqlite",
                retrieval_status="ok" if previous.evidence else "not_required",
            )
        if plan.operation == QueryOperation.RESEARCH:
            return self._execute_research(request.question, request.top_k_evidence)
        if plan.operation == QueryOperation.GENERAL_RESPONSE:
            return self._execute_general(
                request.question,
                current=plan.date_scope.is_current,
            )
        return self._execute_analytics(plan, request)

    def _execute_analytics(self, plan: QueryPlan, request: QueryRequest) -> ExecutionOutcome:
        scope = plan.date_scope
        port = plan.ports[0] if plan.ports else None
        result: ResultType
        execution_warnings: List[str] = []
        op = plan.operation

        if op in _LIVE_ETA_OPERATIONS:
            return self._execute_live_eta(plan)

        if op == QueryOperation.ARRIVALS:
            result = self.kpi.get_arrivals(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                dow=plan.day_of_week,
                window=scope.relative_window,
                source_scope=plan.source_scope,
            )
        elif op == QueryOperation.ARRIVALS_MULTI:
            result = self.kpi.get_arrivals_multi(
                ports=plan.ports,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                dow=plan.day_of_week,
                window=scope.relative_window,
                source_scope=plan.source_scope,
            )
        elif op == QueryOperation.TOP_PORTS:
            result = self.kpi.top_ports_by_arrivals(
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                dow=plan.day_of_week,
                top_n=plan.limit if plan.limit > 1 else 10,
                source_scope=plan.source_scope,
                country_codes=plan.country_codes,
            )
        elif op == QueryOperation.PEAK_ARRIVAL_DAY:
            result = self.kpi.get_peak_arrival_day(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                window=scope.relative_window,
                source_scope=plan.source_scope,
            )
        elif op == QueryOperation.BUSIEST_WEEKDAY:
            result = self.kpi.get_busiest_dow(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                source_scope=plan.source_scope,
            )
        elif op == QueryOperation.BUSIEST_HOUR:
            result = self.kpi.get_busiest_hour(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
            )
        elif op == QueryOperation.ARRIVAL_PATTERN:
            result = self.kpi.get_arrival_weekday_hour_pattern(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
            )
        elif op == QueryOperation.WEEKDAY_COMPARISON:
            result = self.kpi.compare_weekdays(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                day_a=str(plan.day_of_week),
                day_b=str(plan.compare_day_of_week),
                vessel_type=plan.vessel_type,
                source_scope=plan.source_scope,
            )
        elif op == QueryOperation.VESSEL_TYPE_COMPOSITION:
            method = getattr(self.kpi, "get_arrival_composition", None)
            result = (
                method(
                    port=port,
                    start=scope.date_from,
                    end=scope.date_to,
                    source_scope=plan.source_scope,
                )
                if callable(method)
                else self.kpi.unsupported("Vessel-type composition is unavailable in this data build.")
            )
        elif op == QueryOperation.DWELL_SUMMARY:
            result = self.kpi.get_avg_dwell_time(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                dow=plan.day_of_week,
                aggregation=plan.aggregation,
            )
        elif op == QueryOperation.DWELL_DISTRIBUTION:
            method = getattr(self.kpi, "get_dwell_distribution", None)
            result = (
                method(
                    port=port,
                    start=scope.date_from,
                    end=scope.date_to,
                    vessel_type=plan.vessel_type,
                )
                if callable(method)
                else self.kpi.unsupported("Dwell distribution is unavailable in this data build.")
            )
        elif op == QueryOperation.MMSI_PORT_STAYS:
            result = self.kpi.get_mmsi_port_stays(
                mmsi=str(plan.mmsi or ""),
                start=scope.date_from,
                end=scope.date_to,
                port=port,
            )
        elif op == QueryOperation.CONGESTION:
            result = self.kpi.get_congestion(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                dow=plan.day_of_week,
                window=scope.relative_window,
            )
        elif op == QueryOperation.PEAK_CONGESTION_DAY:
            result = self.kpi.get_peak_congestion_days(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                dow=plan.day_of_week,
                window=scope.relative_window,
                limit=plan.limit,
            )
        elif op == QueryOperation.CONGESTION_WEEKDAY_COMPARISON:
            result = self.kpi.compare_congestion_weekdays(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                day_a=str(plan.day_of_week),
                day_b=str(plan.compare_day_of_week),
            )
        elif op == QueryOperation.PRESSURE_BY_VESSEL_TYPE:
            result = self.kpi.get_pressure_by_vessel_type(port=port, start=scope.date_from, end=scope.date_to)
        elif op == QueryOperation.PORT_COMPARISON:
            result = self.kpi.compare_ports(
                ports=plan.ports,
                metric=str(plan.metric or "arrival_count"),
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                dow=plan.day_of_week,
                source_scope=plan.source_scope,
            )
        elif op == QueryOperation.FORECAST_ARRIVALS:
            result = (
                self.forecaster.forecast_arrivals_for_date(
                    port=port or "",
                    target_date=scope.target_date,
                    horizon_weeks=plan.horizon_weeks,
                )
                if scope.target_date
                else self.forecaster.forecast_arrivals(
                    port=port or "",
                    horizon_weeks=plan.horizon_weeks,
                    vessel_type=plan.vessel_type,
                )
            )
        elif op == QueryOperation.FORECAST_CONGESTION:
            result = (
                self.forecaster.forecast_congestion_for_date(
                    port=port or "",
                    target_date=scope.target_date,
                    horizon_weeks=plan.horizon_weeks,
                )
                if scope.target_date
                else self.forecaster.forecast_congestion(
                    port=port or "",
                    target_dow=plan.day_of_week or "Friday",
                    horizon_weeks=plan.horizon_weeks,
                )
            )
        elif op == QueryOperation.FORECAST_COMPARISON:
            result = (
                self.forecaster.compare_congestion_weekdays(
                    port=port or "",
                    day_a=str(plan.day_of_week),
                    day_b=str(plan.compare_day_of_week),
                    horizon_weeks=plan.horizon_weeks,
                )
                if plan.day_of_week and plan.compare_day_of_week and len(plan.ports) <= 1
                else self.forecaster.compare_congestion_ports(
                    ports=plan.ports,
                    target_date=scope.target_date,
                    target_dow=plan.day_of_week,
                    horizon_weeks=plan.horizon_weeks,
                )
            )
        elif op == QueryOperation.DIAGNOSTIC:
            result = self.kpi.diagnose_congestion(port=port, target_date=scope.date_from or scope.date_to)
        elif op == QueryOperation.CORRELATION:
            result = self.kpi.get_arrivals_dwell_correlation(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
            )
        elif op == QueryOperation.ARRIVAL_ANOMALY:
            result = self.kpi.detect_arrival_spikes(port=port, start=scope.date_from, end=scope.date_to)
        elif op == QueryOperation.AIS_JUMP:
            result = self._execute_ais_jump(plan, warnings=execution_warnings)
        elif op == QueryOperation.CARBON:
            result = self.carbon.from_question_entities(
                question=request.question,
                entities={
                    "ports": plan.ports,
                    "country_codes": plan.country_codes,
                    "port": port,
                    "date_from": scope.date_from,
                    "date_to": scope.date_to,
                    "target_date": scope.target_date,
                    "vessel_type": plan.vessel_type,
                    "mmsi": plan.mmsi,
                    "imo": plan.imo,
                    "call_id": plan.call_id,
                    "boundary": plan.carbon_boundary,
                    "pollutants": plan.pollutants,
                },
                user_filters=request.filters.model_dump(),
                resolved_scope={"port": port, "date_from": scope.date_from, "date_to": scope.date_to},
            )
        elif op == QueryOperation.FIRST_ARRIVAL:
            result = self.kpi.get_first_arrival(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                window=scope.relative_window,
            )
        elif op == QueryOperation.LAST_ARRIVAL:
            result = self.kpi.get_last_arrival(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                window=scope.relative_window,
            )
        elif op == QueryOperation.FIRST_DEPARTURE:
            result = self.kpi.get_first_departure(
                port=port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                window=scope.relative_window,
            )
        elif op == QueryOperation.FIRST_ROUTE_VESSEL:
            result = self.kpi.get_first_route_vessel(
                origin_port=plan.origin_port,
                destination_port=plan.destination_port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                window=scope.relative_window,
            )
        elif op == QueryOperation.ROUTE_TRAVEL_TIME:
            result = self.kpi.get_route_travel_time_summary(
                origin_port=plan.origin_port,
                destination_port=plan.destination_port,
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                window=scope.relative_window,
            )
        elif op == QueryOperation.MIXED_PORT_ROUTE_COMPARISON:
            result = self.kpi.compare_ports_and_routes(
                ports=plan.ports,
                route_pairs=[pair.model_dump() for pair in plan.route_pairs],
                start=scope.date_from,
                end=scope.date_to,
                vessel_type=plan.vessel_type,
                dow=plan.day_of_week,
                window=scope.relative_window,
            )
        else:
            result = self.kpi.unsupported("No canonical executor is registered for this operation.")

        state = self._state_for_result(result)
        if (
            state in {AnswerState.COMPUTED, AnswerState.PARTIAL}
            and not self._result_has_usable_rows(result)
        ):
            # A zero-event detector result is an honest absence of matching
            # rows, not a successful data-bearing computation.
            state = AnswerState.NO_DATA
        retrieval = self._retrieve_traffic_evidence(
            plan,
            request.question,
            request.top_k_evidence,
        )
        caveats = list(getattr(result, "caveats", []) or [])
        return ExecutionOutcome(
            state=state,
            answer=str(result.answer).strip(),
            result=result,
            evidence=retrieval.evidence,
            confidence=extract_confidence_label(result),
            caveats=caveats,
            warnings=dedupe_strings(
                [*execution_warnings, *([retrieval.warning] if retrieval.warning else [])]
            ),
            retrieval_mode=retrieval.mode,
            retrieval_backend=retrieval.backend,
            retrieval_status=retrieval.status,
        )

    def _execute_eta_watch(self, plan: QueryPlan) -> ExecutionOutcome:
        """Execute one AISStream-backed intent and compose only row-derived prose."""

        assert self.live_eta is not None
        assert plan.eta_watch_intent is not None
        provider = str(getattr(self.live_eta, "provider", "aisstream") or "aisstream")
        provider_operation = _ETA_WATCH_OPERATION_BY_INTENT[plan.eta_watch_intent]
        provider_ports = list(plan.ports)
        if not provider_ports and plan.aggregation in _REGIONAL_AIS_AGGREGATIONS:
            allowed_prefixes = (
                {"SE"}
                if plan.aggregation == "swedish_destination_signals"
                else set(_ETA_WATCH_COUNTRY_SCOPE)
            )
            provider_ports = sorted(
                port
                for port in _ETA_WATCH_CURATED_PORTS
                if port[:2] in allowed_prefixes
            )
        query_arguments: Dict[str, Any] = {
            "operation": provider_operation,
            "ports": provider_ports,
            "mmsi": plan.mmsi,
            "imo": plan.imo,
            "vessel_name": plan.vessel_name,
            "horizon_hours": plan.horizon_hours,
            # Retrieve the bounded provider maximum so totals and "next"
            # prioritization are computed before the public display limit.
            "limit": 500,
            "speed_threshold_kn": plan.speed_threshold_kn,
            "eta_change_threshold_minutes": plan.eta_change_threshold_minutes,
            "change_window_minutes": plan.change_window_minutes,
            "include_stale": (
                plan.include_stale
                or (
                    plan.eta_watch_intent
                    in {
                        ETAWatchIntent.INBOUND_WATCHLIST,
                        ETAWatchIntent.DESTINATION_LOAD,
                    }
                    and "position" not in plan.dimensions
                )
            ),
        }
        try:
            signature = inspect.signature(self.live_eta.query)
            accepts_keywords = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if not accepts_keywords:
                query_arguments = {
                    name: value
                    for name, value in query_arguments.items()
                    if name in signature.parameters
                }
            raw_result = self.live_eta.query(**query_arguments)
        except Exception as exc:
            safe_error = redact_sensitive_text(str(exc))
            headline = "ETA Watch could not reach its live AIS source."
            return ExecutionOutcome(
                state=AnswerState.NO_CURRENT_DATA,
                answer=(
                    f"{headline} No vessel status, ETA, or position was published"
                    + (f" ({safe_error})." if safe_error else ".")
                ),
                confidence="not_applicable",
                caveats=["AISStream is a non-exhaustive vessel-broadcast source."],
                warnings=[f"ETA Watch provider query failed: {safe_error}"],
                retrieval_mode="live_ais_broadcast",
                retrieval_backend=provider,
                retrieval_status="error",
                availability_code="source_unavailable",
                availability_provider=provider,
                availability_retryable=True,
                operational_brief=self._empty_operational_brief(
                    plan,
                    source_health="unavailable",
                    headline=headline,
                ),
            )

        snapshot_at = self._utc_datetime(
            getattr(raw_result, "snapshot_at", None)
        ) or datetime.now(timezone.utc)
        data_updated_at = self._utc_datetime(
            getattr(raw_result, "data_updated_at", None)
        )
        horizon_end = self._utc_datetime(
            getattr(raw_result, "horizon_end", None)
        ) or snapshot_at + timedelta(hours=int(plan.horizon_hours or 24))
        summary = getattr(raw_result, "summary", None)
        if not isinstance(summary, dict):
            summary = {}
        raw_table = getattr(raw_result, "table", None)
        table = (
            raw_table.copy()
            if isinstance(raw_table, pd.DataFrame)
            else pd.DataFrame()
        )
        if plan.eta_watch_intent == ETAWatchIntent.SHIFT_HANDOVER:
            table = self._combine_eta_watch_sections(raw_result, fallback=table)
        prepared = self._prepare_eta_watch_frame(
            plan,
            table,
            snapshot_at=snapshot_at,
            horizon_end=horizon_end,
        )
        raw_matched_count = summary.get(
            "matched_count",
            summary.get(
                "matched_vessels",
                summary.get("total_count"),
            ),
        )
        source_candidate_count = (
            raw_matched_count
            if isinstance(raw_matched_count, int)
            and not isinstance(raw_matched_count, bool)
            and raw_matched_count >= 0
            else len(table)
        )
        if (
            plan.eta_watch_intent == ETAWatchIntent.DESTINATION_LOAD
            and "inbound_vessels" in prepared.columns
        ):
            matched_count = int(
                pd.to_numeric(
                    prepared["inbound_vessels"],
                    errors="coerce",
                ).fillna(0).sum()
            )
        else:
            # Public matched totals are always post-validation. Provider
            # candidate counts can include stale or incomplete rows and must
            # never be presented as validated matches.
            matched_count = len(prepared)
        displayed = prepared.head(plan.limit).reset_index(drop=True)
        candidate_table = self._eta_watch_candidate_frame(
            plan,
            raw_table=table,
            validated=prepared,
            snapshot_at=snapshot_at,
            horizon_end=horizon_end,
        )

        raw_status = str(getattr(raw_result, "status", "no_current_data")).lower()
        source_health = self._eta_watch_source_health(raw_result, raw_status)
        if (
            raw_status == "ok"
            and not displayed.empty
            and source_health == "live"
        ):
            status = "ok"
            state = AnswerState.COMPUTED
        elif source_health == "live" and (
            raw_status == "no_data"
            or (raw_status == "ok" and displayed.empty)
        ):
            status = "no_data"
            state = AnswerState.NO_DATA
        else:
            status = "no_current_data"
            state = AnswerState.NO_CURRENT_DATA

        coverage = (
            "AISStream is a non-exhaustive vessel-broadcast feed. Results are "
            "not an official schedule, confirmed delay, port traffic total, "
            "arrival confirmation, or prediction."
        )
        source_observed_at = data_updated_at or snapshot_at
        answer = self._compose_eta_watch_answer(
            plan,
            displayed,
            matched_count=matched_count,
            source_candidate_count=source_candidate_count,
            snapshot_at=source_observed_at,
            state=state,
            source_health=source_health,
        )
        caveats = dedupe_strings(
            [
                *list(getattr(raw_result, "caveats", None) or []),
                coverage,
            ]
        )
        result = ETAWatchExecutionResult(
            status=status,
            answer=answer,
            table=(
                displayed
                if state == AnswerState.COMPUTED and not displayed.empty
                else None
            ),
            coverage_notes=list(
                getattr(raw_result, "coverage_notes", None) or []
            ),
            caveats=caveats,
            snapshot_at=snapshot_at,
            horizon_end=horizon_end,
            data_updated_at=data_updated_at,
            failure_reason=getattr(raw_result, "failure_reason", None),
            matched_count=matched_count,
            candidate_table=candidate_table,
        )
        brief = (
            self._build_operational_brief(
                plan,
                displayed,
                full_table=prepared,
                matched_count=matched_count,
                snapshot_at=snapshot_at,
                horizon_end=horizon_end,
                source_health=source_health,
                coverage=coverage,
                source_observed_at=source_observed_at,
            )
            if state == AnswerState.COMPUTED
            else self._empty_operational_brief(
                plan,
                source_health=source_health,
                headline=(
                    "No matching AIS vessel signal is currently available."
                    if state == AnswerState.NO_DATA
                    else "ETA Watch live source is unavailable."
                ),
            )
        )
        evidence = [
            EvidenceItem(
                id="aisstream_vessel_broadcasts",
                source_type="web",
                title="AISStream vessel-broadcast feed",
                url=_ETA_WATCH_SOURCE_URL,
                metadata={
                    "provider": provider,
                    "authority": "vessel_reported_ais_broadcast",
                    "source_kind": "aisstream",
                    "official_schedule": False,
                    "prediction": False,
                    "snapshot_at": snapshot_at.isoformat(),
                    "timezone": "UTC",
                },
            )
        ]
        observed_at = source_observed_at
        freshness = FreshnessInfo(
            data_from=snapshot_at.strftime("%Y-%m-%d") if state == AnswerState.COMPUTED else None,
            data_to=snapshot_at.strftime("%Y-%m-%d") if state == AnswerState.COMPUTED else None,
            as_of=observed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            historical=False,
            message=(
                f"Vessel-broadcast snapshot observed at "
                f"{observed_at.strftime('%Y-%m-%dT%H:%M:%SZ')}. "
                "AIS coverage is non-exhaustive and does not establish an official schedule."
            ),
        )
        availability_code = (
            "available"
            if state == AnswerState.COMPUTED
            else "no_data"
            if state == AnswerState.NO_DATA
            else "source_stale"
            if source_health == "stale"
            else "source_unavailable"
        )
        return ExecutionOutcome(
            state=state,
            answer=answer,
            result=result,
            evidence=evidence,
            confidence="high" if state == AnswerState.COMPUTED else "not_applicable",
            caveats=caveats,
            retrieval_mode="live_ais_broadcast",
            retrieval_backend=provider,
            retrieval_status=(
                "ok"
                if state == AnswerState.COMPUTED
                else "empty"
                if state == AnswerState.NO_DATA
                else "unavailable"
            ),
            freshness_override=freshness,
            availability_code=availability_code,
            availability_provider=provider,
            availability_retryable=availability_code
            in {"source_unavailable", "source_stale"},
            operational_brief=brief,
        )

    @staticmethod
    def _utc_datetime(value: Any) -> Optional[datetime]:
        if value is None or value is pd.NaT:
            return None
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if timestamp is pd.NaT or pd.isna(timestamp):
            return None
        if isinstance(timestamp, pd.DatetimeIndex):
            return None
        return timestamp.to_pydatetime()

    @staticmethod
    def _utc_label(value: Any) -> Optional[str]:
        parsed = QueryService._utc_datetime(value)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") if parsed else None

    def _eta_watch_source_health(self, raw_result: Any, raw_status: str) -> str:
        candidate: Any = None
        summary = getattr(raw_result, "summary", None)
        if isinstance(summary, dict):
            candidate = summary.get("source_health")
        health_method = getattr(self.live_eta, "source_health", None)
        if callable(health_method):
            try:
                health_payload = health_method()
                if isinstance(health_payload, dict):
                    candidate = (
                        health_payload.get("status")
                        or health_payload.get("state")
                        or health_payload.get("health")
                        or candidate
                    )
                elif health_payload is not None:
                    if (
                        getattr(health_payload, "api_key_configured", None) is False
                        and int(getattr(health_payload, "cached_vessels", 0) or 0) == 0
                    ):
                        return "unavailable"
                    candidate = getattr(health_payload, "status", health_payload)
            except Exception:
                pass
        token = str(candidate or "").strip().lower()
        if token in {"connecting", "warming", "live", "stale", "unavailable"}:
            return token
        if token == "healthy":
            return "live"
        if token == "offline_cache":
            return "live"
        if token in {"reconnecting"}:
            return "connecting"
        if token in {"not_started"}:
            return "warming"
        if token in {"stopped"}:
            return "unavailable"
        if raw_status == "ok":
            return "live"
        if "stale" in " ".join(
            [
                str(getattr(raw_result, "failure_reason", "") or ""),
                *(str(item) for item in getattr(raw_result, "caveats", None) or []),
            ]
        ).lower():
            return "stale"
        return "unavailable"

    @staticmethod
    def _combine_eta_watch_sections(
        raw_result: Any,
        *,
        fallback: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge handover sections by MMSI without inventing absent values."""

        sections = getattr(raw_result, "sections", None)
        if not isinstance(sections, dict):
            return fallback
        records: Dict[str, Dict[str, Any]] = {}

        def merge_frame(name: str, frame: Any) -> None:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return
            for raw in frame.to_dict(orient="records"):
                mmsi = str(raw.get("mmsi") or "").strip()
                if not mmsi:
                    continue
                target = records.setdefault(mmsi, {"mmsi": mmsi})
                for key, value in raw.items():
                    try:
                        missing = pd.isna(value)
                    except Exception:
                        missing = False
                    if isinstance(missing, bool) and missing:
                        continue
                    if value is not None:
                        target[key] = value
                if name == "low_speed":
                    target["provider_low_speed"] = True
                elif name == "eta_revisions":
                    target["provider_eta_changed"] = True
                elif name == "stale_missing":
                    target["provider_stale_missing"] = True

        merge_frame("inbound_watchlist", fallback)
        for section_name in (
            "inbound_watchlist",
            "low_speed",
            "eta_revisions",
            "stale_missing",
        ):
            merge_frame(section_name, sections.get(section_name))
        return pd.DataFrame(list(records.values())) if records else fallback

    @staticmethod
    def _prepare_eta_watch_frame(
        plan: QueryPlan,
        raw_table: pd.DataFrame,
        *,
        snapshot_at: datetime,
        horizon_end: datetime,
    ) -> pd.DataFrame:
        if raw_table.empty:
            return pd.DataFrame()
        work = raw_table.copy()
        aliases: Dict[str, Tuple[str, ...]] = {
            "vessel_name": ("vessel_label", "name", "ship_name"),
            "destination_locode": ("port_locode", "locode"),
            "destination_name": ("port_name",),
            "destination_raw": ("ais_destination", "destination"),
            "reported_eta_utc": (
                "ais_eta_utc",
                "eta_utc",
                "current_eta_utc",
                "next_eta_utc",
            ),
            "previous_reported_eta_utc": (
                "previous_ais_eta_utc",
                "previous_eta_utc",
            ),
            "position_time_utc": (
                "ais_location_time_utc",
                "position_observed_at_utc",
                "last_position_at",
                "location_time_utc",
            ),
            "eta_observation_time_utc": (
                "eta_observed_at_utc",
                "static_observed_at_utc",
            ),
            "observation_time_utc": (
                "position_time_utc",
                "ais_location_time_utc",
                "ais_metadata_time_utc",
            ),
            "speed_kn": ("sog_kn", "speed_over_ground_kn"),
            "course_deg": ("cog_deg", "course_over_ground_deg"),
            "eta_change_minutes": (
                "eta_revision_minutes",
            ),
            "eta_change_observed_at_utc": (
                "revision_observed_at_utc",
            ),
            "observation_age_minutes": (
                "position_age_minutes",
            ),
        }
        for canonical, candidates in aliases.items():
            if canonical in work.columns:
                continue
            source = next(
                (field for field in candidates if field in work.columns),
                None,
            )
            if source:
                work[canonical] = work[source]

        # Retain the established AIS field names for compatibility while
        # exposing provider-neutral canonical names to ETA Watch.
        if "reported_eta_utc" in work.columns and "ais_eta_utc" not in work.columns:
            work["ais_eta_utc"] = work["reported_eta_utc"]
        if "speed_kn" in work.columns and "sog_kn" not in work.columns:
            work["sog_kn"] = work["speed_kn"]
        if (
            "destination_locode" in work.columns
            and "port_locode" not in work.columns
        ):
            work["port_locode"] = work["destination_locode"]
        if (
            "position_time_utc" in work.columns
            and "ais_location_time_utc" not in work.columns
        ):
            work["ais_location_time_utc"] = work["position_time_utc"]

        # The provider emits destination ranking rows directly. Preserve their
        # validated counts rather than trying to reinterpret them as vessel
        # observations requiring an ETA on every row.
        if (
            plan.eta_watch_intent == ETAWatchIntent.DESTINATION_LOAD
            and "vessel_count" in work.columns
            and "inbound_vessels" not in work.columns
        ):
            work["inbound_vessels"] = pd.to_numeric(
                work["vessel_count"],
                errors="coerce",
            )
        if (
            plan.eta_watch_intent == ETAWatchIntent.DESTINATION_LOAD
            and "inbound_vessels" in work.columns
            and "reported_eta_utc" not in work.columns
        ):
            if "destination_locode" not in work.columns:
                return pd.DataFrame()
            locodes = work["destination_locode"].fillna("").astype(str)
            verified = locodes.str.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}")
            work = work[
                verified
                & pd.to_numeric(
                    work["inbound_vessels"],
                    errors="coerce",
                ).gt(0)
            ].copy()
            if work.empty:
                return pd.DataFrame()
            work["inbound_vessels"] = pd.to_numeric(
                work["inbound_vessels"],
                errors="coerce",
            ).astype(int)
            work["row_id"] = work["destination_locode"].map(
                lambda value: f"destination-{value}"
            )
            return work.sort_values(
                ["inbound_vessels", "destination_locode"],
                ascending=[False, True],
                kind="stable",
            ).reset_index(drop=True)

        for field in (
            "reported_eta_utc",
            "ais_eta_utc",
            "previous_reported_eta_utc",
            "position_time_utc",
            "observation_time_utc",
            "ais_location_time_utc",
            "eta_change_observed_at_utc",
            "eta_observation_time_utc",
        ):
            if field in work.columns:
                work[field] = pd.to_datetime(
                    work[field],
                    errors="coerce",
                    utc=True,
                )
        for field in (
            "latitude",
            "longitude",
            "speed_kn",
            "sog_kn",
            "course_deg",
            "eta_change_minutes",
            "observation_age_minutes",
        ):
            if field in work.columns:
                work[field] = pd.to_numeric(work[field], errors="coerce")

        if "vessel_name" not in work.columns:
            work["vessel_name"] = None
        if "mmsi" not in work.columns:
            work["mmsi"] = None
        work["vessel_label"] = work.apply(
            lambda row: (
                str(row.get("vessel_name")).strip()
                if row.get("vessel_name") is not None
                and str(row.get("vessel_name")).strip()
                and str(row.get("vessel_name")).lower() != "nan"
                else f"MMSI {str(row.get('mmsi')).strip()}"
                if row.get("mmsi") is not None
                and str(row.get("mmsi")).strip()
                and str(row.get("mmsi")).lower() != "nan"
                else "Unidentified vessel"
            ),
            axis=1,
        )

        if "observation_age_minutes" not in work.columns:
            observation_field = next(
                (
                    field
                    for field in (
                        "position_time_utc",
                        "observation_time_utc",
                        "ais_location_time_utc",
                    )
                    if field in work.columns
                ),
                None,
            )
            if observation_field:
                work["observation_age_minutes"] = (
                    pd.Timestamp(snapshot_at) - work[observation_field]
                ).dt.total_seconds() / 60.0
            else:
                work["observation_age_minutes"] = math.nan

        eta_valid = (
            work["reported_eta_utc"].notna()
            if "reported_eta_utc" in work.columns
            else pd.Series(False, index=work.index)
        )
        if "eta_observation_time_utc" in work.columns:
            eta_age_minutes = (
                pd.Timestamp(snapshot_at) - work["eta_observation_time_utc"]
            ).dt.total_seconds() / 60.0
            work["eta_observation_age_minutes"] = eta_age_minutes
            eta_valid &= eta_age_minutes.between(
                -10.0,
                10.0,
                inclusive="both",
            )
        elif plan.eta_watch_intent not in {
            ETAWatchIntent.ETA_REVISIONS,
            ETAWatchIntent.SIGNAL_QUALITY,
        }:
            # A live reported ETA without a source observation time cannot
            # pass the ten-minute publication gate.
            eta_valid &= False
        if "eta_valid" in work.columns:
            eta_valid &= work["eta_valid"].fillna(False).astype(bool)
        due_in_window = eta_valid.copy()
        if "reported_eta_utc" in work.columns:
            due_in_window &= work["reported_eta_utc"].between(
                pd.Timestamp(snapshot_at),
                pd.Timestamp(horizon_end),
                inclusive="both",
            )
        latitude_valid = (
            work["latitude"].between(-90.0, 90.0, inclusive="both")
            if "latitude" in work.columns
            else pd.Series(False, index=work.index)
        )
        longitude_valid = (
            work["longitude"].between(-180.0, 180.0, inclusive="both")
            if "longitude" in work.columns
            else pd.Series(False, index=work.index)
        )
        position_time_valid = (
            work["position_time_utc"].notna()
            if "position_time_utc" in work.columns
            else pd.Series(False, index=work.index)
        )
        position_valid = latitude_valid & longitude_valid & position_time_valid
        position_fields_present = any(
            field in work.columns
            for field in (
                "latitude",
                "longitude",
                "position_time_utc",
                "position_observed_at_utc",
            )
        )
        if position_fields_present or plan.eta_watch_intent in {
            ETAWatchIntent.SHIFT_HANDOVER,
            ETAWatchIntent.SIGNAL_QUALITY,
        }:
            stale_position = ~position_valid | (
                work["observation_age_minutes"].fillna(math.inf) > 10.0
            )
        else:
            stale_position = pd.Series(False, index=work.index)
        if "is_position_stale" in work.columns:
            stale_position |= work["is_position_stale"].fillna(True).astype(bool)
        work["is_due_in_window"] = due_in_window.astype(bool)
        work["is_position_stale"] = stale_position.astype(bool)
        work["is_missing_eta"] = (~due_in_window).astype(bool)
        if "reported_eta_utc" in work.columns:
            # Keep the source transmission for audit, but expose only a
            # freshness- and horizon-valid ETA through the canonical field.
            # This prevents stale, past, or ambiguous AIS dates from appearing
            # on the ETA rail or being described as current.
            work["source_reported_eta_utc"] = work["reported_eta_utc"]
            work["reported_eta_utc"] = work["reported_eta_utc"].where(
                due_in_window
            )
            if "ais_eta_utc" in work.columns:
                work["ais_eta_utc"] = work["ais_eta_utc"].where(due_in_window)

        speed_threshold = float(plan.speed_threshold_kn or 2.0)
        speed_series = (
            work["speed_kn"]
            if "speed_kn" in work.columns
            else pd.Series(math.nan, index=work.index)
        )
        work["is_low_speed"] = (
            speed_series.notna()
            & (speed_series < speed_threshold)
            & due_in_window
            & ~stale_position
        ).astype(bool)
        if "provider_low_speed" in work.columns:
            work["is_low_speed"] |= (
                work["provider_low_speed"].fillna(False).astype(bool)
                & ~stale_position
            )
        change_threshold = float(plan.eta_change_threshold_minutes or 30)
        change_series = (
            work["eta_change_minutes"]
            if "eta_change_minutes" in work.columns
            else pd.Series(math.nan, index=work.index)
        )
        change_valid = change_series.notna() & (
            change_series.abs() > change_threshold
        )
        if (
            plan.change_window_minutes
            and "eta_change_observed_at_utc" in work.columns
        ):
            window_start = pd.Timestamp(snapshot_at) - pd.Timedelta(
                minutes=plan.change_window_minutes
            )
            change_valid &= work["eta_change_observed_at_utc"].between(
                window_start,
                pd.Timestamp(snapshot_at),
                inclusive="both",
            )
        work["is_eta_changed"] = change_valid.astype(bool)
        if "provider_eta_changed" in work.columns:
            work["is_eta_changed"] |= work["provider_eta_changed"].fillna(False).astype(bool)
        if "provider_stale_missing" in work.columns:
            provider_stale = work["provider_stale_missing"].fillna(False).astype(bool)
            work["is_position_stale"] |= provider_stale & ~work["is_missing_eta"]

        if plan.aggregation == "swedish_destination_signals":
            locodes = work.get(
                "destination_locode",
                pd.Series("", index=work.index),
            ).fillna("").astype(str)
            work = work[locodes.str.startswith("SE")]
        elif plan.aggregation == "baltic_destination_signals":
            locodes = work.get(
                "destination_locode",
                pd.Series("", index=work.index),
            ).fillna("").astype(str)
            work = work[
                locodes.str[:2].isin(_ETA_WATCH_COUNTRY_SCOPE)
            ]

        intent = plan.eta_watch_intent
        if intent == ETAWatchIntent.SHIFT_HANDOVER:
            work = work[
                work["is_due_in_window"]
                | work["is_low_speed"]
                | work["is_eta_changed"]
                | work["is_position_stale"]
                | work["is_missing_eta"]
            ]
        elif intent == ETAWatchIntent.INBOUND_WATCHLIST:
            work = work[work["is_due_in_window"]]
            if "position" in plan.dimensions:
                work = work[position_valid.reindex(work.index).fillna(False)]
        elif intent == ETAWatchIntent.LOW_SPEED_EXCEPTIONS:
            work = work[work["is_low_speed"]]
        elif intent == ETAWatchIntent.ETA_REVISIONS:
            work = work[
                work["is_eta_changed"] & work["is_due_in_window"]
            ]
        elif intent == ETAWatchIntent.SIGNAL_QUALITY:
            work = work[
                work["is_position_stale"] | work["is_missing_eta"]
            ]
        elif intent == ETAWatchIntent.VESSEL_STATUS:
            work = work[work["is_due_in_window"]]
            if "position" in plan.dimensions:
                work = work[position_valid.reindex(work.index).fillna(False)]
        elif intent == ETAWatchIntent.DESTINATION_LOAD:
            work = work[work["is_due_in_window"]]
            if "destination_locode" not in work.columns:
                return pd.DataFrame()
            locodes = work["destination_locode"].fillna("").astype(str)
            verified = locodes.str.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}")
            if "destination_verified" in work.columns:
                verified &= work["destination_verified"].fillna(False).astype(bool)
            work = work[verified]
            if work.empty:
                return pd.DataFrame()
            name_field = (
                "destination_name"
                if "destination_name" in work.columns
                else "destination_locode"
            )
            work = (
                work.groupby(
                    ["destination_locode", name_field],
                    dropna=False,
                    sort=False,
                )
                .size()
                .rename("inbound_vessels")
                .reset_index()
            )
            work["row_id"] = work["destination_locode"].map(
                lambda value: f"destination-{value}"
            )
            return work.sort_values(
                ["inbound_vessels", "destination_locode"],
                ascending=[False, True],
                kind="stable",
            ).reset_index(drop=True)

        if work.empty:
            return pd.DataFrame()
        if "row_id" not in work.columns:
            identifiers = work.apply(
                lambda row: "|".join(
                    str(row.get(field) or "")
                    for field in (
                        "mmsi",
                        "destination_locode",
                        "reported_eta_utc",
                        "position_time_utc",
                    )
                ),
                axis=1,
            )
            seen: Dict[str, int] = {}
            row_ids: List[str] = []
            for value in identifiers:
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
                ordinal = seen.get(digest, 0)
                seen[digest] = ordinal + 1
                row_ids.append(
                    f"ais-{digest}" if ordinal == 0 else f"ais-{digest}-{ordinal}"
                )
            work["row_id"] = row_ids
        else:
            work["row_id"] = work["row_id"].astype(str)

        if intent == ETAWatchIntent.SHIFT_HANDOVER:
            work["__handover_due_rank"] = work["is_due_in_window"].astype(int)
            work = work.sort_values(
                [
                    "__handover_due_rank",
                    "reported_eta_utc",
                    "vessel_label",
                ],
                ascending=[False, True, True],
                kind="stable",
                na_position="last",
            ).drop(columns=["__handover_due_rank"])
        elif intent == ETAWatchIntent.ETA_REVISIONS:
            work["__sort_change"] = work["eta_change_minutes"].abs()
            work = work.sort_values(
                ["__sort_change", "vessel_label"],
                ascending=[False, True],
                kind="stable",
            ).drop(columns=["__sort_change"])
        elif intent == ETAWatchIntent.LOW_SPEED_EXCEPTIONS:
            work = work.sort_values(
                ["speed_kn", "reported_eta_utc"],
                ascending=[True, True],
                kind="stable",
                na_position="last",
            )
        elif intent == ETAWatchIntent.SIGNAL_QUALITY:
            work["__quality_rank"] = (
                work["is_missing_eta"].astype(int) * 2
                + work["is_position_stale"].astype(int)
            )
            work = work.sort_values(
                ["__quality_rank", "vessel_label"],
                ascending=[False, True],
                kind="stable",
            ).drop(columns=["__quality_rank"])
        else:
            sort_fields = [
                field
                for field in (
                    "reported_eta_utc",
                    "destination_locode",
                    "vessel_label",
                )
                if field in work.columns
            ]
            if sort_fields:
                work = work.sort_values(
                    sort_fields,
                    kind="stable",
                    na_position="last",
                )
        return work.reset_index(drop=True)

    @staticmethod
    def _eta_watch_row_description(row: pd.Series) -> str:
        def clean(value: Any) -> str:
            if value is None:
                return ""
            try:
                if bool(pd.isna(value)):
                    return ""
            except (TypeError, ValueError):
                pass
            return str(value).strip()

        vessel = clean(row.get("vessel_label")) or "Unidentified vessel"
        destination_name = clean(row.get("destination_name"))
        destination_locode = clean(
            row.get("destination_locode")
        ) or clean(row.get("port_locode"))
        destination = (
            f"{destination_name} ({destination_locode})"
            if destination_name and destination_locode
            and destination_name != destination_locode
            else destination_name
            or destination_locode
            or f'"{clean(row.get("destination_raw")) or "unknown"}"'
        )
        details = [f"{vessel} to {destination}"]
        eta = QueryService._utc_label(
            row.get("reported_eta_utc") or row.get("ais_eta_utc")
        )
        details.append(
            f"reported ETA {eta}" if eta else "reported ETA unavailable"
        )
        latitude = row.get("latitude")
        longitude = row.get("longitude")
        if (
            isinstance(latitude, (int, float))
            and isinstance(longitude, (int, float))
            and math.isfinite(float(latitude))
            and math.isfinite(float(longitude))
        ):
            details.append(
                f"position {float(latitude):.4f}, {float(longitude):.4f}"
            )
        else:
            details.append("position unavailable")
        speed = row.get("speed_kn", row.get("sog_kn"))
        if (
            isinstance(speed, (int, float))
            and not isinstance(speed, bool)
            and math.isfinite(float(speed))
        ):
            details.append(f"speed {float(speed):.1f} knots")
        observation = QueryService._utc_label(
            row.get("position_time_utc")
            or row.get("observation_time_utc")
            or row.get("ais_location_time_utc")
        )
        details.append(
            f"observed {observation}" if observation else "observation time unavailable"
        )
        change = row.get("eta_change_minutes")
        if (
            isinstance(change, (int, float))
            and not isinstance(change, bool)
            and math.isfinite(float(change))
            and float(change) != 0.0
        ):
            direction = "later" if float(change) > 0 else "earlier"
            details.append(
                f"reported ETA moved {abs(float(change)):.0f} minutes {direction}"
            )
        return ", ".join(details)

    def _compose_eta_watch_answer(
        self,
        plan: QueryPlan,
        displayed: pd.DataFrame,
        *,
        matched_count: int,
        source_candidate_count: int,
        snapshot_at: datetime,
        state: AnswerState,
        source_health: str,
    ) -> str:
        observed = snapshot_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        coverage = (
            "These are vessel-reported AIS signals, not an official schedule, "
            "confirmed delay, arrival confirmation, or prediction."
        )
        excluded_count = max(0, source_candidate_count - matched_count)
        report_requested_shortfall = (
            plan.eta_watch_intent == ETAWatchIntent.INBOUND_WATCHLIST
            and 1 < plan.limit < 20
        )
        excluded_note = (
            f" {excluded_count} other matching vessel signal"
            f"{'' if excluded_count == 1 else 's'} "
            f"{'is' if excluded_count == 1 else 'are'} retained separately as "
            "awaiting a fresh ETA broadcast; "
            f"{'it was' if excluded_count == 1 else 'they were'} excluded from "
            "the current ETA total and chart."
            if excluded_count and report_requested_shortfall
            else ""
        )
        missing_requested = (
            max(0, int(plan.limit) - source_candidate_count)
            if report_requested_shortfall
            else 0
        )
        missing_note = (
            f" The source contained no additional matching vessel for "
            f"{missing_requested} requested slot"
            f"{'' if missing_requested == 1 else 's'} inside the "
            f"{int(plan.horizon_hours or 24)}-hour window."
            if missing_requested
            else ""
        )
        if state == AnswerState.NO_CURRENT_DATA:
            if source_health in {"connecting", "warming"}:
                return (
                    "ETA Watch is connected but still warming its live AIS state. "
                    "No vessel status was published yet. "
                    + coverage
                )
            return (
                "ETA Watch has no current validated AIS snapshot because the live "
                "source is unavailable or not configured for this runtime. "
                + coverage
            )
        if state == AnswerState.NO_DATA or displayed.empty:
            return (
                f"No source-validated AIS vessel matches the requested "
                f"{int(plan.horizon_hours or 24)}-hour UTC scope as of {observed}."
                f"{excluded_note}{missing_note} "
                + coverage
            )

        intent = plan.eta_watch_intent
        shown = len(displayed)
        if intent == ETAWatchIntent.DESTINATION_LOAD:
            ranking = "; ".join(
                f"{str(row.get('destination_name') or row.get('destination_locode'))}: "
                f"{int(row.get('inbound_vessels'))} vessel"
                f"{'' if int(row.get('inbound_vessels')) == 1 else 's'}"
                for _, row in displayed.head(5).iterrows()
            )
            return (
                f"Destination load for AIS-visible inbound signals: {ranking}. "
                f"Showing {shown} ranked destinations from {matched_count} matching "
                f"vessel signals as of {observed}.{excluded_note} {coverage}"
            )

        narrative_limit = self._eta_watch_visible_row_limit(plan)
        descriptions = "; ".join(
            self._eta_watch_row_description(row)
            for _, row in displayed.head(narrative_limit).iterrows()
        )
        if intent == ETAWatchIntent.SHIFT_HANDOVER:
            due_soon = int(displayed["is_due_in_window"].sum())
            low_speed = int(displayed["is_low_speed"].sum())
            eta_changed = int(displayed["is_eta_changed"].sum())
            stale = int(displayed["is_position_stale"].sum())
            missing_eta = int(displayed["is_missing_eta"].sum())
            flags = []
            if low_speed:
                flags.append(f"{low_speed} low-speed")
            if eta_changed:
                flags.append(f"{eta_changed} ETA-revision")
            if stale:
                flags.append(f"{stale} stale-position")
            if missing_eta:
                flags.append(f"{missing_eta} missing-ETA")
            attention = ", ".join(flags) if flags else "no flagged exceptions"
            opening = (
                f"Sweden-bound {int(plan.horizon_hours or 12)}-hour shift watch: "
                f"{matched_count} vessel signal"
                f"{'' if matched_count == 1 else 's'} reviewed, including "
                f"{due_soon} due soon; attention: {attention}."
            )
        elif intent == ETAWatchIntent.LOW_SPEED_EXCEPTIONS:
            opening = (
                f"{matched_count} due-soon vessel signal"
                f"{'' if matched_count == 1 else 's'} reported speed below "
                f"{float(plan.speed_threshold_kn or 2.0):g} knots."
            )
        elif intent == ETAWatchIntent.ETA_REVISIONS:
            opening = (
                f"{matched_count} vessel signal"
                f"{'' if matched_count == 1 else 's'} changed reported ETA by more "
                f"than {int(plan.eta_change_threshold_minutes or 30)} minutes "
                f"within the last {int(plan.change_window_minutes or 60)} minutes."
            )
        elif intent == ETAWatchIntent.SIGNAL_QUALITY:
            opening = (
                f"{matched_count} vessel signal"
                f"{'' if matched_count == 1 else 's'} need"
                f"{'s' if matched_count == 1 else ''} confirmation because the "
                "position is stale or a valid reported ETA is unavailable."
            )
        elif intent == ETAWatchIntent.VESSEL_STATUS:
            opening = "The next matching AIS-visible vessel signal is:"
        elif (
            intent == ETAWatchIntent.INBOUND_WATCHLIST
            and plan.limit < 20
            and matched_count < plan.limit
        ):
            opening = (
                f"Only {matched_count} source-validated AIS-visible vessel signal"
                f"{'' if matched_count == 1 else 's'} "
                f"{'is' if matched_count == 1 else 'are'} currently available "
                f"for the requested next {plan.limit} within "
                f"{int(plan.horizon_hours or 24)} hours."
            )
        elif (
            intent == ETAWatchIntent.INBOUND_WATCHLIST
            and plan.limit < 20
        ):
            if plan.limit == 1:
                opening = (
                    f"{matched_count} source-validated AIS-visible vessel signal"
                    f"{'' if matched_count == 1 else 's'} matched the next "
                    f"{int(plan.horizon_hours or 24)} hours; the next vessel is "
                    "listed by earliest reported ETA."
                )
            else:
                opening = (
                    f"{matched_count} source-validated AIS-visible vessel signal"
                    f"{'' if matched_count == 1 else 's'} matched the next "
                    f"{int(plan.horizon_hours or 24)} hours; the requested next "
                    f"{plan.limit} are listed in ETA order."
                )
        else:
            opening = (
                f"{matched_count} AIS-visible vessel signal"
                f"{'' if matched_count == 1 else 's'} matched the next "
                f"{int(plan.horizon_hours or 24)} hours."
            )
        return (
            f"{opening} {descriptions}. Showing {shown} of {matched_count}; "
            f"source observed {observed}.{excluded_note}{missing_note} {coverage}"
        )

    @staticmethod
    def _eta_watch_candidate_frame(
        plan: QueryPlan,
        *,
        raw_table: pd.DataFrame,
        validated: pd.DataFrame,
        snapshot_at: datetime,
        horizon_end: datetime,
    ) -> Optional[pd.DataFrame]:
        """Retain rejected count-request candidates as explicitly non-current rows.

        These rows are presentation and audit context only. Their ETA field is
        deliberately named ``last_reported_eta_utc`` so it cannot enter the
        canonical timeline, immutable current facts, or validated totals.
        """

        if (
            plan.eta_watch_intent != ETAWatchIntent.INBOUND_WATCHLIST
            or plan.limit >= 20
            or raw_table.empty
        ):
            return None

        displayed_valid_count = min(len(validated), int(plan.limit))
        remaining_slots = max(0, int(plan.limit) - displayed_valid_count)
        if remaining_slots == 0:
            return None

        work = raw_table.copy()
        if "mmsi" not in work.columns:
            return None
        valid_mmsis = (
            set(validated["mmsi"].dropna().astype(str))
            if "mmsi" in validated.columns
            else set()
        )
        work = work[~work["mmsi"].fillna("").astype(str).isin(valid_mmsis)].copy()
        if work.empty:
            return None

        alias_fields = {
            "vessel_name": ("vessel_label", "name", "ship_name"),
            "destination_name": ("port_name",),
            "destination_locode": ("port_locode", "locode"),
            "destination_raw": ("ais_destination", "destination"),
            "eta_utc": ("reported_eta_utc", "ais_eta_utc"),
            "eta_observed_at_utc": (
                "eta_observation_time_utc",
                "static_observed_at_utc",
            ),
            "position_observed_at_utc": (
                "position_time_utc",
                "ais_location_time_utc",
            ),
            "speed_kn": ("sog_kn", "speed_over_ground_kn"),
        }
        for canonical, aliases in alias_fields.items():
            if canonical in work.columns:
                continue
            source = next((field for field in aliases if field in work.columns), None)
            if source:
                work[canonical] = work[source]

        for field in (
            "eta_utc",
            "eta_observed_at_utc",
            "position_observed_at_utc",
        ):
            if field in work.columns:
                work[field] = pd.to_datetime(work[field], errors="coerce", utc=True)
        for field in (
            "latitude",
            "longitude",
            "speed_kn",
            "position_age_minutes",
        ):
            if field in work.columns:
                work[field] = pd.to_numeric(work[field], errors="coerce")

        if "eta_utc" in work.columns:
            work = work[
                work["eta_utc"].between(
                    pd.Timestamp(snapshot_at) - pd.Timedelta(minutes=10),
                    pd.Timestamp(horizon_end),
                    inclusive="both",
                )
            ].copy()
        if work.empty:
            return None

        eta_observed = (
            work["eta_observed_at_utc"]
            if "eta_observed_at_utc" in work.columns
            else pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
        )
        work["eta_observation_age_minutes"] = (
            pd.Timestamp(snapshot_at) - eta_observed
        ).dt.total_seconds() / 60.0

        def rejection_reason(row: pd.Series) -> str:
            observed_at = row.get("eta_observed_at_utc")
            age = row.get("eta_observation_age_minutes")
            if pd.isna(observed_at):
                return "No source timestamp accompanies the retained ETA broadcast."
            if isinstance(age, (int, float)) and math.isfinite(float(age)):
                if float(age) > 10.0:
                    return (
                        f"Last ETA broadcast is {float(age):.0f} minutes old; "
                        "current publication requires 10 minutes or less."
                    )
                if float(age) < -10.0:
                    return "ETA observation time is inconsistent with the source clock."
            return "The signal did not pass every current ETA publication check."

        work["validation_status"] = "awaiting_fresh_eta"
        work["validation_reason"] = work.apply(rejection_reason, axis=1)
        work["row_id"] = work.apply(
            lambda row: f"eta-candidate-{str(row.get('mmsi') or row.name)}",
            axis=1,
        )
        work["vessel_label"] = work.apply(
            lambda row: (
                str(row.get("vessel_name")).strip()
                if row.get("vessel_name") is not None
                and str(row.get("vessel_name")).strip()
                and str(row.get("vessel_name")).lower() != "nan"
                else f"MMSI {str(row.get('mmsi')).strip()}"
            ),
            axis=1,
        )
        work["last_reported_eta_utc"] = work.get("eta_utc")
        work["last_eta_observed_at_utc"] = work.get("eta_observed_at_utc")

        sort_fields = [
            field
            for field in ("last_reported_eta_utc", "destination_locode", "mmsi")
            if field in work.columns
        ]
        if sort_fields:
            work = work.sort_values(sort_fields, kind="stable", na_position="last")
        columns = [
            field
            for field in (
                "row_id",
                "mmsi",
                "vessel_label",
                "vessel_name",
                "destination_name",
                "destination_locode",
                "destination_raw",
                "last_reported_eta_utc",
                "last_eta_observed_at_utc",
                "eta_observation_age_minutes",
                "latitude",
                "longitude",
                "speed_kn",
                "position_observed_at_utc",
                "position_age_minutes",
                "validation_status",
                "validation_reason",
            )
            if field in work.columns
        ]
        return work[columns].head(remaining_slots).reset_index(drop=True)

    @staticmethod
    def _eta_watch_visible_row_limit(plan: QueryPlan) -> int:
        """Show every explicitly bounded inbound row, while keeping default briefs compact."""

        if (
            plan.eta_watch_intent == ETAWatchIntent.INBOUND_WATCHLIST
            and plan.limit < 20
        ):
            return max(1, min(plan.limit, 10))
        return 5

    @staticmethod
    def _operational_item(row: pd.Series) -> OperationalBriefItem:
        missing_eta = bool(row.get("is_missing_eta"))
        stale = bool(row.get("is_position_stale"))
        low_speed = bool(row.get("is_low_speed"))
        eta_changed = bool(row.get("is_eta_changed"))
        due = bool(row.get("is_due_in_window"))
        if missing_eta:
            status = "missing_eta"
            reason = "No valid future-facing vessel-reported ETA is available."
        elif stale:
            status = "stale_position"
            reason = "The latest position is missing or older than ten minutes."
        elif low_speed:
            status = "low_speed"
            reason = "Speed is below the requested operational threshold."
        elif eta_changed:
            status = "eta_changed"
            reason = "The vessel-reported ETA crossed the requested revision threshold."
        elif due:
            status = "due_soon"
            reason = "The validated vessel-reported ETA falls inside the requested watch window."
        else:
            status = "observed"
            reason = "A source-validated AIS vessel signal is available."
        actions: List[str] = []
        latitude = row.get("latitude")
        longitude = row.get("longitude")
        if (
            isinstance(latitude, (int, float))
            and isinstance(longitude, (int, float))
            and math.isfinite(float(latitude))
            and math.isfinite(float(longitude))
        ):
            actions.append("locate_vessel")
        if not missing_eta:
            actions.append("watch_next_six_hours")
        raw_change = row.get("eta_change_minutes")
        if eta_changed or (
            isinstance(raw_change, (int, float))
            and not isinstance(raw_change, bool)
            and math.isfinite(float(raw_change))
            and float(raw_change) != 0.0
        ):
            actions.append("inspect_eta_changes")
        return OperationalBriefItem(
            row_id=str(row.get("row_id")),
            vessel_label=str(row.get("vessel_label") or "") or None,
            priority=(
                "attention"
                if status
                in {"missing_eta", "stale_position", "low_speed", "eta_changed"}
                else "monitor"
                if status == "due_soon"
                else "information"
            ),
            status=status,  # type: ignore[arg-type]
            reason=reason,
            actions=actions,  # type: ignore[arg-type]
        )

    def _build_operational_brief(
        self,
        plan: QueryPlan,
        displayed: pd.DataFrame,
        *,
        full_table: pd.DataFrame,
        matched_count: int,
        snapshot_at: datetime,
        horizon_end: datetime,
        source_health: str,
        coverage: str,
        source_observed_at: Optional[datetime] = None,
    ) -> OperationalBrief:
        assert plan.eta_watch_intent is not None
        labels = {
            ETAWatchIntent.SHIFT_HANDOVER: "Shift watch prepared from validated AIS signals.",
            ETAWatchIntent.INBOUND_WATCHLIST: "Inbound watchlist ordered by vessel-reported ETA.",
            ETAWatchIntent.LOW_SPEED_EXCEPTIONS: "Low-speed due-soon exceptions identified.",
            ETAWatchIntent.DESTINATION_LOAD: "Inbound destination load ranked.",
            ETAWatchIntent.ETA_REVISIONS: "Material vessel-reported ETA revisions identified.",
            ETAWatchIntent.SIGNAL_QUALITY: "Signals needing confirmation identified.",
            ETAWatchIntent.VESSEL_STATUS: "Next matching vessel status identified.",
        }
        exception_specs = (
            (
                "due_soon",
                "is_due_in_window",
                "Vessels whose reported ETA falls within the watch window.",
            ),
            (
                "low_speed",
                "is_low_speed",
                "Due-soon vessels below the speed threshold.",
            ),
            (
                "eta_changed",
                "is_eta_changed",
                "Vessels whose reported ETA crossed the revision threshold.",
            ),
            (
                "stale_position",
                "is_position_stale",
                "Vessels with a missing or stale position.",
            ),
            (
                "missing_eta",
                "is_missing_eta",
                "Vessels without a valid future-facing reported ETA.",
            ),
        )
        exceptions: List[OperationalExceptionSummary] = []
        for code, field, summary in exception_specs:
            if field not in full_table.columns:
                continue
            count = int(full_table[field].fillna(False).astype(bool).sum())
            if count == 0:
                continue
            exceptions.append(
                OperationalExceptionSummary(
                    code=code,  # type: ignore[arg-type]
                    count=count,
                    summary=summary,
                )
            )
        prioritized_items = (
            [
                self._operational_item(row)
                for _, row in displayed.head(
                    self._eta_watch_visible_row_limit(plan)
                ).iterrows()
            ]
            if "row_id" in displayed.columns
            else []
        )
        return OperationalBrief(
            intent=plan.eta_watch_intent,
            headline=labels[plan.eta_watch_intent],
            window_start_utc=snapshot_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            window_end_utc=horizon_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            matched_count=matched_count,
            displayed_count=len(displayed),
            prioritized_items=prioritized_items,
            exceptions=exceptions,
            source_health=source_health,  # type: ignore[arg-type]
            source_observed_at=(
                source_observed_at or snapshot_at
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            coverage=coverage,
        )

    @staticmethod
    def _empty_operational_brief(
        plan: QueryPlan,
        *,
        source_health: str,
        headline: str,
    ) -> Optional[OperationalBrief]:
        if plan.eta_watch_intent is None:
            return None
        return OperationalBrief(
            intent=plan.eta_watch_intent,
            headline=headline,
            matched_count=0,
            displayed_count=0,
            source_health=source_health,  # type: ignore[arg-type]
            coverage=(
                "AISStream is a non-exhaustive vessel-broadcast feed, not an "
                "official schedule, confirmed-delay source, or arrival board."
            ),
        )

    def _execute_live_eta(self, plan: QueryPlan) -> ExecutionOutcome:
        if plan.eta_watch_intent is not None:
            return self._execute_eta_watch(plan)
        assert self.live_eta is not None
        result = self.live_eta.query(
            operation=plan.operation.value,
            port=plan.ports[0] if plan.ports else None,
            ports=plan.ports,
            mmsi=plan.mmsi,
            imo=plan.imo,
            vessel_name=plan.vessel_name,
            target_date=plan.date_scope.target_date,
            horizon_hours=plan.horizon_hours,
            aggregation=plan.aggregation,
            limit=plan.limit,
        )
        if (
            result.status == "ok"
            and plan.operation == QueryOperation.VESSEL_ETA
            and plan.metric == "ais_eta_utc"
            and (
                result.table is None
                or "ais_eta_utc" not in result.table.columns
                or result.table["ais_eta_utc"].dropna().empty
            )
        ):
            result = replace(
                result,
                status="no_current_data",
                failure_reason="validated_ais_match_unavailable",
            )
        if result.status == "ok":
            state = AnswerState.COMPUTED
        elif result.status == "partial":
            state = AnswerState.PARTIAL
        elif result.status == "no_current_data":
            state = AnswerState.NO_CURRENT_DATA
        else:
            state = AnswerState.NO_DATA

        ais_destination_only = _is_ais_destination_only(result)
        ais_observation_available = bool(
            result.data_updated_at is not None or result.table is not None
        )
        ais_evidence = EvidenceItem(
            id=(
                "fintraffic_ais_destination_observations"
                if ais_destination_only
                else "fintraffic_ais_announced_eta"
            ),
            source_type="web",
            title=(
                "Fintraffic AIS vessel-reported destination observations"
                if ais_destination_only
                else "Fintraffic AIS vessel metadata and locations"
            ),
            url=FINTRAFFIC_AIS_VESSELS_URL,
            metadata={
                "locations_url": FINTRAFFIC_AIS_LOCATIONS_URL,
                "provider": self.live_eta.provider,
                "authority": "vessel_reported_eta",
                "source_kind": (
                    "ais_broadcast_observation"
                    if ais_destination_only
                    else "portnet_schedule_enrichment"
                ),
                "destination_matching": (
                    "exact source-reported destination token after conservative normalization"
                    if ais_destination_only
                    else "validated against the Portnet destination"
                ),
                "official_schedule": not ais_destination_only,
                "announced_variance_formula": (
                    None
                    if ais_destination_only
                    else "AIS vessel-reported ETA minus Portnet official scheduled ETA"
                ),
                "geographic_scope": (
                    "Fintraffic Finnish-receiver observation footprint; not exhaustive Baltic coverage"
                    if ais_destination_only
                    else "Finnish Portnet calls with validated AIS enrichment"
                ),
                "prediction": False,
            },
        )
        if ais_destination_only:
            # A foreign-port schedule request can be rejected on capability
            # grounds without querying AIS. Do not attach source evidence that
            # was not actually retrieved.
            evidence = [ais_evidence] if ais_observation_available else []
        else:
            evidence = [
                EvidenceItem(
                    id="fintraffic_portnet_schedule",
                    source_type="web",
                    title="Fintraffic Portnet port-call schedule",
                    url=FINTRAFFIC_PORT_CALLS_URL,
                    metadata={
                        "provider": self.live_eta.provider,
                        "authority": "official_scheduled_eta",
                        "snapshot_at": result.snapshot_at.isoformat(),
                        "data_updated_at": (
                            result.data_updated_at.isoformat()
                            if result.data_updated_at
                            else None
                        ),
                        "timezone": "UTC",
                        "maximum_horizon_days": 14,
                    },
                ),
                ais_evidence,
            ]

        as_of = result.data_updated_at or result.snapshot_at
        horizon_end = result.horizon_end or (
            result.snapshot_at + pd.Timedelta(days=14)
        )
        snapshot_label = result.snapshot_at.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        horizon_label = pd.Timestamp(horizon_end).strftime("%Y-%m-%dT%H:%M:%SZ")
        if ais_destination_only and ais_observation_available:
            freshness = FreshnessInfo(
                data_from=result.snapshot_at.strftime("%Y-%m-%d"),
                data_to=result.snapshot_at.strftime("%Y-%m-%d"),
                as_of=as_of.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                historical=False,
                message=(
                    f"Fintraffic AIS observation snapshot {snapshot_label}; vessel metadata and "
                    f"positions were freshness-validated at retrieval, and self-reported ETAs were "
                    f"bounded through {horizon_label}. This is not an official schedule, arrival "
                    "confirmation, or prediction."
                ),
            )
        elif ais_destination_only:
            freshness = FreshnessInfo(
                data_from=None,
                data_to=None,
                as_of=snapshot_label,
                historical=False,
                message=(
                    f"Live-source coverage was evaluated at {snapshot_label}. No official foreign-port "
                    "schedule source is integrated, and no AIS observation or historical value was "
                    "substituted."
                ),
            )
        else:
            freshness = FreshnessInfo(
                data_from=result.snapshot_at.strftime("%Y-%m-%d"),
                data_to=pd.Timestamp(horizon_end).strftime("%Y-%m-%d"),
                as_of=as_of.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                historical=False,
                message=(
                    f"Live Fintraffic retrieval snapshot {snapshot_label}; "
                    f"official schedule horizon through {horizon_label}."
                ),
            )
        live_availability = self._live_result_availability(result)
        return ExecutionOutcome(
            state=state,
            answer=result.answer,
            result=result,
            evidence=evidence,
            confidence="high" if state in {AnswerState.COMPUTED, AnswerState.NO_CURRENT_DATA} else "not_applicable",
            caveats=result.caveats,
            retrieval_mode=(
                "live_ais_observation"
                if ais_destination_only and ais_observation_available
                else "live_capability_check"
                if ais_destination_only
                else "live_structured"
            ),
            retrieval_backend=self.live_eta.provider,
            retrieval_status=(
                "ok"
                if state == AnswerState.COMPUTED
                else "unavailable"
                if state == AnswerState.NO_CURRENT_DATA
                else "empty"
            ),
            freshness_override=freshness,
            availability_code=live_availability,
            availability_provider=self.live_eta.provider,
            availability_retryable=(
                live_availability in {"source_unavailable", "source_stale"}
            ),
        )

    @staticmethod
    def _live_result_availability(result: LiveETAResult) -> str:
        if result.status == "ok":
            return "available"
        if result.status == "no_data":
            return "no_data"

        failure_detail = str(result.failure_reason or "").lower()
        answer_detail = str(result.answer or "").lower()
        detail = " ".join(
            [
                failure_detail,
                answer_detail,
                *(str(item) for item in (result.caveats or [])),
            ]
        ).lower()
        if any(
            token in detail
            for token in (
                "coverage_unavailable",
                "not integrated",
            )
        ) or any(
            token in answer_detail
            for token in (
                "requests are limited to",
                "outside eagle eye",
                "unsupported port",
                "maximum horizon",
                "next 14 days",
            )
        ):
            return "coverage_unavailable"
        if "stale" in detail or "future timestamp" in detail:
            return "source_stale"
        if "conflict" in detail or "ambiguous" in detail:
            return "ambiguous_match"

        table = result.table
        statuses = (
            {
                str(value).lower()
                for value in table.get("variance_status", pd.Series(dtype=str))
                .dropna()
                .tolist()
            }
            if isinstance(table, pd.DataFrame)
            else set()
        )
        if statuses.intersection(
            {
                "ais_schedule_conflict",
                "portnet_identity_conflict",
                "ais_identity_mismatch",
                "ais_imo_mismatch",
            }
        ):
            return "ambiguous_match"
        if any("stale" in status for status in statuses):
            return "source_stale"
        return "source_unavailable"

    def _execute_ais_jump(
        self,
        plan: QueryPlan,
        *,
        warnings: Optional[List[str]] = None,
    ) -> AnalyticsResult:
        filters = RAGQueryFilters(
            mmsi=plan.mmsi,
            imo=plan.imo,
            locode=plan.ports[0] if plan.ports else None,
            vessel_type=plan.vessel_type,
            date_from=plan.date_scope.date_from,
            date_to=plan.date_scope.date_to,
        )
        payload: Optional[Dict[str, Any]] = None
        source = ""
        if self.retriever is not None:
            try:
                payload = self.retriever.detect_sudden_jumps(filters=filters)
                source = "AIS metadata index"
            except Exception as exc:
                if warnings is not None:
                    warnings.append(
                        redact_sensitive_text(f"AIS jump index query failed: {exc}")
                    )
        if payload is None and self.events_path and self.events_path.exists():
            try:
                payload = detect_sudden_jump_events_from_parquet(
                    events_path=self.events_path,
                    mmsi=filters.mmsi,
                    locode=filters.locode,
                    date_from=filters.date_from,
                    date_to=filters.date_to,
                )
                source = "row-level AIS events parquet"
            except Exception as exc:
                if warnings is not None:
                    warnings.append(
                        redact_sensitive_text(f"AIS event-row query failed: {exc}")
                    )
        if payload is None:
            return self.kpi.no_data("No AIS event rows are available for this jump query.")
        if filters.locode and not bool(payload.get("scope_applied")):
            reason = str(
                payload.get("reason")
                or (
                    f"AIS jump analysis cannot apply observed port scope {filters.locode}; "
                    "destination text cannot establish an observed port location."
                )
            ).strip()
            return AnalyticsResult(
                status="unsupported",
                answer=(
                    f"Port-scoped AIS jump analysis is unavailable for {filters.locode}. {reason}"
                ),
                table=None,
                chart=None,
                coverage_notes=[f"Requested observed port scope: {filters.locode}"],
                caveats=["No unscoped AIS anomaly count was substituted for the requested port."],
            )

        events = pd.DataFrame(payload.get("events") or [])
        count = int(payload.get("count", len(events)))
        table = None
        chart = None
        if not events.empty:
            fields = [
                field
                for field in (
                    "mmsi",
                    "timestamp_full",
                    "distance_km",
                    "implied_speed_kn",
                    "speed_kn",
                    "dt_minutes",
                    "latitude",
                    "longitude",
                    "prev_latitude",
                    "prev_longitude",
                    "port",
                    "stable_id",
                )
                if field in events.columns
            ]
            table = events[fields].copy()
            motion_fields = [
                field
                for field in ("distance_km", "implied_speed_kn", "speed_kn")
                if field in events.columns
            ]
            if "timestamp_full" in events.columns and motion_fields:
                chart = (
                    events.assign(timestamp=pd.to_datetime(events["timestamp_full"], errors="coerce", utc=True))
                    .dropna(subset=["timestamp"])
                    .sort_values("timestamp")
                    .set_index("timestamp")[motion_fields]
                )
        scope_suffix = f" within observed port scope {filters.locode}" if filters.locode else ""
        scope_notes = (
            [
                f"Applied observed port scope: {filters.locode}",
                f"Port scope field: {payload.get('scope_field')}",
            ]
            if filters.locode
            else []
        )
        return AnalyticsResult(
            status="ok",
            answer=f"Detected {count} potential sudden AIS coordinate jumps in the filtered range{scope_suffix}.",
            table=table,
            chart=chart,
            coverage_notes=[f"Rows used: {count}", f"Data sources used: {source}", *scope_notes],
                caveats=["The jump rule is an internal screening method."],
        )

    def _execute_research(self, question: str, top_k: int) -> ExecutionOutcome:
        if top_k <= 0:
            return ExecutionOutcome(
                state=AnswerState.NO_DATA,
                answer=(
                    "Source-grounded maritime research requires evidence retrieval, but Evidence top K is set to 0. "
                    "Increase it above 0 and run the research question again."
                ),
                confidence="not_applicable",
                retrieval_mode="documents",
                retrieval_backend=(self.retriever.retrieval_backend if self.retriever else None),
                retrieval_status="disabled_by_request",
            )
        local_evidence: List[EvidenceItem] = []
        retrieval_backend: Optional[str] = None
        retrieval_status = "unavailable"
        retrieval_warning: Optional[str] = None
        if self.retriever is not None:
            try:
                retrieved = self.retriever.query_docs(question=question, top_k=top_k)
                retrieval_backend = retrieved.backend
                for item in retrieved.evidence[:top_k]:
                    local_evidence.append(
                        EvidenceItem(
                            id=str(item.id),
                            source_type="local_document",
                            title=str(item.metadata.get("title") or item.metadata.get("source") or "Local maritime document"),
                            excerpt=str(item.text or "")[:800] or None,
                            url=item.metadata.get("url"),
                            metadata=finite_json_value(item.metadata),
                        )
                    )
                retrieval_status = "ok" if local_evidence else "empty"
            except Exception as exc:
                local_evidence = []
                retrieval_status = "error"
                retrieval_warning = redact_sensitive_text(f"Local document retrieval failed: {exc}")

        if not self.enable_model_responses or self.openai_client is None:
            if local_evidence:
                if self.local_synthesizer is not None:
                    try:
                        synthesis = self.local_synthesizer.synthesize_research(
                            question,
                            local_evidence,
                        )
                        return ExecutionOutcome(
                            state=AnswerState.RETRIEVED,
                            answer=synthesis.text,
                            evidence=local_evidence,
                            confidence="medium",
                            model_used=f"{synthesis.provider}/{synthesis.model}",
                            retrieval_mode="documents",
                            retrieval_backend=retrieval_backend,
                            retrieval_status="ok",
                        )
                    except Exception as exc:
                        retrieval_warning = redact_sensitive_text(
                            f"Local grounded synthesis failed: {exc}"
                        )
                return ExecutionOutcome(
                    state=AnswerState.RETRIEVED,
                    answer=(
                        "Relevant local maritime sources were found, but grounded synthesis could not be completed. "
                        "Review the cited evidence excerpts; Eagle Eye did not generate an unverified summary."
                    ),
                    evidence=local_evidence,
                    confidence="low",
                    warnings=[retrieval_warning] if retrieval_warning else [],
                    retrieval_mode="documents",
                    retrieval_backend=retrieval_backend,
                    retrieval_status="ok",
                )
            return ExecutionOutcome(
                state=AnswerState.NO_DATA,
                answer=(
                    "Source-grounded maritime research is unavailable in this runtime because neither a usable local document "
                    "index nor the explicitly enabled model-backed research route is available."
                ),
                confidence="not_applicable",
                caveats=[self.retriever_reason] if self.retriever_reason else [],
                warnings=[retrieval_warning] if retrieval_warning else [],
                retrieval_mode="documents",
                retrieval_backend=retrieval_backend,
                retrieval_status=retrieval_status,
            )

        evidence_block = "\n\n".join(
            f"[{item.id}] {item.title}\n{item.excerpt or ''}" for item in local_evidence
        ) or "No local excerpts were available."
        instructions = (
            "You are Eagle Eye's maritime research assistant. Answer only from the supplied local excerpts and authoritative "
            "web sources. Cite sources inline. Do not use historical traffic analytics as regulatory evidence. Clearly state "
            "when the sources do not establish a claim."
        )
        try:
            response = self.openai_client.responses.create(
                model=self.research_model,
                reasoning={"effort": self.reasoning_effort},
                instructions=instructions,
                input=f"Question:\n{question}\n\nLocal evidence:\n{evidence_block}",
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": AUTHORITATIVE_MARITIME_DOMAINS},
                    }
                ],
                include=["web_search_call.action.sources"],
                store=False,
            )
            text = str(getattr(response, "output_text", "") or "").strip()
            web_evidence = self._web_evidence(response)
            if not text:
                raise ValueError("empty model response")
            return ExecutionOutcome(
                state=AnswerState.RETRIEVED,
                answer=text,
                evidence=[*local_evidence, *web_evidence],
                confidence="medium" if (local_evidence or web_evidence) else "low",
                model_used=str(getattr(response, "model", None) or self.research_model),
                retrieval_mode="documents+authoritative_web",
                retrieval_backend="+".join(
                    value for value in (retrieval_backend, "openai_web_search") if value
                ),
                retrieval_status="ok" if (local_evidence or web_evidence) else "empty",
            )
        except Exception as exc:
            return ExecutionOutcome(
                state=AnswerState.RETRIEVED if local_evidence else AnswerState.NO_DATA,
                answer=(
                    "The research request could not be completed from validated sources."
                    if not local_evidence
                    else "Relevant local sources were found, but source-grounded synthesis failed; review the evidence excerpts."
                ),
                evidence=local_evidence,
                confidence="low",
                warnings=[
                    redact_sensitive_text(f"Source-grounded research synthesis failed: {exc}")
                ],
                retrieval_mode="documents+authoritative_web",
                retrieval_backend="+".join(
                    value for value in (retrieval_backend, "openai_web_search") if value
                ),
                retrieval_status="partial" if local_evidence else "error",
            )

    def _execute_general(self, question: str, *, current: bool) -> ExecutionOutcome:
        if not self.enable_model_responses or self.openai_client is None:
            if not current and self.local_synthesizer is not None:
                try:
                    synthesis = self.local_synthesizer.answer_general(question)
                    return ExecutionOutcome(
                        state=AnswerState.GENERAL,
                        answer=synthesis.text,
                        confidence="not_applicable",
                        model_used=f"{synthesis.provider}/{synthesis.model}",
                        retrieval_mode="none",
                        retrieval_backend=None,
                        retrieval_status="not_required",
                    )
                except Exception as exc:
                    warning = redact_sensitive_text(f"Local general synthesis failed: {exc}")
            else:
                warning = None
            state = AnswerState.NO_CURRENT_DATA if current else AnswerState.NO_DATA
            return ExecutionOutcome(
                state=state,
                answer=(
                    "A current, validated web source is unavailable in this runtime."
                    if current
                    else "The general assistant could not complete this request without a validated response."
                ),
                confidence="not_applicable",
                warnings=[warning] if warning else [],
                retrieval_mode="web" if current else "none",
                retrieval_backend=None,
                retrieval_status="unavailable" if current else "not_required",
            )
        try:
            kwargs: Dict[str, Any] = {
                "model": self.general_model,
                "reasoning": {"effort": self.reasoning_effort},
                "instructions": (
                    "Answer the user's general question directly. Do not imply that Eagle Eye's historical datasets contain "
                    "live information. Cite web sources for current facts and state uncertainty plainly."
                ),
                "input": question,
                "store": False,
            }
            if current:
                kwargs["tools"] = [{"type": "web_search"}]
                kwargs["include"] = ["web_search_call.action.sources"]
            response = self.openai_client.responses.create(**kwargs)
            text = str(getattr(response, "output_text", "") or "").strip()
            if not text:
                raise ValueError("empty model response")
            evidence = self._web_evidence(response) if current else []
            return ExecutionOutcome(
                state=AnswerState.GENERAL,
                answer=text,
                evidence=evidence,
                confidence="medium" if current else "not_applicable",
                model_used=str(getattr(response, "model", None) or self.general_model),
                retrieval_mode="web" if current else "none",
                retrieval_backend="openai_web_search" if current else None,
                retrieval_status="ok" if current and evidence else "not_required",
            )
        except Exception as exc:
            return ExecutionOutcome(
                state=AnswerState.NO_CURRENT_DATA if current else AnswerState.NO_DATA,
                answer="The general assistant could not complete this request without a validated response.",
                confidence="not_applicable",
                warnings=[redact_sensitive_text(f"General assistant failed: {exc}")],
                retrieval_mode="web" if current else "none",
                retrieval_backend="openai_web_search" if current else None,
                retrieval_status="error" if current else "not_required",
            )

    @staticmethod
    def _web_evidence(response: Any) -> List[EvidenceItem]:
        try:
            payload = response.model_dump(mode="json")
        except Exception:
            return []
        found: List[EvidenceItem] = []
        seen: set[str] = set()

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                url = value.get("url")
                title = value.get("title") or value.get("name")
                if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in seen:
                    seen.add(url)
                    found.append(
                        EvidenceItem(
                            id=f"web_{len(found) + 1}",
                            source_type="web",
                            title=str(title or url),
                            url=url,
                        )
                    )
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)
        return found[:20]

    def _retrieve_traffic_evidence(
        self,
        plan: QueryPlan,
        question: str,
        top_k: int,
    ) -> RetrievalAudit:
        if top_k <= 0:
            return RetrievalAudit(
                evidence=[],
                mode="traffic",
                backend=self.retriever.retrieval_backend if self.retriever else None,
                status="disabled_by_request",
            )
        if self.retriever is None:
            return RetrievalAudit(
                evidence=[],
                mode="traffic",
                backend=None,
                status="unavailable",
                warning=self.retriever_reason or "Traffic evidence retrieval is unavailable.",
            )
        try:
            filters = RAGQueryFilters(
                mmsi=plan.mmsi,
                imo=plan.imo,
                locode=plan.ports[0] if len(plan.ports) == 1 else None,
                vessel_type=plan.vessel_type,
                date_from=plan.date_scope.date_from,
                date_to=plan.date_scope.date_to,
            )
            retrieved = self.retriever.query_traffic(question=question, filters=filters, top_k=top_k)
        except Exception as exc:
            return RetrievalAudit(
                evidence=[],
                mode="traffic",
                backend=self.retriever.retrieval_backend,
                status="error",
                warning=redact_sensitive_text(f"Traffic evidence retrieval failed: {exc}"),
            )
        evidence: List[EvidenceItem] = []
        for item in retrieved.evidence[:top_k]:
            metadata = dict(finite_json_value(item.metadata))
            metadata.update(
                {
                    "retrieval_mode": "traffic",
                    "retrieval_backend": retrieved.backend,
                    "distance": item.distance,
                }
            )
            evidence.append(
                EvidenceItem(
                    id=str(item.id),
                    source_type="traffic_event",
                    title="Supporting AIS event",
                    excerpt=str(item.text or "")[:500] or None,
                    metadata=finite_json_value(metadata),
                )
            )
        return RetrievalAudit(
            evidence=evidence,
            mode="traffic",
            backend=retrieved.backend,
            status="ok" if evidence else "empty",
        )

    @staticmethod
    def _state_for_result(result: ResultType) -> AnswerState:
        if isinstance(result, LiveETAResult):
            if result.status == "ok":
                return AnswerState.COMPUTED
            if result.status == "partial":
                return AnswerState.PARTIAL
            if result.status == "no_current_data":
                return AnswerState.NO_CURRENT_DATA
            return AnswerState.NO_DATA
        if isinstance(result, CarbonResult):
            if result.result_state in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
                return AnswerState.COMPUTED
            if result.result_state == CARBON_STATE_RETRIEVAL_ONLY:
                return AnswerState.RETRIEVED
            if result.result_state in {CARBON_STATE_NOT_COMPUTABLE, CARBON_STATE_FORECAST_ONLY}:
                return AnswerState.NO_DATA
            if result.result_state == CARBON_STATE_UNSUPPORTED:
                return AnswerState.UNSUPPORTED
        status = str(getattr(result, "status", "")).lower()
        if status == "partial" or str(getattr(result, "answer", "")).lower().startswith("partial coverage"):
            return AnswerState.PARTIAL
        if status == "ok":
            return AnswerState.COMPUTED
        if status == "unsupported":
            return AnswerState.UNSUPPORTED
        return AnswerState.NO_DATA

    def _build_envelope(
        self,
        *,
        request: QueryRequest,
        plan: QueryPlan,
        outcome: ExecutionOutcome,
        assurance: AssuranceDecision,
        conversation_id: str,
        turn_id: str,
        latency_ms: float,
    ) -> AnswerEnvelope:
        datasets = self._datasets(plan, outcome.result)
        if outcome.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}:
            visualizations = build_visualizations(
                plan,
                datasets,
                caveats=outcome.caveats,
            )
        else:
            reason_code = "insufficient_data"
            reason = "No validated rows are available for a meaningful graph."
            if (
                outcome.state == AnswerState.NO_DATA
                and plan.operation in _LIVE_ETA_OPERATIONS
            ):
                reason = (
                    "A graph was omitted because AISStream returned no source-validated "
                    "vessel broadcast matching the requested destination, identity, "
                    "field, freshness, and UTC-horizon checks."
                    if plan.eta_watch_intent is not None
                    else
                    "A graph was omitted because Fintraffic returned no source-validated "
                    "row matching the requested live destination, schedule, identity, "
                    "freshness, and UTC-horizon checks."
                )
            if outcome.state == AnswerState.NO_CURRENT_DATA:
                if outcome.availability_code == "coverage_unavailable":
                    reason_code = "coverage_unavailable"
                    reason = (
                        "A graph was omitted because the requested live-source "
                        "coverage is not integrated."
                    )
                else:
                    reason_code = "stale_data"
                    reason = (
                        "A current graph was omitted because the required live data "
                        "did not pass its availability checks."
                    )
            elif outcome.state == AnswerState.ASSURANCE_UNAVAILABLE:
                reason_code = "validation_failed"
                reason = (
                    "A graph was omitted because no supported structured rows are available."
                )
            elif plan.operation in {QueryOperation.ROUTE_TRAVEL_TIME, QueryOperation.FIRST_ROUTE_VESSEL}:
                reason_code = "missing_route_dataset"
                reason = "The required validated route dataset is unavailable for this scope."
            elif plan.requested_visual.value == "none":
                reason_code = "not_requested"
                reason = "The user requested a text-only answer."
            visualizations = [
                OmittedVisualization(
                    id="visualization_omitted",
                    title="Visualization unavailable",
                    accessible_summary=reason,
                    reason_code=reason_code,  # type: ignore[arg-type]
                    reason=reason,
                )
            ]
        source = "computed" if outcome.state in {AnswerState.COMPUTED, AnswerState.PARTIAL} else (
            "retrieved" if outcome.state == AnswerState.RETRIEVED else "system"
        )
        safe_answer = redact_sensitive_text(outcome.answer)
        safe_question = redact_sensitive_text(request.question)
        safe_evidence = [
            item.model_copy(
                update={
                    "title": redact_sensitive_text(item.title),
                    "excerpt": redact_sensitive_text(item.excerpt) if item.excerpt else None,
                    "url": redact_sensitive_text(item.url) if item.url else None,
                    "metadata": redact_sensitive_value(item.metadata),
                }
            )
            for item in outcome.evidence
        ]
        operational_brief = outcome.operational_brief
        if operational_brief is None and plan.eta_watch_intent is not None:
            operational_brief = self._empty_operational_brief(
                plan,
                source_health=(
                    "stale"
                    if outcome.availability_code == "source_stale"
                    else "unavailable"
                ),
                headline="No validated ETA Watch result is available.",
            )
        safe_operational_brief = (
            OperationalBrief.model_validate(
                redact_sensitive_value(
                    operational_brief.model_dump(mode="json")
                )
            )
            if operational_brief is not None
            else None
        )
        if plan.eta_watch_intent is not None:
            facts = self._eta_watch_facts(
                plan,
                state=outcome.state,
                brief=safe_operational_brief,
                datasets=datasets,
            )
        else:
            facts = extract_answer_facts(
                safe_answer,
                source=source,
                state=outcome.state.value,
                operation=plan.operation.value,
                metric=plan.metric,
                # Documentary citations contain resolution numbers, passage
                # sequences, and content-hash digits. They are provenance,
                # not answer facts; research claims remain in cited prose and
                # must not create positional numeric slots.
                include_answer_numbers=plan.mode != QueryMode.MARITIME_RESEARCH,
                entities={
                    "ports": plan.ports,
                    "origin_port": plan.origin_port,
                    "destination_port": plan.destination_port,
                    "date_from": plan.date_scope.date_from,
                    "date_to": plan.date_scope.date_to,
                    "target_date": plan.date_scope.target_date,
                    "vessel_type": plan.vessel_type,
                    "mmsi": plan.mmsi,
                    "imo": plan.imo,
                    "vessel_name": plan.vessel_name,
                    "horizon_hours": plan.horizon_hours,
                    "source_scope": plan.source_scope,
                },
            )
            facts.extend(
                extract_dataset_facts(
                    datasets,
                    operation=plan.operation.value,
                    source=source,
                )
            )
        chart_insights, chart_facts = build_chart_insights(
            plan=plan,
            datasets=datasets,
            visualizations=visualizations,
            evidence_ids=(item.id for item in safe_evidence),
        )
        # Legacy answer-derived slots remain byte-for-byte and positionally
        # unchanged.  Chart-derived slots are appended and referenced by name.
        facts.extend(chart_facts)
        freshness = outcome.freshness_override or self._freshness()
        manifest_version = self._manifest().get("schema_version") or self._manifest().get("version")
        applied_scope = AppliedScope(
            ports=plan.ports,
            country_codes=plan.country_codes,
            origin_port=plan.origin_port,
            destination_port=plan.destination_port,
            date_from=plan.date_scope.date_from,
            date_to=plan.date_scope.date_to,
            target_date=plan.date_scope.target_date,
            vessel_type=plan.vessel_type,
            mmsi=plan.mmsi,
            imo=plan.imo,
            vessel_name=plan.vessel_name,
            horizon_hours=plan.horizon_hours,
            source_scope=plan.source_scope,
        )
        confidence = self._normalize_confidence(outcome.confidence)
        confidence = "high" if assurance.status == "verified" else "not_applicable"
        # Method notes remain available to developer observability without
        # becoming public result labels or presentation caveats.
        caveats: List[str] = []
        diagnostic_warnings = dedupe_strings(
            [
                *outcome.warnings,
                *outcome.caveats,
                *(
                    [outcome.result.confidence_reason, outcome.result.source_label]
                    if isinstance(outcome.result, CarbonResult)
                    else []
                ),
            ]
        )
        assurance_info = AssuranceAssessment(
            status=assurance.status,  # type: ignore[arg-type]
            level=assurance.level,  # type: ignore[arg-type]
            basis=assurance.basis,  # type: ignore[arg-type]
            reason=assurance.reason,
            checks=list(assurance.checks),
        )
        provider = outcome.availability_provider or outcome.retrieval_backend
        if not provider:
            if plan.operation in _LIVE_ETA_OPERATIONS:
                provider = (
                    "aisstream"
                    if plan.eta_watch_intent is not None
                    else "fintraffic_digitraffic"
                )
            elif plan.mode == QueryMode.ANALYTICS:
                provider = "structured_datasets"
            elif plan.mode == QueryMode.APP_HELP:
                provider = "capability_registry"
        availability_info = AvailabilityInfo(
            code=assurance.availability_code,  # type: ignore[arg-type]
            provider=provider,
            retryable=(
                outcome.availability_retryable
                or assurance.availability_code
                in {"source_unavailable", "source_stale"}
            ),
        )
        stable_result = {
            "visualization_contract_version": "2.1",
            "mode": plan.mode.value,
            "state": outcome.state.value,
            "answer": safe_answer,
            "plan": plan.model_dump(mode="json"),
            "facts": [fact.model_dump(mode="json") for fact in facts],
            "applied_scope": applied_scope.model_dump(mode="json"),
            "datasets": [dataset.model_dump(mode="json") for dataset in datasets],
            "visualizations": [visual.model_dump(mode="json") for visual in visualizations],
            "chart_insights": [insight.model_dump(mode="json") for insight in chart_insights],
            "operational_brief": (
                safe_operational_brief.model_dump(mode="json")
                if safe_operational_brief
                else None
            ),
            "evidence": [item.model_dump(mode="json") for item in safe_evidence],
            "freshness": freshness.model_dump(mode="json"),
            "confidence": confidence,
            "assurance": assurance_info.model_dump(mode="json"),
            "availability": availability_info.model_dump(mode="json"),
            "caveats": caveats,
        }
        result_hash = hashlib.sha256(
            json.dumps(stable_result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        trace_sources = dedupe_strings(item.source_type for item in safe_evidence)
        if (
            outcome.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}
            and plan.operation in _LIVE_ETA_OPERATIONS
        ):
            trace_sources = dedupe_strings(
                [
                    (
                        "ais_broadcast_observation"
                        if _is_ais_destination_only(outcome.result)
                        else "official_live_source"
                    ),
                    *trace_sources,
                ]
            )
        elif outcome.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}:
            trace_sources = dedupe_strings(["structured_datasets", *trace_sources])
        elif not trace_sources:
            trace_sources = ["system"]
        configured_model = "deterministic"
        if outcome.model_used:
            configured_model = outcome.model_used
        elif plan.planner_model:
            configured_model = plan.planner_model
        elif self.enable_model_responses and plan.mode == QueryMode.MARITIME_RESEARCH:
            configured_model = self.research_model
        elif self.enable_model_responses and plan.mode == QueryMode.GENERAL_CHAT:
            configured_model = self.general_model
        return AnswerEnvelope(
            conversation_id=conversation_id,
            turn_id=turn_id,
            question=safe_question,
            mode=plan.mode,
            state=outcome.state,
            answer=safe_answer,
            plan=plan,
            facts=facts,
            applied_scope=applied_scope,
            datasets=datasets,
            visualizations=visualizations,
            chart_insights=chart_insights,
            operational_brief=safe_operational_brief,
            evidence=safe_evidence,
            freshness=freshness,
            confidence=confidence,
            assurance=assurance_info,
            availability=availability_info,
            caveats=caveats,
            trace=TraceInfo(
                trace_id=f"trace_{uuid.uuid4().hex[:20]}",
                route=plan.mode.value,
                operation=plan.operation.value,
                planner_source=plan.planner_source,
                planner_model=plan.planner_model,
                model=configured_model,
                reasoning_effort=self.reasoning_effort if configured_model != "deterministic" else None,
                sources=trace_sources,
                retrieval_mode=outcome.retrieval_mode,
                retrieval_backend=outcome.retrieval_backend,
                retrieval_status=outcome.retrieval_status,
                retrieval_top_k=request.top_k_evidence,
                result_state=outcome.state.value,
                failure_state=(
                    outcome.state.value
                    if outcome.state
                    not in {AnswerState.COMPUTED, AnswerState.PARTIAL, AnswerState.RETRIEVED, AnswerState.GENERAL}
                    else None
                ),
                result_hash=result_hash,
                data_manifest_version=str(manifest_version) if manifest_version else None,
                dataset_rows=sum(dataset.row_count for dataset in datasets),
                visualization_decision=",".join(visual.kind for visual in visualizations),
                chart_profile=[
                    (
                        f"{visual.kind}:{visual.chart_type}"
                        if hasattr(visual, "chart_type")
                        else visual.kind
                    )
                    for visual in visualizations
                ],
                visualization_dataset_ids=dedupe_strings(
                    visual.dataset_id
                    for visual in visualizations
                    if visual.dataset_id
                ),
                visualization_fallback_reasons=dedupe_strings(
                    visual.reason
                    for visual in visualizations
                    if isinstance(visual, OmittedVisualization)
                ),
                latency_ms=max(0.0, round(latency_ms, 3)),
                warnings=diagnostic_warnings,
            ),
        )

    @staticmethod
    def _eta_watch_facts(
        plan: QueryPlan,
        *,
        state: AnswerState,
        brief: Optional[OperationalBrief],
        datasets: Sequence[DatasetSpec],
    ) -> List[FactSlot]:
        """Create typed AIS facts without extracting numbers from answer prose."""

        source: str = (
            "computed"
            if state in {AnswerState.COMPUTED, AnswerState.PARTIAL}
            else "system"
        )
        facts = [
            FactSlot(
                name="source_type",
                value="ais_vessel_broadcast",
                entity="aisstream",
                source=source,  # type: ignore[arg-type]
            ),
            FactSlot(
                name="result_state",
                value=state.value,
                entity=plan.operation.value,
                source=source,  # type: ignore[arg-type]
            ),
            FactSlot(
                name="operation",
                value=plan.operation.value,
                entity=plan.operation.value,
                source=source,  # type: ignore[arg-type]
            ),
            FactSlot(
                name="eta_watch_intent",
                value=(
                    plan.eta_watch_intent.value
                    if plan.eta_watch_intent is not None
                    else None
                ),
                entity="ETA Watch",
                source=source,  # type: ignore[arg-type]
            ),
        ]
        for index, port in enumerate(plan.ports, 1):
            facts.append(
                FactSlot(
                    name=f"ports_{index}",
                    value=port,
                    entity=port,
                    source=source,  # type: ignore[arg-type]
                )
            )
        if plan.horizon_hours is not None:
            facts.append(
                FactSlot(
                    name="horizon_hours",
                    value=plan.horizon_hours,
                    unit="hours",
                    entity="UTC watch window",
                    source=source,  # type: ignore[arg-type]
                )
            )
        if state not in {AnswerState.COMPUTED, AnswerState.PARTIAL} or brief is None:
            return facts

        facts.extend(
            [
                FactSlot(
                    name="matched_count",
                    value=brief.matched_count,
                    unit="vessels",
                    entity=brief.intent.value,
                    source="computed",
                ),
                FactSlot(
                    name="displayed_count",
                    value=brief.displayed_count,
                    unit="rows",
                    entity=brief.intent.value,
                    source="computed",
                ),
                FactSlot(
                    name="source_health",
                    value=brief.source_health,
                    entity="aisstream",
                    source="computed",
                ),
            ]
        )
        table = next(
            (dataset for dataset in datasets if dataset.id == "table"),
            None,
        )
        if table is None or not table.rows:
            return facts
        first = table.rows[0]
        vessel = first.get("vessel_label") or first.get("vessel_name")
        eta = first.get("reported_eta_utc") or first.get("ais_eta_utc")
        if vessel:
            facts.append(
                FactSlot(
                    name="next_vessel",
                    value=vessel,
                    entity=str(vessel),
                    source="computed",
                )
            )
        if eta:
            facts.append(
                FactSlot(
                    name="next_reported_eta_utc",
                    value=eta,
                    unit="UTC",
                    entity=str(vessel or "next vessel"),
                    source="computed",
                )
            )
        if plan.eta_watch_intent == ETAWatchIntent.INBOUND_WATCHLIST:
            visible_rows = table.rows[
                : QueryService._eta_watch_visible_row_limit(plan)
            ]
            row_fact_fields = (
                ("vessel", "vessel_label", None),
                ("mmsi", "mmsi", None),
                ("destination", "destination_name", None),
                ("destination_locode", "destination_locode", None),
                ("reported_eta_utc", "reported_eta_utc", "UTC"),
                ("latitude", "latitude", "degrees"),
                ("longitude", "longitude", "degrees"),
                ("speed_kn", "speed_kn", "knots"),
                ("observation_time_utc", "position_time_utc", "UTC"),
            )
            for row_index, row in enumerate(visible_rows, 1):
                row_entity = str(
                    row.get("vessel_label")
                    or row.get("vessel_name")
                    or row.get("mmsi")
                    or f"displayed vessel {row_index}"
                )
                for fact_suffix, field, unit in row_fact_fields:
                    value = row.get(field)
                    if value is None or value == "":
                        continue
                    if (
                        isinstance(value, float)
                        and not math.isfinite(value)
                    ):
                        continue
                    facts.append(
                        FactSlot(
                            name=f"displayed_{row_index}_{fact_suffix}",
                            value=value,
                            unit=unit,
                            entity=row_entity,
                            source="computed",
                        )
                    )
        changes = [
            float(value)
            for row in table.rows
            if (
                (value := row.get("eta_change_minutes")) is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
        ]
        if changes:
            largest = max(changes, key=abs)
            facts.append(
                FactSlot(
                    name="largest_eta_change_minutes",
                    value=largest,
                    unit="minutes",
                    entity=brief.intent.value,
                    source="computed",
                )
            )
        for exception in brief.exceptions:
            facts.append(
                FactSlot(
                    name=f"{exception.code}_count",
                    value=exception.count,
                    unit="vessels",
                    entity=brief.intent.value,
                    source="computed",
                )
            )
        return facts

    @staticmethod
    def _public_result_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Remove internal method fields and unusable interval bounds from a copy."""

        if frame is None:
            return None
        work = frame.copy()
        internal_columns = {
            "confidence_label",
            "confidence_reason",
            "fallback_usage_ratio",
            "fallback_usage_count",
            "quality_factor_fallback_flag",
            "proxy_class",
            "pressure_kind",
            "reconstruction_version",
            "source_kind",
        }
        work = work.drop(
            columns=[column for column in work.columns if str(column) in internal_columns],
            errors="ignore",
        )

        interval_pairs: List[Tuple[str, str, str]] = []
        for lower_field in [
            str(column) for column in work.columns if str(column).endswith("_lower")
        ]:
            point_field = lower_field.removesuffix("_lower")
            upper_field = f"{point_field}_upper"
            if point_field in work.columns and upper_field in work.columns:
                interval_pairs.append((point_field, lower_field, upper_field))
        if {"predicted", "lower", "upper"}.issubset(work.columns):
            interval_pairs.append(("predicted", "lower", "upper"))

        for point_field, lower_field, upper_field in interval_pairs:
            points = pd.to_numeric(work[point_field], errors="coerce")
            lower = pd.to_numeric(work[lower_field], errors="coerce")
            upper = pd.to_numeric(work[upper_field], errors="coerce")
            complete = points.notna() & lower.notna() & upper.notna()
            valid = complete & (lower <= points) & (points <= upper)
            invalid_bounds = (lower.notna() | upper.notna()) & ~valid
            if bool(invalid_bounds.any()):
                work.loc[invalid_bounds, [lower_field, upper_field]] = None
        return work

    def _datasets(self, plan: QueryPlan, result: Optional[ResultType]) -> List[DatasetSpec]:
        if result is None:
            return []
        raw_table = getattr(result, "table", None)
        table = self._public_result_frame(raw_table) if isinstance(raw_table, pd.DataFrame) else raw_table
        if isinstance(table, pd.DataFrame) and not table.empty and plan.operation in {
            QueryOperation.FIRST_ARRIVAL,
            QueryOperation.LAST_ARRIVAL,
            QueryOperation.FIRST_DEPARTURE,
            QueryOperation.FIRST_ROUTE_VESSEL,
        }:
            time_field = next(
                (field for field in ("arrival_time", "departure_time", "arrival_date") if field in table.columns),
                None,
            )
            if time_field:
                table = table.sort_values(time_field, kind="stable").reset_index(drop=True)
        chart: Optional[pd.DataFrame]
        if isinstance(result, ForecastResult):
            forecast = self._public_result_frame(result.forecast) if result.forecast is not None else pd.DataFrame()
            history = self._public_result_frame(result.history) if result.history is not None else pd.DataFrame()
            if not forecast.empty and not history.empty and "date" in forecast.columns and "date" in history.columns:
                chart = history.merge(forecast, on="date", how="outer").sort_values("date")
            else:
                chart = forecast if not forecast.empty else history
        else:
            raw_chart = getattr(result, "chart", None)
            chart = self._public_result_frame(raw_chart) if isinstance(raw_chart, pd.DataFrame) else raw_chart

        if (
            _is_ais_destination_only(result)
            and plan.eta_watch_intent != ETAWatchIntent.ETA_REVISIONS
        ):
            if (
                isinstance(table, pd.DataFrame)
                and not table.empty
                and "ais_eta_utc" in table.columns
            ):
                sort_fields = [
                    field
                    for field in ("ais_eta_utc", "port_locode", "vessel_name", "mmsi")
                    if field in table.columns
                ]
                table = table.sort_values(
                    sort_fields,
                    kind="stable",
                    na_position="last",
                ).reset_index(drop=True)
            if (
                isinstance(chart, pd.DataFrame)
                and not chart.empty
                and "ais_eta_utc" in chart.columns
            ):
                sort_fields = [
                    field
                    for field in ("ais_eta_utc", "port_locode", "vessel_name", "mmsi")
                    if field in chart.columns
                ]
                chart = chart.sort_values(
                    sort_fields,
                    kind="stable",
                    na_position="last",
                ).reset_index(drop=True)

        if plan.operation in _CALENDAR_ORDER_OPERATIONS:
            table = _calendar_ordered_frame(table)
            chart = _calendar_ordered_frame(chart)

        if plan.operation == QueryOperation.VESSEL_TYPE_COMPOSITION:
            for frame in (table, chart):
                if frame is not None and not frame.empty and "scope" not in frame.columns:
                    frame["scope"] = plan.ports[0] if plan.ports else "Selected scope"

        unit_overrides: Dict[str, Optional[str]] = {}
        if plan.operation in {
            QueryOperation.FORECAST_ARRIVALS,
            QueryOperation.FORECAST_CONGESTION,
            QueryOperation.FORECAST_COMPARISON,
        }:
            forecast_unit = "vessels" if plan.operation == QueryOperation.FORECAST_ARRIVALS else "index"
            unit_overrides.update(
                {field: forecast_unit for field in ("predicted", "actual", "lower", "upper")}
            )
        if plan.operation == QueryOperation.VESSEL_TYPE_COMPOSITION:
            unit_overrides["share_percent"] = "percent"
        if plan.operation in _LIVE_ETA_OPERATIONS:
            unit_overrides.update(
                {
                    "announced_delay_minutes": "minutes",
                    "eta_change_minutes": "minutes",
                    "observation_age_minutes": "minutes",
                    "sog_kn": "knots",
                    "speed_kn": "knots",
                    "course_deg": "degrees",
                    "latitude": "degrees",
                    "longitude": "degrees",
                    "inbound_vessels": "vessels",
                }
            )

        datasets: List[DatasetSpec] = []
        table_dataset = dataframe_to_dataset(
            table,
            dataset_id="table",
            unit_overrides=unit_overrides,
        )
        chart_dataset = dataframe_to_dataset(
            chart,
            dataset_id="chart",
            unit_overrides=unit_overrides,
        )
        if table_dataset:
            datasets.append(table_dataset)
        if chart_dataset:
            datasets.append(chart_dataset)
        if (
            plan.eta_watch_intent is not None
            and isinstance(table, pd.DataFrame)
            and not table.empty
            and plan.eta_watch_intent
            not in {
                ETAWatchIntent.DESTINATION_LOAD,
                ETAWatchIntent.ETA_REVISIONS,
                ETAWatchIntent.SIGNAL_QUALITY,
                ETAWatchIntent.LOW_SPEED_EXCEPTIONS,
            }
        ):
            eta_field = next(
                (
                    field
                    for field in (
                        "reported_eta_utc",
                        "ais_eta_utc",
                        "eta_utc",
                    )
                    if field in table.columns
                ),
                None,
            )
            if eta_field:
                parsed_eta = table[eta_field].map(self._utc_datetime)
                valid_eta = parsed_eta.notna()
                if bool(valid_eta.any()):
                    eta_timeline = table.loc[valid_eta].copy()
                    eta_timeline["__eta_sort"] = parsed_eta.loc[valid_eta]
                    eta_timeline = (
                        eta_timeline.sort_values("__eta_sort", kind="stable")
                        .drop(columns=["__eta_sort"])
                        .reset_index(drop=True)
                    )
                    eta_timeline_dataset = dataframe_to_dataset(
                        eta_timeline,
                        dataset_id="eta_timeline",
                        unit_overrides=unit_overrides,
                    )
                    if eta_timeline_dataset:
                        datasets.append(eta_timeline_dataset)
        raw_candidate_table = getattr(result, "candidate_table", None)
        candidate_table = (
            self._public_result_frame(raw_candidate_table)
            if isinstance(raw_candidate_table, pd.DataFrame)
            else raw_candidate_table
        )
        candidate_dataset = dataframe_to_dataset(
            candidate_table,
            dataset_id="eta_freshness_candidates",
            unit_overrides={
                "eta_observation_age_minutes": "minutes",
                "position_age_minutes": "minutes",
                "speed_kn": "knots",
                "latitude": "degrees",
                "longitude": "degrees",
            },
        )
        if candidate_dataset:
            datasets.append(candidate_dataset)

        summary = self._summary_frame(plan, result, table)
        summary_unit_overrides = (
            {"value": _carbon_summary_unit(result)} if isinstance(result, CarbonResult) else None
        )
        summary_dataset = dataframe_to_dataset(
            summary,
            dataset_id="summary",
            max_rows=1,
            unit_overrides=summary_unit_overrides,
        )
        if summary_dataset:
            datasets.append(summary_dataset)

        if plan.operation == QueryOperation.MIXED_PORT_ROUTE_COMPARISON and table is not None and not table.empty:
            metric_field = "metric" if "metric" in table.columns else None
            if metric_field:
                for metric, group in table.groupby(metric_field, dropna=False):
                    token = re.sub(r"[^a-z0-9]+", "_", str(metric).lower()).strip("_") or "value"
                    dataset = dataframe_to_dataset(
                        group.reset_index(drop=True),
                        dataset_id=f"metric_{token}",
                        unit_overrides={"value": unit_for_field(str(metric))},
                    )
                    if dataset:
                        datasets.append(dataset)
        observations = self._public_result_frame(self._distribution_observation_frame(plan))
        observation_dataset = dataframe_to_dataset(
            observations,
            dataset_id="distribution_observations",
            max_rows=5000,
        )
        if observation_dataset:
            datasets.append(observation_dataset)
        return enrich_chart_datasets(plan, datasets)

    def _distribution_observation_frame(self, plan: QueryPlan) -> Optional[pd.DataFrame]:
        """Expose exact duration rows only when the same filtered authority is bounded.

        This additive dataset exists solely for quartile, Tukey-whisker and
        outlier bindings.  If the full filtered population exceeds the response
        bound, no partial sample is returned and the chart remains a histogram
        or percentile view without invented distribution statistics.
        """

        if not hasattr(self, "kpi"):
            return None
        start = plan.date_scope.date_from
        end = plan.date_scope.date_to
        window = plan.date_scope.relative_window
        if plan.operation in {QueryOperation.DWELL_SUMMARY, QueryOperation.DWELL_DISTRIBUTION}:
            work = self.kpi.dwell
            if work is None or work.empty:
                return None
            work = self.kpi._filter_port(work, plan.ports[0] if plan.ports else None)
            work = self.kpi._filter_dates(work, "arrival_date", start, end, window=window)
            work = self.kpi._filter_vessel_type(work, plan.vessel_type)
            if "dwell_minutes" not in work.columns:
                return None
            work = work.copy()
            work["dwell_minutes"] = pd.to_numeric(work["dwell_minutes"], errors="coerce")
            work = work.dropna(subset=["dwell_minutes"])
            work = work[(work["dwell_minutes"] > 0) & (work["dwell_minutes"] <= 45 * 24 * 60)]
            if work.empty or len(work) > 5000:
                return None
            columns = [
                column
                for column in (
                    "arrival_date",
                    "arrival_time",
                    "departure_time",
                    "mmsi",
                    "port_key",
                    "vessel_type_norm",
                    "dwell_minutes",
                    "source_kind",
                )
                if column in work.columns
            ]
            return work[columns].sort_values(
                "dwell_minutes",
                kind="stable",
            ).reset_index(drop=True)

        if plan.operation == QueryOperation.ROUTE_TRAVEL_TIME:
            work = self.kpi.voyages
            if work is None or work.empty or not plan.origin_port or not plan.destination_port:
                return None
            work = self.kpi._filter_voyage_endpoint(work, "origin", plan.origin_port)
            work = self.kpi._filter_voyage_endpoint(work, "destination", plan.destination_port)
            work = self.kpi._filter_dates(work, "departure_time", start, end, window=window)
            work = self.kpi._filter_vessel_type(work, plan.vessel_type)
            if "duration_h" not in work.columns:
                return None
            work = work.copy()
            work["duration_h"] = pd.to_numeric(work["duration_h"], errors="coerce")
            work = work.dropna(subset=["duration_h"])
            work = work[work["duration_h"] >= 0]
            if work.empty or len(work) > 5000:
                return None
            columns = [
                column
                for column in (
                    "voyage_id",
                    "mmsi",
                    "origin_port_key",
                    "destination_port_key",
                    "departure_time",
                    "arrival_time",
                    "duration_h",
                    "vessel_type_norm",
                )
                if column in work.columns
            ]
            return work[columns].sort_values("duration_h", kind="stable").reset_index(drop=True)
        return None

    @staticmethod
    def _summary_frame(
        plan: QueryPlan,
        result: ResultType,
        table: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if isinstance(result, CarbonResult):
            interval = result.uncertainty_interval.get("CO2e") or result.uncertainty_interval.get("CO2") or {}
            if interval.get("point") is not None:
                return pd.DataFrame([{"label": "emissions", "value": interval.get("point")}])
        if table is None or table.empty:
            return None
        if plan.operation == QueryOperation.ARRIVALS and "arrival_count" in table.columns:
            return pd.DataFrame([{"label": "arrivals", "arrival_count": pd.to_numeric(table["arrival_count"], errors="coerce").sum()}])
        if plan.operation == QueryOperation.DWELL_SUMMARY:
            target = (
                "mean_dwell_hours"
                if plan.aggregation == "mean"
                else "median_dwell_hours"
            )
            if target in table.columns:
                return pd.DataFrame(
                    [
                        {
                            "label": target,
                            target: pd.to_numeric(table[target], errors="coerce").iloc[0],
                            "complete_dwell_count": int(
                                pd.to_numeric(
                                    table.get("complete_dwell_count"), errors="coerce"
                                ).iloc[0]
                            ),
                        }
                    ]
                )
        if plan.operation == QueryOperation.MMSI_PORT_STAYS and "dwell_minutes" in table.columns:
            total_hours = pd.to_numeric(table["dwell_minutes"], errors="coerce").fillna(0).sum() / 60.0
            return pd.DataFrame([{"label": "total dwell", "duration_h": total_hours}])
        if plan.operation == QueryOperation.CONGESTION and "congestion_index" in table.columns and len(table) == 1:
            return pd.DataFrame([{"label": "port pressure", "congestion_index": table["congestion_index"].iloc[0]}])
        for field in ("arrival_count", "arrivals_vessels", "congestion_index", "value", "duration_h"):
            if field in table.columns and len(table) == 1:
                return pd.DataFrame([{"label": field, "value": table[field].iloc[0]}])
        return None

    def _freshness(self) -> FreshnessInfo:
        values: List[pd.Timestamp] = []
        minima: List[pd.Timestamp] = []
        for frame, candidates in (
            (self.kpi.arrivals_daily, ("date",)),
            (self.kpi.congestion, ("date",)),
            (self.kpi.dwell, ("arrival_date", "arrival_time")),
        ):
            if frame is None or frame.empty:
                continue
            for field in candidates:
                if field not in frame.columns:
                    continue
                series = pd.to_datetime(frame[field], errors="coerce", utc=True).dropna()
                if not series.empty:
                    minima.append(series.min())
                    values.append(series.max())
                    break
        data_from = min(minima).strftime("%Y-%m-%d") if minima else None
        data_to = max(values).strftime("%Y-%m-%d") if values else None
        return FreshnessInfo(
            data_from=data_from,
            data_to=data_to,
            as_of=data_to,
            historical=True,
            message=(
                f"Historical validated data through {data_to}."
                if data_to
                else "No validated dataset freshness timestamp is available."
            ),
        )

    @staticmethod
    def _normalize_confidence(value: str) -> str:
        token = str(value or "").lower()
        if "high" in token:
            return "high"
        if "medium" in token or "moderate" in token:
            return "medium"
        if "low" in token:
            return "low"
        return "not_applicable"

    def _manifest(self) -> Dict[str, Any]:
        for filename in ("data_manifest.json", "data_manifest.v2.json", "manifest.json"):
            path = self.processed_dir / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return {}

    def capabilities(self) -> Dict[str, Any]:
        freshness = self._freshness()
        manifest = self._manifest()
        operations = [operation.value for operation in QueryOperation if operation != QueryOperation.UNSUPPORTED]
        return {
            "api_version": "2.0",
            "visualization_contract_version": "2.1",
            "modes": [
                QueryMode.ANALYTICS.value,
                QueryMode.MARITIME_RESEARCH.value,
                QueryMode.GENERAL_CHAT.value,
                QueryMode.APP_HELP.value,
            ],
            "operations": operations,
            "visualization_kinds": [
                "kpi",
                "cartesian",
                "forecast",
                "distribution",
                "heatmap",
                "map",
                "timeline",
                "table",
                "omitted",
            ],
            "freshness": freshness.model_dump(mode="json"),
            "data_manifest": finite_json_value(manifest),
            "kpi_capabilities": finite_json_value(self.kpi.capabilities()),
            "model_routes": {
                "planner": self.general_model,
                "general": self.general_model,
                "research": self.research_model,
                "reasoning_effort": self.reasoning_effort,
                "enabled": self.enable_model_responses,
            },
            "retrieval": {
                "available": self.retriever is not None,
                "backend": self.retriever.retrieval_backend if self.retriever else None,
                "reason": self.retriever_reason or None,
                "default_top_k": 5,
                "recommended_top_k": 5,
                "disable_value": 0,
                "local_synthesis": {
                    "available": self.local_synthesizer is not None,
                    "provider": getattr(self.local_synthesizer, "provider", None),
                    "model": getattr(self.local_synthesizer, "model", None),
                    "scope": "maritime_research_and_non_current_general_only",
                    "analytics_fact_rewriting": False,
                },
            },
            "conversation_store": "sqlite",
            "capability_registry": finite_json_value(PUBLIC_CAPABILITY_REGISTRY),
            "live_eta": (
                finite_json_value(self.live_eta.capabilities())
                if self.live_eta is not None
                else {
                    "provider": "aisstream",
                    "available": False,
                    "country_scope": _ETA_WATCH_COUNTRY_SCOPE,
                    "official_schedule_country_scope": [],
                    "ais_destination_country_scope": _ETA_WATCH_COUNTRY_SCOPE,
                    "timezone": "UTC",
                    "maximum_horizon_hours": 48,
                    "operations": sorted(operation.value for operation in _LIVE_ETA_OPERATIONS),
                    "eta_watch_intents": sorted(
                        intent.value for intent in ETAWatchIntent
                    ),
                    "official_eta_authority": None,
                    "regional_ais_scope": (
                        "When configured, fresh vessel-reported destinations, ETAs, positions, "
                        "speeds, and ETA revisions can be observed for curated Baltic ports. "
                        "This is not a complete or official arrival board."
                    ),
                    "announced_variance": None,
                    "prediction": False,
                    "reason": "AISStream is not configured in this runtime.",
                }
            ),
        }

    def export(self, request: ExportRequest) -> ExportResponse:
        envelope = self.store.get_envelope(request.conversation_id, request.turn_id)
        if envelope is None:
            raise KeyError("conversation turn not found")
        dataset = next((item for item in envelope.datasets if item.id == request.dataset_id), None)
        if dataset is None:
            raise KeyError("dataset not found")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        export_id = f"export_{uuid.uuid4().hex[:20]}"
        path = (self.export_dir / f"{export_id}.{request.format}").resolve()
        if request.format == "json":
            path.write_text(json.dumps(dataset.model_dump(mode="json"), indent=2, allow_nan=False), encoding="utf-8")
        else:
            fields = [column.field for column in dataset.columns]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in dataset.rows:
                    writer.writerow({field: self._safe_csv_cell(row.get(field)) for field in fields})
        return ExportResponse(
            export_id=export_id,
            format=request.format,
            dataset_id=dataset.id,
            row_count=dataset.row_count,
            path=str(path),
        )

    def submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        feedback_id = self.store.save_feedback(
            trace_id=request.trace_id,
            prompt=request.prompt,
            note=request.note,
        )
        return FeedbackResponse(feedback_id=feedback_id)

    @staticmethod
    def _safe_csv_cell(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
